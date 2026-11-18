// src/kernels/cuda/p100_mmvq_f16.cu - p100k: FP16 mat-vec for Q5_K / Q6_K on the Tesla P100 (compute capability 6.0).
//
// The kernels are p100k llama.cpp's ggml/src/ggml-cuda/mmvq-f16.cu (MIT; ours, github.com/willtejeda/llama.cpp branch
// p100k), copied unchanged except for the small shim below. sm_60 has no DP4A, so the q8_1 integer mat-vec is
// ALU-bound there (emulated dot: four VMAD per four multiply-adds); this path converts each weight pair to FP16 once
// and does one HFMA2 per two multiply-adds, with the activations prescaled to FP16 by a power of two per 1024-value
// window. Measured in llama.cpp on these cards: faster than the integer path at 2..12 columns (a speculative verify)
// and closer to an FP32 reference (NMSE ~4.6e-7 vs ~1.3e-4 for q8_1 activations).
// Strata use: verify-path dense projections (core/verify.cpp), STRATA_P100_MMVQ_F16=0 turns it off.
#include "strata/kernels/p100_mmvq_f16.hpp"

#if defined(__HIPCC__) || defined(STRATA_USE_HIP)
namespace strata::kernels {   // HIP: no P100 path
bool p100_mmvq_f16_on() { return false; }
std::size_t p100_mmvq_f16_scratch_bytes(int, int) { return 0; }
bool p100_mmvq_f16_prep(const float*, int, int, void*, std::size_t, void*) { return false; }
bool p100_mmvq_f16(int, const void*, float*, int, int, int, const void*, void*) { return false; }
}
#else

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <type_traits>
#include <utility>

namespace {
// ---- shim for the llama.cpp code below
#define WARP_SIZE 32
#define GGML_ABORT(...) (std::fprintf(stderr, "p100_mmvq_f16: " __VA_ARGS__), std::fputc('\n', stderr), std::abort())
#define CUDA_CHECK(x) do { const cudaError_t e_ = (x); if (e_ != cudaSuccess) GGML_ABORT("%s", cudaGetErrorString(e_)); } while (0)
static __device__ __forceinline__ float ggml_cuda_op_silu_single(float x) { return x / (1.0f + expf(-x)); }

#define MMVQ_F16_NW  2   // warps per block
#define MMVQ_F16_RPW 4   // rows per warp
#define MMVQ_F16_NBF 4   // q6_K blocks per window (fold)
static constexpr int MMVQ_F16_WIN = MMVQ_F16_NBF*256;
#define MMVQ_F16_MAXK 12 // widest single launch: wider, the 4-row tile runs out of registers

// Columns are independent (each one's arithmetic never touches another's), so splitting a batch into
// launches over column ranges gives the same results.
template <typename F>
static void mmvq_f16_columns(const int64_t ncols, F && f) {
    const int64_t n0 = ncols <= MMVQ_F16_MAXK ? ncols : (ncols + 1)/2;
    for (int64_t c0 = 0; c0 < ncols; c0 += n0) {
        f(c0, std::min(n0, ncols - c0));
    }
}

template <typename F>
static void mmvq_f16_nc(const int64_t nc, F && f) {
    switch (nc) {
        case 2:  f(std::integral_constant<int, 2>{});  break;
        case 3:  f(std::integral_constant<int, 3>{});  break;
        case 4:  f(std::integral_constant<int, 4>{});  break;
        case 5:  f(std::integral_constant<int, 5>{});  break;
        case 6:  f(std::integral_constant<int, 6>{});  break;
        case 7:  f(std::integral_constant<int, 7>{});  break;
        case 8:  f(std::integral_constant<int, 8>{});  break;
        case 9:  f(std::integral_constant<int, 9>{});  break;
        case 10: f(std::integral_constant<int, 10>{}); break;
        case 11: f(std::integral_constant<int, 11>{}); break;
        case 12: f(std::integral_constant<int, 12>{}); break;
        default: GGML_ABORT("mmvq_f16: %d columns", (int) nc);
    }
}

// MSVC cannot fold a constexpr captured from an enclosing lambda in template-argument position (it
// reads it through the closure), so extra tags go to the callback as parameters.
template <typename F, typename... T>
static void mmvq_f16_nc_fwd(const int64_t nc, F && f, T &&... t) {
    switch (nc) {
        case 1:  f(std::integral_constant<int, 1>{},  t...); break;
        case 2:  f(std::integral_constant<int, 2>{},  t...); break;
        case 3:  f(std::integral_constant<int, 3>{},  t...); break;
        case 4:  f(std::integral_constant<int, 4>{},  t...); break;
        case 5:  f(std::integral_constant<int, 5>{},  t...); break;
        case 6:  f(std::integral_constant<int, 6>{},  t...); break;
        case 7:  f(std::integral_constant<int, 7>{},  t...); break;
        case 8:  f(std::integral_constant<int, 8>{},  t...); break;
        case 9:  f(std::integral_constant<int, 9>{},  t...); break;
        case 10: f(std::integral_constant<int, 10>{}, t...); break;
        case 11: f(std::integral_constant<int, 11>{}, t...); break;
        case 12: f(std::integral_constant<int, 12>{}, t...); break;
        default: GGML_ABORT("mmvq_f16: %d columns", (int) nc);
    }
}

// one warp per (window, column)
static __global__ void mmvq_f16_prep(const float * __restrict__ X, const int64_t sx, __half * __restrict__ XS,
                                     float * __restrict__ S, const int K) {
    const int w = blockIdx.x*blockDim.y + threadIdx.y, c = blockIdx.y, nw = (K + MMVQ_F16_WIN - 1)/MMVQ_F16_WIN;
    if (w >= nw) {
        return;
    }
    const float * x = X + c*sx + (int64_t) w*MMVQ_F16_WIN;
    constexpr int PER = MMVQ_F16_WIN/32;
    const int len = min(MMVQ_F16_WIN, K - w*MMVQ_F16_WIN);
    float v[PER];
    float m = 0.0f;
#pragma unroll
    for (int j = 0; j < PER/4; ++j) {
        float4 f = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        if (j*128 < len) {
            f = __ldg((const float4 *) x + j*32 + threadIdx.x);
        }
        v[4*j] = f.x; v[4*j + 1] = f.y; v[4*j + 2] = f.z; v[4*j + 3] = f.w;
        m = fmaxf(m, fmaxf(fmaxf(fabsf(f.x), fabsf(f.y)), fmaxf(fabsf(f.z), fabsf(f.w))));
    }
#pragma unroll
    for (int o = 16; o; o >>= 1) {
        m = fmaxf(m, __shfl_xor_sync(0xFFFFFFFF, m, o));
    }
    int e = 0;
    if (m > 0.0f) {
        frexpf(m, &e);
    }
    const float inv = ldexpf(1.0f, -e);
    __half * xs = XS + (int64_t) c*K + (int64_t) w*MMVQ_F16_WIN;
#pragma unroll
    for (int j = 0; j < PER/4; ++j) {
        if (j*128 >= len) {
            break;
        }
        const __half2 a = __floats2half2_rn(v[4*j]*inv, v[4*j + 1]*inv), b = __floats2half2_rn(v[4*j + 2]*inv, v[4*j + 3]*inv);
        uint2 u;
        u.x = *(const uint32_t *) &a;
        u.y = *(const uint32_t *) &b;
        ((uint2 *) xs)[j*32 + threadIdx.x] = u;
    }
    if (threadIdx.x == 0) {
        S[c*nw + w] = ldexpf(1.0f, e - 10); // also undoes the 1024 on the weights
    }
}

// 32-bit field at byte offset OFF of block B, relative to a 4-aligned window base
template <int B, int OFF>
static __device__ __forceinline__ uint32_t mmvq_f16_fld(const uint32_t * wb, const int lane_words) {
    constexpr int A = 210*B + OFF;
    if constexpr ((A & 3) == 0) {
        return wb[A/4 + lane_words];
    } else {
        return __byte_perm(wb[A/4 + lane_words], wb[A/4 + 1 + lane_words], 0x5432);
    }
}

template <int NC, int RPW, int NBLK, int B = 0>
static __device__ __forceinline__ void mmvq_f16_blocks(
        const uint32_t * const * wb, const uint2 * const * scs, const __half * const * xw, __half2 (&t)[NC][RPW],
        const int P, const int ql_w, const int qh_w, const int vh_shift, const int sp) {
    if constexpr (B < NBLK) {
        const __half2 k1056 = __float2half2_rn(1056.0f);
        __half2 xa[NC][2], xb[NC][2];
#pragma unroll
        for (int c = 0; c < NC; ++c) {
            const __half * xs = xw[c] + B*256;
            const uint2 a = __ldg((const uint2 *) (xs + P)), bb = __ldg((const uint2 *) (xs + P + 64));
            xa[c][0] = *(const __half2 *) &a.x;  xa[c][1] = *(const __half2 *) &a.y;
            xb[c][0] = *(const __half2 *) &bb.x; xb[c][1] = *(const __half2 *) &bb.y;
        }
#pragma unroll
        for (int i = 0; i < RPW; ++i) {
            const uint32_t vl = mmvq_f16_fld<B, 0>(wb[i], ql_w);
            const uint32_t vh = mmvq_f16_fld<B, 128>(wb[i], qh_w) >> vh_shift;
            // scales so and so+4, precomputed per window by mmvq_f16_scales (the same half ops)
            const uint2 sAB = scs[i][B*8 + sp];
            const __half2 sA = *(const __half2 *) &sAB.x, sB = *(const __half2 *) &sAB.y;
            const uint32_t qa = (vl & 0x0F0F0F0F) | ((vh << 4) & 0x30303030);
            const uint32_t qb = ((vl >> 4) & 0x0F0F0F0F) | (vh & 0x30303030);
            // half(1024 + q) per byte, minus 1056, times the scale
            const uint32_t h0 = __byte_perm(qa, 0x64646464u, 0x5140), h1 = __byte_perm(qa, 0x64646464u, 0x5342);
            const uint32_t h2 = __byte_perm(qb, 0x64646464u, 0x5140), h3 = __byte_perm(qb, 0x64646464u, 0x5342);
            const __half2 w0 = __hmul2(__hsub2(*(const __half2 *) &h0, k1056), sA);
            const __half2 w1 = __hmul2(__hsub2(*(const __half2 *) &h1, k1056), sA);
            const __half2 w2 = __hmul2(__hsub2(*(const __half2 *) &h2, k1056), sB);
            const __half2 w3 = __hmul2(__hsub2(*(const __half2 *) &h3, k1056), sB);
#pragma unroll
            for (int c = 0; c < NC; ++c) {
                __half2 u = B == 0 ? __hmul2(w0, xa[c][0]) : __hfma2(w0, xa[c][0], t[c][i]);
                u = __hfma2(w1, xa[c][1], u);
                u = __hfma2(w2, xb[c][0], u);
                t[c][i] = __hfma2(w3, xb[c][1], u);
            }
        }
        mmvq_f16_blocks<NC, RPW, NBLK, B + 1>(wb, scs, xw, t, P, ql_w, qh_w, vh_shift, sp);
    }
}

// A block's 16 scales, as the (sA, sB) half2 pairs its lanes use: lane l of the warp computes pair
// p = l % 8 (scales j and j + 4, j = p < 4 ? p : p + 4) of block l / 8 of the window, once, instead
// of every lane recomputing its pair for every block (~12 instructions per lane, block and row).
// The half arithmetic is exactly the per-lane version's: sc2 = (half(1152 + sc) - 1152) * (d*1024).
template <int RPW>
static __device__ __forceinline__ void mmvq_f16_scales(const uint32_t * const * wb, uint2 (*scst)[MMVQ_F16_NBF*8],
                                                       const int nblk, const int lane) {
    const int B = lane >> 3, p = lane & 7, j = p < 4 ? p : p + 4;
    if (B >= nblk) {
        return;
    }
    const __half2 k1152 = __float2half2_rn(1152.0f), k1024 = __float2half2_rn(1024.0f);
#pragma unroll
    for (int i = 0; i < RPW; ++i) {
        const uint8_t * bb = (const uint8_t *) wb[i] + 210*B;
        const uint32_t sab = (uint32_t) bb[192 + j] | ((uint32_t) bb[196 + j] << 8);
        const uint32_t sh  = __byte_perm(sab ^ 0x8080u, 0x64646464u, 0x5140); // half(1024 + 128 + sc)
        const uint32_t d16 = *(const uint16_t *) (bb + 208);
        const uint32_t dd  = d16 | (d16 << 16);
        const __half2 d2  = __hmul2(*(const __half2 *) &dd, k1024);
        const __half2 sc2 = __hmul2(__hsub2(*(const __half2 *) &sh, k1152), d2);
        const __half2 sA  = __low2half2(sc2), sB = __high2half2(sc2);
        scst[i][B*8 + p] = make_uint2(*(const uint32_t *) &sA, *(const uint32_t *) &sB);
    }
}

// The gated delta net's gate and beta in one launch (GATE): rows [0, rows1) are alpha's (W, Y),
// giving softplus(alpha + b) * m; the rest are beta's (W2, Y2), giving sigmoid(beta). The epilogues
// are the unfused ADD, SOFTPLUS, MUL and SIGMOID kernels' operations, so results are bit-identical.
struct mmvq_f16_gate {
    const uint8_t * W2;
    float *         Y2;
    const float *   b;
    const float *   m;
    int             rows1;
};

static __device__ __forceinline__ float mmvq_f16_softplus(const float x) {
    return (x > 20.0f) ? x : logf(1.0f + expf(x)); // op_softplus (unary.cu)
}

// GLU: the FFN's gate and up matvecs and the SWIGLU after them in one launch. Each warp takes RPW/2
// output rows: its first RPW/2 row slots walk the gate matrix (W), the rest the same rows of up
// (gate.W2), and the epilogue writes silu(gate) * up with the unfused SWIGLU kernel's expression.
// Every row's arithmetic is the unfused kernel's, so the result is bit-identical.
template <int NC, int RPW, int NWT, bool KS, int MINB = 12/NWT, bool GATE = false, bool GLU = false>
__launch_bounds__(NWT*WARP_SIZE, MINB) // 12/NWT: 168 registers, no spills: 6 blocks per SM (OPTLOG 210)
static __global__ void mmvq_f16_q6_K(const uint8_t * __restrict__ W, const int64_t row_bytes, const __half * __restrict__ XS,
                                     const float * __restrict__ S, float * __restrict__ Y, const int64_t sy,
                                     int rows, const int K, const mmvq_f16_gate gate = {}) {
    static_assert(!GATE || (KS && RPW == 1), "GATE: split-K, one row per block");
    static_assert(!GLU || (!KS && !GATE && RPW % 2 == 0), "GLU: row-parallel, gate/up row pairs");
    constexpr int ORW = GLU ? RPW/2 : RPW; // output rows per warp
    constexpr int RPB = NWT*RPW, WB = MMVQ_F16_NBF*210, NU = (WB + 15 + 15)/16;
    const int lane = threadIdx.x, wid = threadIdx.y;
    const int nb = K/256, nw = (nb + MMVQ_F16_NBF - 1)/MMVQ_F16_NBF;
    // KS (split K): the block's warps share its RPW rows and take every NWT-th window each
    int row0 = KS ? blockIdx.x*RPW : GLU ? (blockIdx.x*NWT + wid)*ORW : blockIdx.x*RPB + wid*RPW;
    bool second = false;
    if constexpr (GATE) {
        if (row0 >= gate.rows1) {
            second = true;
            row0 -= gate.rows1;
            rows -= gate.rows1;
            W = gate.W2;
            Y = gate.Y2;
        } else {
            rows = gate.rows1;
        }
    }

    __shared__ uint4 wst[NWT][RPW][NU];

    // this lane's 8 values of a block: P..P+3 (scale so) and P+64..P+67 (scale so+4)
    const int iqs = lane, P = 128*(iqs/16) + 4*(iqs % 16);
    const int ql_w = iqs, qh_w = 8*(iqs/16) + iqs % 8, vh_shift = 2*((iqs % 16)/8), so = 8*(iqs/16) + (iqs % 16)/4;
    const int sp = so < 8 ? so : so - 4; // this lane's scale pair (so, so + 4), see mmvq_f16_scales
    __shared__ uint2 scst[NWT][RPW][MMVQ_F16_NBF*8];
    const uint2 * scs[RPW];
#pragma unroll
    for (int i = 0; i < RPW; ++i) {
        scs[i] = scst[wid][i];
    }

    float acc[NC][RPW];
#pragma unroll
    for (int c = 0; c < NC; ++c) {
#pragma unroll
        for (int i = 0; i < RPW; ++i) {
            acc[c][i] = 0.0f;
        }
    }
    const char * rp[RPW];
#pragma unroll
    for (int i = 0; i < RPW; ++i) {
        if constexpr (GLU) {
            rp[i] = (const char *) (i < ORW ? W : gate.W2) + (int64_t) min(row0 + i % ORW, rows - 1)*row_bytes;
        } else {
            rp[i] = (const char *) W + (int64_t) min(row0 + i, rows - 1)*row_bytes + (KS ? (int64_t) wid*WB : 0);
        }
    }
    const __half * xw[NC];
#pragma unroll
    for (int c = 0; c < NC; ++c) {
        xw[c] = XS + (int64_t) c*K + (KS ? (int64_t) wid*MMVQ_F16_WIN : 0);
    }

    for (int win = KS ? wid : 0; win < nw; win += KS ? NWT : 1) {
        const int nblk = min(MMVQ_F16_NBF, nb - win*MMVQ_F16_NBF);
        __syncwarp();
        const uint32_t * wb[RPW];
#pragma unroll
        for (int i = 0; i < RPW; ++i) {
            const char * g = rp[i];
            rp[i] += KS ? (int64_t) NWT*WB : WB;
            const int m = (int) ((uintptr_t) g & 15); // a multiple of 4
            wb[i] = (const uint32_t *) wst[wid][i] + (m >> 2);
            const uint4 * g16 = (const uint4 *) (g - m);
            const int nu = (m + nblk*210 + 15)/16;
#pragma unroll
            for (int r = 0; r < (NU + 31)/32; ++r) {
                const int k = r*32 + lane;
                if (k < nu) {
                    wst[wid][i][k] = __ldg(g16 + k);
                }
            }
        }
        __syncwarp();
        mmvq_f16_scales<RPW>(wb, scst[wid], nblk, lane);
        __syncwarp();
        __half2 t[NC][RPW];
        switch (nblk) {
            case 4: mmvq_f16_blocks<NC, RPW, 4>(wb, scs, xw, t, P, ql_w, qh_w, vh_shift, sp); break;
            default: mmvq_f16_blocks<NC, RPW, 2>(wb, scs, xw, t, P, ql_w, qh_w, vh_shift, sp); break; // nb % 4 == 2
        }
#pragma unroll
        for (int c = 0; c < NC; ++c) {
            xw[c] += KS ? NWT*MMVQ_F16_WIN : MMVQ_F16_WIN;
            const float s = __ldg(S + c*nw + win);
#pragma unroll
            for (int i = 0; i < RPW; ++i) {
                const __half2 u = __hadd2(t[c][i], __lowhigh2highlow(t[c][i]));
                acc[c][i] = fmaf(s, __low2float(u), acc[c][i]);
            }
        }
    }
    if constexpr (KS) {
        // per-warp partials, then warp 0 adds them in a fixed order
        __shared__ float red[NWT][NC][RPW];
#pragma unroll
        for (int c = 0; c < NC; ++c) {
#pragma unroll
            for (int i = 0; i < RPW; ++i) {
                float v = acc[c][i];
#pragma unroll
                for (int o = 16; o; o >>= 1) {
                    v += __shfl_xor_sync(0xFFFFFFFF, v, o);
                }
                if (lane == 0) {
                    red[wid][c][i] = v;
                }
            }
        }
        __syncthreads();
        if (wid == 0 && lane < NC*RPW) {
            const int c = lane / RPW, i = lane % RPW;
            float v = 0.0f;
#pragma unroll
            for (int w = 0; w < NWT; ++w) {
                v += red[w][c][i];
            }
            if (row0 + i < rows) {
                if constexpr (GATE) {
                    const int r = row0 + i;
                    v = second ? 1.0f / (1.0f + expf(-v)) : mmvq_f16_softplus(v + gate.b[r]) * gate.m[r];
                }
                Y[c*sy + row0 + i] = v;
            }
        }
        return;
    }
#pragma unroll
    for (int c = 0; c < NC; ++c) {
        float v[RPW];
#pragma unroll
        for (int i = 0; i < RPW; ++i) {
            v[i] = acc[c][i];
#pragma unroll
            for (int o = 16; o; o >>= 1) {
                v[i] += __shfl_xor_sync(0xFFFFFFFF, v[i], o);
            }
        }
#pragma unroll
        for (int i = 0; i < ORW; ++i) {
            if (lane == i && row0 + i < rows) {
                if constexpr (GLU) {
                    Y[c*sy + row0 + i] = ggml_cuda_op_silu_single(v[i]) * v[i + ORW]; // op_swiglu (unary.cu)
                } else {
                    Y[c*sy + row0 + i] = v[i];
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------------
// Every other weight type: one kernel, one small unpack per type.
//
// The q6_K kernel above is scheduled around q6_K's 210-byte block. This one takes any type through a
// per-type unpack of 8 consecutive weights of a block (a "chunk"), written from ggml's reference
// dequantization (ggml-quants.c), and shares the rest: the prescaled fp16 activation and its cache,
// the per-window fp16 chains folded into fp32, the column split. Numerics per weight:
//   q    -> half(1024 + q) by byte permute (exact), minus the bias (exact)
//   w    = (q - bias) * half(d*1024*sc) [+ half(-dmin*1024*m)]: one or two roundings, as on q6_K
// Lane l takes chunks l, l + 32, l + 64, l + 96 of each 1024-value window: 16 HFMA2 per window and
// column, as on q6_K, then the fold into fp32 with the window's activation scale.

static __device__ __forceinline__ uint32_t mmvq_f16_ld2(const uint8_t * p) { // 2-byte aligned
    const uint16_t * q = (const uint16_t *) p;
    return (uint32_t) q[0] | ((uint32_t) q[1] << 16);
}
static __device__ __forceinline__ uint32_t mmvq_f16_ld4(const uint8_t * p) { // 4-byte aligned
    return *(const uint32_t *) p;
}
// 4 bits (bit k of b) -> bit 4 of byte k
static __device__ __forceinline__ uint32_t mmvq_f16_spread4(const uint32_t b) {
    return ((b & 1) << 4) | ((b & 2) << 11) | ((b & 4) << 18) | ((b & 8) << 25);
}
// 4 nibble indices (one per byte) -> kvalues_iq4nl[i] + 128 per byte
static __device__ __forceinline__ uint32_t mmvq_f16_iq4(const uint32_t v) {
    const uint32_t x = v & 0x07070707, y = x | (x >> 4);
    const uint32_t sel = (y & 0xFF) | ((y >> 8) & 0xFF00);
    const uint32_t lo = __byte_perm(0x3f2d1801u, 0x766a5d4fu, sel), hi = __byte_perm(0xa6998d81u, 0xf1d9c5b5u, sel);
    const uint32_t m = ((v >> 3) & 0x01010101) * 0xFF;
    return (lo & ~m) | (hi & m);
}
// 8 bytes at any byte address (2 aligned words + funnel shifts), e.g. an mxfp4 block's qs
static __device__ __forceinline__ void mmvq_f16_ld8u(const uint8_t * p, uint32_t & a0, uint32_t & a1) {
    const uint32_t * q = (const uint32_t *) ((uintptr_t) p & ~(uintptr_t) 3);
    const uint32_t sh = ((uintptr_t) p & 3) * 8;
    const uint32_t w0 = q[0], w1 = q[1], w2 = q[2];
    a0 = __funnelshift_r(w0, w1, sh);
    a1 = __funnelshift_r(w1, w2, sh);
}
// 8 bytes (q per byte, element order) -> 8 weights (q - bias)*s, or (q - 1024 + 1024 - bias)*s + mn
template <bool MIN>
static __device__ __forceinline__ void mmvq_f16_w8(const uint32_t v0, const uint32_t v1, const float bias, const float s, const float mn,
                                                   __half2 (&w)[4]) {
    const __half2 kb = __float2half2_rn(1024.0f + bias), s2 = __float2half2_rn(s);
    const uint32_t h[4] = { __byte_perm(v0, 0x64646464u, 0x5140), __byte_perm(v0, 0x64646464u, 0x5342),
                            __byte_perm(v1, 0x64646464u, 0x5140), __byte_perm(v1, 0x64646464u, 0x5342) };
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        const __half2 q = __hsub2(*(const __half2 *) &h[k], kb);
        if constexpr (MIN) {
            w[k] = __hfma2(q, s2, __float2half2_rn(mn));
        } else {
            w[k] = __hmul2(q, s2);
        }
    }
}
static __device__ __forceinline__ void mmvq_f16_sm_k4(const int j, const uint8_t * q, int & d, int & m) { // get_scale_min_k4
    if (j < 4) {
        d = q[j] & 63; m = q[j + 4] & 63;
    } else {
        d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        m = (q[j + 4] >>  4) | ((q[j - 0] >> 6) << 4);
    }
}
static __device__ __forceinline__ float mmvq_f16_h(const uint8_t * p) {
    return __half2float(*(const __half *) p);
}


template <typename F>
static void mmvq_f16_gen_columns(const int64_t ncols, F && f) {
    const int64_t n0 = ncols <= 8 ? ncols : (ncols + 1)/2;
    for (int64_t c0 = 0; c0 < ncols; c0 += n0) {
        const int64_t n = std::min(n0, ncols - c0);
        switch (n) {
            case 1: f(c0, std::integral_constant<int, 1>{}); break;
            case 2: f(c0, std::integral_constant<int, 2>{}); break;
            case 3: f(c0, std::integral_constant<int, 3>{}); break;
            case 4: f(c0, std::integral_constant<int, 4>{}); break;
            case 5: f(c0, std::integral_constant<int, 5>{}); break;
            case 6: f(c0, std::integral_constant<int, 6>{}); break;
            case 7: f(c0, std::integral_constant<int, 7>{}); break;
            case 8: f(c0, std::integral_constant<int, 8>{}); break;
            default: GGML_ABORT("mmvq_f16_gen: %d columns", (int) n);
        }
    }
}

// See mmvq_f16_nc_fwd: extra tags are forwarded to the callback as parameters.
template <typename F, typename... T>
static void mmvq_f16_gen_columns_fwd(const int64_t ncols, F && f, T &&... t) {
    const int64_t n0 = ncols <= 8 ? ncols : (ncols + 1)/2;
    for (int64_t c0 = 0; c0 < ncols; c0 += n0) {
        const int64_t n = std::min(n0, ncols - c0);
        switch (n) {
            case 1: f(c0, std::integral_constant<int, 1>{}, t...); break;
            case 2: f(c0, std::integral_constant<int, 2>{}, t...); break;
            case 3: f(c0, std::integral_constant<int, 3>{}, t...); break;
            case 4: f(c0, std::integral_constant<int, 4>{}, t...); break;
            case 5: f(c0, std::integral_constant<int, 5>{}, t...); break;
            case 6: f(c0, std::integral_constant<int, 6>{}, t...); break;
            case 7: f(c0, std::integral_constant<int, 7>{}, t...); break;
            case 8: f(c0, std::integral_constant<int, 8>{}, t...); break;
            default: GGML_ABORT("mmvq_f16_gen: %d columns", (int) n);
        }
    }
}

// ---------------------------------------------------------------------------------------------------
// q5_K: a hand-scheduled kernel (the generic one decodes scales and unpacks bits per 8-value chunk).
//
// q5_K's 176-byte block is a multiple of 16, so every block is 16-byte aligned and the lanes load
// their bytes directly (no shared staging). A warp covers a 1024-value window = 4 blocks, 8 lanes per
// block; lane g = lane % 8 of a block owns 16 bytes of qs (bytes 16g.., sub-block pair j = g/2: 16
// low-nibble values of sub-block 2j and the 16 high-nibble values of sub-block 2j+1) and 16 bytes of
// qh: 32 values per row and window = 16 HFMA2 per column, as on q6_K. Each lane decodes the scale and
// min of its own two sub-blocks. Numerics per weight are the generic kernel's:
//   w = (half(1024 + q) - 1024) * half(d*1024*sc) + half(-dmin*1024*m), q = nibble | qh bit << 4
// then a chain of 16 HFMA2 per lane and window, folded into fp32 with the window's scale.
#ifndef K5_DEC
#define K5_DEC 0
#endif
template <int NC, int RPW, int NWT, bool KS, int MINB, bool GLU>
__launch_bounds__(NWT*WARP_SIZE, MINB)
static __global__ void mmvq_f16_q5_K(const uint8_t * __restrict__ W, const uint8_t * __restrict__ W2, const int64_t row_bytes,
                                     const __half * __restrict__ XS, const float * __restrict__ S, float * __restrict__ Y,
                                     const int64_t sy, const int rows, const int K) {
    static_assert(!GLU || (!KS && RPW % 2 == 0), "GLU: row-parallel, gate/up row pairs");
    constexpr int ORW = GLU ? RPW/2 : RPW, RPB = NWT*RPW, WB = 4*176;
    // 7+ columns on the 4-row tile: fold each half-window's 8-HFMA2 chain into fp32 right away instead
    // of holding NC*RPW half2 chains over the window (8704x5120 at 10 columns 234 -> 205 us, NMSE
    // 1.14e-6 -> 8.4e-7); fewer columns or rows keep the window-long chains (faster there)
    constexpr bool FOLD = NC >= 7 && RPW == 4;
    const int lane = threadIdx.x, wid = threadIdx.y;
    const int nb = K/256, nw = (nb + 3)/4;
    const int row0 = KS ? blockIdx.x*RPW : GLU ? (blockIdx.x*NWT + wid)*ORW : blockIdx.x*RPB + wid*RPW;
    const int B = lane >> 3, g = lane & 7, j = g >> 1, h = g & 1;
    const int rl = (4 - 2*j) & 31, rh = (3 - 2*j) & 31; // rotate qh's bit 2j / 2j+1 to bit 4 of each byte
    const bool hiS = j >= 2;                             // sub-blocks 4..7: the packed scale layout
    const int sh0 = 16*(j & 1);
    const char * rp[RPW];
#pragma unroll
    for (int i = 0; i < RPW; ++i) {
        const uint8_t * base = GLU && i >= ORW ? W2 : W;
        rp[i] = (const char *) base + (int64_t) min(row0 + (GLU ? i % ORW : i), rows - 1)*row_bytes + (KS ? (int64_t) wid*WB : 0) + B*176;
    }
    const __half * xw[NC];
#pragma unroll
    for (int c = 0; c < NC; ++c) {
        xw[c] = XS + (int64_t) c*K + (KS ? (int64_t) wid*MMVQ_F16_WIN : 0) + B*256 + 64*j + 16*h;
    }
    float acc[NC][RPW];
#pragma unroll
    for (int c = 0; c < NC; ++c) {
#pragma unroll
        for (int i = 0; i < RPW; ++i) {
            acc[c][i] = 0.0f;
        }
    }
    const __half2 k1024 = __float2half2_rn(1024.0f), kdm = __halves2half2(__float2half(1024.0f), __float2half(-1024.0f));

    for (int win = KS ? wid : 0; win < nw; win += KS ? NWT : 1) {
        const bool act = win*4 + B < nb;
        __half2 t[FOLD ? 1 : NC][RPW];
#pragma unroll
        for (int c = 0; c < (FOLD ? 1 : NC); ++c) {
#pragma unroll
            for (int i = 0; i < RPW; ++i) {
                t[c][i] = __float2half2_rn(0.0f);
            }
        }
        if (act) {
            uint4 qs[RPW], qh[RPW];
            __half2 sL[RPW], mL[RPW], sH[RPW], mH[RPW];
#pragma unroll
            for (int i = 0; i < RPW; ++i) {
                qs[i] = __ldg((const uint4 *) (rp[i] + 48 + 16*g));
                qh[i] = __ldg((const uint4 *) (rp[i] + 16 + 16*h));
                const uint4 hd = __ldg((const uint4 *) rp[i]);
#if K5_DEC == 0
                const float d = __half2float(__ushort_as_half((unsigned short) (hd.x & 0xFFFF)));
                const float dm = __half2float(__ushort_as_half((unsigned short) (hd.x >> 16)));
                {
                    const uint32_t x0 = hd.y >> sh0, x1 = hd.z >> sh0, x2 = hd.w >> sh0;
                    const uint32_t sc = hiS ? ((x2 & 15) | ((x0 >> 2) & 0x30)) : (x0 & 63);
                    const uint32_t m  = hiS ? (((x2 >> 4) & 15) | ((x1 >> 2) & 0x30)) : (x1 & 63);
                    sL[i] = __float2half2_rn(d*1024.0f*(float) sc);
                    mL[i] = __float2half2_rn(-dm*1024.0f*(float) m);
                }
                {
                    const uint32_t x0 = hd.y >> (sh0 + 8), x1 = hd.z >> (sh0 + 8), x2 = hd.w >> (sh0 + 8);
                    const uint32_t sc = hiS ? ((x2 & 15) | ((x0 >> 2) & 0x30)) : (x0 & 63);
                    const uint32_t m  = hiS ? (((x2 >> 4) & 15) | ((x1 >> 2) & 0x30)) : (x1 & 63);
                    sH[i] = __float2half2_rn(d*1024.0f*(float) sc);
                    mH[i] = __float2half2_rn(-dm*1024.0f*(float) m);
                }
#else
                // scale and min of sub-blocks 2j (lo) and 2j+1 (hi) as exact half integers, times the
                // half(d*1024) / half(-dmin*1024): the product d*1024*sc is exact in fp32 and so is the
                // half product, so the one rounding to half is the same as the generic kernel's
                const uint32_t x0 = hd.y >> sh0, x1 = hd.z >> sh0, x2 = hd.w >> sh0;
                const uint32_t pks = hiS ? ((x2 & 0x0F0Fu) | ((x0 >> 2) & 0x3030u)) : (x0 & 0x3F3Fu);
                const uint32_t pkm = hiS ? (((x2 >> 4) & 0x0F0Fu) | ((x1 >> 2) & 0x3030u)) : (x1 & 0x3F3Fu);
                const uint32_t word = pks | (pkm << 16);
                const uint32_t dd32 = hd.x;
                const __half2 dd = __hmul2(*(const __half2 *) &dd32, kdm);
                const __half2 dB = __low2half2(dd), nB = __high2half2(dd);
                const uint32_t c0 = __byte_perm(word, 0x64646464u, 0x4040), c1 = __byte_perm(word, 0x64646464u, 0x4141);
                const uint32_t c2 = __byte_perm(word, 0x64646464u, 0x4242), c3 = __byte_perm(word, 0x64646464u, 0x4343);
                sL[i] = __hmul2(__hsub2(*(const __half2 *) &c0, k1024), dB);
                sH[i] = __hmul2(__hsub2(*(const __half2 *) &c1, k1024), dB);
                mL[i] = __hmul2(__hsub2(*(const __half2 *) &c2, k1024), nB);
                mH[i] = __hmul2(__hsub2(*(const __half2 *) &c3, k1024), nB);
#endif
            }
#pragma unroll
            for (int m = 0; m < 2; ++m) {
                __half2 wl[RPW][4], wh[RPW][4];
#pragma unroll
                for (int i = 0; i < RPW; ++i) {
#pragma unroll
                    for (int e = 0; e < 2; ++e) {
                        const uint32_t q  = m == 0 ? (e == 0 ? qs[i].x : qs[i].y) : (e == 0 ? qs[i].z : qs[i].w);
                        const uint32_t hq = m == 0 ? (e == 0 ? qh[i].x : qh[i].y) : (e == 0 ? qh[i].z : qh[i].w);
                        const uint32_t lo = (q & 0x0F0F0F0Fu) | (__funnelshift_l(hq, hq, rl) & 0x10101010u);
                        const uint32_t hi = ((q >> 4) & 0x0F0F0F0Fu) | (__funnelshift_l(hq, hq, rh) & 0x10101010u);
                        const uint32_t a0 = __byte_perm(lo, 0x64646464u, 0x5140), a1 = __byte_perm(lo, 0x64646464u, 0x5342);
                        const uint32_t b0 = __byte_perm(hi, 0x64646464u, 0x5140), b1 = __byte_perm(hi, 0x64646464u, 0x5342);
                        wl[i][2*e]   = __hfma2(__hsub2(*(const __half2 *) &a0, k1024), sL[i], mL[i]);
                        wl[i][2*e+1] = __hfma2(__hsub2(*(const __half2 *) &a1, k1024), sL[i], mL[i]);
                        wh[i][2*e]   = __hfma2(__hsub2(*(const __half2 *) &b0, k1024), sH[i], mH[i]);
                        wh[i][2*e+1] = __hfma2(__hsub2(*(const __half2 *) &b1, k1024), sH[i], mH[i]);
                    }
                }
#pragma unroll
                for (int c = 0; c < NC; ++c) {
                    const uint4 xl = __ldg((const uint4 *) (xw[c] + 8*m)), xh = __ldg((const uint4 *) (xw[c] + 32 + 8*m));
                    const __half2 xlv[4] = { *(const __half2 *) &xl.x, *(const __half2 *) &xl.y, *(const __half2 *) &xl.z, *(const __half2 *) &xl.w };
                    const __half2 xhv[4] = { *(const __half2 *) &xh.x, *(const __half2 *) &xh.y, *(const __half2 *) &xh.z, *(const __half2 *) &xh.w };
                    const float sw = FOLD ? __ldg(S + c*nw + win) : 0.0f;
#pragma unroll
                    for (int i = 0; i < RPW; ++i) {
                        __half2 u = FOLD ? __float2half2_rn(0.0f) : t[FOLD ? 0 : c][i];
#pragma unroll
                        for (int k = 0; k < 4; ++k) {
                            u = __hfma2(wl[i][k], xlv[k], u);
                        }
#pragma unroll
                        for (int k = 0; k < 4; ++k) {
                            u = __hfma2(wh[i][k], xhv[k], u);
                        }
                        if constexpr (FOLD) {
                            const __half2 v = __hadd2(u, __lowhigh2highlow(u));
                            acc[c][i] = fmaf(sw, __low2float(v), acc[c][i]);
                        } else {
                            t[c][i] = u;
                        }
                    }
                }
            }
        }
#pragma unroll
        for (int i = 0; i < RPW; ++i) {
            rp[i] += KS ? (int64_t) NWT*WB : WB;
        }
#pragma unroll
        for (int c = 0; c < NC; ++c) {
            xw[c] += KS ? NWT*MMVQ_F16_WIN : MMVQ_F16_WIN;
            if constexpr (!FOLD) {
                const float s = __ldg(S + c*nw + win);
#pragma unroll
                for (int i = 0; i < RPW; ++i) {
                    const __half2 u = __hadd2(t[c][i], __lowhigh2highlow(t[c][i]));
                    acc[c][i] = fmaf(s, __low2float(u), acc[c][i]);
                }
            }
        }
    }
    if constexpr (KS) {
        __shared__ float red[NWT][NC][RPW];
#pragma unroll
        for (int c = 0; c < NC; ++c) {
#pragma unroll
            for (int i = 0; i < RPW; ++i) {
                float v = acc[c][i];
#pragma unroll
                for (int o = 16; o; o >>= 1) {
                    v += __shfl_xor_sync(0xFFFFFFFF, v, o);
                }
                if (lane == 0) {
                    red[wid][c][i] = v;
                }
            }
        }
        __syncthreads();
        if (wid == 0 && lane < NC*RPW) {
            const int c = lane / RPW, i = lane % RPW;
            float v = 0.0f;
#pragma unroll
            for (int w = 0; w < NWT; ++w) {
                v += red[w][c][i];
            }
            if (row0 + i < rows) {
                Y[c*sy + row0 + i] = v;
            }
        }
        return;
    }
#pragma unroll
    for (int c = 0; c < NC; ++c) {
        float v[RPW];
#pragma unroll
        for (int i = 0; i < RPW; ++i) {
            v[i] = acc[c][i];
#pragma unroll
            for (int o = 16; o; o >>= 1) {
                v[i] += __shfl_xor_sync(0xFFFFFFFF, v[i], o);
            }
        }
#pragma unroll
        for (int i = 0; i < ORW; ++i) {
            if (lane == i && row0 + i < rows) {
                if constexpr (GLU) {
                    Y[c*sy + row0 + i] = ggml_cuda_op_silu_single(v[i]) * v[i + ORW];
                } else {
                    Y[c*sy + row0 + i] = v[i];
                }
            }
        }
    }
}

struct mmvq_f16_q5_K_cfg { int rpw, nwt, ks, minb; };

// launch geometry per column count and row count
static mmvq_f16_q5_K_cfg mmvq_f16_q5_K_pick(const int nc, const int64_t rows, const bool glu) {
#ifdef K5_FORCE
    if (glu) { return { K5_G_RPW, K5_G_NWT, 0, K5_G_MINB }; }
    return { K5_F_RPW, K5_F_NWT, K5_F_KS, K5_F_MINB };
#else
    // p100k: STRATA_K5_CFG="rpw,nwt,ks,minb" (sweeps only): one of the compiled configurations (mmvq_f16_q5_K_run), not GLU
    static const mmvq_f16_q5_K_cfg forced = [] {
        mmvq_f16_q5_K_cfg c{0, 0, 0, 0};
        if (const char* v = std::getenv("STRATA_K5_CFG")) std::sscanf(v, "%d,%d,%d,%d", &c.rpw, &c.nwt, &c.ks, &c.minb);
        return c;
    }();
    if (forced.rpw > 0 && !glu) return forced;
    // from the k5s sweep (hx harness, us): the 2-warp tile spills past 8 columns (8704x5120 at 10:
    // 314 -> 239 with 4x1x8); small row counts want fewer rows per warp, and a K-split below 256 rows
    if (rows < 256) {
        return glu ? mmvq_f16_q5_K_cfg{ 2, 1, 0, 16 } : mmvq_f16_q5_K_cfg{ 1, 4, 1, 3 };
    }
    if (rows < 1024) {
        return nc <= 5 ? mmvq_f16_q5_K_cfg{ 2, 2, 0, 8 } : mmvq_f16_q5_K_cfg{ 2, 1, 0, 16 };
    }
    // p100k (Gyro-S shapes, 10240/6144/12288 x 2560 and 2560 x 6144, sweep 11-18): the 2-warp tile also wins at 3-5 columns
    // (the verify windows), 5-20% (5 columns: 285 -> 254 us over the four); 6+ columns keep the 1-warp tile (spills above)
    return nc <= 5 ? mmvq_f16_q5_K_cfg{ 4, 2, 0, 6 } : mmvq_f16_q5_K_cfg{ 4, 1, 0, 8 };
#endif
}

template <int NC, int RPW, int NWT, int KS, int MINB, bool GLU>
static bool mmvq_f16_q5_K_try(const mmvq_f16_q5_K_cfg c, const uint8_t * W, const uint8_t * W2, const int64_t row_bytes, const __half * xs,
                              const float * sc, float * Y, const int64_t sy, const int64_t rows, const int K, cudaStream_t stream) {
    if (c.rpw != RPW || c.nwt != NWT || c.ks != KS || c.minb != MINB) {
        return false;
    }
    constexpr int ORW = GLU ? RPW/2 : RPW;
    const int g = KS ? (int) ((rows + RPW - 1)/RPW) : (int) ((rows + NWT*ORW - 1)/(NWT*ORW));
    mmvq_f16_q5_K<NC, RPW, NWT, KS != 0, MINB, GLU><<<g, dim3(WARP_SIZE, NWT), 0, stream>>>(W, W2, row_bytes, xs, sc, Y, sy, (int) rows, K);
    return true;
}

template <int NC, bool GLU>
static void mmvq_f16_q5_K_run(const mmvq_f16_q5_K_cfg c, const uint8_t * W, const uint8_t * W2, const int64_t row_bytes, const __half * xs,
                              const float * sc, float * Y, const int64_t sy, const int64_t rows, const int K, cudaStream_t stream) {
#define K5T(R, N, S, M) mmvq_f16_q5_K_try<NC, R, N, S, M, GLU>(c, W, W2, row_bytes, xs, sc, Y, sy, rows, K, stream)
#ifdef K5_FORCE
    if constexpr (GLU) { K5T(K5_G_RPW, K5_G_NWT, 0, K5_G_MINB); } else { K5T(K5_F_RPW, K5_F_NWT, K5_F_KS, K5_F_MINB); }
#else
    if constexpr (!GLU) {
        if (K5T(1, 4, 1, 3)) { return; }
    }
    K5T(4, 2, 0, 6) || K5T(4, 1, 0, 8) || K5T(2, 2, 0, 8) || K5T(2, 1, 0, 16);
#endif
#undef K5T
}

// every column count, and the launch loop over column ranges
static bool mmvq_f16_q5_K_launch(const uint8_t * W, const uint8_t * W2, const int64_t row_bytes, const __half * xs, const float * sc,
                                 float * Y, const int64_t sy, const int64_t rows, const int64_t K, const int64_t ncols, const bool glu,
                                 cudaStream_t stream) {
    if (((uintptr_t) W & 15) || (glu && ((uintptr_t) W2 & 15)) || (row_bytes & 15) || ((uintptr_t) xs & 15)) {
        return false;
    }
    const int nw = (int) ((K + MMVQ_F16_WIN - 1)/MMVQ_F16_WIN);
    mmvq_f16_columns(ncols, [&](const int64_t c0, const int64_t n) {
        const mmvq_f16_q5_K_cfg c = mmvq_f16_q5_K_pick((int) n, rows, glu);
        const __half * x = xs + c0*K;
        const float * s = sc + c0*nw;
        float * y = Y + c0*sy;
#define K5N(NCV) case NCV: if (glu) { mmvq_f16_q5_K_run<NCV, true>(c, W, W2, row_bytes, x, s, y, sy, rows, (int) K, stream); } \
                           else { mmvq_f16_q5_K_run<NCV, false>(c, W, W2, row_bytes, x, s, y, sy, rows, (int) K, stream); } break;
        switch (n) {
            K5N(1) K5N(2) K5N(3) K5N(4) K5N(5) K5N(6) K5N(7) K5N(8) K5N(9) K5N(10) K5N(11) K5N(12)
            default: GGML_ABORT("mmvq_f16_q5_K: %d columns", (int) n);
        }
#undef K5N
    });
    CUDA_CHECK(cudaGetLastError());
    return true;
}


// ---------------------------------------------------------------------------------------------
// IQ4_NL / IQ4_XS: dedicated kernel. A lane owns 16 weights of one 32-weight (sub-)block: 8 qs bytes,
// i.e. the elements 8h..8h+7 (low nibbles) and 16+8h..16+8h+7 (high nibbles). The 16-entry table is
// looked up with byte_perm straight from the packed nibbles (no unpacking): the low class (index < 8)
// by the sign-replicate mode (bit 3 of the selector nibble) giving zero for the other class, the high
// class from a flipped selector, plus 0x80 for the high class, so one lookup gives tab + 128 per
// byte in element-pair order. The weights are half(1024 + byte) - 1152 = the exact table values and
// the block scale is applied once to each lane's 16-term fp16 partial sum. A window is 32 blocks.

// ---- Strata side
int nsm_of(int dev) { int n = 0; cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, dev); return n; }

void launch_q6(const uint8_t* W, int64_t row_bytes, const __half* xs_ptr, const float* sc_ptr, float* Y, int64_t sy,
               int64_t rows, int64_t K, int64_t ncols, int nw, cudaStream_t stream, int dev) {
    int64_t c0 = 0, nc = ncols;
    auto launch = [&](auto rpw, auto nwt, auto ks, auto minb) {
        mmvq_f16_nc_fwd(nc, [&](auto ncc, auto rpw_, auto nwt_, auto ks_, auto minb_) {
            constexpr int  NC   = decltype(ncc)::value;
            constexpr int  RPW  = decltype(rpw_)::value;
            constexpr int  NWT  = decltype(nwt_)::value;
            constexpr bool KS   = decltype(ks_)::value;
            constexpr int  MINB = decltype(minb_)::value;
            const dim3 bdk(WARP_SIZE, NWT);
            const int g = KS ? (int) ((rows + RPW - 1)/RPW) : (int) ((rows + NWT*RPW - 1)/(NWT*RPW));
            mmvq_f16_q6_K<NC, RPW, NWT, KS, MINB><<<g, bdk, 0, stream>>>(W, row_bytes, xs_ptr + c0*K, sc_ptr + c0*nw, Y + c0*sy, sy, (int) rows, (int) K);
        }, rpw, nwt, ks, minb);
    };
    using I = std::integral_constant<int, 1>;
    mmvq_f16_columns(ncols, [&](const int64_t c0_, const int64_t n_) {
        c0 = c0_;
        nc = n_;
        if (nc > 5) {
            if (rows >= 3072) launch(std::integral_constant<int, MMVQ_F16_RPW>{}, I{}, std::false_type{}, std::integral_constant<int, 8>{});
            else if (rows >= 256) launch(std::integral_constant<int, 2>{}, I{}, std::false_type{}, std::integral_constant<int, 12>{});
            else launch(I{}, std::integral_constant<int, 4>{}, std::true_type{}, std::integral_constant<int, 3>{});
            return;
        }
        if (rows >= 3072) {
            const int64_t nblk = (rows + MMVQ_F16_NW*MMVQ_F16_RPW - 1)/(MMVQ_F16_NW*MMVQ_F16_RPW);
            const int64_t nsm  = nsm_of(dev);
            const int64_t w6 = (nblk + 6*nsm - 1)/(6*nsm), w7 = (nblk + 7*nsm - 1)/(7*nsm);
            if (w7 < w6) launch(std::integral_constant<int, MMVQ_F16_RPW>{}, std::integral_constant<int, MMVQ_F16_NW>{}, std::false_type{}, std::integral_constant<int, 7>{});
            else launch(std::integral_constant<int, MMVQ_F16_RPW>{}, std::integral_constant<int, MMVQ_F16_NW>{}, std::false_type{}, std::integral_constant<int, 6>{});
        } else if (rows >= 256) {
            launch(std::integral_constant<int, 2>{}, I{}, std::false_type{}, std::integral_constant<int, 16>{});
        } else {
            launch(I{}, std::integral_constant<int, 4>{}, std::true_type{}, std::integral_constant<int, 3>{});
        }
    });
}
}  // namespace

namespace strata::kernels {

bool p100_mmvq_f16_on() {
    static const bool on = [] {
        const char* e = std::getenv("STRATA_P100_MMVQ_F16");
        if (e && e[0] == '0') return false;
        int dev = 0, major = 0, minor = 0;
        if (cudaGetDevice(&dev) != cudaSuccess) return false;
        cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
        cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev);
        const bool v = major == 6 && minor == 0;
        if (v) std::fprintf(stderr, "verify: dense Q5_K/Q6_K mat-vecs in FP16 (Tesla P100; STRATA_P100_MMVQ_F16=0 for q8_1)\n");
        return v;
    }();
    return on;
}

std::size_t p100_mmvq_f16_scratch_bytes(int max_in, int max_cols) {
    const int nw = (max_in + MMVQ_F16_WIN - 1) / MMVQ_F16_WIN;
    return std::size_t(max_cols) * max_in * sizeof(__half) + std::size_t(max_cols) * nw * sizeof(float) + 256;
}

bool p100_mmvq_f16_prep(const float* x, int n_in, int ncols, void* scratch, std::size_t cap, void* stream) {
    if (!scratch || ncols < 1 || ncols > 12 || n_in % 512 != 0) return false;   // (1 column: the Q6_K heads, below)
    if (reinterpret_cast<std::uintptr_t>(x) % 16 != 0 || reinterpret_cast<std::uintptr_t>(scratch) % 16 != 0) return false;
    const int nw = (n_in + MMVQ_F16_WIN - 1) / MMVQ_F16_WIN;
    if (p100_mmvq_f16_scratch_bytes(n_in, ncols) > cap) return false;
    __half* xs = static_cast<__half*>(scratch);
    float* sc = reinterpret_cast<float*>(static_cast<char*>(scratch) + std::size_t(ncols) * n_in * sizeof(__half));
    const dim3 pb(WARP_SIZE, 4), pg((nw + 3) / 4, ncols);
    mmvq_f16_prep<<<pg, pb, 0, static_cast<cudaStream_t>(stream)>>>(x, n_in, xs, sc, n_in);
    return cudaPeekAtLastError() == cudaSuccess;
}

bool p100_mmvq_f16(int ggml_type, const void* weights, float* y, int n_in, int n_out, int ncols, const void* scratch,
                   void* stream) {
    if (ggml_type != 13 && ggml_type != 14) return false;          // Q5_K, Q6_K
    if (!scratch || ncols < 1 || ncols > 12 || n_out < 16 || n_in % 512 != 0) return false;
    // one column: only the big Q6_K heads (the MTP draft head, 106k rows: 1.11 -> ~0.45 ms on a P100); the small Q5_K / Q6_K
    // projections stay on the q8_1 kernels at one column, where those were measured as fast or faster
    if (ncols == 1 && (ggml_type != 14 || n_out < 32768)) return false;
    if (reinterpret_cast<std::uintptr_t>(weights) % 4 != 0) return false;
    int dev = 0;
    cudaGetDevice(&dev);
    const int nw = (n_in + MMVQ_F16_WIN - 1) / MMVQ_F16_WIN;
    const __half* xs = static_cast<const __half*>(scratch);
    const float* sc = reinterpret_cast<const float*>(static_cast<const char*>(scratch) + std::size_t(ncols) * n_in * sizeof(__half));
    const auto s = static_cast<cudaStream_t>(stream);
    const auto* W = static_cast<const uint8_t*>(weights);
    if (ggml_type == 14) {
        launch_q6(W, int64_t(n_in) / 256 * 210, xs, sc, y, n_out, n_out, n_in, ncols, nw, s, dev);
    } else if (!mmvq_f16_q5_K_launch(W, nullptr, int64_t(n_in) / 256 * 176, xs, sc, y, n_out, n_out, n_in, ncols, false, s)) {
        return false;
    }
    static const bool trace = std::getenv("STRATA_P100_MMVQ_F16_TRACE") != nullptr;
    if (trace) {
        static int said = 0;
        if (said < 8) { ++said; std::fprintf(stderr, "p100_mmvq_f16: type %d n_in %d n_out %d cols %d\n", ggml_type, n_in, n_out, ncols); }
    }
    return cudaPeekAtLastError() == cudaSuccess;
}

}  // namespace strata::kernels
#endif  // !HIP
