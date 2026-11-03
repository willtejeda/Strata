// Strata on Pascal (sm_60, Tesla P100): DP4A arrived with sm_61. Below that, __dp4a is the same sum one byte at a
// time, as ggml-cuda's ggml_cuda_dp4a does on these cards. Elsewhere this header is empty.
#pragma once
#if defined(__CUDACC__) && !defined(STRATA_USE_HIP) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 610
#include <cstdint>

__device__ __forceinline__ int strata_dp4a(int a, int b, int c) {
    const int8_t* va = reinterpret_cast<const int8_t*>(&a);
    const int8_t* vb = reinterpret_cast<const int8_t*>(&b);
    return c + va[0] * vb[0] + va[1] * vb[1] + va[2] * vb[2] + va[3] * vb[3];
}

__device__ __forceinline__ unsigned int strata_dp4a(unsigned int a, unsigned int b, unsigned int c) {
    const uint8_t* va = reinterpret_cast<const uint8_t*>(&a);
    const uint8_t* vb = reinterpret_cast<const uint8_t*>(&b);
    return c + va[0] * vb[0] + va[1] * vb[1] + va[2] * vb[2] + va[3] * vb[3];
}

#define __dp4a strata_dp4a
#endif

// Tensor cores (WMMA) start at sm_70, and <mma.h> defines nvcuda::wmma only there. The engine already picks its
// non-tensor-core path at run time on older cards (ds4_gguf.cu: compute capability major >= 7, or
// STRATA_DS4_NO_TC=1), so the WMMA kernels only need to compile on Pascal; these stand-ins trap if one ever ran.
#if defined(__CUDACC__) && !defined(STRATA_USE_HIP) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 700
namespace nvcuda {
namespace wmma {
struct row_major {};
struct col_major {};
struct matrix_a {};
struct matrix_b {};
struct accumulator {};
enum layout_t { mem_row_major, mem_col_major };
template <typename Use, int M, int N, int K, typename T, typename Layout = void>
struct fragment {
    static constexpr int num_elements = 8;
    T x[num_elements];
};
template <typename F, typename P>
__device__ inline void load_matrix_sync(F&, const P*, unsigned) { __trap(); }
template <typename F, typename P>
__device__ inline void load_matrix_sync(F&, const P*, unsigned, layout_t) { __trap(); }
template <typename P, typename F>
__device__ inline void store_matrix_sync(P*, const F&, unsigned, layout_t) { __trap(); }
template <typename D, typename A, typename B, typename C>
__device__ inline void mma_sync(D&, const A&, const B&, const C&) { __trap(); }
template <typename F, typename V>
__device__ inline void fill_fragment(F&, const V&) { __trap(); }
}  // namespace wmma
}  // namespace nvcuda
#endif
