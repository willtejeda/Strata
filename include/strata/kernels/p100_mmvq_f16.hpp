// include/strata/kernels/p100_mmvq_f16.hpp - p100k: FP16 dense mat-vec for Q5_K / Q6_K on the Tesla P100.
#pragma once
#include <cstddef>

namespace strata::kernels {
bool p100_mmvq_f16_on();                                  // compute capability 6.0 and not STRATA_P100_MMVQ_F16=0
std::size_t p100_mmvq_f16_scratch_bytes(int max_in, int max_cols);   // device scratch for one prepared input
// convert an input's ncols float columns (stride n_in) to the FP16 form the kernel reads, into scratch
bool p100_mmvq_f16_prep(const float* x, int n_in, int ncols, void* scratch, std::size_t cap, void* stream);
// y[c * n_out + row] = W[row] . x[c] for the input prepared in scratch; false: not handled (use the q8_1 path)
bool p100_mmvq_f16(int ggml_type, const void* weights, float* y, int n_in, int n_out, int ncols, const void* scratch,
                   void* stream);
}  // namespace strata::kernels
