#pragma once

// ninfer::ops::detail - private launch prototypes for causal_conv1d.

#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops::detail {

inline constexpr std::int32_t kCausalConvSequenceMaxTokens = 64;
inline constexpr std::int32_t kCausalConvParallelMaxTokens = 16;

void causal_conv1d_prefill_launch(const Tensor& x, const Tensor& weight,
                                  const Tensor& conv_state_in, Tensor& conv_state_out, Tensor& out,
                                  cudaStream_t stream);
void causal_conv1d_sequence_launch(const Tensor& x, const Tensor& weight,
                                   const Tensor& conv_state_in, Tensor& conv_state_out, Tensor& out,
                                   cudaStream_t stream);
void causal_conv1d_smallt_launch(const Tensor& x, const Tensor& weight, const Tensor& conv_state_in,
                                 Tensor& conv_state_out, Tensor& out, cudaStream_t stream);
void causal_conv1d_decode_launch(const Tensor& x, const Tensor& weight, const Tensor& conv_state_in,
                                 Tensor& conv_state_out, Tensor& out, cudaStream_t stream);
void causal_conv1d_snapshot_launch(const Tensor& x, const Tensor& weight, Tensor& conv_states,
                                    const Tensor& valid_columns, const Tensor& initial_state_slots,
                                    const Tensor& snapshot_base_slots, Tensor& out,
                                    cudaStream_t stream);

// I8 conv state form: conv_state_in/out are I8 [C,3] code planes (may alias) and
// conv_scale is the FP16 [C/128] per-128-channel-group scale of the *input* window.
// Computes the convolution from dequantized history and republishes the trailing
// window with a fresh per-group scale (see causal_conv1d_i8_*_kernel).
void causal_conv1d_i8_launch(const Tensor& x, const Tensor& weight, const Tensor& conv_state_in,
                             Tensor& conv_state_out, const Tensor& conv_scale, Tensor& out,
                             cudaStream_t stream);

} // namespace ninfer::ops::detail
