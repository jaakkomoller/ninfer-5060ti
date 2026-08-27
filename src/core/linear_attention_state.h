#pragma once

#include "core/layout.h"
#include "core/tensor.h"

#include <cuda_runtime_api.h>

#include <cstdint>
#include <vector>

namespace ninfer {

// I8 conv storage groups 128 consecutive channels under one FP16 scale.
inline constexpr std::int32_t kLinearAttentionConvScaleGroup = 128;

struct LinearAttentionStatePoolSpec {
    std::uint32_t layers        = 0;
    std::int32_t conv_channels  = 0;
    std::int32_t conv_width     = 0;
    std::int32_t value_heads    = 0;
    std::int32_t value_head_dim = 0;
    std::int32_t key_head_dim   = 0;
    std::int32_t slot_count     = 1;
    // Persistent conv storage dtype: BF16 stores the exact window values; I8
    // stores one signed code per window value plus one FP16 scale per
    // kLinearAttentionConvScaleGroup-channel group and slot (element =
    // code * scale). The conv scale plane is allocated and managed by this
    // pool. The conv state is a sliding width-(conv_width) window, so consumer
    // kernels shift existing codes verbatim and quantize only the incoming
    // value under the current group scale; a fresh per-group scale is written
    // by the prefill state-publication kernel at every chunk boundary.
    DType conv_dtype            = DType::BF16;
    // Persistent storage dtype for the recurrent state. Computation stays in FP32
    // regardless: each consumer kernel loads to FP32, computes in FP32, then
    // converts back to this dtype when storing. BF16 halves the persistent
    // Linear Attention footprint at the cost of one rounding per save/load.
    // I8 stores one signed code per element plus one FP16 scale per
    // (value_head, dv-row) of 128 dk values (element = code * scale); the
    // scale plane is allocated and managed by this pool.
    DType recurrent_dtype       = DType::FP32;
};

struct LinearAttentionStatePoolLayout {
    LinearAttentionStatePoolSpec spec;
    std::vector<LayoutRegion> conv;
    std::vector<LayoutRegion> recurrent;
    // Present only for I8 recurrent storage: one FP16 [value_head_dim, value_heads, slot]
    // scale plane per layer, empty otherwise.
    std::vector<LayoutRegion> recurrent_scale;
    // Present only for I8 conv storage: one FP16 [conv_channels/128, slot]
    // scale plane per layer, empty otherwise.
    std::vector<LayoutRegion> conv_scale;
};

struct LinearAttentionStateAllLayersView {
    Tensor conv_layer0;
    Tensor recurrent_layer0;
    Tensor recurrent_scale_layer0;
    Tensor conv_scale_layer0;
    std::int64_t conv_layer_stride_bytes            = 0;
    std::int64_t recurrent_layer_stride_bytes       = 0;
    std::int64_t recurrent_scale_layer_stride_bytes = 0;
    std::int64_t conv_scale_layer_stride_bytes      = 0;
    LinearAttentionStatePoolSpec spec;
};

[[nodiscard]] LinearAttentionStatePoolLayout
plan_linear_attention_state_pool(LayoutBuilder& builder, const LinearAttentionStatePoolSpec& spec);

/**
 * Fixed-capacity physical storage for model-level Linear Attention state images.
 *
 * One logical slot selects the same frontier across every layer's convolution and recurrent
 * component. The pool owns no slot roles, validity, request metadata, allocation policy, or CUDA
 * stream. Construction binds caller-owned backing without mutating it.
 */
struct LinearAttentionStatePool {
    std::vector<Tensor> conv;
    std::vector<Tensor> recurrent;
    std::vector<Tensor> recurrent_scale;
    std::vector<Tensor> conv_scale;
    LinearAttentionStatePoolSpec spec;

    LinearAttentionStatePool() = default;
    LinearAttentionStatePool(DeviceSpan backing, const LinearAttentionStatePoolLayout& layout);

    [[nodiscard]] std::uint32_t layer_count() const noexcept;
    [[nodiscard]] std::int32_t slot_count() const noexcept;
    [[nodiscard]] bool recurrent_is_i8() const noexcept;
    [[nodiscard]] bool conv_is_i8() const noexcept;
    [[nodiscard]] std::int64_t conv_slot_stride_elements() const noexcept;
    [[nodiscard]] std::int64_t recurrent_slot_stride_elements() const noexcept;
    [[nodiscard]] std::int32_t conv_scale_groups() const noexcept;
    [[nodiscard]] LinearAttentionStateAllLayersView all_layers_view() const;
    [[nodiscard]] Tensor conv_slot(std::uint32_t layer, std::int32_t slot) const;
    [[nodiscard]] Tensor recurrent_slot(std::uint32_t layer, std::int32_t slot) const;
    // I8 only: the FP16 per-(value_head, dv-row) scale for one recurrent slot.
    [[nodiscard]] Tensor recurrent_scale_slot(std::uint32_t layer, std::int32_t slot) const;
    // I8 only: the full FP16 [value_head_dim, value_heads, slot_count] scale plane.
    [[nodiscard]] Tensor recurrent_scale_layer(std::uint32_t layer) const;
    // I8 conv only: the FP16 per-128-channel-group scale for one conv slot.
    [[nodiscard]] Tensor conv_scale_slot(std::uint32_t layer, std::int32_t slot) const;
    // I8 conv only: the full FP16 [groups, slot_count] scale plane for one layer.
    [[nodiscard]] Tensor conv_scale_layer(std::uint32_t layer) const;

    void copy_slot(std::int32_t src, std::int32_t dst, cudaStream_t stream = nullptr);
    void zero_slot(std::int32_t slot, cudaStream_t stream = nullptr);
};

} // namespace ninfer
