#include "ops/gdn_input_proj/gdn_projected_conv.h"

#include "core/device.h"
#include "ops/common/math.cuh"
#include "ops/gdn_input_proj/gdn_conv.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

template <bool ConvI8, int Channels, int QueryRows, int KeyRows, int ValueRows, int StaticWidth,
          class Publish>
__global__ void gdn_projected_conv_kernel(
    const __nv_bfloat16* __restrict__ projected, const __nv_bfloat16* __restrict__ conv_weight,
    const void* __restrict__ state_read, const __half* __restrict__ scale_read,
    const std::int32_t* __restrict__ valid_columns, const std::int32_t* __restrict__ initial_state_slots,
    __nv_bfloat16* __restrict__ query, __nv_bfloat16* __restrict__ key,
    __nv_bfloat16* __restrict__ value, std::int32_t width, Publish publish) {
    static_assert(Channels == QueryRows + KeyRows + ValueRows);
    const std::int32_t row = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
    if (row >= Channels) { return; }
    const std::int32_t batch = static_cast<std::int32_t>(blockIdx.y);
    if constexpr (StaticWidth != 0) { width = StaticWidth; }

    std::int32_t valid                 = valid_columns == nullptr ? width : valid_columns[batch];
    valid                              = valid < 0 ? 0 : (valid > width ? width : valid);
    constexpr std::int64_t slot_stride = static_cast<std::int64_t>(Channels) * 3;
    const std::int64_t initial_base =
        static_cast<std::int64_t>(initial_state_slots[batch]) * slot_stride;
    const float w0 = __bfloat162float(conv_weight[row]);
    const float w1 = __bfloat162float(conv_weight[Channels + row]);
    const float w2 = __bfloat162float(conv_weight[2LL * Channels + row]);
    const float w3 = __bfloat162float(conv_weight[3LL * Channels + row]);

    if constexpr (!ConvI8) {
        const auto* state_bf16 = static_cast<const __nv_bfloat16*>(state_read);
        float s0               = __bfloat162float(state_bf16[initial_base + row]);
        float s1               = __bfloat162float(state_bf16[initial_base + Channels + row]);
        float s2               = __bfloat162float(state_bf16[initial_base + 2LL * Channels + row]);

        for (std::int32_t token = 0; token < width; ++token) {
            const std::int64_t column = static_cast<std::int64_t>(batch) * width + token;
            if (token >= valid) {
                if (row < QueryRows) {
                    query[column * QueryRows + row] = __float2bfloat16_rn(0.0F);
                } else if (row < QueryRows + KeyRows) {
                    key[column * KeyRows + row - QueryRows] = __float2bfloat16_rn(0.0F);
                } else {
                    value[column * ValueRows + row - QueryRows - KeyRows] = __float2bfloat16_rn(0.0F);
                }
                continue;
            }

            const float p              = __bfloat162float(projected[column * Channels + row]);
            float conv                 = fmaf(w0, s0, 0.0F);
            conv                       = fmaf(w1, s1, conv);
            conv                       = fmaf(w2, s2, conv);
            conv                       = fmaf(w3, p, conv);
            const __nv_bfloat16 output = __float2bfloat16_rn(silu(conv));
            if (row < QueryRows) {
                query[column * QueryRows + row] = output;
            } else if (row < QueryRows + KeyRows) {
                key[column * KeyRows + row - QueryRows] = output;
            } else {
                value[column * ValueRows + row - QueryRows - KeyRows] = output;
            }
            publish.publish(token, batch, row, s1, s2, p);
            s0 = s1;
            s1 = s2;
            s2 = p;
        }
    } else {
        const auto* state_codes = static_cast<const std::int8_t*>(state_read);
        constexpr std::int32_t kScaleGroups = Channels / 128;
        const __half scale16 =
            scale_read[static_cast<std::int64_t>(initial_state_slots[batch]) * kScaleGroups +
                       (row >> 7)];
        const float scale = __half2float(scale16);
        std::int32_t c0    = static_cast<std::int8_t>(state_codes[initial_base + row]);
        std::int32_t c1    = static_cast<std::int8_t>(state_codes[initial_base + Channels + row]);
        std::int32_t c2    = static_cast<std::int8_t>(state_codes[initial_base + 2LL * Channels + row]);
        float s0           = static_cast<float>(c0) * scale;
        float s1           = static_cast<float>(c1) * scale;
        float s2           = static_cast<float>(c2) * scale;

        for (std::int32_t token = 0; token < width; ++token) {
            const std::int64_t column = static_cast<std::int64_t>(batch) * width + token;
            if (token >= valid) {
                if (row < QueryRows) {
                    query[column * QueryRows + row] = __float2bfloat16_rn(0.0F);
                } else if (row < QueryRows + KeyRows) {
                    key[column * KeyRows + row - QueryRows] = __float2bfloat16_rn(0.0F);
                } else {
                    value[column * ValueRows + row - QueryRows - KeyRows] = __float2bfloat16_rn(0.0F);
                }
                continue;
            }

            const float p               = __bfloat162float(projected[column * Channels + row]);
            float conv                  = fmaf(w0, s0, 0.0F);
            conv                        = fmaf(w1, s1, conv);
            conv                        = fmaf(w2, s2, conv);
            conv                        = fmaf(w3, p, conv);
            const __nv_bfloat16 output  = __float2bfloat16_rn(silu(conv));
            if (row < QueryRows) {
                query[column * QueryRows + row] = output;
            } else if (row < QueryRows + KeyRows) {
                key[column * KeyRows + row - QueryRows] = output;
            } else {
                value[column * ValueRows + row - QueryRows - KeyRows] = output;
            }
            const std::int32_t pc = static_cast<std::int32_t>(i8_quantize(p, scale));
            publish.publish(token, batch, row, c1, c2, pc, p, scale16);
            s0 = s1;
            s1 = s2;
            s2 = p;
            c0 = c1;
            c1 = c2;
            c2 = pc;
        }
    }
}

template <bool ConvI8, int Channels, int QueryRows, int KeyRows, int ValueRows, class Publish>
void launch(const Tensor& projected, const Tensor& conv_weight, const Tensor& state_read,
            const Tensor& scale_read, const Tensor& valid_columns,
            const Tensor& initial_state_slots, Tensor& query, Tensor& key, Tensor& value,
            Publish publish, cudaStream_t stream) {
    constexpr int kDefaultThreads = 256;
    const std::int32_t width      = projected.ne[1];
    const std::int32_t batch      = projected.ne[2];
    if constexpr (Channels == 10240) {
        if (width == 4 && batch == 1) {
            constexpr int kT4Threads = 64;
            gdn_projected_conv_kernel<ConvI8, Channels, QueryRows, KeyRows, ValueRows, 4>
                <<<(Channels + kT4Threads - 1) / kT4Threads, kT4Threads, 0, stream>>>(
                    static_cast<const __nv_bfloat16*>(projected.data),
                    static_cast<const __nv_bfloat16*>(conv_weight.data),
                    state_read.data,
                    static_cast<const __half*>(scale_read.data),
                    valid_columns.data == nullptr
                        ? nullptr
                        : static_cast<const std::int32_t*>(valid_columns.data),
                    static_cast<const std::int32_t*>(initial_state_slots.data),
                    static_cast<__nv_bfloat16*>(query.data), static_cast<__nv_bfloat16*>(key.data),
                    static_cast<__nv_bfloat16*>(value.data), width, publish);
            CUDA_CHECK(cudaGetLastError());
            return;
        }
    }
    const dim3 grid((Channels + kDefaultThreads - 1) / kDefaultThreads,
                    static_cast<unsigned>(batch));
    gdn_projected_conv_kernel<ConvI8, Channels, QueryRows, KeyRows, ValueRows, 0>
        <<<grid, kDefaultThreads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(projected.data),
            static_cast<const __nv_bfloat16*>(conv_weight.data), state_read.data,
            static_cast<const __half*>(scale_read.data),
            valid_columns.data == nullptr ? nullptr
                                          : static_cast<const std::int32_t*>(valid_columns.data),
            static_cast<const std::int32_t*>(initial_state_slots.data),
            static_cast<__nv_bfloat16*>(query.data), static_cast<__nv_bfloat16*>(key.data),
            static_cast<__nv_bfloat16*>(value.data), width, publish);
    CUDA_CHECK(cudaGetLastError());
}

template <class Publish, bool I8>
void dispatch(const Tensor& projected, const Tensor& conv_weight, const Tensor& state_read,
              const Tensor& scale_read, const Tensor& valid_columns,
              const Tensor& initial_state_slots, Tensor& query, Tensor& key, Tensor& value,
              Publish publish, cudaStream_t stream) {
    if (projected.ne[0] == 10240 && query.ne[0] == 2048 && key.ne[0] == 2048 &&
        value.ne[0] == 6144) {
        launch<I8, 10240, 2048, 2048, 6144>(projected, conv_weight, state_read, scale_read,
                                            valid_columns, initial_state_slots, query, key, value,
                                            publish, stream);
        return;
    }
    if (projected.ne[0] == 8192 && query.ne[0] == 2048 && key.ne[0] == 2048 &&
        value.ne[0] == 4096) {
        launch<I8, 8192, 2048, 2048, 4096>(projected, conv_weight, state_read, scale_read,
                                           valid_columns, initial_state_slots, query, key, value,
                                           publish, stream);
        return;
    }
    throw std::invalid_argument("GDN projected-conv received an unregistered geometry");
}

} // namespace

void gdn_projected_conv_snapshot_launch(const Tensor& projected, const Tensor& conv_weight,
                                        Tensor& conv_states, const Tensor& conv_scale,
                                        const Tensor& valid_columns,
                                        const Tensor& initial_state_slots,
                                        const Tensor& snapshot_base_slots, Tensor& query,
                                        Tensor& key, Tensor& value, cudaStream_t stream) {
    if (conv_states.dtype == DType::I8) {
        const SnapshotHistoryPublishI8 publish{
            static_cast<std::int8_t*>(conv_states.data),
            static_cast<const std::int32_t*>(snapshot_base_slots.data), projected.ne[0],
            static_cast<std::int32_t>(projected.ne[0] / 128), static_cast<__half*>(conv_scale.data)};
        dispatch<SnapshotHistoryPublishI8, true>(
            projected, conv_weight, conv_states, conv_scale, valid_columns, initial_state_slots,
            query, key, value, publish, stream);
        return;
    }
    dispatch<SnapshotHistoryPublish, false>(
        projected, conv_weight, conv_states, conv_scale, valid_columns, initial_state_slots,
        query, key, value,
        SnapshotHistoryPublish{static_cast<__nv_bfloat16*>(conv_states.data),
                               static_cast<const std::int32_t*>(snapshot_base_slots.data),
                               projected.ne[0]},
        stream);
}

void gdn_projected_conv_record_launch(const Tensor& conv_record, const Tensor& conv_weight,
                                      const Tensor& conv_states, const Tensor& conv_scale,
                                      const Tensor& valid_columns,
                                      const Tensor& initial_state_slots, Tensor& query, Tensor& key,
                                      Tensor& value, cudaStream_t stream) {
    if (conv_states.dtype == DType::I8) {
        dispatch<NoHistoryPublishI8, true>(
            conv_record, conv_weight, conv_states, conv_scale, valid_columns,
            initial_state_slots, query, key, value, NoHistoryPublishI8{}, stream);
        return;
    }
    dispatch<NoHistoryPublish, false>(conv_record, conv_weight, conv_states, conv_scale,
                                      valid_columns, initial_state_slots, query, key, value,
                                      NoHistoryPublish{}, stream);
}

} // namespace ninfer::ops::detail