#pragma once

#include "ops/common/math.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>

namespace ninfer::ops::detail {

struct SnapshotHistoryPublish {
    __nv_bfloat16* state_write;
    const std::int32_t* snapshot_base_slots;
    std::int32_t channels;

    __device__ __forceinline__ void publish(std::int32_t token, std::int32_t batch,
                                            std::int32_t row, float s1, float s2, float p) const {
        const std::int64_t slot_stride = static_cast<std::int64_t>(channels) * 3;
        const std::int64_t base =
            static_cast<std::int64_t>(snapshot_base_slots[batch] + token) * slot_stride;
        state_write[base + row]                  = __float2bfloat16_rn(s1);
        state_write[base + channels + row]       = __float2bfloat16_rn(s2);
        state_write[base + 2LL * channels + row] = __float2bfloat16_rn(p);
    }
};

struct RecordColumnPublish {
    __nv_bfloat16* record;
    std::int32_t channels;
    std::int32_t width;

    __device__ __forceinline__ void publish(std::int32_t token, std::int32_t batch,
                                            std::int32_t row, float, float, float p) const {
        const std::int64_t column       = static_cast<std::int64_t>(batch) * width + token;
        record[column * channels + row] = __float2bfloat16_rn(p);
    }
};

struct NoHistoryPublish {
    __device__ __forceinline__ void publish(std::int32_t, std::int32_t, std::int32_t, float, float,
                                            float) const {}
};

// I8 conv-state publish forms. The conv state window is stored as I8 codes with one
// FP16 scale per 128-channel group and slot (element = code * scale); the epilogue
// shifts existing codes verbatim and passes the incoming value's code, quantized
// under the group scale, plus the group scale itself (snapshot slots copy it).
// `p` is the raw FP32 incoming value (kept for the BF16 conv record).
struct SnapshotHistoryPublishI8 {
    std::int8_t* codes;
    const std::int32_t* snapshot_base_slots;
    std::int32_t channels;
    std::int32_t scale_groups;
    __half* scale_write;

    __device__ __forceinline__ void publish(std::int32_t token, std::int32_t batch,
                                            std::int32_t row, std::int32_t c1, std::int32_t c2,
                                            std::int32_t pc, float, __half scale16) const {
        const std::int64_t slot      = snapshot_base_slots[batch] + token;
        const std::int64_t slot_base = slot * static_cast<std::int64_t>(channels) * 3;
        codes[slot_base + row]                  = static_cast<std::int8_t>(c1);
        codes[slot_base + channels + row]       = static_cast<std::int8_t>(c2);
        codes[slot_base + 2LL * channels + row] = static_cast<std::int8_t>(pc);
        if ((row & 127) == 0) {
            scale_write[slot * scale_groups + (row >> 7)] = scale16;
        }
    }
};

struct RecordColumnPublishI8 {
    __nv_bfloat16* record;
    std::int32_t channels;
    std::int32_t width;

    __device__ __forceinline__ void publish(std::int32_t token, std::int32_t batch,
                                            std::int32_t row, std::int32_t, std::int32_t,
                                            std::int32_t, float p, __half) const {
        const std::int64_t column       = static_cast<std::int64_t>(batch) * width + token;
        record[column * channels + row] = __float2bfloat16_rn(p);
    }
};

struct NoHistoryPublishI8 {
    __device__ __forceinline__ void
    publish(std::int32_t, std::int32_t, std::int32_t, std::int32_t, std::int32_t, std::int32_t,
            float, __half) const {}
};

// Device-side implementation detail shared by exact packed projection kernels. Projection
// accumulators stay in the route's existing private precision; Publish changes only the side
// effect after the convolution has consumed that accumulator. ConvI8 selects the I8 conv
// state form: the window is stored as I8 codes with a per-128-channel-group FP16 scale
// plane (slot-major: each slot owns a contiguous [groups] block); history taps load as
// code * scale, existing codes shift verbatim, and only the incoming value is quantized
// (i8_quantize) under the current group scale.
template <class Publish, bool ConvI8 = false>
struct GdnConvEpilogue {
    const __nv_bfloat16* conv_weight;
    const __nv_bfloat16* state_read;
    const std::int32_t* initial_slots;
    const std::int32_t* valid_columns;
    __nv_bfloat16* query;
    __nv_bfloat16* key;
    __nv_bfloat16* value;
    std::int32_t channels;
    std::int32_t query_rows;
    std::int32_t key_rows;
    std::int32_t value_rows;
    std::int32_t global_row_offset;
    std::int32_t width;
    std::int32_t batch_row;
    Publish publish;
    // ConvI8 only: the I8 codes plane reinterpreting state_read and the FP16 scale plane.
    // The plane is stored slot-major: slot s owns the contiguous [groups] block
    // [s * scale_groups, (s + 1) * scale_groups), element (group, slot) at slot * scale_groups + group.
    const std::int8_t* i8_state_read;
    const __half* scale_read;
    std::int32_t scale_groups;

    template <int Tokens>
    __device__ __forceinline__ void store(std::int32_t local_row,
                                          const float (&projected)[Tokens]) const {
        static_assert(Tokens >= 1);
        const std::int32_t row         = global_row_offset + local_row;
        const std::int64_t slot_stride = static_cast<std::int64_t>(channels) * 3;
        const std::int64_t initial_base =
            static_cast<std::int64_t>(initial_slots[batch_row]) * slot_stride;
        std::int32_t valid = valid_columns == nullptr ? Tokens : valid_columns[batch_row];
        valid              = valid < 0 ? 0 : (valid > Tokens ? Tokens : valid);

        const float w0 = __bfloat162float(conv_weight[row]);
        const float w1 = __bfloat162float(conv_weight[channels + row]);
        const float w2 = __bfloat162float(conv_weight[2LL * channels + row]);
        const float w3 = __bfloat162float(conv_weight[3LL * channels + row]);

        if constexpr (!ConvI8) {
            float s0       = __bfloat162float(state_read[initial_base + row]);
            float s1       = __bfloat162float(state_read[initial_base + channels + row]);
            float s2       = __bfloat162float(state_read[initial_base + 2LL * channels + row]);

#pragma unroll
            for (int token = 0; token < Tokens; ++token) {
                const std::int64_t column = static_cast<std::int64_t>(batch_row) * width + token;
                if (token >= valid) {
                    if (row < query_rows) {
                        query[column * query_rows + row] = __float2bfloat16_rn(0.0F);
                    } else if (row < query_rows + key_rows) {
                        key[column * key_rows + row - query_rows] = __float2bfloat16_rn(0.0F);
                    } else {
                        value[column * value_rows + row - query_rows - key_rows] =
                            __float2bfloat16_rn(0.0F);
                    }
                    continue;
                }

                const float p              = projected[token];
                float conv                 = fmaf(w0, s0, 0.0F);
                conv                       = fmaf(w1, s1, conv);
                conv                       = fmaf(w2, s2, conv);
                conv                       = fmaf(w3, p, conv);
                const __nv_bfloat16 output = __float2bfloat16_rn(silu(conv));
                if (row < query_rows) {
                    query[column * query_rows + row] = output;
                } else if (row < query_rows + key_rows) {
                    key[column * key_rows + row - query_rows] = output;
                } else {
                    value[column * value_rows + row - query_rows - key_rows] = output;
                }

                publish.publish(token, batch_row, row, s1, s2, p);
                s0 = s1;
                s1 = s2;
                s2 = p;
            }
        } else {
            const std::int32_t group    = row >> 7;
            const __half scale16        = scale_read[
                static_cast<std::int64_t>(initial_slots[batch_row]) * scale_groups + group];
            const float scale = __half2float(scale16);
            std::int32_t c0    = static_cast<std::int8_t>(i8_state_read[initial_base + row]);
            std::int32_t c1    = static_cast<std::int8_t>(i8_state_read[initial_base + channels + row]);
            std::int32_t c2    = static_cast<std::int8_t>(
                i8_state_read[initial_base + 2LL * channels + row]);
            float s0 = static_cast<float>(c0) * scale;
            float s1 = static_cast<float>(c1) * scale;
            float s2 = static_cast<float>(c2) * scale;

#pragma unroll
            for (int token = 0; token < Tokens; ++token) {
                const std::int64_t column = static_cast<std::int64_t>(batch_row) * width + token;
                if (token >= valid) {
                    if (row < query_rows) {
                        query[column * query_rows + row] = __float2bfloat16_rn(0.0F);
                    } else if (row < query_rows + key_rows) {
                        key[column * key_rows + row - query_rows] = __float2bfloat16_rn(0.0F);
                    } else {
                        value[column * value_rows + row - query_rows - key_rows] =
                            __float2bfloat16_rn(0.0F);
                    }
                    continue;
                }

                const float p               = projected[token];
                float conv                  = fmaf(w0, s0, 0.0F);
                conv                        = fmaf(w1, s1, conv);
                conv                        = fmaf(w2, s2, conv);
                conv                        = fmaf(w3, p, conv);
                const __nv_bfloat16 output  = __float2bfloat16_rn(silu(conv));
                if (row < query_rows) {
                    query[column * query_rows + row] = output;
                } else if (row < query_rows + key_rows) {
                    key[column * key_rows + row - query_rows] = output;
                } else {
                    value[column * value_rows + row - query_rows - key_rows] = output;
                }

                const std::int32_t pc = static_cast<std::int32_t>(i8_quantize(p, scale));
                publish.publish(token, batch_row, row, c1, c2, pc, p, scale16);
                s0 = s1;
                s1 = s2;
                s2 = p;
                c0 = c1;
                c1 = c2;
                c2 = pc;
            }
        }
    }
};

} // namespace ninfer::ops::detail