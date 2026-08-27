#pragma once

#include "ops/common/bf16_vector.cuh"
#include "ops/linear_attention/gated_delta_net/common.cuh"
#include "ops/linear_attention/gated_delta_net/launch.h"

#include <cuda_bf16.h>

#include <cstdint>

namespace ninfer::ops::detail::gated_delta_net {

inline constexpr int kDvPerWarp = 4;
inline constexpr int kNumWarps  = 4;
inline constexpr int kBlockDv   = kNumWarps * kDvPerWarp;
inline constexpr int kQkPerLane = kStateDim / kWarpSize;

static_assert(kStateDim % kWarpSize == 0);
static_assert(kQkPerLane == 4);
static_assert(kStateDim % kBlockDv == 0);

__device__ __forceinline__ void load_qk_lane(float (&reg)[kQkPerLane], const float* base,
                                             std::uint32_t dqk_base) {
    store_vec(reg, load_vec<float4>(base + dqk_base));
}

__device__ __forceinline__ void store_qk_lane(const float (&reg)[kQkPerLane],
                                              const float* base, std::uint32_t dqk_base) {
    auto* dst = const_cast<float*>(base) + dqk_base;
    *reinterpret_cast<float4*>(dst) = load_vec<float4>(reg);
}

// BF16 storage variants: load 4 BF16 elements into a FP32[kQkPerLane] register
// array, and write a FP32[kQkPerLane] register array back to 4 BF16 elements.
// Rounds-to-nearest-even on store, matching the persistent storage dtype.
__device__ __forceinline__ void load_qk_lane(float (&reg)[kQkPerLane], const __nv_bfloat16* base,
                                             std::uint32_t dqk_base) {
    const Bf16x4Pack packed = load_vec<Bf16x4Pack>(base + dqk_base);
    const float2 lo        = bf16x2_to_float2(packed.pair[0]);
    const float2 hi        = bf16x2_to_float2(packed.pair[1]);
    reg[0]                = lo.x;
    reg[1]                = lo.y;
    reg[2]                = hi.x;
    reg[3]                = hi.y;
}

__device__ __forceinline__ void store_qk_lane(const float (&reg)[kQkPerLane],
                                              const __nv_bfloat16* base, std::uint32_t dqk_base) {
    Bf16x4Pack packed;
    packed.pair[0] = __floats2bfloat162_rn(reg[0], reg[1]);
    packed.pair[1] = __floats2bfloat162_rn(reg[2], reg[3]);
    auto* dst = const_cast<__nv_bfloat16*>(base) + dqk_base;
    store_vec(dst, packed);
}

// I8 storage variants: one signed code per element plus one FP16 scale per
// (value_head, dv row) of kStateDim dk values; element = code * scale. Loads convert
// code * scale to FP32. Stores expect the caller to have reduced the absolute row max
// across the full warp (all kStateDim dk values); the row scale is row_max / 127 rounded
// to FP16, codes are round-to-nearest-even and clamped to [-127, 127].
__device__ __forceinline__ void load_qk_lane(float (&reg)[kQkPerLane], const std::int8_t* base,
                                             std::uint32_t dqk_base, float scale) {
    const std::uint32_t packed =
        load_vec<std::uint32_t>(reinterpret_cast<const std::uint32_t*>(base + dqk_base));
#pragma unroll
    for (int i = 0; i < kQkPerLane; ++i) {
        const std::int8_t code = static_cast<std::int8_t>((packed >> (8 * i)) & 0xffU);
        reg[i] = static_cast<float>(code) * scale;
    }
}

__device__ __forceinline__ void store_qk_lane(const float (&reg)[kQkPerLane],
                                              const std::int8_t* base, std::uint32_t dqk_base,
                                              float row_max, __half* row_scale) {
    const __half scale16 = __float2half_rn(row_max / 127.0f);
    const float scale    = __half2float(scale16);
    std::uint32_t packed = 0;
#pragma unroll
    for (int i = 0; i < kQkPerLane; ++i) {
        int code = 0;
        if (scale != 0.0f) {
            code = static_cast<int>(__float2int_rn(reg[i] / scale));
            if (code > 127) { code = 127; }
            if (code < -127) { code = -127; }
        }
        packed |= static_cast<std::uint32_t>(static_cast<std::uint8_t>(code)) << (8 * i);
    }
    store_vec(reinterpret_cast<std::uint32_t*>(const_cast<std::int8_t*>(base) + dqk_base), packed);
    if (row_scale != nullptr) { *row_scale = scale16; }
}

__global__ void __launch_bounds__(kWarpSize* kNumWarps, 2)
    recurrent_fp32_kernel(const float* __restrict__ q, const float* __restrict__ k,
                          const float* __restrict__ v, const float* __restrict__ g,
                          const float* __restrict__ beta, float* __restrict__ ssm_state,
                          float* __restrict__ out, std::int64_t T, head_map heads, float scale) {
    const int lane           = threadIdx.x;
    const int warp_id        = threadIdx.y;
    const std::uint32_t h_v  = static_cast<std::uint32_t>(blockIdx.x);
    const std::uint32_t h_qk = static_cast<std::uint32_t>(heads.qk_head(static_cast<int>(h_v)));

    const std::uint32_t dv_base =
        static_cast<std::uint32_t>(blockIdx.z * kBlockDv + warp_id * kDvPerWarp);
    const std::uint32_t dqk_base = static_cast<std::uint32_t>(lane * kQkPerLane);

    float* state_h = ssm_state + static_cast<std::int64_t>(h_v) * kStateDim * kStateDim;

    __align__(16) float s_tile[kDvPerWarp][kQkPerLane];
#pragma unroll
    for (int r = 0; r < kDvPerWarp; ++r) {
        load_qk_lane(s_tile[r], state_h + static_cast<std::int64_t>(dv_base + r) * kStateDim,
                     dqk_base);
    }

    __align__(16) float k_reg[kQkPerLane];
    load_qk_lane(k_reg, k + static_cast<std::int64_t>(h_qk) * kStateDim, dqk_base);

    for (std::int64_t t = 0; t < T; ++t) {
        const float* v_t          = v + (t * heads.H_v + h_v) * kStateDim;
        const std::int64_t gb_off = t * heads.H_v + h_v;
        const float beta_val      = beta[gb_off];
        const float alpha         = expf(g[gb_off]);

        float v_local = 0.0f;
        if (lane < kDvPerWarp) { v_local = v_t[dv_base + lane]; }

#pragma unroll
        for (int r = 0; r < kDvPerWarp; ++r) {
            float partial = 0.0f;
#pragma unroll
            for (int c = 0; c < kQkPerLane; ++c) { partial += s_tile[r][c] * k_reg[c]; }
            partial = warp_sum<kWarpSize>(partial);

            const float v_r   = __shfl_sync(0xffffffff, v_local, r, kWarpSize);
            const float delta = beta_val * (v_r - alpha * partial);

#pragma unroll
            for (int c = 0; c < kQkPerLane; ++c) {
                s_tile[r][c] = alpha * s_tile[r][c] + delta * k_reg[c];
            }
        }

        if (t + 1 < T) {
            load_qk_lane(k_reg, k + ((t + 1) * heads.H_qk + h_qk) * kStateDim, dqk_base);
        }

        __align__(16) float q_reg[kQkPerLane];
        load_qk_lane(q_reg, q + (t * heads.H_qk + h_qk) * kStateDim, dqk_base);

        float attn_val = 0.0f;
#pragma unroll
        for (int r = 0; r < kDvPerWarp; ++r) {
            float partial = 0.0f;
#pragma unroll
            for (int c = 0; c < kQkPerLane; ++c) { partial += s_tile[r][c] * q_reg[c]; }
            partial = warp_sum<kWarpSize>(partial);
            if (lane == r) { attn_val = partial; }
        }

        if (lane < kDvPerWarp) {
            out[(t * heads.H_v + h_v) * kStateDim + dv_base + lane] = attn_val * scale;
        }
    }

#pragma unroll
    for (int r = 0; r < kDvPerWarp; ++r) {
        store_qk_lane(s_tile[r], state_h + static_cast<std::int64_t>(dv_base + r) * kStateDim,
                      dqk_base);
    }
}

inline constexpr float kQkL2NormEps = 1.0e-6f;

struct RawQkLane {
    Bf16x4Pack bits;
    float value[kQkPerLane];
};

struct RawValueLane {
    __nv_bfloat16 bits;
    float value;
};

struct RawGatePair {
    uint2 bits;
    float g;
    float beta;
};

__device__ __forceinline__ RawQkLane load_raw_qk_lane(const __nv_bfloat16* base,
                                                      std::uint32_t dqk_base) {
    RawQkLane out;
    out.bits        = load_vec<Bf16x4Pack>(base + dqk_base);
    const float2 lo = bf16x2_to_float2(out.bits.pair[0]);
    const float2 hi = bf16x2_to_float2(out.bits.pair[1]);
    out.value[0]    = lo.x;
    out.value[1]    = lo.y;
    out.value[2]    = hi.x;
    out.value[3]    = hi.y;
    return out;
}

template <bool Normalize>
__device__ __forceinline__ void normalize_qk_lane(float (&value)[kQkPerLane], int lane) {
    if constexpr (Normalize) {
        float sum = 0.0f;
#pragma unroll
        for (int i = 0; i < kQkPerLane; ++i) { sum += value[i] * value[i]; }
        sum       = warp_reduce_sum(sum);
        float inv = lane == 0 ? rsqrtf(sum + kQkL2NormEps) : 0.0f;
        inv       = __shfl_sync(kFullWarpMask, inv, 0);
#pragma unroll
        for (int i = 0; i < kQkPerLane; ++i) { value[i] *= inv; }
    }
}

__device__ __forceinline__ RawValueLane load_value_lane(const __nv_bfloat16* base, int lane,
                                                        std::uint32_t dv_base) {
    RawValueLane out{__float2bfloat16(0.0f), 0.0f};
    if (lane < kDvPerWarp) {
        out.bits  = base[dv_base + lane];
        out.value = __bfloat162float(out.bits);
    }
    return out;
}

__device__ __forceinline__ RawGatePair load_source_gate(const float* g, const float* beta,
                                                        std::int64_t offset) {
    const float g_value    = g[offset];
    const float beta_value = beta[offset];
    return {make_uint2(__float_as_uint(g_value), __float_as_uint(beta_value)), g_value, beta_value};
}

__device__ __forceinline__ RawGatePair load_record_gate(const uint2* gate, std::int64_t offset) {
    const uint2 bits = load_vec<uint2>(gate + offset);
    return {bits, __uint_as_float(bits.x), __uint_as_float(bits.y)};
}

__device__ __forceinline__ void apply_gdn_transition(float (&state)[kDvPerWarp][kQkPerLane],
                                                     const float (&key)[kQkPerLane], float v_local,
                                                     float g, float beta) {
    const float alpha = expf(g);

#pragma unroll
    for (int r = 0; r < kDvPerWarp; ++r) {
        float partial = 0.0f;
#pragma unroll
        for (int c = 0; c < kQkPerLane; ++c) { partial += state[r][c] * key[c]; }
        partial = warp_sum<kWarpSize>(partial);

        const float v_r   = __shfl_sync(0xffffffff, v_local, r, kWarpSize);
        const float delta = beta * (v_r - alpha * partial);

#pragma unroll
        for (int c = 0; c < kQkPerLane; ++c) { state[r][c] = alpha * state[r][c] + delta * key[c]; }
    }
}

template <bool Normalize>
__device__ __forceinline__ void readout_and_store(float (&state)[kDvPerWarp][kQkPerLane],
                                                  const __nv_bfloat16* query, __nv_bfloat16* output,
                                                  std::uint32_t dqk_base, std::uint32_t dv_base,
                                                  int lane, float scale) {
    RawQkLane q = load_raw_qk_lane(query, dqk_base);
    normalize_qk_lane<Normalize>(q.value, lane);

    float attn_val = 0.0f;
#pragma unroll
    for (int r = 0; r < kDvPerWarp; ++r) {
        float partial = 0.0f;
#pragma unroll
        for (int c = 0; c < kQkPerLane; ++c) { partial += state[r][c] * q.value[c]; }
        partial = warp_sum<kWarpSize>(partial);
        if (lane == r) { attn_val = partial; }
    }
    if (lane < kDvPerWarp) { output[dv_base + lane] = __float2bfloat16(attn_val * scale); }
}

__device__ __forceinline__ float lane_row_max(const float (&row)[kQkPerLane]) {
    float m = fabsf(row[0]);
#pragma unroll
    for (int c = 1; c < kQkPerLane; ++c) { m = fmaxf(m, fabsf(row[c])); }
    return warp_max(m);
}

template <bool NormalizeQK, class StateReadPtr, class StateWritePtr>
__global__ void __launch_bounds__(kWarpSize* kNumWarps, 2)
    recurrent_bf16_direct_kernel(const __nv_bfloat16* __restrict__ q,
                                 const __nv_bfloat16* __restrict__ k,
                                 const __nv_bfloat16* __restrict__ v, const float* __restrict__ g,
                                 const float* __restrict__ beta,
                                 StateReadPtr __restrict__ state_read,
                                 StateWritePtr __restrict__ state_write,
                                 const __half* __restrict__ state_scale,
                                 __half* __restrict__ state_scale_out,
                                 __nv_bfloat16* __restrict__ out,
                                 std::int32_t width, head_map heads, float scale) {
    const int lane           = threadIdx.x;
    const int warp_id        = threadIdx.y;
    const std::uint32_t h_v  = static_cast<std::uint32_t>(blockIdx.x);
    const std::uint32_t h_qk = static_cast<std::uint32_t>(heads.qk_head(static_cast<int>(h_v)));
    const std::uint32_t dv_base =
        static_cast<std::uint32_t>(blockIdx.z * kBlockDv + warp_id * kDvPerWarp);
    const std::uint32_t dqk_base = static_cast<std::uint32_t>(lane * kQkPerLane);
    auto read_h = state_read + static_cast<std::int64_t>(h_v) * kStateDim * kStateDim;

    __align__(16) float state[kDvPerWarp][kQkPerLane];
    if constexpr (std::is_same_v<std::decay_t<StateReadPtr>, const std::int8_t*>) {
        const __half* read_scale_h = state_scale + static_cast<std::int64_t>(h_v) * kStateDim;
        float read_scale[kDvPerWarp];
#pragma unroll
        for (int r = 0; r < kDvPerWarp; ++r) {
            read_scale[r] = __half2float(read_scale_h[dv_base + r]);
        }
#pragma unroll
        for (int r = 0; r < kDvPerWarp; ++r) {
            load_qk_lane(state[r], read_h + static_cast<std::int64_t>(dv_base + r) * kStateDim,
                         dqk_base, read_scale[r]);
        }
    } else {
#pragma unroll
        for (int r = 0; r < kDvPerWarp; ++r) {
            load_qk_lane(state[r], read_h + static_cast<std::int64_t>(dv_base + r) * kStateDim,
                         dqk_base);
        }
    }

    RawQkLane key = load_raw_qk_lane(k + static_cast<std::int64_t>(h_qk) * kStateDim, dqk_base);
    normalize_qk_lane<NormalizeQK>(key.value, lane);
    for (std::int32_t token = 0; token < width; ++token) {
        const std::int64_t column = token;
        const RawGatePair gate    = load_source_gate(g, beta, column * heads.H_v + h_v);
        const RawValueLane value =
            load_value_lane(v + (column * heads.H_v + h_v) * kStateDim, lane, dv_base);
        apply_gdn_transition(state, key.value, value.value, gate.g, gate.beta);

        if (token + 1 < width) {
            key = load_raw_qk_lane(k + ((column + 1) * heads.H_qk + h_qk) * kStateDim, dqk_base);
            normalize_qk_lane<NormalizeQK>(key.value, lane);
        }

        readout_and_store<NormalizeQK>(state, q + (column * heads.H_qk + h_qk) * kStateDim,
                                       out + (column * heads.H_v + h_v) * kStateDim, dqk_base,
                                       dv_base, lane, scale);
    }

    auto write_h = state_write + static_cast<std::int64_t>(h_v) * kStateDim * kStateDim;
    if constexpr (std::is_same_v<std::decay_t<StateWritePtr>, std::int8_t*>) {
        __half* write_scale_h = state_scale_out + static_cast<std::int64_t>(h_v) * kStateDim;
#pragma unroll
        for (int r = 0; r < kDvPerWarp; ++r) {
            const float row_max = lane_row_max(state[r]);
            store_qk_lane(state[r], write_h + static_cast<std::int64_t>(dv_base + r) * kStateDim,
                          dqk_base, row_max, lane == 0 ? write_scale_h + dv_base + r : nullptr);
        }
    } else {
#pragma unroll
        for (int r = 0; r < kDvPerWarp; ++r) {
            store_qk_lane(state[r], write_h + static_cast<std::int64_t>(dv_base + r) * kStateDim,
                          dqk_base);
        }
    }
}

enum class RecurrentMode {
    Snapshot,
    Record,
    Fold,
};

struct RecurrentCoordinates {
    int lane;
    int warp;
    std::int32_t batch;
    std::int32_t layer;
    std::int32_t state_tile;
    std::uint32_t value_head;
    std::uint32_t qk_head;
    std::uint32_t dv_base;
    std::uint32_t dqk_base;
};

__device__ __forceinline__ RecurrentCoordinates make_coordinates(std::int32_t batch,
                                                                 std::int32_t layer,
                                                                 std::int32_t state_tile,
                                                                 head_map heads) {
    const int lane                 = threadIdx.x;
    const int warp                 = threadIdx.y;
    const std::uint32_t value_head = static_cast<std::uint32_t>(blockIdx.x);
    const std::uint32_t qk_head =
        static_cast<std::uint32_t>(heads.qk_head(static_cast<int>(value_head)));
    const std::uint32_t dv_base =
        static_cast<std::uint32_t>(state_tile * kBlockDv + warp * kDvPerWarp);
    return {lane,    warp,       batch,
            layer,   state_tile, value_head,
            qk_head, dv_base,    static_cast<std::uint32_t>(lane * kQkPerLane)};
}

template <bool Batched, bool Masked, class StatePtr = float>
struct SnapshotAccess {
    using state_ptr_t = StatePtr;
    const __nv_bfloat16* q;
    const __nv_bfloat16* k;
    const __nv_bfloat16* v;
    const float* g;
    const float* beta;
    StatePtr states;
    // FP16 per-(value_head, dv row) scale plane [value_head_dim, value_heads, slots]; null for
    // non-I8 state storage.
    __half* state_scale;
    const std::int32_t* valid_columns;
    const std::int32_t* initial_slots;
    const std::int32_t* snapshot_bases;
    __nv_bfloat16* out;
    head_map heads;
    std::int32_t width;
    std::int64_t state_slot_stride;
    float scale;

    __device__ __forceinline__ RecurrentCoordinates coordinates() const {
        const std::int32_t batch = Batched ? static_cast<std::int32_t>(blockIdx.y) : 0;
        return make_coordinates(batch, 0, static_cast<std::int32_t>(blockIdx.z), heads);
    }

    __device__ __forceinline__ std::int32_t
    active_columns(const RecurrentCoordinates& coord) const {
        if constexpr (Masked) { return valid_columns[coord.batch]; }
        return width;
    }

    __device__ __forceinline__ std::int64_t column(const RecurrentCoordinates& coord,
                                                   std::int32_t token) const {
        return static_cast<std::int64_t>(coord.batch) * width + token;
    }

    __device__ __forceinline__ StatePtr
    state_read_base(const RecurrentCoordinates& coord) const {
        return states +
               static_cast<std::int64_t>(initial_slots[coord.batch]) * state_slot_stride +
               static_cast<std::int64_t>(coord.value_head) * kStateDim * kStateDim;
    }

    __device__ __forceinline__ std::int64_t scale_slot_stride() const {
        return static_cast<std::int64_t>(kStateDim) * heads.H_v;
    }

    __device__ __forceinline__ const __half*
    state_scale_read_base(const RecurrentCoordinates& coord) const {
        return state_scale +
               static_cast<std::int64_t>(initial_slots[coord.batch]) * scale_slot_stride() +
               static_cast<std::int64_t>(coord.value_head) * kStateDim;
    }

    __device__ __forceinline__ __half*
    state_scale_write_base(const RecurrentCoordinates& coord, std::int32_t token) const {
        return state_scale +
               static_cast<std::int64_t>(snapshot_bases[coord.batch] + token) * scale_slot_stride() +
               static_cast<std::int64_t>(coord.value_head) * kStateDim;
    }

    __device__ __forceinline__ const __nv_bfloat16* key_ptr(const RecurrentCoordinates& coord,
                                                            std::int32_t token) const {
        return k + (column(coord, token) * heads.H_qk + coord.qk_head) * kStateDim;
    }

    __device__ __forceinline__ const __nv_bfloat16* value_ptr(const RecurrentCoordinates& coord,
                                                              std::int32_t token) const {
        return v + (column(coord, token) * heads.H_v + coord.value_head) * kStateDim;
    }

    __device__ __forceinline__ RawGatePair load_gate(const RecurrentCoordinates& coord,
                                                     std::int32_t token) const {
        return load_source_gate(g, beta, column(coord, token) * heads.H_v + coord.value_head);
    }

    __device__ __forceinline__ const __nv_bfloat16* query_ptr(const RecurrentCoordinates& coord,
                                                              std::int32_t token) const {
        return q + (column(coord, token) * heads.H_qk + coord.qk_head) * kStateDim;
    }

    __device__ __forceinline__ __nv_bfloat16* output_ptr(const RecurrentCoordinates& coord,
                                                         std::int32_t token) const {
        return out + (column(coord, token) * heads.H_v + coord.value_head) * kStateDim;
    }

    __device__ __forceinline__ void
    store_snapshot(const RecurrentCoordinates& coord, std::int32_t token,
                   const float (&state)[kDvPerWarp][kQkPerLane]) const {
        auto* snapshot = states +
                         static_cast<std::int64_t>(snapshot_bases[coord.batch] + token) *
                             state_slot_stride +
                         static_cast<std::int64_t>(coord.value_head) * kStateDim * kStateDim;
        if constexpr (std::is_same_v<StatePtr, std::int8_t*>) {
            __half* snapshot_scale = state_scale_write_base(coord, token);
#pragma unroll
            for (int r = 0; r < kDvPerWarp; ++r) {
                const float row_max = lane_row_max(state[r]);
                store_qk_lane(state[r],
                              snapshot + static_cast<std::int64_t>(coord.dv_base + r) * kStateDim,
                              coord.dqk_base, row_max,
                              coord.lane == 0 ? snapshot_scale + coord.dv_base + r : nullptr);
            }
        } else {
#pragma unroll
            for (int r = 0; r < kDvPerWarp; ++r) {
                store_qk_lane(state[r],
                              snapshot + static_cast<std::int64_t>(coord.dv_base + r) * kStateDim,
                              coord.dqk_base);
            }
        }
    }
};

template <bool Masked, class StatePtr = float>
struct RecordAccess {
    using state_ptr_t = StatePtr;
    const __nv_bfloat16* q;
    const __nv_bfloat16* k;
    const __nv_bfloat16* v;
    const float* g;
    const float* beta;
    StatePtr states;
    // FP16 per-(value_head, dv row) scale plane [value_head_dim, value_heads, slots] (read-only
    // for the record pass); null for non-I8 state storage.
    const __half* state_scale;
    const std::int32_t* valid_columns;
    const std::int32_t* initial_slots;
    __nv_bfloat16* key_record;
    __nv_bfloat16* value_record;
    uint2* gate_record;
    __nv_bfloat16* out;
    head_map heads;
    std::int32_t width;
    std::int64_t state_slot_stride;
    float scale;

    __device__ __forceinline__ RecurrentCoordinates coordinates() const {
        return make_coordinates(static_cast<std::int32_t>(blockIdx.y), 0,
                                static_cast<std::int32_t>(blockIdx.z), heads);
    }

    __device__ __forceinline__ std::int32_t
    active_columns(const RecurrentCoordinates& coord) const {
        if constexpr (Masked) { return valid_columns[coord.batch]; }
        return width;
    }

    __device__ __forceinline__ std::int64_t column(const RecurrentCoordinates& coord,
                                                   std::int32_t token) const {
        return static_cast<std::int64_t>(coord.batch) * width + token;
    }

    __device__ __forceinline__ StatePtr
    state_read_base(const RecurrentCoordinates& coord) const {
        return states + static_cast<std::int64_t>(initial_slots[coord.batch]) * state_slot_stride +
               static_cast<std::int64_t>(coord.value_head) * kStateDim * kStateDim;
    }

    __device__ __forceinline__ const __half*
    state_scale_read_base(const RecurrentCoordinates& coord) const {
        return state_scale +
               static_cast<std::int64_t>(initial_slots[coord.batch]) *
                   (static_cast<std::int64_t>(kStateDim) * heads.H_v) +
               static_cast<std::int64_t>(coord.value_head) * kStateDim;
    }

    __device__ __forceinline__ const __nv_bfloat16* key_ptr(const RecurrentCoordinates& coord,
                                                            std::int32_t token) const {
        return k + (column(coord, token) * heads.H_qk + coord.qk_head) * kStateDim;
    }

    __device__ __forceinline__ const __nv_bfloat16* value_ptr(const RecurrentCoordinates& coord,
                                                              std::int32_t token) const {
        return v + (column(coord, token) * heads.H_v + coord.value_head) * kStateDim;
    }

    __device__ __forceinline__ RawGatePair load_gate(const RecurrentCoordinates& coord,
                                                     std::int32_t token) const {
        return load_source_gate(g, beta, column(coord, token) * heads.H_v + coord.value_head);
    }

    __device__ __forceinline__ const __nv_bfloat16* query_ptr(const RecurrentCoordinates& coord,
                                                              std::int32_t token) const {
        return q + (column(coord, token) * heads.H_qk + coord.qk_head) * kStateDim;
    }

    __device__ __forceinline__ __nv_bfloat16* output_ptr(const RecurrentCoordinates& coord,
                                                         std::int32_t token) const {
        return out + (column(coord, token) * heads.H_v + coord.value_head) * kStateDim;
    }

    __device__ __forceinline__ void store_key(const RecurrentCoordinates& coord, std::int32_t token,
                                              const RawQkLane& raw) const {
        if (coord.state_tile == 0 && coord.warp == 0 &&
            static_cast<int>(coord.value_head) % heads.group_size() == 0) {
            __nv_bfloat16* destination =
                key_record + (column(coord, token) * heads.H_qk + coord.qk_head) * kStateDim;
            store_vec(destination + coord.dqk_base, raw.bits);
        }
    }

    __device__ __forceinline__ void store_value(const RecurrentCoordinates& coord,
                                                std::int32_t token, const RawValueLane& raw) const {
        if (coord.lane < kDvPerWarp) {
            __nv_bfloat16* destination =
                value_record + (column(coord, token) * heads.H_v + coord.value_head) * kStateDim;
            destination[coord.dv_base + coord.lane] = raw.bits;
        }
    }

    __device__ __forceinline__ void store_gate(const RecurrentCoordinates& coord,
                                               std::int32_t token, const RawGatePair& raw) const {
        if (coord.state_tile == 0 && coord.warp == 0 && coord.lane == 0) {
            gate_record[column(coord, token) * heads.H_v + coord.value_head] = raw.bits;
        }
    }
};

template <int Layers, int QkHeads, int ValueHeads, int ConvChannels>
struct FoldGeometry {
    static constexpr int kLayers       = Layers;
    static constexpr int kQkHeads      = QkHeads;
    static constexpr int kValueHeads   = ValueHeads;
    static constexpr int kConvChannels = ConvChannels;
    static_assert(ValueHeads % QkHeads == 0);
    static_assert(ConvChannels % 128 == 0);
};

using FoldGeometry48x48 = FoldGeometry<48, 16, 48, 10240>;
using FoldGeometry30x32 = FoldGeometry<30, 16, 32, 8192>;
using FoldGeometry24x32 = FoldGeometry<24, 16, 32, 8192>;

template <class Geometry, class StatePtr = float>
struct FoldAccess {
    using state_ptr_t = StatePtr;
    const __nv_bfloat16* key_record;
    const __nv_bfloat16* value_record;
    const uint2* gate_record;
    const __nv_bfloat16* conv_record;
    StatePtr recurrent_layer0;
    // FP16 per-(value_head, dv row) scale plane, layer 0; null for non-I8 state storage.
    __half* recurrent_scale_layer0;
    __nv_bfloat16* conv_layer0;
    std::int64_t recurrent_layer_stride;
    std::int64_t recurrent_scale_layer_stride;
    std::int64_t conv_layer_stride;
    std::int32_t record_capacity;
    std::int32_t width;
    GdnReplayFoldKernelRows rows;
    // I8 conv state only: the FP16 conv scale plane, layer 0 (null for BF16 conv
    // storage). The plane is slot-major: slot s owns the contiguous [groups] block
    // [s * conv_scale_groups, (s + 1) * conv_scale_groups), element (group, slot) at
    // slot * conv_scale_groups + group. When non-null, conv_layer0 holds I8 codes (one
    // byte per element) and conv_layer_stride is in two-byte units; conv history is
    // published as shifted codes plus the record values quantized under the slot's
    // group scale, which is read-only and sticky.
    const __half* conv_scale_layer0;
    std::int64_t conv_scale_layer_stride;
    std::int32_t conv_scale_groups;

    __device__ __forceinline__ RecurrentCoordinates coordinates() const {
        const std::int32_t batch       = static_cast<std::int32_t>(blockIdx.y);
        const std::int32_t layer_tile  = static_cast<std::int32_t>(blockIdx.z);
        const int lane                 = threadIdx.x;
        const int warp                 = threadIdx.y;
        const std::int32_t state_tile  = layer_tile & 7;
        const std::uint32_t value_head = static_cast<std::uint32_t>(blockIdx.x);
        constexpr std::uint32_t kGroup = Geometry::kValueHeads / Geometry::kQkHeads;
        const std::uint32_t qk_head    = value_head / kGroup;
        const std::uint32_t dv_base =
            static_cast<std::uint32_t>(state_tile * kBlockDv + warp * kDvPerWarp);
        return {lane,
                warp,
                batch,
                layer_tile >> 3,
                state_tile,
                value_head,
                qk_head,
                dv_base,
                static_cast<std::uint32_t>(lane * kQkPerLane)};
    }

    __device__ __forceinline__ std::int32_t
    active_columns(const RecurrentCoordinates& coord) const {
        return rows.row[coord.batch].commit_columns;
    }

    __device__ __forceinline__ std::int64_t record_outer(const RecurrentCoordinates& coord) const {
        return static_cast<std::int64_t>(coord.layer) * record_capacity + coord.batch;
    }

    __device__ __forceinline__ StatePtr state_read_base(const RecurrentCoordinates& coord) const {
        const std::int64_t slot_stride =
            static_cast<std::int64_t>(Geometry::kValueHeads) * kStateDim * kStateDim;
        return recurrent_layer0 + static_cast<std::int64_t>(coord.layer) * recurrent_layer_stride +
               static_cast<std::int64_t>(rows.row[coord.batch].linear_state_slot) * slot_stride +
               static_cast<std::int64_t>(coord.value_head) * kStateDim * kStateDim;
    }

    __device__ __forceinline__ __half*
    state_scale_base(const RecurrentCoordinates& coord) const {
        const std::int64_t slot_stride =
            static_cast<std::int64_t>(Geometry::kValueHeads) * kStateDim;
        return recurrent_scale_layer0 +
               static_cast<std::int64_t>(coord.layer) * recurrent_scale_layer_stride +
               static_cast<std::int64_t>(rows.row[coord.batch].linear_state_slot) * slot_stride +
               static_cast<std::int64_t>(coord.value_head) * kStateDim;
    }

    __device__ __forceinline__ const __half*
    state_scale_read_base(const RecurrentCoordinates& coord) const {
        return state_scale_base(coord);
    }

    __device__ __forceinline__ const __nv_bfloat16* key_ptr(const RecurrentCoordinates& coord,
                                                            std::int32_t token) const {
        const std::int64_t column = record_outer(coord) * width + token;
        return key_record + (column * Geometry::kQkHeads + coord.qk_head) * kStateDim;
    }

    __device__ __forceinline__ const __nv_bfloat16* value_ptr(const RecurrentCoordinates& coord,
                                                              std::int32_t token) const {
        const std::int64_t column = record_outer(coord) * width + token;
        return value_record + (column * Geometry::kValueHeads + coord.value_head) * kStateDim;
    }

    __device__ __forceinline__ RawGatePair load_gate(const RecurrentCoordinates& coord,
                                                     std::int32_t token) const {
        const std::int64_t column = record_outer(coord) * width + token;
        return load_record_gate(gate_record, column * Geometry::kValueHeads + coord.value_head);
    }

    __device__ __forceinline__ void
    store_final_state(const RecurrentCoordinates& coord,
                      const float (&state)[kDvPerWarp][kQkPerLane]) const {
        auto* destination = state_read_base(coord);
        if constexpr (std::is_same_v<StatePtr, std::int8_t*>) {
            __half* destination_scale = state_scale_base(coord);
#pragma unroll
            for (int r = 0; r < kDvPerWarp; ++r) {
                const float row_max = lane_row_max(state[r]);
                store_qk_lane(state[r],
                              destination + static_cast<std::int64_t>(coord.dv_base + r) * kStateDim,
                              coord.dqk_base, row_max,
                              coord.lane == 0 ? destination_scale + coord.dv_base + r : nullptr);
            }
        } else {
#pragma unroll
            for (int r = 0; r < kDvPerWarp; ++r) {
                store_qk_lane(state[r],
                              destination + static_cast<std::int64_t>(coord.dv_base + r) * kStateDim,
                              coord.dqk_base);
            }
        }
    }

    __device__ __forceinline__ void publish_final_conv_history(const RecurrentCoordinates& coord,
                                                               std::int32_t commit) const {
        const std::int32_t tile_block =
            static_cast<std::int32_t>(coord.value_head) * 8 + coord.state_tile;
        if (tile_block >= Geometry::kConvChannels / 128) { return; }

        const std::int32_t tid     = coord.warp * kWarpSize + coord.lane;
        const std::int32_t channel = tile_block * 128 + tid;
        if (conv_scale_layer0 == nullptr) {
            __nv_bfloat16* history =
                conv_layer0 + static_cast<std::int64_t>(coord.layer) * conv_layer_stride +
                static_cast<std::int64_t>(rows.row[coord.batch].linear_state_slot) *
                    (3LL * Geometry::kConvChannels) +
                channel;
            const __nv_bfloat16* record =
                conv_record + record_outer(coord) * width * Geometry::kConvChannels + channel;

            __nv_bfloat16 h0;
            __nv_bfloat16 h1;
            __nv_bfloat16 h2;
            if (commit == 1) {
                h0 = history[Geometry::kConvChannels];
                h1 = history[2LL * Geometry::kConvChannels];
                h2 = record[0];
            } else if (commit == 2) {
                h0 = history[2LL * Geometry::kConvChannels];
                h1 = record[0];
                h2 = record[Geometry::kConvChannels];
            } else {
                h0 = record[static_cast<std::int64_t>(commit - 3) * Geometry::kConvChannels];
                h1 = record[static_cast<std::int64_t>(commit - 2) * Geometry::kConvChannels];
                h2 = record[static_cast<std::int64_t>(commit - 1) * Geometry::kConvChannels];
            }
            history[0]                             = h0;
            history[Geometry::kConvChannels]       = h1;
            history[2LL * Geometry::kConvChannels] = h2;
            return;
        }

        // I8 conv storage: the per-(128-channel group, slot) FP16 scale is read-only and sticky;
        // old codes shift verbatim and only the incoming record taps are quantized under the
// slot's group scale. Code layer stride is stored in two-byte units; for I8 storage each
        // code is one byte, so the per-layer code count is 2 * conv_layer_stride. The per-slot
        // layout is tap-major [3, C]: tap h of channel c sits at slot * 3*C + h*C + c.
        constexpr std::int64_t kSlotStride = 3LL * Geometry::kConvChannels;
        constexpr std::int64_t kTapStride  = Geometry::kConvChannels;
        const std::int32_t slot            = rows.row[coord.batch].linear_state_slot;
        std::int8_t* codes = reinterpret_cast<std::int8_t*>(conv_layer0) +
                              static_cast<std::int64_t>(coord.layer) * (2 * conv_layer_stride) +
                              static_cast<std::int64_t>(slot) * kSlotStride + channel;
        const __nv_bfloat16* record =
            conv_record + record_outer(coord) * width * Geometry::kConvChannels + channel;
        const float scale =
            __half2float(conv_scale_layer0[static_cast<std::int64_t>(coord.layer) *
                                               conv_scale_layer_stride +
                                           static_cast<std::int64_t>(slot) * conv_scale_groups +
                                           tile_block]);
        const auto q = [&record, scale](std::int32_t token) {
            return i8_quantize(__bfloat162float(record[static_cast<std::int64_t>(token) *
                                                       Geometry::kConvChannels]),
                               scale);
        };

        std::int8_t h0;
        std::int8_t h1;
        std::int8_t h2;
        if (commit == 1) {
            h0 = codes[kTapStride];
            h1 = codes[2 * kTapStride];
            h2 = q(0);
        } else if (commit == 2) {
            h0 = codes[2 * kTapStride];
            h1 = q(0);
            h2 = q(1);
        } else {
            h0 = q(commit - 3);
            h1 = q(commit - 2);
            h2 = q(commit - 1);
        }
        codes[0]             = h0;
        codes[kTapStride]    = h1;
        codes[2 * kTapStride] = h2;
    }
};

template <RecurrentMode Mode, bool NormalizeInputs, class Access>
__device__ __forceinline__ void recurrent_bf16_body(const Access& access,
                                                    const RecurrentCoordinates& coord,
                                                    std::int32_t width, std::int32_t valid) {
    if constexpr (Mode == RecurrentMode::Fold) {
        if (valid == 0) { return; }
    }

    const auto initial = access.state_read_base(coord);
    __align__(16) float state[kDvPerWarp][kQkPerLane];
    if constexpr (std::is_same_v<typename Access::state_ptr_t, std::int8_t*> ||
                  std::is_same_v<typename Access::state_ptr_t, const std::int8_t*>) {
        const __half* read_scale = access.state_scale_read_base(coord);
        float read_scale_row[kDvPerWarp];
#pragma unroll
        for (int r = 0; r < kDvPerWarp; ++r) {
            read_scale_row[r] = __half2float(read_scale[coord.dv_base + r]);
        }
#pragma unroll
        for (int r = 0; r < kDvPerWarp; ++r) {
            load_qk_lane(state[r], initial + static_cast<std::int64_t>(coord.dv_base + r) * kStateDim,
                         coord.dqk_base, read_scale_row[r]);
        }
    } else {
#pragma unroll
        for (int r = 0; r < kDvPerWarp; ++r) {
            load_qk_lane(state[r], initial + static_cast<std::int64_t>(coord.dv_base + r) * kStateDim,
                         coord.dqk_base);
        }
    }

    RawQkLane key = load_raw_qk_lane(access.key_ptr(coord, 0), coord.dqk_base);
    if constexpr (Mode == RecurrentMode::Record) { access.store_key(coord, 0, key); }
    normalize_qk_lane<NormalizeInputs>(key.value, coord.lane);

    for (std::int32_t token = 0; token < valid; ++token) {
        const RawGatePair gate = access.load_gate(coord, token);
        const RawValueLane value =
            load_value_lane(access.value_ptr(coord, token), coord.lane, coord.dv_base);
        if constexpr (Mode == RecurrentMode::Record) {
            access.store_value(coord, token, value);
            access.store_gate(coord, token, gate);
        }

        apply_gdn_transition(state, key.value, value.value, gate.g, gate.beta);

        if (token + 1 < valid) {
            key = load_raw_qk_lane(access.key_ptr(coord, token + 1), coord.dqk_base);
            if constexpr (Mode == RecurrentMode::Record) {
                access.store_key(coord, token + 1, key);
            }
            normalize_qk_lane<NormalizeInputs>(key.value, coord.lane);
        }

        if constexpr (Mode != RecurrentMode::Fold) {
            readout_and_store<NormalizeInputs>(state, access.query_ptr(coord, token),
                                               access.output_ptr(coord, token), coord.dqk_base,
                                               coord.dv_base, coord.lane, access.scale);
        }
        if constexpr (Mode == RecurrentMode::Snapshot) {
            access.store_snapshot(coord, token, state);
        }
    }

    if constexpr (Mode == RecurrentMode::Fold) {
        access.store_final_state(coord, state);
        access.publish_final_conv_history(coord, valid);
    } else {
        if (coord.lane < kDvPerWarp) {
            for (std::int32_t token = valid; token < width; ++token) {
                access.output_ptr(coord, token)[coord.dv_base + coord.lane] =
                    __float2bfloat16(0.0f);
            }
        }
    }
}

template <bool NormalizeInputs, bool Batched, bool Masked, class StatePtr>
__global__ void __launch_bounds__(kWarpSize* kNumWarps, 2)
    recurrent_snapshot_kernel(SnapshotAccess<Batched, Masked, StatePtr> access) {
    static_assert(!Masked || Batched);
    const RecurrentCoordinates coord = access.coordinates();
    recurrent_bf16_body<RecurrentMode::Snapshot, NormalizeInputs>(access, coord, access.width,
                                                                  access.active_columns(coord));
}

template <bool Masked, class StatePtr>
__global__ void __launch_bounds__(kWarpSize* kNumWarps, 2)
    recurrent_record_kernel(RecordAccess<Masked, StatePtr> access) {
    const RecurrentCoordinates coord = access.coordinates();
    recurrent_bf16_body<RecurrentMode::Record, true>(access, coord, access.width,
                                                     access.active_columns(coord));
}

template <class Geometry, class StatePtr>
__global__ void __launch_bounds__(kWarpSize* kNumWarps, 2)
    recurrent_fold_kernel(const __grid_constant__ FoldAccess<Geometry, StatePtr> access) {
    const RecurrentCoordinates coord = access.coordinates();
    recurrent_bf16_body<RecurrentMode::Fold, true>(access, coord, access.width,
                                                   access.active_columns(coord));
}

} // namespace ninfer::ops::detail::gated_delta_net
