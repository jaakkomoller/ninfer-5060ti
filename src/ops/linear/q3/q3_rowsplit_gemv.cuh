#pragma once

// Q3G64 RowSplit x BF16 GEMV (T = 1 residual dot product).
//
// out[Rows] = W[Rows, K] * x[K]
//
// One row per CTA, eight warps per row, dynamic group ownership. Raw Q3 codes
// (24 bytes per 64-code group) and adjacent FP16 scales are staged per tile
// with 16-byte synchronous vector loads. Because a Q3 group is 24 bytes, code
// transfers move whole two-group units of 48 bytes so every 16-byte vector
// stays globally 16-byte aligned. Lanes decode two codes per group directly
// from the packed shared bytes and accumulate FP32 FMA.

#include "core/pdl.cuh"
#include "ops/common/memory.cuh"
#include "ops/common/warp.cuh"
#include "ops/linear/q3/q3_rowsplit_storage.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops::detail {

template <int RowsPerCta_, int WarpsPerRow_, int GroupsPerWarpTile_, int LaunchBoundsMinBlocks_>
struct Q3RowSplitGemvSchedule {
    static constexpr int kRowsPerCta            = RowsPerCta_;
    static constexpr int kWarpsPerRow           = WarpsPerRow_;
    static constexpr int kGroupsPerWarpTile     = GroupsPerWarpTile_;
    static constexpr int kLaunchBoundsMinBlocks = LaunchBoundsMinBlocks_;

    static constexpr int kCtaWarps = kRowsPerCta * kWarpsPerRow;
    static constexpr int kThreads  = kCtaWarps * 32;
    static constexpr int kCodeBytesPerTile =
        kGroupsPerWarpTile * Q3RowSplitStorage::kCodeBytesPerGroup;
    static constexpr int kCodeVecsPerTile = kCodeBytesPerTile / static_cast<int>(sizeof(uint4));

    static_assert(kCtaWarps <= 32, "Q3 GEMV cannot exceed the CUDA CTA warp limit");
    static_assert(kThreads <= 1024, "Q3 GEMV cannot exceed the CUDA CTA thread limit");
    static_assert((kGroupsPerWarpTile % 2) == 0,
                  "Q3 GEMV group tiles must contain whole two-group units");
    static_assert(kCodeVecsPerTile <= 32,
                  "Q3 GEMV code tile must fit one vector per lane");
    static_assert(kCodeBytesPerTile % static_cast<int>(sizeof(uint4)) == 0,
                  "Q3 GEMV code tile must decompose into complete 16-byte vectors");
    static_assert(kLaunchBoundsMinBlocks >= 1,
                  "Q3 GEMV launch-bounds occupancy must be positive");
};

// Decodes the two codes owned by `lane` (bit ranges [6*lane, 6*lane+3) and
// [6*lane+3, 6*lane+6) of the 24-byte group word) from packed shared bytes.
__device__ __forceinline__ void q3_gemv_decode_lane_pair(const std::uint8_t* group_base, int lane,
                                                         int& code0, int& code1) {
    const int bit0   = 6 * lane;
    const int byte0  = bit0 >> 3;
    const int shift0 = bit0 & 7;
    std::uint32_t word0 = static_cast<std::uint32_t>(group_base[byte0]);
    if (shift0 >= 6) { word0 |= static_cast<std::uint32_t>(group_base[byte0 + 1]) << 8; }
    code0 = static_cast<int>(word0 >> shift0) & 0x7u;

    const int bit1   = bit0 + 3;
    const int byte1  = bit1 >> 3;
    const int shift1 = bit1 & 7;
    std::uint32_t word1 = static_cast<std::uint32_t>(group_base[byte1]);
    if (shift1 >= 6) { word1 |= static_cast<std::uint32_t>(group_base[byte1 + 1]) << 8; }
    code1 = static_cast<int>(word1 >> shift1) & 0x7u;
}

template <class Schedule>
__device__ __forceinline__ float q3_gemv_dot_byte_sync(std::uint8_t* __restrict__ shared_codes,
                                                       const __nv_bfloat16* __restrict__ activation,
                                                       const std::uint8_t* __restrict__ code_row,
                                                       const std::uint8_t* __restrict__ scale_row,
                                                       int group_begin, int group_end, int lane) {
    constexpr int kGroupsPerTile = Schedule::kGroupsPerWarpTile;
    constexpr int kVecsPerUnit   = 3;  // one two-group unit is 48 bytes = 3 x 16 B

    float accumulator0 = 0.0f;
    float accumulator1 = 0.0f;

    for (int tile_group_begin = group_begin; tile_group_begin < group_end;
         tile_group_begin += kGroupsPerTile) {
        const int active_groups = min(kGroupsPerTile, group_end - tile_group_begin);
        const int units         = active_groups / 2;
        const int residual      = active_groups - units * 2;
        const int code_vecs     = units * kVecsPerUnit;

        if (lane < code_vecs) {
            const int unit     = lane / kVecsPerUnit;
            const int offset   = (tile_group_begin + unit * 2) * Q3RowSplitStorage::kCodeBytesPerGroup +
                                 (lane % kVecsPerUnit) * 16;
            *reinterpret_cast<uint4*>(shared_codes + lane * 16) =
                *reinterpret_cast<const uint4*>(code_row + offset);
        } else if (residual != 0) {
            const int offset = (tile_group_begin + units * 2) *
                                   Q3RowSplitStorage::kCodeBytesPerGroup +
                               (lane - code_vecs) * 8;
            if (lane - code_vecs == 0) {
                *reinterpret_cast<uint4*>(shared_codes + lane * 16) =
                    *reinterpret_cast<const uint4*>(code_row + offset);
            } else {
                *reinterpret_cast<std::uint64_t*>(shared_codes + lane * 16) =
                    *reinterpret_cast<const std::uint64_t*>(code_row + offset);
            }
        }
        __syncwarp();

        std::uint16_t lane_scale_bits = 0;
        if (lane < active_groups) {
            lane_scale_bits = load_vec<std::uint16_t>(
                scale_row + static_cast<std::int64_t>(tile_group_begin + lane) *
                                Q3RowSplitStorage::kScaleBytesPerGroup);
        }

        const auto* activation_pairs = reinterpret_cast<const __nv_bfloat162*>(activation);
#pragma unroll 1
        for (int local_group = 0; local_group < active_groups; local_group += 2) {
            const std::uint16_t scale_bits0 =
                static_cast<std::uint16_t>(__shfl_sync(kFullWarpMask, lane_scale_bits, local_group));
            const std::uint16_t scale_bits1 = static_cast<std::uint16_t>(
                __shfl_sync(kFullWarpMask, lane_scale_bits, local_group + 1));
            const float scale0 = __half2float(__ushort_as_half(scale_bits0));
            const float scale1 = __half2float(__ushort_as_half(scale_bits1));

            const std::uint8_t* group0 = shared_codes + local_group * Q3RowSplitStorage::kCodeBytesPerGroup;
            const std::uint8_t* group1 = group0 + Q3RowSplitStorage::kCodeBytesPerGroup;
            int code00, code01, code10, code11;
            q3_gemv_decode_lane_pair(group0, lane, code00, code01);
            q3_gemv_decode_lane_pair(group1, lane, code10, code11);

            const int k_begin =
                (tile_group_begin + local_group) * Q3RowSplitStorage::kGroupK + lane * 2;
            const int k_begin_next = k_begin + Q3RowSplitStorage::kGroupK;
            const float2 x0 = __bfloat1622float2(activation_pairs[k_begin >> 1]);
            const float2 x1 = __bfloat1622float2(activation_pairs[k_begin_next >> 1]);
            accumulator0 = fmaf(static_cast<float>(code00 - 4) * scale0, x0.x, accumulator0);
            accumulator0 = fmaf(static_cast<float>(code01 - 4) * scale0, x0.y, accumulator0);
            accumulator1 = fmaf(static_cast<float>(code10 - 4) * scale1, x1.x, accumulator1);
            accumulator1 = fmaf(static_cast<float>(code11 - 4) * scale1, x1.y, accumulator1);
        }
        __syncwarp();
    }
    return accumulator0 + accumulator1;
}

template <bool SplitOutput, int SplitRow>
__device__ __forceinline__ void q3_gemv_store(__nv_bfloat16* out, __nv_bfloat16* out_tail,
                                              int output_row, float value) {
    if constexpr (SplitOutput) {
        if (output_row < SplitRow) {
            out[output_row] = __float2bfloat16(value);
        } else {
            out_tail[output_row - SplitRow] = __float2bfloat16(value);
        }
    } else {
        out[output_row] = __float2bfloat16(value);
    }
}

// clang-format off
struct Q3GemvStoreEpilogue {
    template <bool SplitOutput, int SplitRow>
    __device__ __forceinline__ void operator()(__nv_bfloat16* out, __nv_bfloat16* out_tail,
                                               int row, float value) const {
        q3_gemv_store<SplitOutput, SplitRow>(out, out_tail, row, value);
    }
};

template <class Schedule, bool SplitOutput = false, int SplitRow = 0,
          class Epilogue = Q3GemvStoreEpilogue, bool TriggerPdl = false, bool JoinPdl = false>
__global__ __launch_bounds__(Schedule::kThreads, Schedule::kLaunchBoundsMinBlocks)
void q3_rowsplit_gemv_kernel(
    const __nv_bfloat16* __restrict__ x,
    const std::uint8_t* __restrict__ codes,
    const std::uint8_t* __restrict__ scales,
    __nv_bfloat16* __restrict__ out,
    __nv_bfloat16* __restrict__ out_tail,
    std::int32_t rows,
    std::int32_t k,
    Epilogue epilogue = {}) {
    // clang-format on
    constexpr int kRowsPerCta  = Schedule::kRowsPerCta;
    constexpr int kWarpsPerRow = Schedule::kWarpsPerRow;
    static_assert(!SplitOutput || SplitRow > 0,
                  "split-output Q3 GEMV requires a positive compile-time seam");

    if constexpr (TriggerPdl) {
        if (threadIdx.x == 0) { pdl::trigger_dependents(); }
    }

    __shared__ __align__(16) std::uint8_t
        shared_codes[kWarpsPerRow][Schedule::kCodeBytesPerTile];
    __shared__ float row_partials[kRowsPerCta][kWarpsPerRow];

    const int lane       = static_cast<int>(threadIdx.x) & 31;
    const int cta_warp   = static_cast<int>(threadIdx.x) >> 5;
    const int row_in_cta = cta_warp / kWarpsPerRow;
    const int warp_in_row = cta_warp % kWarpsPerRow;
    const int row        = static_cast<int>(blockIdx.x) * kRowsPerCta + row_in_cta;
    if constexpr (kRowsPerCta > 1) {
        if (row >= rows) { return; }
    }

    const int groups_per_row = k / Q3RowSplitStorage::kGroupK;
    const int pairs_per_row  = groups_per_row / 2;
    const int pair_begin     = pairs_per_row * warp_in_row / kWarpsPerRow;
    const int pair_end       = pairs_per_row * (warp_in_row + 1) / kWarpsPerRow;
    const int group_begin    = pair_begin * 2;
    const int group_end      = pair_end * 2;

    const std::uint8_t* code_row = codes + static_cast<std::int64_t>(row) * groups_per_row *
                                                Q3RowSplitStorage::kCodeBytesPerGroup;
    const std::uint8_t* scale_row = scales + static_cast<std::int64_t>(row) * groups_per_row *
                                                  Q3RowSplitStorage::kScaleBytesPerGroup;

    float accumulator = 0.0f;
    if (group_begin < group_end) {
        accumulator = q3_gemv_dot_byte_sync<Schedule>(shared_codes[warp_in_row], x, code_row,
                                                      scale_row, group_begin, group_end, lane);
    }

    accumulator = warp_reduce_sum(accumulator);
    if constexpr (kWarpsPerRow == 1) {
        if (lane == 0) {
            epilogue.template operator()<SplitOutput, SplitRow>(out, out_tail, row, accumulator);
        }
    } else {
        if (lane == 0) { row_partials[row_in_cta][warp_in_row] = accumulator; }
        __syncthreads();

        if (warp_in_row == 0 && lane == 0) {
            float row_accumulator = 0.0f;
#pragma unroll
            for (int warp = 0; warp < kWarpsPerRow; ++warp) {
                row_accumulator += row_partials[row_in_cta][warp];
            }
            epilogue.template operator()<SplitOutput, SplitRow>(out, out_tail, row,
                                                                row_accumulator);
        }
    }
    if constexpr (JoinPdl) { pdl::wait_for_dependencies(); }
}

} // namespace ninfer::ops::detail