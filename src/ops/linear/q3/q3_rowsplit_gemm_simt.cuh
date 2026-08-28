#pragma once

// Q3G64 RowSplit x BF16 SIMT GEMM (small token counts).
//
// out[Rows, Cols] = W[Rows, K] * x[K, Cols]
//
// One warp owns one output row across ColsPerTile columns. Raw Q3 codes (24
// bytes per 64-code group) and adjacent FP16 scales are staged per row with
// cp.async; codes move in whole two-group units of 48 bytes so every 16-byte
// vector stays globally 16-byte aligned. Lanes decode one 24-bit word (eight
// codes) from the packed shared bytes and accumulate FP32 FMA.
//
// K is staged in whole quant groups. A predicated final stage still copies
// complete two-group units and aligned 4-byte pairs of FP16 scales.

#include "core/pdl.cuh"
#include "ops/common/math.cuh"
#include "ops/common/memory.cuh"
#include "ops/common/warp.cuh"
#include "ops/linear/q3/q3_rowsplit_storage.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <type_traits>

namespace ninfer::ops::detail {

template <int RowsPerCta_, int ColsPerTile_, int GroupsPerStage_, int PipelineStages_,
          Cache CodeCache_, int LaunchBoundsMinBlocks_>
struct Q3RowSplitSimtGemmSchedule {
    static constexpr int kRowsPerCta            = RowsPerCta_;
    static constexpr int kColsPerTile           = ColsPerTile_;
    static constexpr int kGroupsPerStage        = GroupsPerStage_;
    static constexpr int kPipelineStages        = PipelineStages_;
    static constexpr Cache kCodeCache           = CodeCache_;
    static constexpr int kLaunchBoundsMinBlocks = LaunchBoundsMinBlocks_;

    static constexpr int kCtaWarps = kRowsPerCta;
    static constexpr int kThreads  = kCtaWarps * 32;
    static constexpr int kStageK   = kGroupsPerStage * Q3RowSplitStorage::kGroupK;
    static constexpr int kCodeUnitsPerStage = kGroupsPerStage / 2;
    static constexpr int kCodeVecsPerStage  = kCodeUnitsPerStage * 3;
    static constexpr int kScalePairsPerStage = kGroupsPerStage / 2;
    static constexpr int kCodePhases = (kGroupsPerStage + 3) / 4;
    static constexpr int kSharedBytes =
        kRowsPerCta * kPipelineStages *
        (kGroupsPerStage * Q3RowSplitStorage::kCodeBytesPerGroup +
         kScalePairsPerStage * static_cast<int>(sizeof(std::uint32_t)));

    static_assert(kRowsPerCta > 0 && kRowsPerCta <= 32);
    static_assert(kColsPerTile > 0 && kColsPerTile <= 8);
    static_assert(kGroupsPerStage > 0 && (kGroupsPerStage % 2) == 0,
                  "Q3 SIMT stages load two-group code units and scale pairs");
    static_assert(kPipelineStages >= 2 && kPipelineStages <= 8,
                  "Q3 SIMT cp.async pipeline depth must fit cp_wait");
    static_assert(kLaunchBoundsMinBlocks >= 1);
    static_assert(kCodeVecsPerStage * static_cast<int>(sizeof(uint4)) ==
                      kGroupsPerStage * Q3RowSplitStorage::kCodeBytesPerGroup,
                  "Q3 code groups must decompose into complete 16-byte vectors");
    static_assert(kCodeVecsPerStage <= 32,
                  "Q3 SIMT code stage must map one vector per lane");
    static_assert(kSharedBytes <= 48 * 1024,
                  "Q3 SIMT staged shared memory exceeds the static 48 KiB budget");
};

template <class Schedule, bool FullStage>
__device__ __forceinline__ void q3_simt_issue_stage(uint4* __restrict__ shared_codes,
                                                    std::uint32_t* __restrict__ shared_scales,
                                                    const std::uint8_t* __restrict__ code_row,
                                                    const std::uint8_t* __restrict__ scale_row,
                                                    int stage, int active_groups, int lane) {
    constexpr int kCodeVecs   = Schedule::kCodeVecsPerStage;
    constexpr int kScalePairs = Schedule::kScalePairsPerStage;

    const std::int64_t group0        = static_cast<std::int64_t>(stage) * Schedule::kGroupsPerStage;
    const std::uint8_t* stage_codes  = code_row + group0 * Q3RowSplitStorage::kCodeBytesPerGroup;
    const std::uint8_t* stage_scales = scale_row + group0 * Q3RowSplitStorage::kScaleBytesPerGroup;

    const int active_units = FullStage ? Schedule::kCodeUnitsPerStage : active_groups / 2;
    const int active_vecs  = FullStage ? kCodeVecs : active_units * 3;
    for (int vec = lane; vec < kCodeVecs; vec += 32) {
        if (FullStage || vec < active_vecs) {
            const int unit   = vec / 3;
            const int offset = unit * 2 * Q3RowSplitStorage::kCodeBytesPerGroup + (vec % 3) * 16;
            if constexpr (Schedule::kCodeCache == Cache::cg) {
                cp_async<16, Cache::cg>(&shared_codes[vec], stage_codes + offset);
            } else {
                cp_async<16, Cache::ca>(&shared_codes[vec], stage_codes + offset);
            }
        } else {
            shared_codes[vec] = uint4{0u, 0u, 0u, 0u};
        }
    }

    const int active_scale_pairs = FullStage ? kScalePairs : active_groups / 2;
    for (int pair = lane; pair < kScalePairs; pair += 32) {
        if (FullStage || pair < active_scale_pairs) {
            cp_async<4>(&shared_scales[pair], stage_scales + static_cast<std::int64_t>(pair) * 4);
        } else {
            shared_scales[pair] = 0u;
        }
    }
    cp_commit();
}

template <class Schedule, bool FullStage, bool FullCols>
__device__ __forceinline__ void
q3_simt_consume_stage(const __nv_bfloat16* __restrict__ x, std::int32_t k, int col0,
                      int active_cols, int stage, int active_groups,
                      const uint4* __restrict__ shared_codes,
                      const std::uint32_t* __restrict__ shared_scales, int lane,
                      float (&acc)[Schedule::kColsPerTile]) {
    constexpr int kCols       = Schedule::kColsPerTile;
    constexpr int kCodePhases = Schedule::kCodePhases;
    const auto* packed        = reinterpret_cast<const std::uint8_t*>(shared_codes);

#pragma unroll
    for (int phase = 0; phase < kCodePhases; ++phase) {
        const int group        = phase * 4 + (lane >> 3);
        const int stage_groups = FullStage ? Schedule::kGroupsPerStage : active_groups;
        if (group < stage_groups) {
            const int word_index   = (phase * 4 + (lane >> 3)) * Q3RowSplitStorage::kWordsPerGroup +
                                     (lane & 7);
            const std::uint32_t word = q3_load_word24(packed, word_index);
            const std::uint32_t scale_pair = shared_scales[group >> 1];
            const std::uint16_t scale_bits =
                static_cast<std::uint16_t>(scale_pair >> ((group & 1) * 16));

            float weights[8];
            Q3SimtDecodeAtom::decode_eight(word, scale_bits, weights);

            const std::int64_t xk = static_cast<std::int64_t>(stage) * Schedule::kStageK +
                                    static_cast<std::int64_t>(phase) * 256 + lane * 8;
#pragma unroll
            for (int col = 0; col < kCols; ++col) {
                if (FullCols || col < active_cols) {
                    const uint4 values =
                        load_vec<uint4>(x + static_cast<std::int64_t>(col0 + col) * k + xk);
                    const float2 x0 = bf16x2_bits_to_float2(values.x);
                    const float2 x1 = bf16x2_bits_to_float2(values.y);
                    const float2 x2 = bf16x2_bits_to_float2(values.z);
                    const float2 x3 = bf16x2_bits_to_float2(values.w);
                    acc[col]        = fmaf(weights[0], x0.x, acc[col]);
                    acc[col]        = fmaf(weights[1], x0.y, acc[col]);
                    acc[col]        = fmaf(weights[2], x1.x, acc[col]);
                    acc[col]        = fmaf(weights[3], x1.y, acc[col]);
                    acc[col]        = fmaf(weights[4], x2.x, acc[col]);
                    acc[col]        = fmaf(weights[5], x2.y, acc[col]);
                    acc[col]        = fmaf(weights[6], x3.x, acc[col]);
                    acc[col]        = fmaf(weights[7], x3.y, acc[col]);
                }
            }
        }
    }
}

struct Q3SimtStoreEpilogue {
    template <bool SplitOutput, int SplitRow, int Cols>
    __device__ __forceinline__ void
    operator()(__nv_bfloat16* out, __nv_bfloat16* out_tail, std::int32_t out_ld,
               std::int32_t out_tail_ld, std::int32_t row, std::int32_t col0,
               std::int32_t active_cols, const float (&values)[Cols]) const {
#pragma unroll
        for (int col = 0; col < Cols; ++col) {
            if (col >= active_cols) { continue; }
            if constexpr (SplitOutput) {
                if (row < SplitRow) {
                    out[static_cast<std::int64_t>(col0 + col) * out_ld + row] =
                        __float2bfloat16(values[col]);
                } else {
                    out_tail[static_cast<std::int64_t>(col0 + col) * out_tail_ld + row - SplitRow] =
                        __float2bfloat16(values[col]);
                }
            } else {
                out[static_cast<std::int64_t>(col0 + col) * out_ld + row] =
                    __float2bfloat16(values[col]);
            }
        }
    }
};

template <class Schedule, bool Full, bool SplitOutput = false, int SplitRow = 0,
          class Epilogue = Q3SimtStoreEpilogue, bool TriggerPdl = false, bool JoinPdl = false>
__global__ __launch_bounds__(
    Schedule::kThreads,
    Schedule::
        kLaunchBoundsMinBlocks) void q3_rowsplit_gemm_simt_kernel(const __nv_bfloat16* __restrict__ x,
                                                                   const std::
                                                                       uint8_t* __restrict__ codes,
                                                                   const std::
                                                                       uint8_t* __restrict__ scales,
                                                                   __nv_bfloat16* __restrict__ out,
                                                                   __nv_bfloat16* __restrict__ out_tail,
                                                                   std::int32_t out_ld,
                                                                   std::int32_t out_tail_ld,
                                                                   std::int32_t rows, std::int32_t k,
                                                                   std::int32_t cols,
                                                                   std::int32_t padded_k,
                                                                   Epilogue epilogue = {}) {
    static_assert(!SplitOutput || SplitRow > 0,
                  "split-output Q3 SIMT requires a positive compile-time seam");

    if constexpr (TriggerPdl) {
        if (threadIdx.x == 0) { pdl::trigger_dependents(); }
    }

    constexpr bool kFull              = Full;
    constexpr int kRowsPerCta         = Schedule::kRowsPerCta;
    constexpr int kColsPerTile        = Schedule::kColsPerTile;
    constexpr int kGroupsPerStage     = Schedule::kGroupsPerStage;
    constexpr int kPipelineStages     = Schedule::kPipelineStages;
    constexpr int kPipelinePrefetch   = kPipelineStages - 1;
    constexpr int kCodeVecsPerStage   = Schedule::kCodeVecsPerStage;
    constexpr int kScalePairsPerStage = Schedule::kScalePairsPerStage;

    __shared__ __align__(16) uint4 shared_codes[kRowsPerCta][kPipelineStages][kCodeVecsPerStage];
    __shared__ __align__(16)
        std::uint32_t shared_scales[kRowsPerCta][kPipelineStages][kScalePairsPerStage];

    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const int row  = static_cast<int>(blockIdx.x) * kRowsPerCta + warp;
    if constexpr (!kFull) {
        if (row >= rows) { return; }
    }

    const int col0        = static_cast<int>(blockIdx.y) * kColsPerTile;
    const int active_cols = kFull ? kColsPerTile : min(kColsPerTile, cols - col0);

    const int padded_groups = padded_k / Q3RowSplitStorage::kGroupK;
    const int groups        = k / Q3RowSplitStorage::kGroupK;
    const int stages =
        kFull ? groups / kGroupsPerStage : (groups + kGroupsPerStage - 1) / kGroupsPerStage;

    const std::uint8_t* code_row = codes + static_cast<std::int64_t>(row) * padded_groups *
                                                Q3RowSplitStorage::kCodeBytesPerGroup;
    const std::uint8_t* scale_row = scales + static_cast<std::int64_t>(row) * padded_groups *
                                                  Q3RowSplitStorage::kScaleBytesPerGroup;

    float acc[kColsPerTile];
#pragma unroll
    for (int col = 0; col < kColsPerTile; ++col) { acc[col] = 0.0f; }

#pragma unroll
    for (int prefetch = 0; prefetch < kPipelinePrefetch; ++prefetch) {
        if (prefetch < stages) {
            const int active_groups =
                kFull ? kGroupsPerStage : min(kGroupsPerStage, groups - prefetch * kGroupsPerStage);
            q3_simt_issue_stage<Schedule, kFull>(shared_codes[warp][prefetch],
                                                 shared_scales[warp][prefetch], code_row, scale_row,
                                                 prefetch, active_groups, lane);
        } else {
            cp_commit();
        }
    }

#pragma unroll 1
    for (int stage = 0; stage < stages; ++stage) {
        const int fetch = stage + kPipelinePrefetch;
        if (fetch < stages) {
            const int active_groups =
                kFull ? kGroupsPerStage : min(kGroupsPerStage, groups - fetch * kGroupsPerStage);
            const int buffer = fetch % kPipelineStages;
            q3_simt_issue_stage<Schedule, kFull>(shared_codes[warp][buffer],
                                                 shared_scales[warp][buffer], code_row, scale_row,
                                                 fetch, active_groups, lane);
        } else {
            cp_commit();
        }

        cp_wait<kPipelinePrefetch>();
        __syncwarp();

        const int active_groups =
            kFull ? kGroupsPerStage : min(kGroupsPerStage, groups - stage * kGroupsPerStage);
        const int buffer = stage % kPipelineStages;
        q3_simt_consume_stage<Schedule, kFull, kFull>(x, k, col0, active_cols, stage, active_groups,
                                                      shared_codes[warp][buffer],
                                                      shared_scales[warp][buffer], lane, acc);
        __syncwarp();
    }

    if constexpr (std::is_same_v<Epilogue, Q3SimtStoreEpilogue>) {
#pragma unroll
        for (int col = 0; col < kColsPerTile; ++col) {
            if (kFull || col < active_cols) {
                const float sum = warp_reduce_sum(acc[col]);
                if (lane == 0) {
                    if constexpr (SplitOutput) {
                        if (row < SplitRow) {
                            out[static_cast<std::int64_t>(col0 + col) * out_ld + row] =
                                __float2bfloat16(sum);
                        } else {
                            out_tail[static_cast<std::int64_t>(col0 + col) * out_tail_ld + row -
                                     SplitRow] =
                                __float2bfloat16(sum);
                        }
                    } else {
                        out[static_cast<std::int64_t>(col0 + col) * out_ld + row] =
                            __float2bfloat16(sum);
                    }
                }
            }
        }
    } else {
        float sums[kColsPerTile];
#pragma unroll
        for (int col = 0; col < kColsPerTile; ++col) {
            sums[col] = (kFull || col < active_cols) ? warp_reduce_sum(acc[col]) : 0.0F;
        }
        if (lane == 0) {
            epilogue.template operator()<SplitOutput, SplitRow>(out, out_tail, out_ld, out_tail_ld,
                                                                row, col0, active_cols, sums);
        }
    }
    if constexpr (JoinPdl) { pdl::wait_for_dependencies(); }
}

} // namespace ninfer::ops::detail