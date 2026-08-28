#include "ops/gdn_input_proj/q4_q5/q4_q5_gdn_input_kernels.h"

#include "core/device.h"
#include "core/pdl.cuh"
#include "ops/common/math.h"
#include "ops/gdn_input_proj/gdn_conv.cuh"
#include "ops/gdn_input_proj/gdn_projected_conv.h"
#include "ops/linear/q4/q4_rowsplit_gemm_simt.cuh"
#include "ops/linear/q4/q4_rowsplit_gemv.cuh"
#include "ops/linear/q5/q5_rowsplit_gemm_simt.cuh"
#include "ops/linear/q5/q5_rowsplit_gemv.cuh"

#include <cuda_bf16.h>

#include <cstdint>
#include <stdexcept>
#include <type_traits>

namespace ninfer::ops::detail {
namespace {

template <int Hidden, int QueryRows, int KeyRows, int ValueRows, int ZRows, int FullSlabs>
struct GdnSnapshotGeometry {
    static constexpr int kHidden      = Hidden;
    static constexpr int kQueryRows   = QueryRows;
    static constexpr int kKeyRows     = KeyRows;
    static constexpr int kValueRows   = ValueRows;
    static constexpr int kZRows       = ZRows;
    static constexpr int kValueZRows  = ValueRows + ZRows;
    static constexpr int kQkRows      = QueryRows + KeyRows;
    static constexpr int kChannels    = QueryRows + KeyRows + ValueRows;
    static constexpr int kValueOffset = QueryRows + KeyRows;
    static constexpr int kFullSlabs   = FullSlabs;
};

using GdnSnapshotGeometry27 = GdnSnapshotGeometry<5120, 2048, 2048, 6144, 6144, 5>;
using GdnSnapshotGeometry9  = GdnSnapshotGeometry<4096, 2048, 2048, 4096, 4096, 4>;

using Q4ScheduleC4 = Q4RowSplitSimtGemmSchedule<8, 4, 16, 2, Cache::ca, 1>;
using Q4ScheduleC8 = Q4RowSplitSimtGemmSchedule<8, 8, 16, 2, Cache::ca, 1>;

enum class PdlOrder {
    Q4ThenQ5,
    Q5ThenQ4,
};

// The conv epilogue type selects the conv state form: GdnConvEpilogue<Publish>
// is the BF16 form, GdnConvEpilogue<PublishI8, true> the I8 codes + scale form.
template <class Geometry, class Epilogue, class Publish>
Epilogue make_epilogue(const Tensor& conv_weight, const Tensor& conv_states,
                       const Tensor& conv_scale, const Tensor& valid_columns,
                       const Tensor& initial_slot, Tensor& query, Tensor& key, Tensor& value,
                       int global_row_offset, Publish publish) {
    Epilogue epilogue{
        static_cast<const __nv_bfloat16*>(conv_weight.data),
        static_cast<const __nv_bfloat16*>(conv_states.data),
        static_cast<const std::int32_t*>(initial_slot.data),
        valid_columns.data == nullptr ? nullptr
                                      : static_cast<const std::int32_t*>(valid_columns.data),
        static_cast<__nv_bfloat16*>(query.data),
        static_cast<__nv_bfloat16*>(key.data),
        static_cast<__nv_bfloat16*>(value.data),
        Geometry::kChannels,
        Geometry::kQueryRows,
        Geometry::kKeyRows,
        Geometry::kValueRows,
        global_row_offset,
        static_cast<std::int32_t>(query.ne[1]),
        0,
        publish,
        static_cast<const std::int8_t*>(conv_states.data),
        static_cast<const __half*>(conv_scale.data),
        static_cast<std::int32_t>(Geometry::kChannels / 128),
    };
    return epilogue;
}

template <class Geometry, class Epilogue>
struct Q4GdnDecodeEpilogue {
    Epilogue conv;

    template <bool, int>
    __device__ __forceinline__ void operator()(__nv_bfloat16*, __nv_bfloat16*, int row,
                                               float value) const {
        const float projected[1]{value};
        conv.store(row, projected);
    }
};

template <class Geometry, int Tokens, class Epilogue>
struct Q4GdnSmallTEpilogue {
    Epilogue conv;

    template <bool, int, int TileCols>
    __device__ __forceinline__ void
    operator()(__nv_bfloat16*, __nv_bfloat16*, std::int32_t, std::int32_t, std::int32_t row,
               std::int32_t, std::int32_t active_cols, const float (&values)[TileCols]) const {
        float projected[Tokens];
#pragma unroll
        for (int token = 0; token < Tokens; ++token) { projected[token] = values[token]; }
        if (active_cols == Tokens) { conv.store(row, projected); }
    }
};

template <class Geometry, class Epilogue>
struct Q5GdnDecodeEpilogue {
    Epilogue conv;
    __nv_bfloat16* z;

    template <bool, int>
    __device__ __forceinline__ void operator()(__nv_bfloat16*, __nv_bfloat16*, int row,
                                               float value) const {
        if (row < Geometry::kValueRows) {
            const float projected[1]{value};
            conv.store(row, projected);
        } else {
            z[row - Geometry::kValueRows] = __float2bfloat16_rn(value);
        }
    }
};

template <class Geometry, int Tokens, class Epilogue>
struct Q5GdnSmallTEpilogue {
    Epilogue conv;
    __nv_bfloat16* z;

    template <bool, int, int ProducedTokens>
    __device__ __forceinline__ void operator()(__nv_bfloat16*, __nv_bfloat16*, std::int32_t,
                                               std::int32_t, std::int32_t row,
                                               const float (&values)[ProducedTokens]) const {
        static_assert(ProducedTokens == Tokens);
        if (row < Geometry::kValueRows) {
            conv.store(row, values);
        } else {
#pragma unroll
            for (int token = 0; token < Tokens; ++token) {
                z[static_cast<std::int64_t>(token) * Geometry::kZRows + row - Geometry::kValueRows] =
                    __float2bfloat16_rn(values[token]);
            }
        }
    }
};

template <class Geometry, class Epilogue>
struct Q4GdnValueZDecodeEpilogue {
    Epilogue conv;
    __nv_bfloat16* z;

    template <bool, int>
    __device__ __forceinline__ void operator()(__nv_bfloat16*, __nv_bfloat16*, int row,
                                               float value) const {
        if (row < Geometry::kValueRows) {
            const float projected[1]{value};
            conv.store(row, projected);
        } else {
            z[row - Geometry::kValueRows] = __float2bfloat16_rn(value);
        }
    }
};

template <class Geometry, int Tokens, class Epilogue>
struct Q4GdnValueZSmallTEpilogue {
    Epilogue conv;
    __nv_bfloat16* z;

    template <bool, int, int TileCols>
    __device__ __forceinline__ void
    operator()(__nv_bfloat16*, __nv_bfloat16*, std::int32_t, std::int32_t, std::int32_t row,
               std::int32_t, std::int32_t active_cols, const float (&values)[TileCols]) const {
        if (row < Geometry::kValueRows) {
            float projected[Tokens];
#pragma unroll
            for (int token = 0; token < Tokens; ++token) { projected[token] = values[token]; }
            if (active_cols == Tokens) { conv.store(row, projected); }
        } else {
#pragma unroll
            for (int token = 0; token < Tokens; ++token) {
                z[static_cast<std::int64_t>(token) * Geometry::kZRows + row - Geometry::kValueRows] =
                    __float2bfloat16_rn(values[token]);
            }
        }
    }
};

template <class Geometry, class Epilogue, bool TriggerPdl, bool JoinPdl, bool Dependent>
void launch_q4_t1(const Tensor& x, const Weight& qk_weight, const Epilogue& qk_epilogue,
                  Tensor& query, cudaStream_t stream) {
    using Schedule = std::conditional_t<Geometry::kHidden == 4096,
                                        Q4GemvR1W8DirectK64Schedule, Q4GemvR1W8DirectSchedule>;
    constexpr int q4_threads = Schedule::kThreads;
    constexpr int q4_blocks  = Geometry::kQkRows / Schedule::kRowsPerCta;
    if constexpr (Dependent) {
        CUDA_CHECK(pdl::launch_dependent(
            {dim3(q4_blocks), dim3(q4_threads), 0, stream},
            q4_rowsplit_gemv_kernel<Schedule, false, 0, Q4GdnDecodeEpilogue<Geometry, Epilogue>,
                                    TriggerPdl, JoinPdl>,
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const std::uint8_t*>(qk_weight.qdata),
            static_cast<const std::uint8_t*>(qk_weight.scales),
            static_cast<__nv_bfloat16*>(query.data), nullptr, Geometry::kQkRows, Geometry::kHidden,
            Q4GdnDecodeEpilogue<Geometry, Epilogue>{qk_epilogue}));
    } else {
        q4_rowsplit_gemv_kernel<Schedule, false, 0, Q4GdnDecodeEpilogue<Geometry, Epilogue>,
                                TriggerPdl, JoinPdl>
            <<<q4_blocks, q4_threads, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(x.data),
                static_cast<const std::uint8_t*>(qk_weight.qdata),
                static_cast<const std::uint8_t*>(qk_weight.scales),
                static_cast<__nv_bfloat16*>(query.data), nullptr, Geometry::kQkRows,
                Geometry::kHidden, Q4GdnDecodeEpilogue<Geometry, Epilogue>{qk_epilogue});
    }
}

template <class Geometry, class Epilogue, bool TriggerPdl, bool JoinPdl, bool Dependent>
void launch_q5_t1(const Tensor& x, const Weight& value_z_weight, const Epilogue& value_epilogue,
                  Tensor& value, Tensor& z, cudaStream_t stream) {
    constexpr int q5_rows_per_block = 16;
    constexpr int q5_threads        = q5_rows_per_block * 32;
    constexpr int q5_blocks         = Geometry::kValueZRows / q5_rows_per_block;
    if constexpr (Dependent) {
        CUDA_CHECK(pdl::launch_dependent(
            {dim3(q5_blocks), dim3(q5_threads), 0, stream},
            q5_rowsplit_gemv_kernel<Geometry::kValueZRows, Geometry::kHidden, q5_rows_per_block, 2, true, false, true,
                                    Geometry::kValueRows, Q5GdnDecodeEpilogue<Geometry, Epilogue>, TriggerPdl, JoinPdl>,
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const std::uint8_t*>(value_z_weight.qdata),
            static_cast<const std::uint8_t*>(value_z_weight.qhigh),
            static_cast<const std::uint8_t*>(value_z_weight.scales),
            static_cast<__nv_bfloat16*>(value.data), static_cast<__nv_bfloat16*>(z.data),
            Q5GdnDecodeEpilogue<Geometry, Epilogue>{value_epilogue, static_cast<__nv_bfloat16*>(z.data)}));
    } else {
        q5_rowsplit_gemv_kernel<Geometry::kValueZRows, Geometry::kHidden, q5_rows_per_block, 2, true, false, true,
                                Geometry::kValueRows, Q5GdnDecodeEpilogue<Geometry, Epilogue>, TriggerPdl, JoinPdl>
            <<<q5_blocks, q5_threads, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(x.data),
                static_cast<const std::uint8_t*>(value_z_weight.qdata),
                static_cast<const std::uint8_t*>(value_z_weight.qhigh),
                static_cast<const std::uint8_t*>(value_z_weight.scales),
                static_cast<__nv_bfloat16*>(value.data), static_cast<__nv_bfloat16*>(z.data),
                Q5GdnDecodeEpilogue<Geometry, Epilogue>{value_epilogue,
                                                        static_cast<__nv_bfloat16*>(z.data)});
    }
}

template <class Geometry, class Epilogue, bool TriggerPdl, bool JoinPdl, bool Dependent>
void launch_q4_vz_t1(const Tensor& x, const Weight& value_z_weight, const Epilogue& value_epilogue,
                     Tensor& value, Tensor& z, cudaStream_t stream) {
    using Schedule = std::conditional_t<Geometry::kHidden == 4096,
                                        Q4GemvR1W8DirectK64Schedule, Q4GemvR1W8DirectSchedule>;
    constexpr int threads = Schedule::kThreads;
    constexpr int blocks  = Geometry::kValueZRows / Schedule::kRowsPerCta;
    using EpilogueT = Q4GdnValueZDecodeEpilogue<Geometry, Epilogue>;
    if constexpr (Dependent) {
        CUDA_CHECK(pdl::launch_dependent(
            {dim3(blocks), dim3(threads), 0, stream},
            q4_rowsplit_gemv_kernel<Schedule, false, 0, EpilogueT, TriggerPdl, JoinPdl>,
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const std::uint8_t*>(value_z_weight.qdata),
            static_cast<const std::uint8_t*>(value_z_weight.scales),
            static_cast<__nv_bfloat16*>(value.data), nullptr, Geometry::kValueZRows,
            Geometry::kHidden, EpilogueT{value_epilogue, static_cast<__nv_bfloat16*>(z.data)}));
    } else {
        q4_rowsplit_gemv_kernel<Schedule, false, 0, EpilogueT, TriggerPdl, JoinPdl>
            <<<blocks, threads, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(x.data),
                static_cast<const std::uint8_t*>(value_z_weight.qdata),
                static_cast<const std::uint8_t*>(value_z_weight.scales),
                static_cast<__nv_bfloat16*>(value.data), nullptr, Geometry::kValueZRows,
                Geometry::kHidden, EpilogueT{value_epilogue, static_cast<__nv_bfloat16*>(z.data)});
    }
}

template <class Geometry, int Tokens, class Q4Schedule, class Epilogue, bool TriggerPdl,
          bool JoinPdl, bool Dependent>
void launch_q4_small_t(const Tensor& x, const Weight& qk_weight, const Epilogue& qk_epilogue,
                       Tensor& query, cudaStream_t stream) {
    const dim3 q4_grid(Geometry::kQkRows / Q4Schedule::kRowsPerCta, 1u, 1u);
    if constexpr (Dependent) {
        CUDA_CHECK(pdl::launch_dependent(
            {q4_grid, dim3(Q4Schedule::kThreads), 0, stream},
            q4_rowsplit_gemm_simt_kernel<Q4Schedule, false, false, 0,
                                         Q4GdnSmallTEpilogue<Geometry, Tokens, Epilogue>, TriggerPdl, JoinPdl>,
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const std::uint8_t*>(qk_weight.qdata),
            static_cast<const std::uint8_t*>(qk_weight.scales),
            static_cast<__nv_bfloat16*>(query.data), nullptr, Geometry::kQueryRows, 0, Geometry::kQkRows, Geometry::kHidden,
            Tokens, Geometry::kHidden, Q4GdnSmallTEpilogue<Geometry, Tokens, Epilogue>{qk_epilogue}));
    } else {
        q4_rowsplit_gemm_simt_kernel<Q4Schedule, false, false, 0,
                                     Q4GdnSmallTEpilogue<Geometry, Tokens, Epilogue>, TriggerPdl, JoinPdl>
            <<<q4_grid, Q4Schedule::kThreads, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(x.data),
                static_cast<const std::uint8_t*>(qk_weight.qdata),
                static_cast<const std::uint8_t*>(qk_weight.scales),
                static_cast<__nv_bfloat16*>(query.data), nullptr, Geometry::kQueryRows, 0,
                Geometry::kQkRows, Geometry::kHidden, Tokens, Geometry::kHidden,
                Q4GdnSmallTEpilogue<Geometry, Tokens, Epilogue>{qk_epilogue});
    }
}

template <class Geometry, int Tokens, class Epilogue, bool TriggerPdl, bool JoinPdl,
          bool Dependent>
void launch_q5_small_t(const Tensor& x, const Weight& value_z_weight,
                       const Epilogue& value_epilogue, Tensor& value, Tensor& z,
                       cudaStream_t stream) {
    constexpr int q5_threads = 4 * 32;
    const dim3 q5_grid(Geometry::kValueZRows, 1u, 1u);
    if constexpr (Dependent) {
        CUDA_CHECK(pdl::launch_dependent(
            {q5_grid, dim3(q5_threads), 0, stream},
            q5_rowsplit_gemm_simt_split4_kernel<Q5RowSplitSimtSchedule, Tokens, Geometry::kFullSlabs, Geometry::kHidden, true,
                                                Geometry::kValueRows, Q5GdnSmallTEpilogue<Geometry, Tokens, Epilogue>,
                                                TriggerPdl, JoinPdl>,
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const std::uint8_t*>(value_z_weight.qdata),
            static_cast<const std::uint8_t*>(value_z_weight.qhigh),
            static_cast<const std::uint8_t*>(value_z_weight.scales),
            static_cast<__nv_bfloat16*>(value.data), static_cast<__nv_bfloat16*>(z.data),
            Geometry::kValueZRows, Geometry::kValueRows, Geometry::kHidden, Tokens, Geometry::kHidden, Geometry::kFullSlabs,
            Q5GdnSmallTEpilogue<Geometry, Tokens, Epilogue>{
                value_epilogue,
                static_cast<__nv_bfloat16*>(z.data),
            }));
    } else {
        q5_rowsplit_gemm_simt_split4_kernel<Q5RowSplitSimtSchedule, Tokens, Geometry::kFullSlabs, Geometry::kHidden, true,
                                            Geometry::kValueRows, Q5GdnSmallTEpilogue<Geometry, Tokens, Epilogue>,
                                            TriggerPdl, JoinPdl>
            <<<q5_grid, q5_threads, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(x.data),
                static_cast<const std::uint8_t*>(value_z_weight.qdata),
                static_cast<const std::uint8_t*>(value_z_weight.qhigh),
                static_cast<const std::uint8_t*>(value_z_weight.scales),
                static_cast<__nv_bfloat16*>(value.data), static_cast<__nv_bfloat16*>(z.data),
                Geometry::kValueZRows, Geometry::kValueRows, Geometry::kHidden, Tokens,
                Geometry::kHidden, Geometry::kFullSlabs,
                Q5GdnSmallTEpilogue<Geometry, Tokens, Epilogue>{value_epilogue,
                                                                static_cast<__nv_bfloat16*>(z.data)});
    }
}

template <class Geometry, int Tokens, class Epilogue, bool TriggerPdl, bool JoinPdl,
          bool Dependent>
void launch_q4_vz_small_t(const Tensor& x, const Weight& value_z_weight,
                          const Epilogue& value_epilogue, Tensor& value, Tensor& z,
                          cudaStream_t stream) {
    using Schedule  = Q4ScheduleC8;
    using EpilogueT = Q4GdnValueZSmallTEpilogue<Geometry, Tokens, Epilogue>;
    constexpr int threads = Schedule::kThreads;
    constexpr int blocks  = Geometry::kValueZRows / Schedule::kRowsPerCta;
    const std::int32_t value_ld =
        static_cast<std::int32_t>(value.nb[1] / sizeof(__nv_bfloat16));
    if constexpr (Dependent) {
        CUDA_CHECK(pdl::launch_dependent(
            {dim3(blocks), dim3(threads), 0, stream},
            q4_rowsplit_gemm_simt_kernel<Schedule, false, false, 0, EpilogueT, TriggerPdl, JoinPdl>,
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const std::uint8_t*>(value_z_weight.qdata),
            static_cast<const std::uint8_t*>(value_z_weight.scales),
            static_cast<__nv_bfloat16*>(value.data), nullptr, value_ld, 0, Geometry::kValueZRows,
            Geometry::kHidden, Tokens, value_z_weight.padded_shape[1],
            EpilogueT{value_epilogue, static_cast<__nv_bfloat16*>(z.data)}));
    } else {
        q4_rowsplit_gemm_simt_kernel<Schedule, false, false, 0, EpilogueT, TriggerPdl, JoinPdl>
            <<<blocks, threads, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(x.data),
                static_cast<const std::uint8_t*>(value_z_weight.qdata),
                static_cast<const std::uint8_t*>(value_z_weight.scales),
                static_cast<__nv_bfloat16*>(value.data), nullptr, value_ld, 0,
                Geometry::kValueZRows, Geometry::kHidden, Tokens, value_z_weight.padded_shape[1],
                EpilogueT{value_epilogue, static_cast<__nv_bfloat16*>(z.data)});
    }
}

template <class Geometry, PdlOrder Order, class Epilogue>
void launch_t1(const Tensor& x, const Weight& qk_weight, const Weight& value_z_weight,
               const Epilogue& qk_epilogue, const Epilogue& value_epilogue, Tensor& query,
               Tensor& value, Tensor& z, cudaStream_t stream) {
    // RTX 5060 Ti 16 GB path: disable PDL on this layer because the consumer-launch
    // attribute fragments the GPU memory allocator and the small headroom on
    // 16 GB class GPUs cannot absorb the extra reservation.
    const bool value_z_q4 = value_z_weight.qtype == QType::Q4G64_F16S;
    if constexpr (Order == PdlOrder::Q5ThenQ4) {
        if (value_z_q4) {
            launch_q4_vz_t1<Geometry, Epilogue, true, false, false>(x, value_z_weight, value_epilogue,
                                                                    value, z, stream);
        } else {
            launch_q5_t1<Geometry, Epilogue, true, false, false>(x, value_z_weight, value_epilogue,
                                                                 value, z, stream);
        }
        launch_q4_t1<Geometry, Epilogue, false, true, false>(x, qk_weight, qk_epilogue, query, stream);
    } else {
        launch_q4_t1<Geometry, Epilogue, true, false, false>(x, qk_weight, qk_epilogue, query, stream);
        if (value_z_q4) {
            launch_q4_vz_t1<Geometry, Epilogue, false, true, false>(x, value_z_weight,
                                                                    value_epilogue, value, z, stream);
        } else {
            launch_q5_t1<Geometry, Epilogue, false, true, false>(x, value_z_weight, value_epilogue,
                                                                 value, z, stream);
        }
    }
}

template <class Geometry, int Tokens, class Q4Schedule, PdlOrder Order, class Epilogue>
void launch_small_t_schedule(const Tensor& x, const Weight& qk_weight, const Weight& value_z_weight,
                             const Epilogue& qk_epilogue, const Epilogue& value_epilogue,
                             Tensor& query, Tensor& value, Tensor& z, cudaStream_t stream) {
    // RTX 5060 Ti 16 GB path: disable PDL on this layer because the consumer-launch
    // attribute fragments the GPU memory allocator and the small headroom on
    // 16 GB class GPUs cannot absorb the extra reservation.
    const bool value_z_q4 = value_z_weight.qtype == QType::Q4G64_F16S;
    if constexpr (Order == PdlOrder::Q5ThenQ4) {
        if (value_z_q4) {
            launch_q4_vz_small_t<Geometry, Tokens, Epilogue, true, false, false>(
                x, value_z_weight, value_epilogue, value, z, stream);
        } else {
            launch_q5_small_t<Geometry, Tokens, Epilogue, true, false, false>(
                x, value_z_weight, value_epilogue, value, z, stream);
        }
        launch_q4_small_t<Geometry, Tokens, Q4Schedule, Epilogue, false, true, false>(
            x, qk_weight, qk_epilogue, query, stream);
    } else {
        launch_q4_small_t<Geometry, Tokens, Q4Schedule, Epilogue, true, false, false>(
            x, qk_weight, qk_epilogue, query, stream);
        if (value_z_q4) {
            launch_q4_vz_small_t<Geometry, Tokens, Epilogue, false, true, false>(
                x, value_z_weight, value_epilogue, value, z, stream);
        } else {
            launch_q5_small_t<Geometry, Tokens, Epilogue, false, true, false>(
                x, value_z_weight, value_epilogue, value, z, stream);
        }
    }
}

template <class Geometry, int Tokens, PdlOrder Order, class Epilogue>
void launch_small_t(const Tensor& x, const Weight& qk_weight, const Weight& value_z_weight,
                    const Epilogue& qk_epilogue, const Epilogue& value_epilogue, Tensor& query,
                    Tensor& value, Tensor& z, cudaStream_t stream) {
    if constexpr (Tokens <= 4) {
        launch_small_t_schedule<Geometry, Tokens, Q4ScheduleC4, Order, Epilogue>(
            x, qk_weight, value_z_weight, qk_epilogue, value_epilogue, query, value, z, stream);
    } else {
        launch_small_t_schedule<Geometry, Tokens, Q4ScheduleC8, Order, Epilogue>(
            x, qk_weight, value_z_weight, qk_epilogue, value_epilogue, query, value, z, stream);
    }
}

template <class Geometry, PdlOrder Order, class Epilogue, class Publish>
void launch_conv(const Tensor& x, const Weight& qk_weight, const Weight& value_z_weight,
                 const Tensor& conv_weight, const Tensor& conv_states, const Tensor& conv_scale,
                 const Tensor& valid_columns, const Tensor& initial_slot, Tensor& query,
                 Tensor& key, Tensor& value, Tensor& z, Publish publish, cudaStream_t stream) {
    const Epilogue qk_epilogue =
        make_epilogue<Geometry, Epilogue>(conv_weight, conv_states, conv_scale, valid_columns,
                                          initial_slot, query, key, value, 0, publish);
    const Epilogue value_epilogue =
        make_epilogue<Geometry, Epilogue>(conv_weight, conv_states, conv_scale, valid_columns,
                                          initial_slot, query, key, value, Geometry::kValueOffset,
                                          publish);

    switch (x.ne[1]) {
    case 1:
        launch_t1<Geometry, Order, Epilogue>(x, qk_weight, value_z_weight, qk_epilogue, value_epilogue,
                                             query, value, z, stream);
        break;
    case 2:
        launch_small_t<Geometry, 2, Order, Epilogue>(x, qk_weight, value_z_weight, qk_epilogue, value_epilogue,
                                                     query, value, z, stream);
        break;
    case 3:
        launch_small_t<Geometry, 3, Order, Epilogue>(x, qk_weight, value_z_weight, qk_epilogue, value_epilogue,
                                                     query, value, z, stream);
        break;
    case 5:
        launch_small_t<Geometry, 5, Order, Epilogue>(x, qk_weight, value_z_weight, qk_epilogue, value_epilogue,
                                                     query, value, z, stream);
        break;
    case 6:
        launch_small_t<Geometry, 6, Order, Epilogue>(x, qk_weight, value_z_weight, qk_epilogue, value_epilogue,
                                                     query, value, z, stream);
        break;
    default:
        throw std::invalid_argument("Q4/Q5 projection-epilogue GDN conv requires T=1..3 or 5..6");
    }
    CUDA_CHECK(cudaGetLastError());
}

template <PdlOrder Order, class Epilogue, class Publish>
void launch_conv_by_geometry(const Tensor& x, const Weight& qk_weight, const Weight& value_z_weight,
                             const Tensor& conv_weight, const Tensor& conv_states,
                             const Tensor& conv_scale, const Tensor& valid_columns,
                             const Tensor& initial_slot, Tensor& query, Tensor& key, Tensor& value,
                             Tensor& z, Publish publish, cudaStream_t stream) {
    if (x.ne[0] == 4096) {
        launch_conv<GdnSnapshotGeometry9, Order, Epilogue>(x, qk_weight, value_z_weight, conv_weight,
                                                           conv_states, conv_scale, valid_columns,
                                                           initial_slot, query, key, value, z,
                                                           publish, stream);
    } else if (x.ne[0] == 5120) {
        launch_conv<GdnSnapshotGeometry27, Order, Epilogue>(x, qk_weight, value_z_weight, conv_weight,
                                                            conv_states, conv_scale, valid_columns,
                                                            initial_slot, query, key, value, z,
                                                            publish, stream);
    } else {
        throw std::invalid_argument("GDN Q4/Q5 projection-epilogue conv: unsupported input width");
    }
}

} // namespace

void q4_q5_gdn_input_conv_snapshot_launch(const Tensor& x, const Weight& qk_weight,
                                          const Weight& value_z_weight, const Tensor& conv_weight,
                                          Tensor& conv_states, const Tensor& conv_scale,
                                          const Tensor& valid_columns, const Tensor& initial_slot,
                                          const Tensor& snapshot_base_slot, Tensor& query,
                                          Tensor& key, Tensor& value, Tensor& z,
                                          cudaStream_t stream) {
    const std::int32_t channels = (x.ne[0] == 4096) ? 8192 : 10240;
    if (conv_states.dtype == DType::I8) {
        const SnapshotHistoryPublishI8 publish{
            static_cast<std::int8_t*>(conv_states.data),
            static_cast<const std::int32_t*>(snapshot_base_slot.data), channels,
            static_cast<std::int32_t>(channels / 128),
            static_cast<__half*>(conv_scale.data)};
        using Epilogue = GdnConvEpilogue<SnapshotHistoryPublishI8, true>;
        if (x.ne[1] == 2) {
            launch_conv_by_geometry<PdlOrder::Q4ThenQ5, Epilogue>(
                x, qk_weight, value_z_weight, conv_weight, conv_states, conv_scale, valid_columns,
                initial_slot, query, key, value, z, publish, stream);
        } else {
            launch_conv_by_geometry<PdlOrder::Q5ThenQ4, Epilogue>(
                x, qk_weight, value_z_weight, conv_weight, conv_states, conv_scale, valid_columns,
                initial_slot, query, key, value, z, publish, stream);
        }
        return;
    }
    const SnapshotHistoryPublish publish{static_cast<__nv_bfloat16*>(conv_states.data),
                                         static_cast<const std::int32_t*>(snapshot_base_slot.data),
                                         channels};
    using Epilogue = GdnConvEpilogue<SnapshotHistoryPublish>;
    if (x.ne[1] == 2) {
        launch_conv_by_geometry<PdlOrder::Q4ThenQ5, Epilogue>(
            x, qk_weight, value_z_weight, conv_weight, conv_states, conv_scale, valid_columns,
            initial_slot, query, key, value, z, publish, stream);
    } else {
        launch_conv_by_geometry<PdlOrder::Q5ThenQ4, Epilogue>(
            x, qk_weight, value_z_weight, conv_weight, conv_states, conv_scale, valid_columns,
            initial_slot, query, key, value, z, publish, stream);
    }
}

void q4_q5_gdn_input_conv_record_launch(const Tensor& x, const Weight& qk_weight,
                                        const Weight& value_z_weight, const Tensor& conv_weight,
                                        const Tensor& conv_states, const Tensor& conv_scale,
                                        const Tensor& valid_columns, const Tensor& initial_slot,
                                        Tensor& conv_record, Tensor& query, Tensor& key,
                                        Tensor& value, Tensor& z, cudaStream_t stream) {
    const std::int32_t channels = (x.ne[0] == 4096) ? 8192 : 10240;
    if (conv_states.dtype == DType::I8) {
        const RecordColumnPublishI8 publish{static_cast<__nv_bfloat16*>(conv_record.data), channels,
                                            static_cast<std::int32_t>(x.ne[1])};
        using Epilogue = GdnConvEpilogue<RecordColumnPublishI8, true>;
        if (x.ne[1] == 2) {
            launch_conv_by_geometry<PdlOrder::Q4ThenQ5, Epilogue>(
                x, qk_weight, value_z_weight, conv_weight, conv_states, conv_scale, valid_columns,
                initial_slot, query, key, value, z, publish, stream);
        } else {
            launch_conv_by_geometry<PdlOrder::Q5ThenQ4, Epilogue>(
                x, qk_weight, value_z_weight, conv_weight, conv_states, conv_scale, valid_columns,
                initial_slot, query, key, value, z, publish, stream);
        }
        return;
    }
    const RecordColumnPublish publish{static_cast<__nv_bfloat16*>(conv_record.data), channels,
                                      static_cast<std::int32_t>(x.ne[1])};
    using Epilogue = GdnConvEpilogue<RecordColumnPublish>;
    if (x.ne[1] == 2) {
        launch_conv_by_geometry<PdlOrder::Q4ThenQ5, Epilogue>(
            x, qk_weight, value_z_weight, conv_weight, conv_states, conv_scale, valid_columns,
            initial_slot, query, key, value, z, publish, stream);
    } else {
        launch_conv_by_geometry<PdlOrder::Q5ThenQ4, Epilogue>(
            x, qk_weight, value_z_weight, conv_weight, conv_states, conv_scale, valid_columns,
            initial_slot, query, key, value, z, publish, stream);
    }
}

} // namespace ninfer::ops::detail