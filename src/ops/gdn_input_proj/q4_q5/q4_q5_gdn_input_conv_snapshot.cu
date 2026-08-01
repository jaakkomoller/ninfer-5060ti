#include "ops/gdn_input_proj/q4_q5/q4_q5_gdn_input_kernels.h"

#include "core/device.h"
#include "ops/common/math.h"
#include "ops/gdn_input_proj/gdn_conv_snapshot.cuh"
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

template <class Geometry>
GdnConvSnapshotEpilogue make_epilogue(const Tensor& conv_weight, Tensor& conv_states,
                                      const Tensor& initial_slot, Tensor& query, Tensor& key,
                                      Tensor& value, int global_row_offset) {
    return {
        static_cast<const __nv_bfloat16*>(conv_weight.data),
        static_cast<__nv_bfloat16*>(conv_states.data),
        static_cast<const std::int32_t*>(initial_slot.data),
        static_cast<__nv_bfloat16*>(query.data),
        static_cast<__nv_bfloat16*>(key.data),
        static_cast<__nv_bfloat16*>(value.data),
        Geometry::kChannels,
        Geometry::kQueryRows,
        Geometry::kKeyRows,
        Geometry::kValueRows,
        global_row_offset,
    };
}

template <class Geometry>
struct Q4GdnDecodeEpilogue {
    GdnConvSnapshotEpilogue conv;

    template <bool, int>
    __device__ __forceinline__ void operator()(__nv_bfloat16*, __nv_bfloat16*, int row,
                                               float value) const {
        const float projected[1]{value};
        conv.store(row, projected);
    }
};

template <class Geometry, int Tokens>
struct Q4GdnSmallTEpilogue {
    GdnConvSnapshotEpilogue conv;

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

template <class Geometry>
struct Q5GdnDecodeEpilogue {
    GdnConvSnapshotEpilogue conv;
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

template <class Geometry, int Tokens>
struct Q5GdnSmallTEpilogue {
    GdnConvSnapshotEpilogue conv;
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

template <class Geometry>
void launch_t1(const Tensor& x, const Weight& qk_weight, const Weight& value_z_weight,
               const GdnConvSnapshotEpilogue& qk_epilogue,
               const GdnConvSnapshotEpilogue& value_epilogue, Tensor& query, Tensor& value,
               Tensor& z, cudaStream_t stream) {
    using Schedule = std::conditional_t<Geometry::kHidden == 4096,
                                        Q4GemvR1W8DirectK64Schedule, Q4GemvR1W8DirectSchedule>;
    constexpr int q4_threads = Schedule::kThreads;
    q4_rowsplit_gemv_kernel<Schedule, false, 0, Q4GdnDecodeEpilogue<Geometry>>
        <<<Geometry::kQkRows / Schedule::kRowsPerCta, q4_threads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const std::uint8_t*>(qk_weight.qdata),
            static_cast<const std::uint8_t*>(qk_weight.scales),
            static_cast<__nv_bfloat16*>(query.data), nullptr, Geometry::kQkRows, Geometry::kHidden,
            Q4GdnDecodeEpilogue<Geometry>{qk_epilogue});

    constexpr int q5_rows_per_block = 16;
    constexpr int q5_threads        = q5_rows_per_block * 32;
    q5_rowsplit_gemv_kernel<Geometry::kValueZRows, Geometry::kHidden, q5_rows_per_block, 2, true,
                            false, true, Geometry::kValueRows, Q5GdnDecodeEpilogue<Geometry>>
        <<<Geometry::kValueZRows / q5_rows_per_block, q5_threads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const std::uint8_t*>(value_z_weight.qdata),
            static_cast<const std::uint8_t*>(value_z_weight.qhigh),
            static_cast<const std::uint8_t*>(value_z_weight.scales),
            static_cast<__nv_bfloat16*>(value.data), static_cast<__nv_bfloat16*>(z.data),
            Q5GdnDecodeEpilogue<Geometry>{value_epilogue, static_cast<__nv_bfloat16*>(z.data)});
}

template <class Geometry, int Tokens, class Q4Schedule>
void launch_small_t_schedule(const Tensor& x, const Weight& qk_weight, const Weight& value_z_weight,
                             const GdnConvSnapshotEpilogue& qk_epilogue,
                             const GdnConvSnapshotEpilogue& value_epilogue, Tensor& query,
                             Tensor& value, Tensor& z, cudaStream_t stream) {
    const dim3 q4_grid(Geometry::kQkRows / Q4Schedule::kRowsPerCta, 1u, 1u);
    q4_rowsplit_gemm_simt_kernel<Q4Schedule, false, false, 0, Q4GdnSmallTEpilogue<Geometry, Tokens>>
        <<<q4_grid, Q4Schedule::kThreads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const std::uint8_t*>(qk_weight.qdata),
            static_cast<const std::uint8_t*>(qk_weight.scales),
            static_cast<__nv_bfloat16*>(query.data), nullptr, Geometry::kQueryRows, 0,
            Geometry::kQkRows, Geometry::kHidden, Tokens, Geometry::kHidden,
            Q4GdnSmallTEpilogue<Geometry, Tokens>{qk_epilogue});

    constexpr int q5_threads = 4 * 32;
    const dim3 q5_grid(Geometry::kValueZRows, 1u, 1u);
    q5_rowsplit_gemm_simt_split4_kernel<Q5RowSplitSimtSchedule, Tokens, Geometry::kFullSlabs,
                                        Geometry::kHidden, true, Geometry::kValueRows,
                                        Q5GdnSmallTEpilogue<Geometry, Tokens>>
        <<<q5_grid, q5_threads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const std::uint8_t*>(value_z_weight.qdata),
            static_cast<const std::uint8_t*>(value_z_weight.qhigh),
            static_cast<const std::uint8_t*>(value_z_weight.scales),
            static_cast<__nv_bfloat16*>(value.data), static_cast<__nv_bfloat16*>(z.data),
            Geometry::kValueZRows, Geometry::kValueRows, Geometry::kHidden, Tokens,
            Geometry::kHidden, Geometry::kFullSlabs,
            Q5GdnSmallTEpilogue<Geometry, Tokens>{
                value_epilogue,
                static_cast<__nv_bfloat16*>(z.data),
            });
}

template <class Geometry, int Tokens>
void launch_small_t(const Tensor& x, const Weight& qk_weight, const Weight& value_z_weight,
                    const GdnConvSnapshotEpilogue& qk_epilogue,
                    const GdnConvSnapshotEpilogue& value_epilogue, Tensor& query, Tensor& value,
                    Tensor& z, cudaStream_t stream) {
    if constexpr (Tokens <= 4) {
        launch_small_t_schedule<Geometry, Tokens, Q4ScheduleC4>(
            x, qk_weight, value_z_weight, qk_epilogue, value_epilogue, query, value, z, stream);
    } else {
        launch_small_t_schedule<Geometry, Tokens, Q4ScheduleC8>(
            x, qk_weight, value_z_weight, qk_epilogue, value_epilogue, query, value, z, stream);
    }
}

template <class Geometry>
void launch_geometry(const Tensor& x, const Weight& qk_weight, const Weight& value_z_weight,
                     const Tensor& conv_weight, Tensor& conv_states, const Tensor& initial_slot,
                     Tensor& query, Tensor& key, Tensor& value, Tensor& z, cudaStream_t stream) {
    const GdnConvSnapshotEpilogue qk_epilogue =
        make_epilogue<Geometry>(conv_weight, conv_states, initial_slot, query, key, value, 0);
    const GdnConvSnapshotEpilogue value_epilogue = make_epilogue<Geometry>(
        conv_weight, conv_states, initial_slot, query, key, value, Geometry::kValueOffset);

    switch (x.ne[1]) {
    case 1:
        launch_t1<Geometry>(x, qk_weight, value_z_weight, qk_epilogue, value_epilogue, query,
                            value, z, stream);
        break;
    case 2:
        launch_small_t<Geometry, 2>(x, qk_weight, value_z_weight, qk_epilogue, value_epilogue,
                                    query, value, z, stream);
        break;
    case 3:
        launch_small_t<Geometry, 3>(x, qk_weight, value_z_weight, qk_epilogue, value_epilogue,
                                    query, value, z, stream);
        break;
    case 5:
        launch_small_t<Geometry, 5>(x, qk_weight, value_z_weight, qk_epilogue, value_epilogue,
                                    query, value, z, stream);
        break;
    case 6:
        launch_small_t<Geometry, 6>(x, qk_weight, value_z_weight, qk_epilogue, value_epilogue,
                                    query, value, z, stream);
        break;
    default:
        throw std::invalid_argument(
            "Q4/Q5 projection-epilogue GDN snapshot requires T=1..3 or 5..6");
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

void q4_q5_gdn_input_conv_snapshot_launch(const Tensor& x, const Weight& qk_weight,
                                          const Weight& value_z_weight, const Tensor& conv_weight,
                                          Tensor& conv_states, const Tensor& initial_slot,
                                          Tensor& query, Tensor& key, Tensor& value, Tensor& z,
                                          cudaStream_t stream) {
    switch (x.ne[0]) {
    case 5120:
        launch_geometry<GdnSnapshotGeometry27>(x, qk_weight, value_z_weight, conv_weight,
                                               conv_states, initial_slot, query, key, value, z,
                                               stream);
        return;
    case 4096:
        launch_geometry<GdnSnapshotGeometry9>(x, qk_weight, value_z_weight, conv_weight,
                                              conv_states, initial_slot, query, key, value, z,
                                              stream);
        return;
    default:
        throw std::invalid_argument("GDN Q4/Q5 snapshot: unsupported input width");
    }
}

void q4_q5_gdn_input_t4_post_snapshot_launch(const Tensor& projected, const Tensor& conv_weight,
                                             Tensor& conv_states, const Tensor& initial_slot,
                                             Tensor& query, Tensor& key, Tensor& value,
                                             cudaStream_t stream) {
    if (projected.ne[1] != 4) {
        throw std::invalid_argument("Q4/Q5 staged GDN post projection requires T=4");
    }
    switch (projected.ne[0]) {
    case 10240: {
        constexpr int threads = 64;
        constexpr int blocks  = (10240 + threads - 1) / threads;
        gdn_projected_conv_snapshot_kernel<10240, 2048, 2048, 6144, 4>
            <<<blocks, threads, 0, stream>>>(static_cast<const __nv_bfloat16*>(projected.data),
                                             static_cast<const __nv_bfloat16*>(conv_weight.data),
                                             static_cast<__nv_bfloat16*>(conv_states.data),
                                             static_cast<const std::int32_t*>(initial_slot.data),
                                             static_cast<__nv_bfloat16*>(query.data),
                                             static_cast<__nv_bfloat16*>(key.data),
                                             static_cast<__nv_bfloat16*>(value.data));
        break;
    }
    case 8192: {
        constexpr int threads = 64;
        constexpr int blocks  = (8192 + threads - 1) / threads;
        gdn_projected_conv_snapshot_kernel<8192, 2048, 2048, 4096, 4>
            <<<blocks, threads, 0, stream>>>(static_cast<const __nv_bfloat16*>(projected.data),
                                             static_cast<const __nv_bfloat16*>(conv_weight.data),
                                             static_cast<__nv_bfloat16*>(conv_states.data),
                                             static_cast<const std::int32_t*>(initial_slot.data),
                                             static_cast<__nv_bfloat16*>(query.data),
                                             static_cast<__nv_bfloat16*>(key.data),
                                             static_cast<__nv_bfloat16*>(value.data));
        break;
    }
    default:
        throw std::invalid_argument("GDN staged snapshot: unsupported projected width");
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
