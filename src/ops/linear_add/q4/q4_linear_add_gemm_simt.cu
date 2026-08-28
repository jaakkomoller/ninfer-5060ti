#include "ops/linear_add/q4/q4_linear_add_kernels.h"

#include "core/device.h"
#include "ops/common/math.h"
#include "ops/common/token_slices.h"
#include "ops/linear/q4/q4_rowsplit_gemm_simt.cuh"

#include <cuda_bf16.h>

#include <cstdint>

namespace ninfer::ops::detail {
namespace {

using Q4LinearAddSimtSchedule = Q4RowSplitSimtGemmSchedule<8, 8, 16, 2, Cache::ca, 1>;

struct Q4LinearAddSimtResidualEpilogue {
    template <bool SplitOutput, int SplitRow, int Cols>
    __device__ __forceinline__ void
    operator()(__nv_bfloat16* out, __nv_bfloat16* out_tail, std::int32_t out_ld,
               std::int32_t out_tail_ld, std::int32_t row, std::int32_t col0,
               std::int32_t active_cols, const float (&values)[Cols]) const {
        static_assert(!SplitOutput,
                      "the Q4 linear_add SIMT residual epilogue is contiguous-only");
        (void)out_tail;
        (void)out_tail_ld;
        (void)SplitRow;
        for (int col = 0; col < Cols; ++col) {
            if (col >= active_cols) { continue; }
            const std::int64_t index = static_cast<std::int64_t>(col0 + col) * out_ld + row;
            out[index]               = __float2bfloat16(__bfloat162float(out[index]) + values[col]);
        }
    }
};

template <class Schedule, bool Full>
void launch_schedule(const Tensor& x, const Weight& w, Tensor& residual_out, cudaStream_t stream) {
    const std::int32_t rows     = residual_out.ne[0];
    const std::int32_t k        = x.ne[0];
    const std::int32_t cols     = x.ne[1];
    const std::int32_t out_ld   = static_cast<std::int32_t>(residual_out.nb[1] / sizeof(__nv_bfloat16));
    const std::int32_t padded_k = w.padded_shape[1];

    const dim3 grid(static_cast<unsigned>(div_up(rows, Schedule::kRowsPerCta)),
                    static_cast<unsigned>(div_up(cols, Schedule::kColsPerTile)), 1u);

    q4_rowsplit_gemm_simt_kernel<Schedule, Full, false, 0, Q4LinearAddSimtResidualEpilogue>
        <<<grid, Schedule::kThreads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(w.qdata),
            static_cast<const std::uint8_t*>(w.scales), static_cast<__nv_bfloat16*>(residual_out.data),
            nullptr, out_ld, 0, rows, k, cols, padded_k, Q4LinearAddSimtResidualEpilogue{});
    CUDA_CHECK(cudaGetLastError());
}

template <class Schedule>
void launch_route(const Tensor& x, const Weight& w, Tensor& residual_out, cudaStream_t stream) {
    const bool full = (residual_out.ne[0] % Schedule::kRowsPerCta) == 0 &&
                      ((x.ne[0] / Q4RowSplitStorage::kGroupK) % Schedule::kGroupsPerStage) == 0 &&
                      (x.ne[1] % Schedule::kColsPerTile) == 0;
    for_each_token_slice(x.ne[1], Schedule::kColsPerTile,
                         [&](std::int32_t offset, std::int32_t count) {
                             const Tensor x_slice = x.slice(1, offset, count);
                             Tensor residual_slice = residual_out.slice(1, offset, count);
                             if (full) {
                                 launch_schedule<Schedule, true>(x_slice, w, residual_slice, stream);
                             } else {
                                 launch_schedule<Schedule, false>(x_slice, w, residual_slice,
                                                                  stream);
                             }
                         });
}

} // namespace

void q4_linear_add_simt_residual_launch(const Tensor& x, const Weight& w, Tensor& residual_out,
                                        cudaStream_t stream) {
    launch_route<Q4LinearAddSimtSchedule>(x, w, residual_out, stream);
}

} // namespace ninfer::ops::detail