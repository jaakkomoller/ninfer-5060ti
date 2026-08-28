#include "ops/linear_add/q4/q4_linear_add_kernels.h"

#include "core/device.h"
#include "ops/common/math.h"
#include "ops/common/token_slices.h"
#include "ops/linear/q4/q4_rowsplit_gemm_mma.cuh"

#include <cuda_bf16.h>

#include <cstdint>

namespace ninfer::ops::detail {
namespace {

using Q4LinearAddMmaR64C32Schedule =
    Q4RowSplitMmaGemmSchedule<64, 32, 64, 16, 8, 2, 2, Q4FragmentPipeline::Serial, Cache::cg,
                              Cache::cg, Q4ScaleLoad::Pair32>;
using Q4LinearAddMmaR64C48Schedule =
    Q4RowSplitMmaGemmSchedule<64, 48, 64, 16, 16, 2, 2, Q4FragmentPipeline::Serial, Cache::cg,
                              Cache::cg, Q4ScaleLoad::Pair32>;
using Q4LinearAddMmaR64C64Schedule =
    Q4RowSplitMmaGemmSchedule<64, 64, 64, 32, 32, 2, 3, Q4FragmentPipeline::PingPong, Cache::ca,
                              Cache::ca, Q4ScaleLoad::Scalar16>;
using Q4LinearAddMmaR64C128Schedule =
    Q4RowSplitMmaGemmSchedule<64, 128, 64, 64, 32, 2, 1, Q4FragmentPipeline::Serial, Cache::cg,
                              Cache::cg, Q4ScaleLoad::Pair32>;

template <class Schedule, bool Full>
void launch_kernel(const Tensor& x, const Weight& w, Tensor& residual_out, cudaStream_t stream) {
    const auto* xp              = static_cast<const __nv_bfloat16*>(x.data);
    const auto* codes           = static_cast<const std::uint8_t*>(w.qdata);
    const auto* scales          = static_cast<const std::uint8_t*>(w.scales);
    auto* out                   = static_cast<__nv_bfloat16*>(residual_out.data);
    const std::int32_t rows     = residual_out.ne[0];
    const std::int32_t k        = x.ne[0];
    const std::int32_t cols     = x.ne[1];
    const std::int32_t padded_k = w.padded_shape[1];
    const dim3 grid(static_cast<unsigned>(div_up(rows, Schedule::kBlockRows)),
                    static_cast<unsigned>(div_up(cols, Schedule::kBlockCols)), 1u);

    q4_rowsplit_gemm_mma_kernel<Schedule, Full, Q4MmaEpilogue::CtaCollectiveResidual>
        <<<grid, Schedule::kThreads, 0, stream>>>(xp, codes, scales, out, rows, k, cols, padded_k);
    CUDA_CHECK(cudaGetLastError());
}

template <class Schedule>
void launch_route(const Tensor& x, const Weight& w, Tensor& residual_out, cudaStream_t stream) {
    const bool full = (w.n % 64) == 0 && (x.ne[1] % Schedule::kBlockCols) == 0 &&
                      w.k == w.padded_shape[1] && (w.k % 64) == 0;
    for_each_token_slice(x.ne[1], Schedule::kBlockCols,
                         [&](std::int32_t offset, std::int32_t count) {
                             const Tensor x_slice = x.slice(1, offset, count);
                             Tensor residual_slice = residual_out.slice(1, offset, count);
                             if (full) {
                                 launch_kernel<Schedule, true>(x_slice, w, residual_slice, stream);
                             } else {
                                 launch_kernel<Schedule, false>(x_slice, w, residual_slice, stream);
                             }
                         });
}

} // namespace

void q4_linear_add_mma_r64_c32_launch(const Tensor& x, const Weight& w, Tensor& residual_out,
                                      cudaStream_t stream) {
    launch_route<Q4LinearAddMmaR64C32Schedule>(x, w, residual_out, stream);
}

void q4_linear_add_mma_r64_c48_launch(const Tensor& x, const Weight& w, Tensor& residual_out,
                                      cudaStream_t stream) {
    launch_route<Q4LinearAddMmaR64C48Schedule>(x, w, residual_out, stream);
}

void q4_linear_add_mma_r64_c64_launch(const Tensor& x, const Weight& w, Tensor& residual_out,
                                      cudaStream_t stream) {
    launch_route<Q4LinearAddMmaR64C64Schedule>(x, w, residual_out, stream);
}

void q4_linear_add_mma_r64_c128_launch(const Tensor& x, const Weight& w, Tensor& residual_out,
                                       cudaStream_t stream) {
    launch_route<Q4LinearAddMmaR64C128Schedule>(x, w, residual_out, stream);
}

} // namespace ninfer::ops::detail