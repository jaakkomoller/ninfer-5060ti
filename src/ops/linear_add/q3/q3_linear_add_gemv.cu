#include "ops/linear_add/q3/q3_linear_add_kernels.h"

#include "core/device.h"
#include "ops/common/math.h"
#include "ops/linear/q3/q3_rowsplit_gemv.cuh"

#include <cuda_bf16.h>

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

// One row per CTA, eight warps per row, dynamic group ownership: the linear_add
// decode shape (K=17408) does not match the fixed static ownership of the
// registered Q3 GEMV schedules, so the row's group range is derived at runtime.
using Q3LinearAddGemvSchedule =
    Q3RowSplitGemvSchedule<1, 8, 16, 1>;

struct Q3LinearAddGemvResidualEpilogue {
    template <bool SplitOutput, int SplitRow>
    __device__ __forceinline__ void operator()(__nv_bfloat16* out, __nv_bfloat16* out_tail,
                                               int row, float value) const {
        static_assert(!SplitOutput,
                      "the Q3 linear_add GEMV residual epilogue is contiguous-only");
        (void)out_tail;
        (void)SplitRow;
        out[row] = __float2bfloat16(__bfloat162float(out[row]) + value);
    }
};

} // namespace

void q3_linear_add_gemv_residual_launch(const Tensor& x, const Weight& w, Tensor& residual_out,
                                        cudaStream_t stream) {
    const auto* xp     = static_cast<const __nv_bfloat16*>(x.data);
    const auto* codes  = static_cast<const std::uint8_t*>(w.qdata);
    const auto* scales = static_cast<const std::uint8_t*>(w.scales);
    auto* out          = static_cast<__nv_bfloat16*>(residual_out.data);

    if (!(w.n == 5120 && w.k == 17408 && w.padded_shape[1] == 17408)) {
        throw std::invalid_argument("q3 linear_add GEMV: unsupported exact shape");
    }

    const dim3 grid(static_cast<unsigned>(residual_out.ne[0] / Q3LinearAddGemvSchedule::kRowsPerCta),
                    1u, 1u);
    q3_rowsplit_gemv_kernel<Q3LinearAddGemvSchedule, false, 0, Q3LinearAddGemvResidualEpilogue>
        <<<grid, Q3LinearAddGemvSchedule::kThreads, 0, stream>>>(
            xp, codes, scales, out, nullptr, residual_out.ne[0], x.ne[0],
            Q3LinearAddGemvResidualEpilogue{});
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail