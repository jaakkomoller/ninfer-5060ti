#include "ops/linear_add/q4/q4_linear_add_kernels.h"

#include "core/device.h"
#include "ops/common/math.h"
#include "ops/linear/q4/q4_rowsplit_gemv.cuh"

#include <cuda_bf16.h>

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

// One row per CTA, eight warps per row, dynamic group ownership: the linear_add
// decode shapes (K=6144/17408) do not match the fixed static ownership of the
// registered Q4 GEMV schedules, so the row's group range is derived at runtime.
using Q4LinearAddGemvSchedule =
    Q4RowSplitGemvSchedule<1, 8, 16, 1, Q4GemvActivationAccess::Direct,
                           Q4GemvLaneMapping::PackedByte2, Q4GemvDecodeMode::ScalarInteger,
                           Q4GemvCodeTransfer::SyncVector16, Q4GemvScaleAccess::Scalar16Shuffle,
                           Cache::ca, 0, 1>;

struct Q4LinearAddGemvResidualEpilogue {
    template <bool SplitOutput, int SplitRow>
    __device__ __forceinline__ void operator()(__nv_bfloat16* out, __nv_bfloat16* out_tail,
                                               int row, float value) const {
        static_assert(!SplitOutput,
                      "the Q4 linear_add GEMV residual epilogue is contiguous-only");
        (void)out_tail;
        (void)SplitRow;
        out[row] = __float2bfloat16(__bfloat162float(out[row]) + value);
    }
};

} // namespace

void q4_linear_add_gemv_residual_launch(const Tensor& x, const Weight& w, Tensor& residual_out,
                                        cudaStream_t stream) {
    const auto* xp     = static_cast<const __nv_bfloat16*>(x.data);
    const auto* codes  = static_cast<const std::uint8_t*>(w.qdata);
    const auto* scales = static_cast<const std::uint8_t*>(w.scales);
    auto* out          = static_cast<__nv_bfloat16*>(residual_out.data);

    if (!((w.n == 5120 && w.k == 6144 && w.padded_shape[1] == 6144) ||
          (w.n == 5120 && w.k == 17408 && w.padded_shape[1] == 17408))) {
        throw std::invalid_argument("q4 linear_add GEMV: unsupported exact shape");
    }

    const dim3 grid(static_cast<unsigned>(residual_out.ne[0] / Q4LinearAddGemvSchedule::kRowsPerCta),
                    1u, 1u);
    q4_rowsplit_gemv_kernel<Q4LinearAddGemvSchedule, false, 0, Q4LinearAddGemvResidualEpilogue>
        <<<grid, Q4LinearAddGemvSchedule::kThreads, 0, stream>>>(
            xp, codes, scales, out, nullptr, residual_out.ne[0], x.ne[0],
            Q4LinearAddGemvResidualEpilogue{});
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail