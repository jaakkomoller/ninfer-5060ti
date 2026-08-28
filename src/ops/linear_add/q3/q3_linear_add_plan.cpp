#include "ops/linear_add/q3/q3_linear_add_plan.h"

#include "ops/linear_add/q3/q3_linear_add_kernels.h"

#include <array>
#include <limits>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

constexpr std::int32_t kAnyCols = std::numeric_limits<std::int32_t>::max();

struct ColsSet {
    std::int32_t first;
    std::int32_t last;

    constexpr bool contains(std::int32_t cols) const noexcept {
        return cols >= first && cols <= last;
    }
};

struct SupportSpec {
    std::int32_t rows;
    std::int32_t k;
    std::int32_t padded_k;
};

struct RouteSpec {
    ColsSet cols;
    Q3LinearAddScheduleId schedule;
};

// The Q3 profile quantizes only the mlp/down role (N=5120, K=17408); every other
// linear_add shape stays Q4/Q5, so this catalog is closed over that single shape.
constexpr std::array<SupportSpec, 1> kSupports{{
    {5120, 17408, 17408},
}};

constexpr std::array<RouteSpec, 6> kK17408Routes{{
    {{1, 1}, Q3LinearAddScheduleId::GemvResidual},
    {{2, 16}, Q3LinearAddScheduleId::SimtResidual},
    {{17, 32}, Q3LinearAddScheduleId::MmaResidualR64C32},
    {{33, 48}, Q3LinearAddScheduleId::MmaResidualR64C48},
    {{49, 128}, Q3LinearAddScheduleId::MmaResidualR64C64},
    {{129, kAnyCols}, Q3LinearAddScheduleId::MmaResidualR64C128},
}};

template <std::size_t N>
constexpr bool catalog_is_closed(const std::array<RouteSpec, N>& routes) noexcept {
    std::int64_t expected = 1;
    for (const RouteSpec& route : routes) {
        if (route.cols.first != expected || route.cols.last < route.cols.first) { return false; }
        expected = static_cast<std::int64_t>(route.cols.last) + 1;
    }
    return routes.back().cols.last == kAnyCols &&
           expected == static_cast<std::int64_t>(kAnyCols) + 1;
}

static_assert(catalog_is_closed(kK17408Routes), "Q3 LinearAdd routes must be exact, contiguous, and closed");

bool supported_shape(const Q3LinearAddProblem& problem) noexcept {
    for (const SupportSpec& support : kSupports) {
        if (problem.rows == support.rows && problem.k == support.k &&
            problem.padded_k == support.padded_k) {
            return true;
        }
    }
    return false;
}

} // namespace

const char* q3_linear_add_schedule_name(Q3LinearAddScheduleId schedule) noexcept {
    switch (schedule) {
    case Q3LinearAddScheduleId::GemvResidual:
        return "linear_add.q3.gemv.residual";
    case Q3LinearAddScheduleId::SimtResidual:
        return "linear_add.q3.simt.residual";
    case Q3LinearAddScheduleId::MmaResidualR64C32:
        return "linear_add.q3.mma.r64.c32.cta_collective_residual";
    case Q3LinearAddScheduleId::MmaResidualR64C48:
        return "linear_add.q3.mma.r64.c48.cta_collective_residual";
    case Q3LinearAddScheduleId::MmaResidualR64C64:
        return "linear_add.q3.mma.r64.c64.cta_collective_residual";
    case Q3LinearAddScheduleId::MmaResidualR64C128:
        return "linear_add.q3.mma.r64.c128.cta_collective_residual";
    }
    return "linear_add.q3.unknown";
}

bool q3_linear_add_admits(const Q3LinearAddProblem& problem) noexcept {
    return supported_shape(problem) && problem.cols >= 1;
}

Q3LinearAddPlan q3_linear_add_resolve_plan(const Q3LinearAddProblem& problem) {
    if (!q3_linear_add_admits(problem)) {
        throw std::invalid_argument("q3 linear_add: exact problem or column count is not admitted");
    }
    for (const RouteSpec& route : kK17408Routes) {
        if (route.cols.contains(problem.cols)) { return {route.schedule, 0}; }
    }
    throw std::logic_error("q3 linear_add: admitted problem has no covering route");
}

std::size_t q3_linear_add_capacity_workspace_bytes(std::int32_t rows, std::int32_t k,
                                                   std::int32_t padded_k, std::int32_t min_cols,
                                                   std::int32_t max_cols) {
    if (min_cols <= 0 || max_cols < min_cols) {
        throw std::invalid_argument("q3 linear_add: invalid column interval");
    }
    (void)q3_linear_add_resolve_plan({rows, k, padded_k, min_cols});
    (void)q3_linear_add_resolve_plan({rows, k, padded_k, max_cols});

    return 0;
}

void q3_linear_add_execute_plan(const Q3LinearAddPlan& plan, const Tensor& x, const Weight& w,
                                Tensor& residual_out, WorkspaceArena& ws, cudaStream_t stream) {
    const Q3LinearAddProblem problem{residual_out.ne[0], x.ne[0], w.padded_shape[1], x.ne[1]};
    const Q3LinearAddPlan resolved = q3_linear_add_resolve_plan(problem);
    if (resolved.schedule != plan.schedule || resolved.workspace_bytes != plan.workspace_bytes) {
        throw std::invalid_argument("q3 linear_add: plan does not match the exact problem");
    }
    (void)ws;

    switch (plan.schedule) {
    case Q3LinearAddScheduleId::GemvResidual:
        q3_linear_add_gemv_residual_launch(x, w, residual_out, stream);
        return;
    case Q3LinearAddScheduleId::SimtResidual:
        q3_linear_add_simt_residual_launch(x, w, residual_out, stream);
        return;
    case Q3LinearAddScheduleId::MmaResidualR64C32:
        q3_linear_add_mma_r64_c32_launch(x, w, residual_out, stream);
        return;
    case Q3LinearAddScheduleId::MmaResidualR64C48:
        q3_linear_add_mma_r64_c48_launch(x, w, residual_out, stream);
        return;
    case Q3LinearAddScheduleId::MmaResidualR64C64:
        q3_linear_add_mma_r64_c64_launch(x, w, residual_out, stream);
        return;
    case Q3LinearAddScheduleId::MmaResidualR64C128:
        q3_linear_add_mma_r64_c128_launch(x, w, residual_out, stream);
        return;
    }
    throw std::logic_error("q3 linear_add: unknown schedule");
}

void q3_linear_add_dispatch(const Tensor& x, const Weight& w, Tensor& residual_out,
                            WorkspaceArena& ws, cudaStream_t stream) {
    const Q3LinearAddProblem problem{residual_out.ne[0], x.ne[0], w.padded_shape[1], x.ne[1]};
    const Q3LinearAddPlan plan = q3_linear_add_resolve_plan(problem);
    q3_linear_add_execute_plan(plan, x, w, residual_out, ws, stream);
}

} // namespace ninfer::ops::detail