#pragma once

#include "core/arena.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::detail {

enum class Q4LinearAddScheduleId {
    GemvResidual,
    SimtResidual,
    MmaResidualR64C32,
    MmaResidualR64C48,
    MmaResidualR64C64,
    MmaResidualR64C128,
};

struct Q4LinearAddProblem {
    std::int32_t rows;
    std::int32_t k;
    std::int32_t padded_k;
    std::int32_t cols;
};

struct Q4LinearAddPlan {
    Q4LinearAddScheduleId schedule;
    std::size_t workspace_bytes;
};

const char* q4_linear_add_schedule_name(Q4LinearAddScheduleId schedule) noexcept;

bool q4_linear_add_admits(const Q4LinearAddProblem& problem) noexcept;
Q4LinearAddPlan q4_linear_add_resolve_plan(const Q4LinearAddProblem& problem);

std::size_t q4_linear_add_capacity_workspace_bytes(std::int32_t rows, std::int32_t k,
                                                   std::int32_t padded_k, std::int32_t min_cols,
                                                   std::int32_t max_cols);

void q4_linear_add_execute_plan(const Q4LinearAddPlan& plan, const Tensor& x, const Weight& w,
                                Tensor& residual_out, WorkspaceArena& ws, cudaStream_t stream);
void q4_linear_add_dispatch(const Tensor& x, const Weight& w, Tensor& residual_out,
                            WorkspaceArena& ws, cudaStream_t stream);

} // namespace ninfer::ops::detail