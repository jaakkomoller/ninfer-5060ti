#include "targets/qwen3_5_9b/impl/variant.h"

#include "ninfer/ops/attn_input_proj.h"
#include "ninfer/ops/gdn_gating_proj.h"
#include "ninfer/ops/gdn_input_proj.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/linear_add.h"
#include "ninfer/ops/linear_pair.h"
#include "ninfer/ops/linear_swiglu.h"
#include "ninfer/ops/mtp_pack.h"
#include "ninfer/ops/residual_add.h"
#include "ninfer/ops/silu_mul.h"

#include <algorithm>
#include <stdexcept>

#define NINFER_QWEN36_VARIANT    ::ninfer::targets::qwen3_5_9b::detail::Variant
#define NINFER_QWEN36_RUNTIME_NS qwen3_5_9b_runtime
#include "targets/qwen3_6/impl/runtime/instantiate.h"

namespace ninfer::targets::qwen3_5_9b::detail {
namespace {

std::vector<GraphFrontierRange>
graph_ranges_through(std::uint32_t max_frontier, const std::vector<std::uint32_t>& preferred_ends) {
    std::vector<GraphFrontierRange> out;
    std::uint32_t begin = 0;
    for (const std::uint32_t preferred_end : preferred_ends) {
        if (begin > max_frontier) { break; }
        const std::uint32_t end = std::min(preferred_end, max_frontier);
        out.push_back({begin, end});
        if (end == max_frontier) { return out; }
        begin = end + 1;
    }
    if (begin <= max_frontier) { out.push_back({begin, max_frontier}); }
    return out;
}

void validate_token_interval(std::int32_t first, std::int32_t last) {
    if (first <= 0 || last < first) {
        throw std::invalid_argument("invalid target leaf token interval");
    }
}

ops::LinearPolicy text_policy(qwen3_6::TextPhase phase, const Weight& weight) {
    return ops::LinearPolicy::A16Only;
}

} // namespace

std::vector<GraphFrontierRange> Variant::ordinary_graph_ranges(std::uint32_t capacity) {
    return graph_ranges_through(capacity - 1, {127, 511, 2047, 4095, 8197, 16389, 32767});
}

std::vector<GraphFrontierRange> Variant::mtp_graph_ranges(std::uint32_t capacity,
                                                           std::uint32_t draft_window) {
    if (draft_window == 0 || 2ULL * draft_window > capacity) { return {}; }
    std::vector<std::uint32_t> ends;
    const auto add_shifted = [&](std::uint32_t visible_end, std::uint32_t offset) {
        if (visible_end >= offset) { ends.push_back(visible_end - offset); }
    };
    for (const std::uint32_t visible_end : {128U, 512U, 2048U, 4096U, 8198U, 16390U, 32768U}) {
        add_shifted(visible_end, 2 * draft_window);
    }
    if (draft_window == 3) {
        add_shifted(1029, draft_window + 1);
    } else if (draft_window == 4) {
        for (const std::uint32_t visible_end : {128U, 512U, 1029U}) {
            add_shifted(visible_end, draft_window + 1);
        }
    } else if (draft_window == 5) {
        for (const std::uint32_t visible_end : {128U, 160U, 2054U, 8198U}) {
            add_shifted(visible_end, draft_window + 1);
        }
    }
    std::sort(ends.begin(), ends.end());
    ends.erase(std::unique(ends.begin(), ends.end()), ends.end());
    return graph_ranges_through(capacity - 2 * draft_window, ends);
}

std::vector<GraphFrontierRange> Variant::dflash_graph_ranges(std::uint32_t, std::uint32_t) {
    return {};
}

void Variant::attach_diagnostics(qwen3_6::Program<Variant>& program, void* context,
                                  qwen3_6::TextTapCallback text,
                                  qwen3_6::VisionTapCallback vision) {
    program.impl_->diagnostic_context    = context;
    program.impl_->diagnostic_text_tap   = text;
    program.impl_->diagnostic_vision_tap = vision;
}

void Variant::detach_diagnostics(qwen3_6::Program<Variant>& program) noexcept {
    program.impl_->diagnostic_context    = nullptr;
    program.impl_->diagnostic_text_tap   = nullptr;
    program.impl_->diagnostic_vision_tap = nullptr;
}

std::size_t Variant::attention_projection_workspace_capacity_bytes(WeightsProfile,
                                                                    qwen3_6::TextPhase,
                                                                    std::int32_t first,
                                                                    std::int32_t last) {
    validate_token_interval(first, last);
    return 0;
}

std::size_t Variant::attention_output_projection_workspace_capacity_bytes(
    WeightsProfile, qwen3_6::TextPhase, std::int32_t first,
    std::int32_t last) {
    validate_token_interval(first, last);
    return ops::linear_add_workspace_capacity_bytes(QType::Q5G64_F16S, TextConfig::hidden,
                                                     TextConfig::query_size,
                                                     ops::LinearPolicy::A16Only, first, last);
}

std::size_t Variant::gdn_input_projection_workspace_capacity_bytes(WeightsProfile,
                                                                    qwen3_6::TextPhase,
                                                                    std::int32_t first,
                                                                    std::int32_t last) {
    validate_token_interval(first, last);
    return 0;
}

std::size_t Variant::gdn_input_projection_snapshot_workspace_capacity_bytes(WeightsProfile,
                                                                             qwen3_6::TextPhase,
                                                                             std::int32_t first,
                                                                             std::int32_t last) {
    validate_token_interval(first, last);
    return 0;
}

std::size_t Variant::gdn_output_projection_workspace_capacity_bytes(WeightsProfile,
                                                                     qwen3_6::TextPhase,
                                                                     std::int32_t first,
                                                                     std::int32_t last) {
    validate_token_interval(first, last);
    return ops::linear_add_workspace_capacity_bytes(QType::Q5G64_F16S, TextConfig::hidden,
                                                     TextConfig::query_size,
                                                     ops::LinearPolicy::A16Only, first, last);
}

std::size_t Variant::gdn_norm_control_projection_workspace_capacity_bytes(std::int32_t first,
                                                                           std::int32_t last) {
    validate_token_interval(first, last);
    WorkspaceLayoutBuilder layout;
    (void)layout.alloc(DType::BF16, {TextConfig::value_dim, last});
    return layout.peak_bytes(1);
}

std::size_t Variant::post_mixer_workspace_capacity_bytes(WeightsProfile, qwen3_6::TextPhase,
                                                          std::int32_t first,
                                                          std::int32_t last) {
    validate_token_interval(first, last);
    WorkspaceLayoutBuilder layout;
    (void)layout.alloc(DType::BF16, {TextConfig::intermediate, last});
    return layout.peak_bytes(1);
}

std::size_t Variant::mtp_post_mixer_workspace_capacity_bytes(std::int32_t first,
                                                              std::int32_t last) {
    validate_token_interval(first, last);
    WorkspaceLayoutBuilder layout;
    (void)layout.alloc(DType::BF16, {TextConfig::mtp_mlp_gate_up_rows, last});
    (void)layout.alloc(DType::BF16, {TextConfig::intermediate, last});
    (void)layout.alloc(DType::BF16, {TextConfig::hidden, last});
    return layout.peak_bytes(1);
}

} // namespace ninfer::targets::qwen3_5_9b::detail