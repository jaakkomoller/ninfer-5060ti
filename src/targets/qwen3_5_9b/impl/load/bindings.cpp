#include "targets/qwen3_5_9b/impl/load/bindings.h"

#include "artifact/typed_binding.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

namespace ninfer::targets::qwen3_5_9b::detail {
namespace {

using artifact::NumericFormat;

bool is_full_layer(std::size_t layer) { return layer >= 3 && (layer - 3) % 4 == 0; }

WeightPlan bind_weight(artifact::Binder& binder, std::string_view name, NumericFormat format,
                       std::initializer_list<std::uint64_t> shape) {
    return WeightPlan{.object = artifact::bind_device_tensor(binder, name, format, shape),
                      .format = format};
}

Weight materialized_weight(const artifact::MaterializedArtifact& materialized,
                           const WeightPlan& plan, std::int32_t rows, std::int32_t columns) {
    return artifact::materialized_weight(materialized, plan.object, plan.format, rows, columns);
}

Weight row_view(const Weight& block, std::int32_t row_begin, std::int32_t row_count) {
    if (row_begin < 0 || row_count <= 0 || row_begin + row_count > block.n ||
        block.layout != QuantLayout::RowSplit) {
        throw std::logic_error("invalid target row view");
    }
    const std::uint64_t groups    = static_cast<std::uint64_t>(block.padded_shape[1] / block.group);
    const std::uint64_t low_group = 32;
    const std::uint64_t high_group = block.qtype == QType::Q5G64_F16S   ? 8
                                      : block.qtype == QType::Q6G64_F16S ? 16
                                                                         : 0;
    const std::uint64_t low_row    = groups * low_group;
    const std::uint64_t high_row   = groups * high_group;
    const std::uint64_t scale_row  = groups * 2;
    Weight out                     = block;
    out.qdata                      = static_cast<const std::byte*>(block.qdata) +
                static_cast<std::uint64_t>(row_begin) * low_row;
    out.qhigh  = high_group == 0 ? nullptr
                                 : static_cast<const std::byte*>(block.qhigh) +
                                      static_cast<std::uint64_t>(row_begin) * high_row;
    out.scales = static_cast<const std::byte*>(block.scales) +
                 static_cast<std::uint64_t>(row_begin) * scale_row;
    out.n               = row_count;
    out.shape[0]        = row_count;
    out.padded_shape[0] = row_count;
    return out;
}

DensePostMixerPayload load_mlp(const MlpPlan& plan,
                               const artifact::MaterializedArtifact& materialized) {
    DensePostMixerPayload out;
    out.gate_up = materialized_weight(materialized, plan.gate_up, 24576, 4096);
    out.down    = materialized_weight(materialized, plan.down, 4096, 12288);
    return out;
}

FullAttentionProjectionPayload
load_attention_projection(const FullAttentionPlan& plan,
                          const artifact::MaterializedArtifact& materialized) {
    const auto* split = std::get_if<SplitAttentionProjectionPlan>(&plan.projection);
    return SplitAttentionProjectionPayload{
        .query_key  = materialized_weight(materialized, split->query_key, 4608, 4096),
        .gate_value = materialized_weight(materialized, split->gate_value, 4608, 4096),
    };
}

GdnInputProjectionPayload
load_gdn_input_projection(const GdnPlan& plan, const artifact::MaterializedArtifact& materialized) {
    const auto* split = std::get_if<SplitGdnInputProjectionPlan>(&plan.input_projection);
    return SplitGdnInputProjectionPayload{
        .query_key = materialized_weight(materialized, split->query_key, 2048, 4096),
        .value_z   = materialized_weight(materialized, split->value_z, 8192, 4096),
    };
}

void bind_text_layers(artifact::Binder& binder, BindingPlan& out) {
    for (std::size_t layer = 0; layer < kTextLayers; ++layer) {
        TextLayerPlan& target    = out.text_layers[layer];
        const std::string prefix = "text/layers/" + std::to_string(layer) + "/";
        target.input_norm        = artifact::bind_device_tensor(binder, prefix + "input_norm",
                                                                NumericFormat::BF16, {4096});
        target.is_full_attention = is_full_layer(layer);
        if (target.is_full_attention) {
            target.attention.projection = SplitAttentionProjectionPlan{
                .query_key  = bind_weight(binder, prefix + "attention/query_key",
                                          NumericFormat::Q4G64_F16S, {4608, 4096}),
                .gate_value = bind_weight(binder, prefix + "attention/gate_value",
                                          NumericFormat::Q5G64_F16S, {4608, 4096}),
            };
            target.attention.query_norm = artifact::bind_device_tensor(
                binder, prefix + "attention/query_norm", NumericFormat::BF16, {256});
            target.attention.key_norm = artifact::bind_device_tensor(
                binder, prefix + "attention/key_norm", NumericFormat::BF16, {256});
            target.attention.output = bind_weight(binder, prefix + "attention/output",
                                                  NumericFormat::Q5G64_F16S, {4096, 4096});
        } else {
            target.gdn.a_log       = artifact::bind_device_tensor(binder, prefix + "gdn/a_log",
                                                                  NumericFormat::FP32, {32});
            target.gdn.dt_bias     = artifact::bind_device_tensor(binder, prefix + "gdn/dt_bias",
                                                                  NumericFormat::FP32, {32});
            target.gdn.convolution = artifact::bind_device_tensor(
                binder, prefix + "gdn/convolution", NumericFormat::BF16, {4, 8192});
            target.gdn.a_projection = artifact::bind_device_tensor(
                binder, prefix + "gdn/a_projection", NumericFormat::BF16, {32, 4096});
            target.gdn.b_projection = artifact::bind_device_tensor(
                binder, prefix + "gdn/b_projection", NumericFormat::BF16, {32, 4096});
            target.gdn.input_projection = SplitGdnInputProjectionPlan{
                .query_key = bind_weight(binder, prefix + "gdn/query_key",
                                         NumericFormat::Q4G64_F16S, {2048, 4096}),
                .value_z   = bind_weight(binder, prefix + "gdn/value_z", NumericFormat::Q5G64_F16S,
                                         {8192, 4096}),
            };
            target.gdn.norm = artifact::bind_device_tensor(binder, prefix + "gdn/norm",
                                                           NumericFormat::BF16, {128});
            target.gdn.output =
                bind_weight(binder, prefix + "gdn/output", NumericFormat::Q5G64_F16S, {4096, 4096});
        }
        target.post_attention_norm = artifact::bind_device_tensor(
            binder, prefix + "post_attention_norm", NumericFormat::BF16, {4096});
        target.mlp.gate_up =
            bind_weight(binder, prefix + "mlp/gate_up", NumericFormat::Q4G64_F16S, {24576, 4096});
        target.mlp.down =
            bind_weight(binder, prefix + "mlp/down", NumericFormat::Q5G64_F16S, {4096, 12288});
    }
}

RuntimeModelView build_runtime(BindingPlan const& plan,
                                const artifact::MaterializedArtifact& materialized) {
    RuntimeModelView view;

    // Token embedding
    view.token_embedding = materialized_weight(materialized, plan.token_embedding, 248320, 4096);

    // Text layers
    for (std::size_t layer = 0; layer < kTextLayers; ++layer) {
        auto& text_layer = view.text_layers[layer];
        text_layer.input_norm =
            Tensor{.data = materialized.device_data(plan.text_layers[layer].input_norm),
                   .bytes = 4096 * 2};
        text_layer.is_full_attention = plan.text_layers[layer].is_full_attention;

        if (plan.text_layers[layer].is_full_attention) {
            auto proj = load_attention_projection(
                plan.text_layers[layer].attention, materialized);
            text_layer.full_attention = std::get<SplitAttentionProjectionPayload>(proj);
            text_layer.full_attention.query_norm.data =
                materialized.device_data(plan.text_layers[layer].attention.query_norm);
            text_layer.full_attention.key_norm.data =
                materialized.device_data(plan.text_layers[layer].attention.key_norm);
        } else {
            auto inp = load_gdn_input_projection(
                plan.text_layers[layer].gdn, materialized);
            text_layer.gdn.input_projection = inp;
            text_layer.gdn.a_log.data = materialized.device_data(plan.text_layers[layer].gdn.a_log);
            text_layer.gdn.dt_bias.data = materialized.device_data(plan.text_layers[layer].gdn.dt_bias);
            text_layer.gdn.a_projection =
                materialized_weight(materialized, plan.text_layers[layer].gdn.a_projection, 32, 4096);
            text_layer.gdn.b_projection =
                materialized_weight(materialized, plan.text_layers[layer].gdn.b_projection, 32, 4096);
            text_layer.gdn.norm.data =
                materialized.device_data(plan.text_layers[layer].gdn.norm);
            text_layer.gdn.output =
                materialized_weight(materialized, plan.text_layers[layer].gdn.output, 4096, 4096);
        }

        text_layer.post_attention_norm.data =
            materialized.device_data(plan.text_layers[layer].post_attention_norm);
        text_layer.mlp = load_mlp(plan.text_layers[layer].mlp, materialized);
    }

    // Final norm, output head, draft head
    view.final_norm.data = materialized.device_data(plan.final_norm);
    view.output_head = materialized_weight(materialized, plan.output_head, 248320, 4096);
    view.draft_head = materialized_weight(materialized, plan.draft_head, 131072, 4096);
    view.draft_head_token_ids.data =
        materialized.device_data(plan.draft_head_token_ids);

    // MTP
    view.mtp.input_projection =
        materialized_weight(materialized, plan.mtp.input_projection, 4096, 8192);
    view.mtp.embedding_norm.data = materialized.device_data(plan.mtp.embedding_norm);
    view.mtp.hidden_norm.data = materialized.device_data(plan.mtp.hidden_norm);
    view.mtp.input_norm.data = materialized.device_data(plan.mtp.input_norm);
    view.mtp.query_key_gate_value.data =
        materialized.device_data(plan.mtp.query_key_gate_value);
    view.mtp.query_norm.data = materialized.device_data(plan.mtp.query_norm);
    view.mtp.key_norm.data = materialized.device_data(plan.mtp.key_norm);
    view.mtp.output =
        materialized_weight(materialized, plan.mtp.output, 4096, 4096);
    view.mtp.post_attention_norm.data = materialized.device_data(plan.mtp.post_attention_norm);
    view.mtp.mlp = load_mlp(plan.mtp.mlp, materialized);
    view.mtp.final_norm.data = materialized.device_data(plan.mtp.final_norm);

    // Vision
    view.vision_backbone.patch_embedding =
        materialized_weight(materialized, plan.vision_backbone.patch_embedding, 1152, 1536);
    view.vision_backbone.patch_embedding_bias.data =
        materialized.device_data(plan.vision_backbone.patch_embedding_bias);
    view.vision_backbone.position_embedding =
        materialized_weight(materialized, plan.vision_backbone.position_embedding, 2304, 1152);

    for (std::size_t i = 0; i < 27; ++i) {
        auto& block = view.vision_backbone.blocks[i];
        std::string prefix = "vision/layers/" + std::to_string(i) + "/";
        (void)prefix;
        block.attention.qkv = materialized_weight(
            materialized, plan.vision_backbone.blocks[i].attention.qkv, 3456, 1152);
        block.attention.qkv_bias.data =
            materialized.device_data(plan.vision_backbone.blocks[i].attention.qkv_bias);
        block.attention.output = materialized_weight(
            materialized, plan.vision_backbone.blocks[i].attention.output, 1152, 1152);
        block.attention.output_bias.data =
            materialized.device_data(plan.vision_backbone.blocks[i].attention.output_bias);
        block.mlp.fc1 = materialized_weight(
            materialized, plan.vision_backbone.blocks[i].mlp.fc1, 4304, 1152);
        block.mlp.fc1_bias.data =
            materialized.device_data(plan.vision_backbone.blocks[i].mlp.fc1_bias);
        block.mlp.fc2 = materialized_weight(
            materialized, plan.vision_backbone.blocks[i].mlp.fc2, 1152, 4304);
        block.mlp.fc2_bias.data =
            materialized.device_data(plan.vision_backbone.blocks[i].mlp.fc2_bias);
        block.norm1.weight.data =
            materialized.device_data(plan.vision_backbone.blocks[i].norm1.weight);
        block.norm1.bias.data =
            materialized.device_data(plan.vision_backbone.blocks[i].norm1.bias);
        block.norm2.weight.data =
            materialized.device_data(plan.vision_backbone.blocks[i].norm2.weight);
        block.norm2.bias.data =
            materialized.device_data(plan.vision_backbone.blocks[i].norm2.bias);
    }

    view.vision_merger.fc1 = materialized_weight(
        materialized, plan.vision_merger_fc1, 4608, 4608);
    view.vision_merger.fc1_bias.data =
        materialized.device_data(plan.vision_merger_fc1_bias);
    view.vision_merger.fc2 = materialized_weight(
        materialized, plan.vision_merger_fc2, 4096, 4608);
    view.vision_merger.fc2_bias.data =
        materialized.device_data(plan.vision_merger_fc2_bias);
    view.vision_merger.norm.weight.data =
        materialized.device_data(plan.vision_merger_norm.weight);
    view.vision_merger.norm.bias.data =
        materialized.device_data(plan.vision_merger_norm.bias);

    return view;
}

} // namespace

ArtifactLoadPlan bind_artifact(artifact::Binder& binder, WeightsProfile weights_profile,
                               qwen3_6::StartupFeatures features) {
    if (weights_profile != WeightsProfile::GroupwiseInt) {
        throw std::invalid_argument("qwen3_5_9b: only GroupwiseInt is supported");
    }

    BindingPlan out;
    out.features = features;

    // Frontend resources
    out.frontend = qwen3_6::bind_frontend_resources(binder, features);

    // Token embedding
    out.token_embedding = bind_weight(binder, "text/token_embedding",
                                      NumericFormat::Q6G64_F16S, {248320, 4096});

    // Text layers
    bind_text_layers(binder, out);

    // Final norm
    out.final_norm = artifact::bind_device_tensor(binder, "text/final_norm",
                                                   NumericFormat::BF16, {4096});

    // Output head
    out.output_head = bind_weight(binder, "text/output_head",
                                   NumericFormat::Q6G64_F16S, {248320, 4096});

    // Draft head
    out.draft_head = artifact::bind_device_tensor(binder, "text/draft_head",
                                                   NumericFormat::Q4G64_F16S, {131072, 4096});
    out.draft_head_token_ids = artifact::bind_device_tensor(binder,
        "text/draft_head_token_ids", NumericFormat::I32, {131072});

    // MTP
    out.mtp.input_projection = bind_weight(binder, "mtp/input_projection",
                                            NumericFormat::W8G32_F16S, {4096, 8192});
    out.mtp.embedding_norm = artifact::bind_device_tensor(binder, "mtp/embedding_norm",
                                                           NumericFormat::BF16, {4096});
    out.mtp.hidden_norm = artifact::bind_device_tensor(binder, "mtp/hidden_norm",
                                                        NumericFormat::BF16, {4096});
    out.mtp.input_norm = artifact::bind_device_tensor(binder, "mtp/layer/input_norm",
                                                       NumericFormat::BF16, {4096});
    out.mtp.query_key_gate_value = artifact::bind_device_tensor(
        binder, "mtp/layer/attention/query_key_gate_value",
        NumericFormat::W8G32_F16S, {12288, 4096});
    out.mtp.query_norm = artifact::bind_device_tensor(binder, "mtp/layer/attention/query_norm",
                                                       NumericFormat::BF16, {256});
    out.mtp.key_norm = artifact::bind_device_tensor(binder, "mtp/layer/attention/key_norm",
                                                     NumericFormat::BF16, {256});
    out.mtp.output = bind_weight(binder, "mtp/layer/attention/output",
                                  NumericFormat::W8G32_F16S, {4096, 4096});
    out.mtp.post_attention_norm = artifact::bind_device_tensor(
        binder, "mtp/layer/post_attention_norm", NumericFormat::BF16, {4096});
    out.mtp.mlp.gate_up = bind_weight(binder, "mtp/layer/mlp/gate_up",
                                       NumericFormat::W8G32_F16S, {24576, 4096});
    out.mtp.mlp.down = bind_weight(binder, "mtp/layer/mlp/down",
                                    NumericFormat::W8G32_F16S, {4096, 12288});
    out.mtp.final_norm = artifact::bind_device_tensor(binder, "mtp/final_norm",
                                                       NumericFormat::BF16, {4096});

    // Vision backbone
    out.vision_backbone = qwen3_6::bind_vision_backbone(binder, features);

    // Vision merger
    out.vision_merger_input = qwen3_6::bind_vision_merger_input(binder, features);
    out.vision_merger_fc1 = artifact::bind_device_tensor(
        binder, "vision/merger/fc1", NumericFormat::W8G32_F16S, {4608, 4608});
    out.vision_merger_fc1_bias = artifact::bind_device_tensor(
        binder, "vision/merger/fc1_bias", NumericFormat::BF16, {4608});
    out.vision_merger_fc2 = artifact::bind_device_tensor(
        binder, "vision/merger/fc2", NumericFormat::W8G32_F16S, {4096, 4608});
    out.vision_merger_fc2_bias = artifact::bind_device_tensor(
        binder, "vision/merger/fc2_bias", NumericFormat::BF16, {4096});
    out.vision_merger_norm = qwen3_6::bind_vision_merger_norm(binder, features);

    // Materialization plan
    ArtifactLoadPlan plan;
    plan.bindings = out;

    for (auto& spec : plan.bindings) {
        plan.materialization |= spec;
    }

    return plan;
}

LoadedModelData::LoadedModelData(BindingPlan plan,
                                  artifact::MaterializedArtifact materialized)
    : backing(std::move(materialized)),
      frontend(qwen3_6::make_frontend_resources(backing, plan)),
      runtime(build_runtime(plan, backing)) {}

} // namespace ninfer::targets::qwen3_5_9b::detail