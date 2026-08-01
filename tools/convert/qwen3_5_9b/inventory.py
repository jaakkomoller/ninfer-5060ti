"""Persistent-object contract for the complete Qwen3.5-9B artifact.

This module contains only target storage roles. Source-checkpoint mapping and
materialization live in the sibling conversion recipe.
"""

from __future__ import annotations

from tools.convert.qwen3_6.common.inventory import (
    BF16,
    CONTIGUOUS_LAYOUT,
    DIRECT_FORMATS,
    FORMAT_NAMES,
    FP32,
    I32,
    LAYOUT_NAMES,
    LogicalAliasSpec,
    LogicalRowViewSpec,
    Q4,
    Q5,
    Q6,
    RESOURCE_ENCODING,
    RESOURCE_SPECS,
    ROW_SPLIT_LAYOUT,
    ResourceSpec,
    StoredObjectSpec,
    TensorSpec,
    VISION_LAYERS,
    W8,
    build_vision_specs,
    tensor_spec,
)


MODEL_ID = "qwen3.5-9b"
WEIGHTS_ID = "groupwise-int"
TARGET_KEY = "qwen3_5_9b"

FULL_ATTENTION_LAYERS = tuple(range(3, 32, 4))
GDN_LAYERS = tuple(layer for layer in range(32) if layer not in FULL_ATTENTION_LAYERS)


_tensor = tensor_spec


def _build_text_core_specs() -> tuple[TensorSpec, ...]:
    specs: list[TensorSpec] = [
        _tensor("text/token_embedding", (248320, 4096), Q6),
    ]

    for layer in range(32):
        prefix = f"text/layers/{layer}/"
        specs.append(_tensor(prefix + "input_norm", (4096,), BF16))

        if layer in FULL_ATTENTION_LAYERS:
            specs.extend(
                (
                    _tensor(prefix + "attention/query_key", (5120, 4096), Q4),
                    _tensor(prefix + "attention/gate_value", (5120, 4096), Q5),
                    _tensor(prefix + "attention/query_norm", (256,), BF16),
                    _tensor(prefix + "attention/key_norm", (256,), BF16),
                    _tensor(prefix + "attention/output", (4096, 4096), Q5),
                )
            )
        else:
            specs.extend(
                (
                    _tensor(prefix + "gdn/a_log", (32,), FP32),
                    _tensor(prefix + "gdn/dt_bias", (32,), FP32),
                    _tensor(prefix + "gdn/convolution", (4, 8192), BF16),
                    _tensor(prefix + "gdn/a_projection", (32, 4096), BF16),
                    _tensor(prefix + "gdn/b_projection", (32, 4096), BF16),
                    _tensor(prefix + "gdn/query_key", (4096, 4096), Q4),
                    _tensor(prefix + "gdn/value_z", (8192, 4096), Q5),
                    _tensor(prefix + "gdn/norm", (128,), BF16),
                    _tensor(prefix + "gdn/output", (4096, 4096), Q5),
                )
            )

        specs.extend(
            (
                _tensor(prefix + "post_attention_norm", (4096,), BF16),
                _tensor(prefix + "mlp/gate_up", (24576, 4096), Q4),
                _tensor(prefix + "mlp/down", (4096, 12288), Q5),
            )
        )

    specs.extend(
        (
            _tensor("text/final_norm", (4096,), BF16),
            _tensor("text/output_head", (248320, 4096), Q6),
        )
    )
    return tuple(specs)


def _build_draft_head_specs() -> tuple[TensorSpec, ...]:
    return (
        _tensor("text/draft_head", (131072, 4096), Q4),
        _tensor("text/draft_head_token_ids", (131072,), I32),
    )


def _build_mtp_specs() -> tuple[TensorSpec, ...]:
    return (
        _tensor("mtp/input_projection", (4096, 8192), W8),
        _tensor("mtp/embedding_norm", (4096,), BF16),
        _tensor("mtp/hidden_norm", (4096,), BF16),
        _tensor("mtp/layer/input_norm", (4096,), BF16),
        _tensor("mtp/layer/attention/query_key_gate_value", (10240, 4096), W8),
        _tensor("mtp/layer/attention/query_norm", (256,), BF16),
        _tensor("mtp/layer/attention/key_norm", (256,), BF16),
        _tensor("mtp/layer/attention/output", (4096, 4096), W8),
        _tensor("mtp/layer/post_attention_norm", (4096,), BF16),
        _tensor("mtp/layer/mlp/gate_up", (24576, 4096), W8),
        _tensor("mtp/layer/mlp/down", (4096, 12288), W8),
        _tensor("mtp/final_norm", (4096,), BF16),
    )


def _build_vision_specs() -> tuple[TensorSpec, ...]:
    return build_vision_specs(4096)


TEXT_CORE_TENSOR_SPECS = _build_text_core_specs()
DRAFT_HEAD_TENSOR_SPECS = _build_draft_head_specs()
MTP_TENSOR_SPECS = _build_mtp_specs()
VISION_TENSOR_SPECS = _build_vision_specs()

TENSOR_SPECS = (
    TEXT_CORE_TENSOR_SPECS
    + DRAFT_HEAD_TENSOR_SPECS
    + MTP_TENSOR_SPECS
    + VISION_TENSOR_SPECS
)
OBJECT_SPECS: tuple[StoredObjectSpec, ...] = RESOURCE_SPECS + TENSOR_SPECS

FORMAT_COUNTS = {
    numeric_format: sum(spec.format == numeric_format for spec in TENSOR_SPECS)
    for numeric_format in FORMAT_NAMES
}
LAYOUT_COUNTS = {
    layout: sum(spec.layout == layout for spec in TENSOR_SPECS)
    for layout in LAYOUT_NAMES
}


LOGICAL_ROW_VIEW_SPECS = (
    LogicalRowViewSpec(
        "text/layers/{l}/attention/query",
        "text/layers/{l}/attention/query_key",
        0,
        4096,
        (4096, 4096),
        FULL_ATTENTION_LAYERS,
    ),
    LogicalRowViewSpec(
        "text/layers/{l}/attention/key",
        "text/layers/{l}/attention/query_key",
        4096,
        5120,
        (1024, 4096),
        FULL_ATTENTION_LAYERS,
    ),
    LogicalRowViewSpec(
        "text/layers/{l}/attention/output_gate",
        "text/layers/{l}/attention/gate_value",
        0,
        4096,
        (4096, 4096),
        FULL_ATTENTION_LAYERS,
    ),
    LogicalRowViewSpec(
        "text/layers/{l}/attention/value",
        "text/layers/{l}/attention/gate_value",
        4096,
        5120,
        (1024, 4096),
        FULL_ATTENTION_LAYERS,
    ),
    LogicalRowViewSpec(
        "text/layers/{l}/gdn/query",
        "text/layers/{l}/gdn/query_key",
        0,
        2048,
        (2048, 4096),
        GDN_LAYERS,
    ),
    LogicalRowViewSpec(
        "text/layers/{l}/gdn/key",
        "text/layers/{l}/gdn/query_key",
        2048,
        4096,
        (2048, 4096),
        GDN_LAYERS,
    ),
    LogicalRowViewSpec(
        "text/layers/{l}/gdn/value",
        "text/layers/{l}/gdn/value_z",
        0,
        4096,
        (4096, 4096),
        GDN_LAYERS,
    ),
    LogicalRowViewSpec(
        "text/layers/{l}/gdn/z",
        "text/layers/{l}/gdn/value_z",
        4096,
        8192,
        (4096, 4096),
        GDN_LAYERS,
    ),
    LogicalRowViewSpec(
        "text/layers/{l}/mlp/gate",
        "text/layers/{l}/mlp/gate_up",
        0,
        12288,
        (12288, 4096),
        tuple(range(32)),
    ),
    LogicalRowViewSpec(
        "text/layers/{l}/mlp/up",
        "text/layers/{l}/mlp/gate_up",
        12288,
        24576,
        (12288, 4096),
        tuple(range(32)),
    ),
    LogicalRowViewSpec(
        "mtp/layer/attention/query",
        "mtp/layer/attention/query_key_gate_value",
        0,
        4096,
        (4096, 4096),
        None,
    ),
    LogicalRowViewSpec(
        "mtp/layer/attention/key",
        "mtp/layer/attention/query_key_gate_value",
        4096,
        5120,
        (1024, 4096),
        None,
    ),
    LogicalRowViewSpec(
        "mtp/layer/attention/output_gate",
        "mtp/layer/attention/query_key_gate_value",
        5120,
        9216,
        (4096, 4096),
        None,
    ),
    LogicalRowViewSpec(
        "mtp/layer/attention/value",
        "mtp/layer/attention/query_key_gate_value",
        9216,
        10240,
        (1024, 4096),
        None,
    ),
    LogicalRowViewSpec(
        "mtp/layer/mlp/gate",
        "mtp/layer/mlp/gate_up",
        0,
        12288,
        (12288, 4096),
        None,
    ),
    LogicalRowViewSpec(
        "mtp/layer/mlp/up",
        "mtp/layer/mlp/gate_up",
        12288,
        24576,
        (12288, 4096),
        None,
    ),
)


ALIAS_SPECS = (
    LogicalAliasSpec("mtp/token_embedding", ("text/token_embedding",)),
    LogicalAliasSpec("mtp/full_output_head", ("text/output_head",)),
    LogicalAliasSpec(
        "mtp/optimized_proposal_head",
        ("text/draft_head", "text/draft_head_token_ids"),
    ),
    LogicalAliasSpec(
        "text/layers/{l}/gdn/channel_major_convolution",
        ("text/layers/{l}/gdn/convolution",),
        layers=GDN_LAYERS,
        axis_order=(1, 0),
    ),
)