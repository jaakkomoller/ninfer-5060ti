"""Persistent-object contract for the complete Qwen3.8-27B artifact.

The graph and all non-vocabulary storage roles are identical to the registered
Qwen3.6-27B groupwise artifact.  The embedding and full output head use the Q6
and Q4 formats supported by the 27B runtime.

The MTP transformer tensors are quantized with a smaller-than-upstream format
mix for the RTX 5060 Ti 16 GB path; the upstream W8 representation is preserved
on targets that do not need this compression.

The five residual/gate text projection roles are stored as Q4G64_F16S instead
of the upstream Q5G64_F16S for the RTX 5060 Ti 16 GB path; the ~1350 MiB of
resident weight this frees raises the context ceiling from ~6K to ~83K tokens
while keeping the INT4 KV and MTP3 decode paths.
"""

from __future__ import annotations

from tools.convert.qwen3_6_27b import inventory as qwen3_6_inventory


MODEL_ID = "qwen3.8-27b"
WEIGHTS_ID = "groupwise-int"
TARGET_KEY = "qwen3_8_27b"

BF16 = qwen3_6_inventory.BF16
FP32 = qwen3_6_inventory.FP32
I32 = qwen3_6_inventory.I32
Q4 = qwen3_6_inventory.Q4
Q5 = qwen3_6_inventory.Q5
Q6 = qwen3_6_inventory.Q6
W8 = qwen3_6_inventory.W8

FORMAT_NAMES = qwen3_6_inventory.FORMAT_NAMES
LAYOUT_NAMES = qwen3_6_inventory.LAYOUT_NAMES
ResourceSpec = qwen3_6_inventory.ResourceSpec
StoredObjectSpec = qwen3_6_inventory.StoredObjectSpec
TensorSpec = qwen3_6_inventory.TensorSpec

FULL_ATTENTION_LAYERS = qwen3_6_inventory.FULL_ATTENTION_LAYERS
GDN_LAYERS = qwen3_6_inventory.GDN_LAYERS
RESOURCE_SPECS = qwen3_6_inventory.RESOURCE_SPECS


def _w8_vocabulary_endpoint(spec: TensorSpec) -> TensorSpec:
    if spec.name == "text/token_embedding":
        return qwen3_6_inventory.tensor_spec(spec.name, spec.shape, Q6)
    if spec.name == "text/output_head":
        return qwen3_6_inventory.tensor_spec(spec.name, spec.shape, Q4)
    return spec


# RTX 5060 Ti-specific MTP format mix. The Qwen3.6-27B inventory declares the
# MTP transformer in W8G32_F16S; the RTX 5060 Ti 16 GB path needs ~180 MiB less
# resident weight so the 27B model fits with MTP enabled. Each entry is
# selected to match a Q4G64_F16S or Q5G64_F16S dispatch case in the runtime.
_MTP_RTX5060TI_FORMAT = {
    "mtp/input_projection": Q4,
    "mtp/layer/attention/query_key_gate_value": Q4,
    "mtp/layer/attention/output": Q4,
    "mtp/layer/mlp/gate_up": Q4,
    "mtp/layer/mlp/down": Q4,
}


def _rtx5060ti_mtp(spec: TensorSpec) -> TensorSpec:
    target = _MTP_RTX5060TI_FORMAT.get(spec.name)
    if target is None:
        return spec
    return qwen3_6_inventory.tensor_spec(spec.name, spec.shape, target)


# RTX 5060 Ti-specific text weight format mix. The Qwen3.6-27B inventory
# declares the five residual/gate text projection roles as Q5G64_F16S; the
# RTX 5060 Ti 16 GB path stores them as Q4G64_F16S. Each entry is selected to
# match a Q4G64_F16S dispatch case in the runtime.
_TEXT_Q4_SUFFIXES = (
    "attention/gate_value",
    "attention/output",
    "gdn/value_z",
    "gdn/output",
    "mlp/down",
)


def _rtx5060ti_text_q4(spec: TensorSpec) -> TensorSpec:
    if spec.name.startswith("text/layers/") and spec.name.endswith(_TEXT_Q4_SUFFIXES):
        return qwen3_6_inventory.tensor_spec(spec.name, spec.shape, Q4)
    return spec


TEXT_CORE_TENSOR_SPECS = tuple(
    _rtx5060ti_text_q4(_w8_vocabulary_endpoint(spec))
    for spec in qwen3_6_inventory.TEXT_CORE_TENSOR_SPECS
)
DRAFT_HEAD_TENSOR_SPECS = qwen3_6_inventory.DRAFT_HEAD_TENSOR_SPECS
MTP_TENSOR_SPECS = tuple(
    _rtx5060ti_mtp(spec) for spec in qwen3_6_inventory.MTP_TENSOR_SPECS
)
VISION_TENSOR_SPECS = qwen3_6_inventory.VISION_TENSOR_SPECS

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

LOGICAL_ROW_VIEW_SPECS = qwen3_6_inventory.LOGICAL_ROW_VIEW_SPECS
ALIAS_SPECS = qwen3_6_inventory.ALIAS_SPECS
