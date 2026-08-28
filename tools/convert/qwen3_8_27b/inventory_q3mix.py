"""Storage contract for the RTX 5060 Ti Q3-mix Qwen3.8-27B artifact.

Identical to the RTX 5060 Ti groupwise profile (five text residual/gate
projections as Q4G64_F16S, MTP as Q4) except the 64 text ``mlp/down``
projections are stored as Q3G64_F16S.  The ~680 MiB of resident weight this
frees buys the ~331 MiB required for the 100K-token MTP3 context on the 16 GB
card.  The artifact carries the registered ``qwen3.8-27b/groupwise-int``
identity; the runtime binder probes the ``mlp/down`` format and dispatches
Q3, Q4, or Q5 accordingly.
"""

from __future__ import annotations

from tools.convert.qwen3_6_27b import inventory as qwen3_6_inventory

from . import inventory

Q3 = "Q3G64_F16S"

MODEL_ID = inventory.MODEL_ID
WEIGHTS_ID = inventory.WEIGHTS_ID
TARGET_KEY = inventory.TARGET_KEY

BF16 = inventory.BF16
FP32 = inventory.FP32
I32 = inventory.I32
Q4 = inventory.Q4
Q5 = inventory.Q5
Q6 = inventory.Q6
W8 = inventory.W8

FORMAT_NAMES = qwen3_6_inventory.FORMAT_NAMES + (Q3,)
LAYOUT_NAMES = inventory.LAYOUT_NAMES
ResourceSpec = inventory.ResourceSpec
StoredObjectSpec = inventory.StoredObjectSpec
TensorSpec = inventory.TensorSpec

FULL_ATTENTION_LAYERS = inventory.FULL_ATTENTION_LAYERS
GDN_LAYERS = inventory.GDN_LAYERS
RESOURCE_SPECS = inventory.RESOURCE_SPECS


def _q3mix_text_mlp_down(spec: TensorSpec) -> TensorSpec:
    if spec.name.startswith("text/layers/") and spec.name.endswith("mlp/down"):
        return qwen3_6_inventory.tensor_spec(spec.name, spec.shape, Q3)
    return spec


TEXT_CORE_TENSOR_SPECS = tuple(
    _q3mix_text_mlp_down(spec) for spec in inventory.TEXT_CORE_TENSOR_SPECS
)
DRAFT_HEAD_TENSOR_SPECS = inventory.DRAFT_HEAD_TENSOR_SPECS
MTP_TENSOR_SPECS = inventory.MTP_TENSOR_SPECS
VISION_TENSOR_SPECS = inventory.VISION_TENSOR_SPECS

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

LOGICAL_ROW_VIEW_SPECS = inventory.LOGICAL_ROW_VIEW_SPECS
ALIAS_SPECS = inventory.ALIAS_SPECS

Q3_TENSOR_SPECS = tuple(spec for spec in TENSOR_SPECS if spec.format == Q3)


def validate_inventory() -> None:
    names = tuple(spec.name for spec in OBJECT_SPECS)
    if len(names) != len(set(names)):
        raise ValueError("Q3-mix inventory contains duplicate names")
    if (
        len(RESOURCE_SPECS),
        len(TEXT_CORE_TENSOR_SPECS),
        len(DRAFT_HEAD_TENSOR_SPECS),
        len(MTP_TENSOR_SPECS),
        len(VISION_TENSOR_SPECS),
        len(TENSOR_SPECS),
        len(OBJECT_SPECS),
        len(Q3_TENSOR_SPECS),
    ) != (6, 771, 2, 12, 333, 1118, 1124, 64):
        raise ValueError("registered Q3-mix inventory is incomplete")
    for spec in Q3_TENSOR_SPECS:
        if not (spec.name.startswith("text/layers/") and spec.name.endswith("mlp/down")):
            raise ValueError(f"unexpected Q3 tensor: {spec.name}")
        if spec.shape != (5120, 17408):
            raise ValueError(f"unexpected Q3 tensor shape: {spec.name} {spec.shape}")
    base = {spec.name: spec for spec in inventory.TENSOR_SPECS}
    for spec in TENSOR_SPECS:
        expected = base[spec.name]
        if spec.format != Q3 and (
            spec.shape,
            spec.format,
            spec.layout,
        ) != (expected.shape, expected.format, expected.layout):
            raise ValueError(f"Q3-mix spec diverges from the base profile: {spec.name}")
    if FORMAT_COUNTS.get(Q3) != 64:
        raise ValueError(f"unexpected Q3 allocation: {FORMAT_COUNTS}")
    if LAYOUT_COUNTS[qwen3_6_inventory.ROW_SPLIT_LAYOUT] != 439:
        raise ValueError(f"unexpected layout allocation: {LAYOUT_COUNTS}")


validate_inventory()


__all__ = [
    "ALIAS_SPECS",
    "BF16",
    "DRAFT_HEAD_TENSOR_SPECS",
    "FORMAT_COUNTS",
    "FORMAT_NAMES",
    "FP32",
    "FULL_ATTENTION_LAYERS",
    "GDN_LAYERS",
    "I32",
    "LAYOUT_COUNTS",
    "LAYOUT_NAMES",
    "LOGICAL_ROW_VIEW_SPECS",
    "MODEL_ID",
    "MTP_TENSOR_SPECS",
    "OBJECT_SPECS",
    "Q3",
    "Q3_TENSOR_SPECS",
    "Q4",
    "Q5",
    "Q6",
    "RESOURCE_SPECS",
    "StoredObjectSpec",
    "TARGET_KEY",
    "TENSOR_SPECS",
    "TEXT_CORE_TENSOR_SPECS",
    "TensorSpec",
    "VISION_TENSOR_SPECS",
    "WEIGHTS_ID",
    "W8",
    "validate_inventory",
]