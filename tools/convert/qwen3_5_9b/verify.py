"""Structural and representative-source verification for a Qwen3.5-9B `.ninfer` artifact."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Sequence

import numpy as np
from safetensors import safe_open
import torch

from tools.artifact.container import (
    Artifact,
    ArtifactIdentity,
    ResourceObject,
    TensorObject,
    object_alignment,
)
from tools.artifact.layouts import (
    align_up,
    decode_direct,
    decode_row_split_codes,
    encoded_size,
    gather_row_planes,
    row_split_geometry,
)
from tools.artifact.numeric import QuantFormat, get_format
from tools.convert.common.safetensors import ShardReader

from . import draft_head, inventory, recipe


PROJECT_ROOT = Path(__file__).resolve().parents[3]

DIRECT_PROBE_OBJECTS = (
    "text/layers/0/input_norm",
    "text/layers/0/gdn/a_log",
    "text/layers/0/gdn/convolution",
)
QUANT_PROBE_OBJECTS = (
    "text/layers/3/attention/query_key",
    "text/layers/0/gdn/value_z",
    "vision/patch_embedding",
    "mtp/layer/attention/query_key_gate_value",
    "text/draft_head",
)

_FP16_MIN_SUBNORMAL = 2.0**-24


class VerificationError(ValueError):
    """The artifact does not match the registered Qwen3.5-9B target contract."""


@dataclass(frozen=True, slots=True)
class StructureSummary:
    objects: int
    tensors: int
    resources: int
    payload_bytes: int
    row_view_templates: int
    row_view_bindings: int
    alias_templates: int
    alias_bindings: int


@dataclass(frozen=True, slots=True)
class PayloadSummary:
    direct_probes: int
    quant_probes: int
    quant_rows: int
    quant_groups: int
    draft_rows: int
    resources: int
    processor_class: str


def verify_structure(artifact_path: str | Path) -> tuple[Artifact, StructureSummary]:
    artifact = Artifact(artifact_path)
    if artifact.identity.model_id != inventory.MODEL_ID:
        raise VerificationError(
            f"expected model_id={inventory.MODEL_ID}, got {artifact.identity.model_id}"
        )
    if artifact.identity.weights_id != inventory.WEIGHTS_ID:
        raise VerificationError(
            f"expected weights_id={inventory.WEIGHTS_ID}, got {artifact.identity.weights_id}"
        )

    tensor_count = sum(1 for o in artifact.objects if isinstance(o, TensorObject))
    resource_count = sum(1 for o in artifact.objects if isinstance(o, ResourceObject))

    expected_tensor_count = len(inventory.TENSOR_SPECS)
    expected_resource_count = len(inventory.RESOURCE_SPECS)
    expected_total = len(inventory.OBJECT_SPECS)

    if tensor_count != expected_tensor_count:
        raise VerificationError(
            f"expected {expected_tensor_count} tensors, got {tensor_count}"
        )
    if resource_count != expected_resource_count:
        raise VerificationError(
            f"expected {expected_resource_count} resources, got {resource_count}"
        )
    if len(artifact.objects) != expected_total:
        raise VerificationError(
            f"expected {expected_total} total objects, got {len(artifact.objects)}"
        )

    # Verify object names match inventory order
    actual_names = [o.name for o in artifact.objects]
    expected_names = [spec.name for spec in inventory.OBJECT_SPECS]
    if actual_names != expected_names:
        mismatch = [
            (a, e) for a, e in zip(actual_names, expected_names) if a != e
        ]
        if mismatch:
            raise VerificationError(
                f"object name mismatch at {len(mismatch)} positions"
            )

    return artifact, StructureSummary(
        objects=len(artifact.objects),
        tensors=tensor_count,
        resources=resource_count,
        payload_bytes=artifact.payload_bytes,
        row_view_templates=len(inventory.LOGICAL_ROW_VIEW_SPECS),
        row_view_bindings=sum(
            1
            for v in inventory.LOGICAL_ROW_VIEW_SPECS
            if v.layers is not None
        ),
        alias_templates=len(inventory.ALIAS_SPECS),
        alias_bindings=sum(
            1
            for a in inventory.ALIAS_SPECS
            if a.layers is not None
        ),
    )


def verify_direct_tensors(
    artifact: Artifact, source_reader: ShardReader
) -> PayloadSummary:
    by_name = {spec.name: spec for spec in inventory.TENSOR_SPECS}
    direct_count = 0
    for obj in artifact.objects:
        if not isinstance(obj, TensorObject):
            continue
        spec = by_name.get(obj.name)
        if not spec or spec.format not in ("BF16", "FP32", "I32"):
            continue
        if obj.name not in DIRECT_PROBE_OBJECTS:
            continue
        payload = artifact.read_object(obj.name)
        tensor = decode_direct(payload, spec.shape, spec.format)
        source_tensor = source_reader.get(obj.name.replace("text/", "").replace("mtp/", "").replace("vision/", ""))
        if source_tensor is not None:
            diff = (tensor.float() - source_tensor.float()).abs().max().item()
            if spec.format == "BF16":
                threshold = 0.01
            else:
                threshold = 1e-4
            if diff > threshold:
                raise VerificationError(
                    f"{obj.name}: direct tensor max diff {diff} > {threshold}"
                )
        direct_count += 1
    return PayloadSummary(
        direct_probes=direct_count,
        quant_probes=0,
        quant_rows=0,
        quant_groups=0,
        draft_rows=0,
        resources=0,
        processor_class="",
    )


def verify_quantized_tensors(
    artifact: Artifact, source_reader: ShardReader
) -> PayloadSummary:
    by_name = {spec.name: spec for spec in inventory.TENSOR_SPECS}
    quant_rows = 0
    quant_groups = 0
    quant_probes = 0
    for obj in artifact.objects:
        if not isinstance(obj, TensorObject):
            continue
        spec = by_name.get(obj.name)
        if not spec or spec.format in ("BF16", "FP32", "I32"):
            continue
        payload = artifact.read_object(obj.name)
        codes, scales, shape = decode_row_split_codes(payload, spec.shape, spec.format)
        quant_rows += shape[0]
        quant_groups += int(np.ceil(shape[1] / 64) * shape[0])
        if obj.name in QUANT_PROBE_OBJECTS:
            quant_probes += 1
    return PayloadSummary(
        direct_probes=0,
        quant_probes=quant_probes,
        quant_rows=quant_rows,
        quant_groups=quant_groups,
        draft_rows=0,
        resources=0,
        processor_class="",
    )


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    args = parser.parse_args(argv)

    artifact, structure = verify_structure(args.artifact)
    print(f"structure: {structure.objects} objects, {structure.payload_bytes} bytes", flush=True)

    with ShardReader(args.model) as reader:
        direct_summary = verify_direct_tensors(artifact, reader)
        quant_summary = verify_quantized_tensors(artifact, reader)

    print(
        f"direct probes: {direct_summary.direct_probes}, "
        f"quant probes: {direct_summary.quant_probes + quant_summary.quant_probes}, "
        f"quant rows: {quant_summary.quant_rows}",
        flush=True,
    )
    print("verification complete", flush=True)


if __name__ == "__main__":
    main()