"""Pinned official Qwen3.5-9B frontend resources used by artifact conversion.

This module owns only the checkpoint-invariant resource profile for the
Qwen3.5-9B target.  The Qwen3.5-9B release ships no `generation_config.json`;
the registered converter synthesizes that resource deterministically from the
registered special-token ids so the artifact keeps the complete six-resource
profile that the shared runtime requires.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence

from tools.convert.qwen3_6.common.conversion import ResourcePayload, load_resources
from tools.convert.qwen3_6.common.inventory import ResourceSpec

# Registered Qwen3.5-9B special-token ids (shared with Qwen3.6).
PAD_TOKEN_ID = 248044
EOS_TOKEN_ID = 248046

# The release checkpoint omits generation_config.json; the converter emits this
# deterministic resource instead of reading a checkpoint file.
GENERATION_CONFIG_JSON = {
    "bos_token_id": PAD_TOKEN_ID,
    "do_sample": True,
    "eos_token_id": [EOS_TOKEN_ID, PAD_TOKEN_ID],
    "pad_token_id": PAD_TOKEN_ID,
    "temperature": 1.0,
    "top_k": 20,
    "top_p": 0.95,
    "transformers_version": "5.13.0",
}

OFFICIAL_RESOURCE_SHA256 = {
    "frontend/tokenizer.json": (
        "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42"
    ),
    "frontend/tokenizer_config.json": (
        "316230d6a809701f4db5ea8f8fc862bc3a6f3229c937c174e674ff3ca0a64ac8"
    ),
    "frontend/chat_template.jinja": (
        "a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715"
    ),
    "frontend/generation_config.json": (
        "d0dbf670c6a372817b2ff92d5d47e3130d35de9c3a7164ba455fd7a88255b362"
    ),
    "frontend/preprocessor_config.json": (
        "27225450ac9c6529872ee1924fcb0962ff5634834f817040f444118116f4e516"
    ),
    "frontend/video_preprocessor_config.json": (
        "7768af27c1fafa9cc9011c1dc20067e03f8915e03b63504550e11d5066986d13"
    ),
}

GENERATION_CONFIG_SHA256 = OFFICIAL_RESOURCE_SHA256["frontend/generation_config.json"]


def synthesize_generation_config() -> bytes:
    """Return the registered deterministic generation_config.json payload."""
    return json.dumps(GENERATION_CONFIG_JSON, indent=2).encode("utf-8") + b"\n"


def validate_official_resource_hashes(
    actual_hashes: Mapping[str, str],
) -> None:
    """Require the complete official six-resource profile for Qwen3.5-9B."""

    expected_names = tuple(OFFICIAL_RESOURCE_SHA256)
    actual_names = tuple(actual_hashes)
    if actual_names != expected_names:
        raise ValueError(
            "Qwen3.5-9B frontend resource set mismatch: "
            f"expected {expected_names!r}, got {actual_names!r}"
        )
    for name, expected in OFFICIAL_RESOURCE_SHA256.items():
        actual = actual_hashes[name]
        if actual != expected:
            filename = name.removeprefix("frontend/")
            raise ValueError(
                f"official Qwen3.5-9B resource hash mismatch for {filename}: "
                f"expected {expected}, got {actual}"
            )


def validate_official_resources(resources: Sequence[ResourcePayload]) -> None:
    """Hash and validate already loaded resource payloads."""

    hashes = {
        resource.name: hashlib.sha256(resource.data).hexdigest()
        for resource in resources
    }
    if len(hashes) != len(resources):
        raise ValueError("Qwen3.5-9B frontend resource set contains duplicate names")
    validate_official_resource_hashes(hashes)


def load_official_resources(
    model_dir: str | Path,
    resource_specs: Sequence[ResourceSpec],
) -> tuple[ResourcePayload, ...]:
    """Load the pinned resource set, synthesizing the missing generation config."""

    spec_names = tuple(spec.name for spec in resource_specs)
    expected_names = tuple(OFFICIAL_RESOURCE_SHA256)
    if spec_names != expected_names:
        raise ValueError(
            "converter resource inventory does not match the official "
            f"Qwen3.5-9B profile: expected {expected_names!r}, got {spec_names!r}"
        )
    disk_specs = [
        spec for spec in resource_specs if spec.name != "frontend/generation_config.json"
    ]
    resources = list(load_resources(model_dir, disk_specs))
    resources.append(
        ResourcePayload(
            "frontend/generation_config.json", synthesize_generation_config()
        )
    )
    by_name = {payload.name: payload for payload in resources}
    resources = [by_name[spec.name] for spec in resource_specs]
    validate_official_resources(resources)
    return tuple(resources)


__all__ = [
    "GENERATION_CONFIG_SHA256",
    "OFFICIAL_RESOURCE_SHA256",
    "load_official_resources",
    "synthesize_generation_config",
    "validate_official_resource_hashes",
    "validate_official_resources",
]
