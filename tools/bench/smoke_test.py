"""Smoke test the benchmark harness on one task across all 4 configs."""

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from mtp_coding_bench import (
    PROMPTS, build_command, parse_summary, BASE_ARTIFACT_DEFAULT,
    ARTIFACT_DEFAULT, BINARY_DEFAULT,
)

ARTIFACT = ARTIFACT_DEFAULT
BASE = BASE_ARTIFACT_DEFAULT
BINARY = BINARY_DEFAULT

for cfg_name, draft, artifact in [
    ("base", None, BASE),
    ("mtp1", 1, ARTIFACT),
    ("mtp2", 2, ARTIFACT),
    ("mtp3", 3, ARTIFACT),
]:
    task = PROMPTS[0]
    cmd = build_command(BINARY, artifact, task["prompt"], task["max_new"],
                        max_context=1024, draft_tokens=draft, extra=[])
    print(f"[{cfg_name}] cmd={cmd[:4]}...")
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    summary = parse_summary(out.stdout + out.stderr)
    print(f"  rc={out.returncode}  "
          f"gen={summary.get('generated tokens', '?')}  "
          f"decode={summary.get('decode speed', '?')}  "
          f"accept={summary.get('mtp acceptance rate', 'n/a')}")
