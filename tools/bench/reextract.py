"""Re-extract assistant text from the run files into a new JSONL."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from mtp_coding_bench import parse_summary, extract_assistant

src = Path("results/mtp_coding/coding_ctx1024.jsonl")
dst = Path("results/mtp_coding/coding_ctx1024_v2.jsonl")
runs_dir = Path("results/mtp_coding/runs/coding_ctx1024")

with src.open() as fh:
    rows = [json.loads(line) for line in fh if line.strip()]

for r in rows:
    fpath = runs_dir / f"{r['task_id']}__{r['config']}__coding_ctx1024_{r.get('idx', 0):02d}.txt"
    if not fpath.exists():
        # Try without idx
        candidates = list(runs_dir.glob(f"{r['task_id']}__{r['config']}__*.txt"))
        if candidates:
            fpath = candidates[0]
    if fpath.exists():
        raw = fpath.read_text()
        summary = parse_summary(raw)
        body, _ = extract_assistant(raw, summary)
        r["raw_response"] = body
    else:
        print(f"missing: {fpath}")

with dst.open("w") as fh:
    for r in rows:
        fh.write(json.dumps(r) + "\n")
print(f"Wrote {len(rows)} rows to {dst}")
