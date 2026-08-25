# MTP Coding Benchmark

Reproducible measurement harness for the Qwen3.8-27B MTP speedup on the
RTX 5060 Ti 16 GB. Compares the same 21 coding tasks (across 10
categories) on the base model, MTP draft=1, MTP draft=2, and MTP draft=3,
plus a separate long-context / OpenCode-like test.

## Run

```bash
# Main sweep (~25 minutes, 84 runs)
python3 tools/bench/mtp_coding_bench.py run --tag coding_ctx1024 \
    --max-context 1024 --timeout 240

# Summarise
python3 tools/bench/mtp_coding_bench.py summarize --results \
    results/mtp_coding/coding_ctx1024.jsonl

# Re-extract code blocks if raw_response was empty (older runs)
python3 tools/bench/reextract.py
python3 tools/bench/mtp_coding_bench.py summarize --results \
    results/mtp_coding/coding_ctx1024_v2.jsonl

# Quality check (inline exec-based pass/fail per task)
python3 tools/bench/quality_check.py --results \
    results/mtp_coding/coding_ctx1024_v2.jsonl \
    --report results/mtp_coding/coding_ctx1024_v2_quality.json

# Long-context / OpenCode-like test (~2 minutes, 2 sizes × 4 configs)
python3 tools/bench/long_context_bench.py
```

## Common flags

All scripts use the same generation flags:

- `--kv-dtype int8`
- `--prefill-chunk 64`
- `--no-prefix-reuse`
- `--no-thinking`
- `--greedy`
- MTP variants add `--spec mtp --draft-tokens N`

The binary is `build/apps/ninfer` and the artifacts are:

- `out/qwen3_8_27b_rtx5060ti.ninfer` (14.96 GB, base only)
- `out/qwen3_8_27b_rtx5060ti_q4mtp.ninfer` (16.95 GB, MTP-capable with
  BF16 GDN recurrent state)

## Scripts in this directory (other than the MTP harness)

The original `tools/bench/` directory contents (corpus baker, ninfer_bench
harness, serve helpers) are unchanged.  They live in the same directory
and are not affected by the MTP coding benchmark scripts.

## MTP harness scripts

- `mtp_coding_bench.py` — the main sweep over the 21 tasks × 4 configs
- `long_context_bench.py` — synthesises a 5-file Python repo and asks the
  model to add a class with a multi-config sweep
- `quality_check.py` — extracts the first Python code block from each
  response and runs a small inline test on a subset of tasks
- `reextract.py` — re-extracts the assistant text from raw run files into a
  fresh JSONL, used when the body-delimiter is wrong in an older run
- `smoke_test.py` — runs one task across all 4 configs as a sanity check
  before launching a full sweep

## Output locations

- `results/mtp_coding/<tag>.jsonl` — one JSON object per line, one per
  (task, config) pair
- `results/mtp_coding/runs/<task>__<config>__<tag>__NN.txt` — raw
  stdout + stderr for each run (kept for debugging)
- `results/mtp_coding/long_ctx/runs/<config>.txt` — same for the
  long-context test
- `results/mtp_coding/long_ctx/long_ctx.json` — long-context summary
- `results/mtp_coding/<tag>_quality.json` — per-task pass/fail
  report from `quality_check.py`
