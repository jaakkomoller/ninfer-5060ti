"""Qwen3.8-27B MTP coding benchmark harness.

Runs the registered artifact across base / MTP draft=1 / 2 / 3 on a curated
set of coding-oriented prompts and records machine-readable JSON results.

Usage:
    python3 tools/bench/mtp_coding_bench.py run --tag run01
    python3 tools/bench/mtp_coding_bench.py summarize results/run01.json
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import re
import shlex
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ARTIFACT_DEFAULT = "out/qwen3_8_27b_rtx5060ti_q4mtp.ninfer"
BASE_ARTIFACT_DEFAULT = "out/qwen3_8_27b_rtx5060ti.ninfer"
BINARY_DEFAULT = "build/apps/ninfer"

# Each task carries:
#   id        short identifier
#   category  one of the categories in the task brief
#   prompt    task prompt (may include embedded code as text)
#   max_new   target generation length to give the model room to think
#   extractor optional function used to extract a runnable snippet
#             from the assistant output for correctness checks
#   validator optional function taking (snippet)->dict of check results

PROMPTS: list[dict[str, Any]] = [
    # ---------- 1. small code generation ----------
    {
        "id": "small_add",
        "category": "small_codegen",
        "prompt": "Write a Python function `add(a, b)` that returns the sum of two numbers. Include a docstring and a short usage example.",
        "max_new": 200,
    },
    {
        "id": "small_factorial",
        "category": "small_codegen",
        "prompt": "Write a Python function `factorial(n)` that computes n! iteratively (no recursion). Raise ValueError for negative inputs. Include a docstring and a usage example.",
        "max_new": 512,
    },
    {
        "id": "small_greet",
        "category": "small_codegen",
        "prompt": "Write a TypeScript function `greet(name: string): string` that returns 'Hello, {name}!' using template literals. Add JSDoc with an example call.",
        "max_new": 220,
    },

    # ---------- 2. medium code generation ----------
    {
        "id": "medium_csv",
        "category": "medium_codegen",
        "prompt": "Write a Python module that reads a CSV file `data.csv` with columns `id,name,score`, computes the average score, and prints the names of entries whose score is above average. Use only the standard library. Include a `if __name__ == '__main__'` guard.",
        "max_new": 600,
    },
    {
        "id": "medium_cli",
        "category": "medium_codegen",
        "prompt": "Write a Bash script that recursively finds all files under the current directory whose name matches `*.log`, prints the file path and line count for each, and exits non-zero if any file has more than 1000 lines. Use `find` and `wc -l`.",
        "max_new": 350,
    },
    {
        "id": "medium_ratelimit",
        "category": "medium_codegen",
        "prompt": "Write a Python class `RateLimiter` with `acquire()` returning True if a request is allowed under a token-bucket algorithm. Constructor takes `rate_per_sec` and `burst`. Use only the standard library, no threading.",
        "max_new": 700,
    },

    # ---------- 3. algorithmic / problem solving ----------
    {
        "id": "algo_twosum",
        "category": "algorithmic",
        "prompt": "Solve LeetCode 1 (Two Sum): given a list `nums` and an integer `target`, return the indices of the two numbers that add up to target. Use a hash map. Each input has exactly one solution. Include type hints and docstring.",
        "max_new": 500,
    },
    {
        "id": "algo_lis",
        "category": "algorithmic",
        "prompt": "Given a list of integers, compute the length of the longest strictly increasing subsequence using the O(n log n) patience-sorting method. Include a short explanation in the docstring.",
        "max_new": 500,
    },
    {
        "id": "algo_paren",
        "category": "algorithmic",
        "prompt": "Given a string containing only the characters '(', ')', '{', '}', '[', ']', determine whether the brackets are balanced. Use a stack. Include 3 example inputs/outputs in the docstring.",
        "max_new": 450,
    },
    {
        "id": "algo_dijkstra",
        "category": "algorithmic",
        "prompt": "Implement Dijkstra's shortest-path algorithm in Python for a graph given as `dict[int, list[tuple[int, int]]]` mapping a node to (neighbor, weight). Return a dict mapping each reachable node to its distance from the source. Include type hints.",
        "max_new": 600,
    },

    # ---------- 4. debugging broken code ----------
    {
        "id": "debug_offbyone",
        "category": "debugging",
        "prompt": "The following function should return the n-th Fibonacci number but returns the wrong value for some inputs. Find the bug and return only the corrected function:\n```python\ndef fib(n):\n    if n < 2:\n        return n\n    a, b = 0, 0\n    for _ in range(n - 1):\n        a, b = b, a + b\n    return b\n```",
        "max_new": 400,
    },
    {
        "id": "debug_index",
        "category": "debugging",
        "prompt": "This code crashes when the input list is empty. Identify the bug and return only the corrected version:\n```python\ndef average(values):\n    total = 0\n    for v in values:\n        total += v\n    return total / len(values)\n```",
        "max_new": 350,
    },
    {
        "id": "debug_offbyone_list",
        "category": "debugging",
        "prompt": "This function should return `nums` rotated right by k positions, but it returns the wrong answer when k is a multiple of len(nums). Find the bug and return only the corrected function:\n```python\ndef rotate(nums, k):\n    k = k % len(nums)\n    return nums[-k:] + nums[:-k]\n```",
        "max_new": 350,
    },

    # ---------- 5. modifying existing code ----------
    {
        "id": "modify_add_typehints",
        "category": "modifying",
        "prompt": "Add PEP-484 type hints to the following function without changing its behaviour:\n```python\ndef add_item(cart, item):\n    if not cart:\n        cart.append(item)\n        return\n    if item in cart:\n        return\n    cart.append(item)\n```",
        "max_new": 350,
    },
    {
        "id": "modify_add_docstring",
        "category": "modifying",
        "prompt": "Add a docstring with parameters, return value and a one-line example to the following function without changing its behaviour:\n```python\ndef clamp(value, low, high):\n    return max(low, min(high, value))\n```",
        "max_new": 300,
    },

    # ---------- 6. refactoring ----------
    {
        "id": "refactor_split",
        "category": "refactoring",
        "prompt": "Refactor the following code so the validation logic lives in a helper function. Do not change behaviour:\n```python\ndef set_age(user, age):\n    if age < 0:\n        raise ValueError('age must be >= 0')\n    if age > 150:\n        raise ValueError('age must be <= 150')\n    user.age = age\n```",
        "max_new": 400,
    },

    # ---------- 7. explaining code ----------
    {
        "id": "explain_regex",
        "category": "explaining",
        "prompt": "Explain in plain English what this regex matches, give two example strings (one that matches, one that does not), and describe each group:\n`^(?:https?://)?(?:www\\.)?([a-z0-9-]+)\\.(com|org|io)/?(?:\\?([^#]*))?(?:#(.*))?$`",
        "max_new": 800,
    },

    # ---------- 8. reasoning over moderate source-code context ----------
    {
        "id": "reason_source_explain",
        "category": "reasoning_source",
        "prompt": "Read the following Python module and explain (a) what each public function does, (b) the bug in `merge()`, and (c) why the bug occurs:\n```python\nclass Index:\n    def __init__(self):\n        self.docs = {}\n        self.sorted_keys = []\n\n    def add(self, key, doc):\n        self.docs[key] = doc\n        self.sorted_keys = sorted(self.docs)\n\n    def merge(self, other):\n        for key in other.sorted_keys:\n            self.docs[key] = other.docs[key]\n            self.sorted_keys = sorted(self.docs)\n```",
        "max_new": 900,
    },

    # ---------- 9. multi-file repository style ----------
    {
        "id": "multifile_mini_repo",
        "category": "multifile",
        "prompt": (
            "You have a tiny Python project with three files. Read them, then add a "
            "`__str__` method to the `Counter` class so printing it shows the items sorted by "
            "count descending (ties broken alphabetically). Do not change the public API "
            "or the test.\n\n"
            "`storage.py`:\n"
            "```python\nclass Counter:\n"
            "    def __init__(self):\n"
            "        self._data = {}\n"
            "    def incr(self, key, by=1):\n"
            "        self._data[key] = self._data.get(key, 0) + by\n"
            "    def get(self, key):\n"
            "        return self._data.get(key, 0)\n"
            "```\n\n"
            "`tests.py`:\n"
            "```python\nfrom storage import Counter\n"
            "c = Counter()\n"
            "c.incr('a'); c.incr('b', 2); c.incr('a')\n"
            "assert c.get('a') == 2\n"
            "```\n\n"
            "`README.md`:\n"
            "```\n"
            "# Storage\n"
            "Counts how many times things happen.\n"
            "```\n\n"
            "Return only the updated `storage.py` plus the new test cases you would add to "
            "`tests.py` for the `__str__` method (no need to print it, just the asserts)."
        ),
        "max_new": 800,
    },

    # ---------- 10. shell / sysadmin ----------
    {
        "id": "shell_find_dup",
        "category": "shell_sysadmin",
        "prompt": "Write a one-line Bash command that finds all duplicate files (same content by sha256) under /var/log and prints each duplicated path grouped per sha256, sorted by group size descending. Do not require root.",
        "max_new": 400,
    },
    {
        "id": "shell_service",
        "category": "shell_sysadmin",
        "prompt": "Write a systemd service unit file `/etc/systemd/system/myapp.service` that runs `/usr/local/bin/myapp --port 8080` as user `myapp`, restarts on failure with a 5-second delay, and only starts after network-online.target. Include comments explaining each directive.",
        "max_new": 800,
    },
]


def build_command(binary: str, artifact: str, prompt: str, max_new: int, *,
                  max_context: int, draft_tokens: int | None,
                  extra: list[str]) -> list[str]:
    cmd = [
        binary,
        artifact,
        "--prompt", prompt,
        "--max-context", str(max_context),
        "--max-new", str(max_new),
        "--kv-dtype", "int8",
        "--prefill-chunk", "64",
        "--no-prefix-reuse",
        "--no-thinking",
        "--greedy",
    ]
    if draft_tokens is not None:
        cmd += ["--spec", "mtp", "--draft-tokens", str(draft_tokens)]
    cmd += extra
    return cmd


# ----- result parsing -----

SUMMARY_RE = re.compile(r"^summary\s+(\S.*?)\s{2,}(\S.*)$")
PromptTokens = "prompt tokens"
DecodeSpeed = "decode speed"
Throughput = "throughput (overall)"
ModelElapsed = "model elapsed"
PrefillSpeed = "prefill speed"
GeneratedTokens = "generated tokens"
FinishReason = "finish reason"
AcceptRate = "mtp acceptance rate"
AcceptLength = "mtp acceptance length"
AcceptByPos = "mtp accepted by pos"
MtpDrafts = "mtp drafted tokens"
MtpAccepts = "mtp accepted tokens"
MtpRounds = "mtp rounds"
MtpFallback = "mtp fallback steps"
MtpWindow = "mtp draft window"
FreeAfterStartup = "free after startup"
RuntimeReservation = "runtime reservation"
KVPayload = "kv cache payload"
MaxContext = "max context"


def parse_summary(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in text.splitlines():
        m = SUMMARY_RE.match(line)
        if not m:
            continue
        key = m.group(1).strip()
        val = m.group(2).strip()
        out[key] = val
    return out


def to_float(s: str) -> float:
    m = re.search(r"-?\d+(?:\.\d+)?", s)
    return float(m.group(0)) if m else float("nan")


def to_int(s: str) -> int:
    m = re.search(r"-?\d+", s)
    return int(m.group(0)) if m else 0


def extract_assistant(raw_stdout: str, summary: dict[str, str]) -> tuple[str, str]:
    # The ninfer CLI prints assistant output to stdout before a set of summary
    # lines starting with "prepare     render/preprocess". The `tensors/resources`
    # summary line that delimits the boundary appears on stderr; in the merged
    # raw_stdout+raw_stderr stream we get the stdout body first, then the
    # loading/tensors-resources/prepare phase markers. Anchor the boundary on
    # the prepare line (the last one that appears right before the decoding
    # summaries).
    marker = "prepare     render/preprocess"
    if marker in raw_stdout:
        # Drop everything from the first occurrence of the marker onward
        # (this is the first such marker — the second is after generation
        # finishes — but we want the model output, which appears before
        # generation stats).
        body = raw_stdout.split(marker, 1)[0]
        return body.strip("\n"), body
    return raw_stdout.strip("\n"), raw_stdout


def parse_acceptance_distribution(accept_by_pos: str, drafted: int) -> list[int]:
    if not accept_by_pos:
        return []
    try:
        parts = [int(p) for p in accept_by_pos.split(",") if p.strip().isdigit()]
    except ValueError:
        return []
    return parts


@dataclasses.dataclass
class RunResult:
    task_id: str
    category: str
    config: str          # base / mtp1 / mtp2 / mtp3
    draft_tokens: int
    prompt_tokens: int
    generated_tokens: int
    prefill_seconds: float
    prefill_tps: float
    decode_seconds: float
    decode_tps: float
    total_seconds: float
    throughput_tps: float
    acceptance_pct: float
    accept_length: float
    accepted_per_round: float
    accept_by_pos: list[int]
    drafted: int
    accepted: int
    rounds: int
    fallback: int
    runtime_mib: float
    free_after_startup_mib: float
    kv_payload_mib: float
    completion: str
    raw_response: str
    extra: dict[str, Any] = dataclasses.field(default_factory=dict)


def run_one(binary: str, artifact: str, task: dict[str, Any], config: str,
            draft_tokens: int | None, max_context: int, timeout: int,
            runs_dir: Path, tag: str, idx: int) -> RunResult:
    runs_dir.mkdir(parents=True, exist_ok=True)
    safe_id = re.sub(r"[^A-Za-z0-9_-]", "_", task["id"])
    out_file = runs_dir / f"{safe_id}__{config}__{tag}_{idx:02d}.txt"
    cmd = build_command(binary, artifact, task["prompt"], task["max_new"],
                        max_context=max_context, draft_tokens=draft_tokens,
                        extra=[])
    print(f"[run] {task['id']} {config} draft={draft_tokens} ctx={max_context}", flush=True)
    t0 = time.time()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=timeout)
        elapsed = time.time() - t0
        rc = proc.returncode
        out = proc.stdout
        err = proc.stderr
        completion = "ok" if rc == 0 else f"fail_rc{rc}"
        if err.strip():
            out = out + "\n[stderr]\n" + err
    except subprocess.TimeoutExpired:
        elapsed = time.time() - t0
        completion = "timeout"
        out = ""
    out_file.write_text(out)
    summary = parse_summary(out)
    raw_response = ""
    if "summary     tensors/resources" in out:
        try:
            raw_response, _ = extract_assistant(out, summary)
        except Exception:
            raw_response = ""
    if PromptTokens in summary and GeneratedTokens in summary:
        prompt_tokens = to_int(summary[PromptTokens])
        generated = to_int(summary[GeneratedTokens])
        prefill_s = to_float(summary.get(PrefillSpeed, "0"))
        decode_s = to_float(summary.get(DecodeSpeed, "0"))
        model_elapsed = to_float(summary.get(ModelElapsed, "0"))
        throughput = to_float(summary.get(Throughput, "0"))
        # Summary lines: "prefill speed 116.37 tok/s"
        # Use model_elapsed to compute total latency.
        prefill_t = 0.0
        for line in out.splitlines():
            if line.startswith("generate    text prefill"):
                try:
                    prefill_t = float(line.split()[2])
                except (IndexError, ValueError):
                    pass
            elif line.startswith("generate    decode"):
                try:
                    decode_t = float(line.split()[2])
                except (IndexError, ValueError):
                    pass
        decode_t_s = decode_t if decode_t else 0.0
        # Accept / accept-length numbers can include "%"
        accept_pct = to_float(summary.get(AcceptRate, "0"))
        accept_len = to_float(summary.get(AcceptLength, "0"))
        accepted = to_int(summary.get(MtpAccepts, "0"))
        drafted = to_int(summary.get(MtpDrafts, "0"))
        rounds = to_int(summary.get(MtpRounds, "0"))
        fallback = to_int(summary.get(MtpFallback, "0"))
        runtime_mib = to_float(summary.get(RuntimeReservation, "0").replace("MiB", ""))
        free_after = to_float(summary.get(FreeAfterStartup, "0").replace("MiB", ""))
        kv_payload = to_float(summary.get(KVPayload, "0").replace("MiB", ""))
        return RunResult(
            task_id=task["id"],
            category=task["category"],
            config=config,
            draft_tokens=draft_tokens or 0,
            prompt_tokens=prompt_tokens,
            generated_tokens=generated,
            prefill_seconds=prefill_t,
            prefill_tps=to_float(summary.get(PrefillSpeed, "0")),
            decode_seconds=decode_t_s,
            decode_tps=decode_s,
            total_seconds=elapsed,
            throughput_tps=throughput,
            acceptance_pct=accept_pct,
            accept_length=accept_len,
            accepted_per_round=(accepted / rounds) if rounds else float("nan"),
            accept_by_pos=parse_acceptance_distribution(summary.get(AcceptByPos, ""), drafted),
            drafted=drafted,
            accepted=accepted,
            rounds=rounds,
            fallback=fallback,
            runtime_mib=runtime_mib,
            free_after_startup_mib=free_after,
            kv_payload_mib=kv_payload,
            completion=completion,
            raw_response=raw_response,
        )
    return RunResult(
        task_id=task["id"],
        category=task["category"],
        config=config,
        draft_tokens=draft_tokens or 0,
        prompt_tokens=0, generated_tokens=0,
        prefill_seconds=0.0, prefill_tps=0.0,
        decode_seconds=0.0, decode_tps=0.0,
        total_seconds=elapsed, throughput_tps=0.0,
        acceptance_pct=0.0, accept_length=0.0,
        accepted_per_round=0.0,
        accept_by_pos=[], drafted=0, accepted=0, rounds=0, fallback=0,
        runtime_mib=0.0, free_after_startup_mib=0.0,
        kv_payload_mib=0.0, completion=completion,
        raw_response=raw_response,
    )


def aggregate(results: list[RunResult]) -> dict[str, dict[str, Any]]:
    by_config: dict[str, list[RunResult]] = {}
    for r in results:
        by_config.setdefault(r.config, []).append(r)
    out: dict[str, dict[str, Any]] = {}
    for cfg, runs in by_config.items():
        decode = [r.decode_tps for r in runs if r.completion == "ok"]
        accept = [r.acceptance_pct for r in runs if r.completion == "ok" and r.draft_tokens > 0]
        avg_acc = [r.accepted_per_round for r in runs if r.completion == "ok" and r.draft_tokens > 0]
        thr = [r.throughput_tps for r in runs if r.completion == "ok"]
        total_s = [r.total_seconds for r in runs if r.completion == "ok"]
        out[cfg] = {
            "n_runs": len(runs),
            "n_ok": sum(1 for r in runs if r.completion == "ok"),
            "decode_mean": statistics.mean(decode) if decode else float("nan"),
            "decode_median": statistics.median(decode) if decode else float("nan"),
            "acceptance_mean": statistics.mean(accept) if accept else float("nan"),
            "avg_acc_per_round_mean": statistics.mean(avg_acc) if avg_acc else float("nan"),
            "throughput_mean": statistics.mean(thr) if thr else float("nan"),
            "total_latency_mean": statistics.mean(total_s) if total_s else float("nan"),
        }
    return out


def cmd_run(args: argparse.Namespace) -> int:
    out_path = Path(args.results_dir) / f"{args.tag}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    runs_dir = Path(args.runs_dir) / args.tag
    configs: list[tuple[str, int | None, str]] = [
        ("base", None, args.base_artifact),
        ("mtp1", 1, args.artifact),
        ("mtp2", 2, args.artifact),
        ("mtp3", 3, args.artifact),
    ]
    with out_path.open("w") as fh:
        for idx, task in enumerate(PROMPTS):
            for cfg_name, draft, artifact in configs:
                result = run_one(args.binary, artifact, task, cfg_name, draft,
                                args.max_context, args.timeout,
                                runs_dir, args.tag, idx)
                d = dataclasses.asdict(result)
                fh.write(json.dumps(d) + "\n")
                fh.flush()
                print(f"  -> {cfg_name:5s}  gen={result.generated_tokens:4d}  "
                      f"decode={result.decode_tps:5.1f} tok/s  "
                      f"acc={result.acceptance_pct:5.1f}%  "
                      f"ok={result.completion == 'ok'}", flush=True)
    print(f"\nResults written to {out_path}")
    return 0


def cmd_summarize(args: argparse.Namespace) -> int:
    rows: list[dict[str, Any]] = []
    with open(args.results) as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    results = [RunResult(**r) for r in rows]
    print(f"Loaded {len(results)} results from {args.results}")
    print()
    agg = aggregate(results)
    print("=" * 72)
    print(f"{'Config':<6} {'n':>4} {'decode_mean':>12} {'decode_med':>12} "
          f"{'thr_mean':>10} {'acc_mean':>10} {'acc/rnd':>9}")
    print("-" * 72)
    for cfg, stats in agg.items():
        print(f"{cfg:<6} {stats['n_runs']:>4} "
              f"{stats['decode_mean']:>12.2f} {stats['decode_median']:>12.2f} "
              f"{stats['throughput_mean']:>10.2f} {stats['acceptance_mean']:>10.2f} "
              f"{stats['avg_acc_per_round_mean']:>9.2f}")
    print()
    # Per category
    print("Per-category decode throughput (tok/s):")
    cats = sorted({r.category for r in results})
    by_cat: dict[str, dict[str, list[float]]] = {}
    for r in results:
        if r.completion != "ok":
            continue
        by_cat.setdefault(r.category, {}).setdefault(r.config, []).append(r.decode_tps)
    print(f"{'Category':<20}", end="")
    for cfg in agg:
        print(f"{cfg:>10}", end="")
    print(f"{'n':>5}")
    for cat in cats:
        print(f"{cat:<20}", end="")
        for cfg in agg:
            vals = by_cat.get(cat, {}).get(cfg, [])
            mean = statistics.mean(vals) if vals else float("nan")
            print(f"{mean:>10.1f}", end="")
        print(f"{len(by_cat.get(cat, {}).get(next(iter(agg)), [])):>5}")
    print()
    # Acceptance breakdown
    print("Acceptance buckets vs mean decode speed:")
    buckets = [(0, 50), (50, 70), (70, 85), (85, 100), (100, 101)]
    print(f"{'Acceptance %':<20}{'Count':>8}{'Decode mean':>14}{'Decode median':>16}")
    bucket_data = {b: [] for b in buckets}
    for r in results:
        if r.completion != "ok" or r.draft_tokens == 0:
            continue
        for lo, hi in buckets:
            if lo <= r.acceptance_pct < hi:
                bucket_data[(lo, hi)].append(r.decode_tps)
                break
    for b in buckets:
        vals = bucket_data[b]
        if vals:
            print(f"[{b[0]:>3}, {b[1]:>3})  {len(vals):>8}"
                  f"{statistics.mean(vals):>14.2f}{statistics.median(vals):>16.2f}")
    return 0


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    pr = sub.add_parser("run", help="run the benchmark sweep")
    pr.add_argument("--binary", default=BINARY_DEFAULT,
                    help="path to the ninfer executable")
    pr.add_argument("--artifact", default=ARTIFACT_DEFAULT)
    pr.add_argument("--base-artifact", default=BASE_ARTIFACT_DEFAULT)
    pr.add_argument("--tag", required=True, help="run identifier")
    pr.add_argument("--max-context", type=int, default=1024)
    pr.add_argument("--timeout", type=int, default=180)
    pr.add_argument("--results-dir", default="results/mtp_coding")
    pr.add_argument("--runs-dir", default="results/mtp_coding/runs")
    pr.set_defaults(func=cmd_run)
    ps = sub.add_parser("summarize", help="summarize a previous run")
    ps.add_argument("--results", required=True, help="results JSONL file")
    ps.set_defaults(func=cmd_summarize)
    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
