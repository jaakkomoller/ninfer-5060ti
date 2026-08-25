"""Run base / MTP1/2/3 on a generated OpenCode-like long source-code context.

Approach: synthesize ~1.5 KB to 4 KB of Python source across 3-5 related
files (a tiny todo-style utility module + tests + README) and then ask the
model to make a specific modification. We measure both prefill and decode
throughput, and report any quality difference.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from mtp_coding_bench import (
    build_command, parse_summary, RunResult, BASE_ARTIFACT_DEFAULT,
    ARTIFACT_DEFAULT, BINARY_DEFAULT, parse_acceptance_distribution,
    to_float, to_int, PromptTokens, DecodeSpeed, GeneratedTokens,
    AcceptRate, AcceptLength, ModelElapsed, PrefillSpeed,
    RuntimeReservation, FreeAfterStartup, KVPayload, MaxContext,
    MtpDrafts, MtpAccepts, MtpRounds, MtpFallback, MtpWindow, AcceptByPos,
)


def gen_tiny_repo(ctx_target_chars: int) -> str:
    """Synthesize a small multi-file Python repo whose total length is
    roughly ctx_target_chars.
    """
    # The repo is parameterizable but we keep the code simple to focus on
    # the measurement rather than the code quality.
    core = '''
## storage.py
```python
"""Tiny storage helpers used by the task tracker."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class Task:
    title: str
    done: bool = False
    priority: int = 0
    tags: List[str] = field(default_factory=list)


class TaskStore:
    def __init__(self) -> None:
        self._tasks: Dict[str, Task] = {}

    def add(self, key: str, task: Task) -> None:
        if key in self._tasks:
            raise KeyError(f"duplicate task key: {key}")
        self._tasks[key] = task

    def get(self, key: str) -> Task:
        return self._tasks[key]

    def mark_done(self, key: str) -> None:
        self._tasks[key].done = True

    def open_tasks(self) -> List[Task]:
        return [t for t in self._tasks.values() if not t.done]

    def by_tag(self, tag: str) -> List[Task]:
        return [t for t in self._tasks.values() if tag in t.tags]
```

## stats.py
```python
"""Statistics helpers for TaskStore output."""

from __future__ import annotations

from typing import Dict, Iterable


def completion_ratio(done: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return done / total


def tag_histogram(tasks: Iterable) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for t in tasks:
        for tag in t.tags:
            counts[tag] = counts.get(tag, 0) + 1
    return counts
```

## cli.py
```python
"""Tiny command line interface around TaskStore."""

from __future__ import annotations

import argparse
import sys

from storage import Task, TaskStore


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="taskmgr")
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("add")
    a.add_argument("key")
    a.add_argument("--title", required=True)
    a.add_argument("--priority", type=int, default=0)
    a.add_argument("--tags", nargs="*", default=[])
    d = sub.add_parser("done")
    d.add_argument("key")
    l = sub.add_parser("list")
    l.add_argument("--open", action="store_true")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    store = TaskStore()
    if args.cmd == "add":
        store.add(args.key, Task(title=args.title, priority=args.priority, tags=args.tags))
        return 0
    if args.cmd == "done":
        store.mark_done(args.key)
        return 0
    if args.cmd == "list":
        tasks = store.open_tasks() if args.open else list(store._tasks.values())
        for t in tasks:
            mark = "x" if t.done else " "
            print(f"[{mark}] {t.title} (p={t.priority}, tags={','.join(t.tags)})")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
```

## tests/test_storage.py
```python
"""Unit tests for the storage helpers."""

from storage import Task, TaskStore


def test_add_and_get():
    s = TaskStore()
    s.add("a", Task(title="A"))
    assert s.get("a").title == "A"


def test_mark_done():
    s = TaskStore()
    s.add("a", Task(title="A"))
    s.mark_done("a")
    assert s.get("a").done is True


def test_open_tasks():
    s = TaskStore()
    s.add("a", Task(title="A"))
    s.add("b", Task(title="B", done=True))
    open_ = s.open_tasks()
    assert len(open_) == 1 and open_[0].title == "A"


def test_by_tag():
    s = TaskStore()
    s.add("a", Task(title="A", tags=["x"]))
    s.add("b", Task(title="B", tags=["y", "x"]))
    got = s.by_tag("x")
    assert {t.title for t in got} == {"A", "B"}
```

## tests/test_stats.py
```python
"""Unit tests for the stats helpers."""

from stats import completion_ratio, tag_histogram


def test_completion_ratio_full():
    assert completion_ratio(3, 3) == 1.0


def test_completion_ratio_empty():
    assert completion_ratio(0, 0) == 0.0


def test_completion_ratio_half():
    assert completion_ratio(1, 2) == 0.5


def test_tag_histogram_counts():
    from storage import Task
    tasks = [Task(title="a", tags=["x", "y"]),
             Task(title="b", tags=["x"])]
    h = tag_histogram(tasks)
    assert h == {"x": 2, "y": 1}
```

## README.md
```
# taskmgr

A tiny task tracker.

## Install

`pip install -e .`

## Usage

```
taskmgr add docs --title "Write docs" --priority 2 --tags docs
taskmgr list --open
taskmgr done docs
```
'''
    base = core.strip()
    # If the user wants a larger context, repeat a "history" append-only log
    # in markdown to bulk up the prompt without changing the rest.  We always
    # pad up to at least ctx_target_chars so the two runs differ in size.
    if len(base) < ctx_target_chars:
        pad = ctx_target_chars - len(base)
        filler_chunks = []
        per_chunk = 240
        for i in range(0, pad, per_chunk):
            filler_chunks.append(
                f"## history/{i // per_chunk:04d}.md\n"
                "```\n"
                f"# session log entry {i // per_chunk}\n"
                "- edited docs/intro.md, fixed typo\n"
                "- reviewed PR #4823, left two comments\n"
                "- rebased feature/auth onto main\n"
                "- ran pytest -x -q (3 passed, 1 skipped, 0 failed)\n"
                f"- bumped version 0.4.{i // per_chunk + 1} -> 0.4.{i // per_chunk + 2}\n"
                "```\n")
        return base + "\n" + "".join(filler_chunks)
    return base


TASK_TEMPLATE = '''You are working in a Python project that has these files:

{repo}

## Task

Add a `TagIndex` class to `stats.py` (and tests for it in `tests/test_stats.py`)
with the following public interface:

```python
class TagIndex:
    """In-memory index from tag -> list[task_key]."""

    def __init__(self) -> None:
        ...

    def add(self, tag: str, key: str) -> None: ...
    def remove(self, tag: str, key: str) -> None: ...
    def keys_for(self, tag: str) -> list[str]: ...
    def tags_for(self, key: str) -> list[str]: ...
    def all_tags(self) -> list[str]: ...
```

- `add` and `remove` must be idempotent for repeated calls of the same pair.
- `keys_for` must return an empty list (never raise) when the tag is unknown.
- `all_tags` must return tags sorted ascending; duplicates allowed (do not deduplicate).
- Do not change any existing file, only add to `stats.py` and `tests/test_stats.py`.
- After your edits, run `python -m pytest tests/ -q` from the project root and
  paste only the PASSED/FAILED summary line.

Return a single code block containing both files in full (first `stats.py`,
then `tests/test_stats.py`), followed by the pytest summary line.
'''


def run_long_context(
    binary: str, artifact: str, base_artifact: str,
    ctx_target_chars: int, max_context: int, max_new: int, timeout: int,
) -> list[RunResult]:
    repo = gen_tiny_repo(ctx_target_chars)
    prompt = TASK_TEMPLATE.format(repo=repo)
    print(f"[long-context] repo chars={len(repo)} prompt chars={len(prompt)}",
          flush=True)

    configs: list[tuple[str, int | None, str]] = [
        ("base", None, base_artifact),
        ("mtp1", 1, artifact),
        ("mtp2", 2, artifact),
        ("mtp3", 3, artifact),
    ]
    results: list[RunResult] = []
    runs_dir = Path("results/mtp_coding/long_ctx/runs")
    runs_dir.mkdir(parents=True, exist_ok=True)
    for cfg_name, draft, art in configs:
        cmd = build_command(binary, art, prompt, max_new,
                            max_context=max_context, draft_tokens=draft,
                            extra=[])
        print(f"[run] {cfg_name} ctx={max_context} draft={draft}", flush=True)
        t0 = time.time()
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True,
                                  timeout=timeout)
            rc = proc.returncode
            out = proc.stdout + proc.stderr
            elapsed = time.time() - t0
            completion = "ok" if rc == 0 else f"fail_rc{rc}"
        except subprocess.TimeoutExpired:
            elapsed = time.time() - t0
            completion = "timeout"
            out = ""
        (runs_dir / f"{cfg_name}.txt").write_text(out)
        summary = parse_summary(out)
        raw = ""
        if "summary     tensors/resources" in out:
            try:
                raw, _ = (
                    out.split("summary     tensors/resources", 1)[1]
                       .split("prepare     render/preprocess", 1)
                )
            except Exception:
                pass
        results.append(RunResult(
            task_id=f"longctx_{ctx_target_chars}",
            category="longctx",
            config=cfg_name,
            draft_tokens=draft or 0,
            prompt_tokens=to_int(summary.get(PromptTokens, "0")),
            generated_tokens=to_int(summary.get(GeneratedTokens, "0")),
            prefill_seconds=0.0,
            prefill_tps=to_float(summary.get(PrefillSpeed, "0")),
            decode_seconds=0.0,
            decode_tps=to_float(summary.get(DecodeSpeed, "0")),
            total_seconds=elapsed,
            throughput_tps=0.0,
            acceptance_pct=to_float(summary.get(AcceptRate, "0")),
            accept_length=to_float(summary.get(AcceptLength, "0")),
            accepted_per_round=0.0,
            accept_by_pos=parse_acceptance_distribution(summary.get(AcceptByPos, ""), 0),
            drafted=to_int(summary.get(MtpDrafts, "0")),
            accepted=to_int(summary.get(MtpAccepts, "0")),
            rounds=to_int(summary.get(MtpRounds, "0")),
            fallback=to_int(summary.get(MtpFallback, "0")),
            runtime_mib=to_float(summary.get(RuntimeReservation, "0").replace("MiB", "")),
            free_after_startup_mib=to_float(summary.get(FreeAfterStartup, "0").replace("MiB", "")),
            kv_payload_mib=to_float(summary.get(KVPayload, "0").replace("MiB", "")),
            completion=completion,
            raw_response=raw,
        ))
        print(f"  -> gen={results[-1].generated_tokens} "
              f"decode={results[-1].decode_tps:.1f} tok/s "
              f"acc={results[-1].acceptance_pct:.1f}%",
              flush=True)
    return results


def main() -> int:
    binary = BINARY_DEFAULT
    artifact = ARTIFACT_DEFAULT
    base_artifact = BASE_ARTIFACT_DEFAULT

    out_dir = Path("results/mtp_coding/long_ctx")
    out_dir.mkdir(parents=True, exist_ok=True)

    runs = []
    # Use a repo that fits comfortably in the budget so all four configs
    # (including MTP3 which has the tightest budget) can produce a complete
    # answer.  The mtp1/mtp2/mtp3 budget measurements showed ctx=1792 fits
    # MTP3 at this prompt size with MTP1/2 succeeding at ctx=2048.
    for target_chars, max_ctx, max_new in [
        (1000, 1536, 400),     # small context, MTP3 fits comfortably
        (1500, 1792, 400),     # moderate, MTP3 at the edge
    ]:
        results = run_long_context(binary, artifact, base_artifact,
                                   ctx_target_chars=target_chars,
                                   max_context=max_ctx, max_new=max_new,
                                   timeout=300)
        runs.append({
            "ctx_target_chars": target_chars,
            "max_context": max_ctx,
            "results": [r.__dict__ for r in results],
        })

    (out_dir / "long_ctx.json").write_text(json.dumps(runs, indent=2))
    print(f"\nWrote {out_dir / 'long_ctx.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
