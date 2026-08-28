"""Q5-vs-Q4 quality benchmark for the RTX 5060 Ti Qwen3.8-27B groupwise profile.

Runs the 21-task coding set from mtp_coding_bench.py plus 14 extended tasks
(C, C++, instruction following, multi-step reasoning, code understanding) on
one artifact at mtp0 (no spec) and mtp3 (draft window 3), with identical
inference settings, and scores every response with a deterministic
per-task validator.

Usage:
    python3 tools/bench/q4q5_quality_bench.py run --artifact out/x.ninfer --tag q4 \
        [--tasks all|orig|new] [--configs mtp0,mtp3]
    python3 tools/bench/q4q5_quality_bench.py score --results a.jsonl b.jsonl
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent))
from mtp_coding_bench import PROMPTS as ORIG_PROMPTS, parse_summary, to_int  # noqa: E402

BINARY_DEFAULT = "build/apps/ninfer"
MAX_CONTEXT = 1024
KV_DTYPE = "int8"

# ---------------------------------------------------------------------------
# Extended task set (14). Every task has a mechanically checkable answer.
# ---------------------------------------------------------------------------

NEW_PROMPTS: list[dict[str, Any]] = [
    {
        "id": "c_gcd",
        "category": "c_codegen",
        "prompt": "Write a C function `int gcd(int a, int b)` that returns the greatest common "
                  "divisor of two non-negative integers (not both zero). Return only the function "
                  "in a ```c code block.",
        "max_new": 250,
    },
    {
        "id": "c_debug_max",
        "category": "c_debug",
        "prompt": "The following C function should return the maximum element of an array of n "
                  "integers, but it gives the wrong answer for some inputs. Find the bug and "
                  "return only the corrected function in a ```c code block:\n"
                  "```c\nint array_max(int *a, int n) {\n"
                  "    int m = 0;\n"
                  "    for (int i = 0; i < n; i++)\n"
                  "        if (a[i] > m) m = a[i];\n"
                  "    return m;\n}\n```",
        "max_new": 300,
    },
    {
        "id": "cpp_miniqueue",
        "category": "cpp_codegen",
        "prompt": "Write a C++ class `MiniQueue` with exactly these public methods:\n"
                  "    void push(int x);\n"
                  "    bool pop(int &x);   // returns false and changes nothing if empty\n"
                  "    int front() const;  // assumes non-empty\n"
                  "    int size() const;\n"
                  "Implement it using std::deque. Return only the class definition in a ```cpp "
                  "code block.",
        "max_new": 400,
    },
    {
        "id": "cpp_debug_sorted",
        "category": "cpp_debug",
        "prompt": "The following C++ function should return true if the vector is sorted in "
                  "non-increasing order, but it misbehaves for some inputs. Find the bug and "
                  "return only the corrected function in a ```cpp code block:\n"
                  "```cpp\nbool is_desc(const std::vector<int> &v) {\n"
                  "    for (size_t i = 0; i < v.size(); i++)\n"
                  "        if (v[i] < v[i + 1]) return false;\n"
                  "    return true;\n}\n```",
        "max_new": 300,
    },
    {
        "id": "cpp_debug_odds",
        "category": "cpp_debug",
        "prompt": "The following C++ function should return the sum of all odd elements (negative "
                  "odd numbers count as odd), but it gives the wrong answer for some inputs. Find "
                  "the bug and return only the corrected function in a ```cpp code block:\n"
                  "```cpp\nint sum_odds(const std::vector<int> &v) {\n"
                  "    int s = 0;\n"
                  "    for (size_t i = 0; i < v.size(); i++)\n"
                  "        if (v[i] % 2 == 1) s += v[i];\n"
                  "    return s;\n}\n```",
        "max_new": 300,
    },
    {
        "id": "modify_c_api",
        "category": "c_modify",
        "prompt": "Modify the following C function so that it copies at most 8 elements (saturate "
                  "n at 8) and RETURNS the number of elements actually copied (change the return "
                  "type to int). Do not use globals. Return only the corrected function in a ```c "
                  "code block:\n```c\nvoid copy_n(int *dst, const int *src, int n) {\n"
                  "    for (int i = 0; i < n; i++) dst[i] = src[i];\n}\n```",
        "max_new": 300,
    },
    {
        "id": "py_lru",
        "category": "py_multistep",
        "prompt": "Implement a Python class `LRUCache` with:\n"
                  "    def __init__(self, capacity)\n"
                  "    def get(self, key)    # returns the value, or -1 if absent\n"
                  "    def put(self, key, value)  # evicts the least-recently-used key when full\n"
                  "Both get and put must run in O(1). Standard library only. Return only the class "
                  "in a ```python code block.",
        "max_new": 700,
    },
    {
        "id": "if_json",
        "category": "instruction_following",
        "prompt": "Follow this instruction exactly. Your entire reply must be a single JSON object "
                  'with exactly one key, "answer", whose value is an integer. Do not output any '
                  "other text, code fences, or explanation. Compute 17 * 23 + 8.",
        "max_new": 60,
    },
    {
        "id": "if_result_line",
        "category": "instruction_following",
        "prompt": "Answer the question below. Your reply must end with exactly one line of the "
                  "form RESULT: <value> where <value> is an integer, and nothing may follow that "
                  "line. Question: how many positive integers divide 120 evenly?",
        "max_new": 300,
    },
    {
        "id": "understand_trace",
        "category": "code_understanding",
        "prompt": "Read this Python function carefully and determine the exact integer it returns "
                  "when called as f(10). Answer with just the number.\n"
                  "```python\ndef f(n):\n"
                  "    s, k = 0, 1\n"
                  "    while k <= n:\n"
                  "        s += k\n"
                  "        k *= 2\n"
                  "    return s\n```",
        "max_new": 120,
    },
    {
        "id": "understand_memo",
        "category": "code_understanding",
        "prompt": "Read this Python code carefully. After executing the two calls `fib(10); "
                  "fib(10)` in this order, what is `len(memo)`? Answer with just the number.\n"
                  "```python\ndef fib(n, memo={}):\n"
                  "    if n < 2: return n\n"
                  "    if n not in memo: memo[n] = fib(n-1, memo) + fib(n-2, memo)\n"
                  "    return memo[n]\n```",
        "max_new": 120,
    },
    {
        "id": "reason_words",
        "category": "multistep_reasoning",
        "prompt": "A train leaves station A at 9:00 travelling toward station B at 60 km/h. A car "
                  "leaves station B at 9:30 travelling toward station A at 90 km/h. The distance "
                  "between A and B is 270 km. At what time (24-hour format HH:MM) do they meet? "
                  "Answer with just the time.",
        "max_new": 150,
    },
    {
        "id": "reason_graph",
        "category": "multistep_reasoning",
        "prompt": "Consider this directed weighted graph (edge -> cost):\n"
                  "A -> B (4)\nA -> C (2)\nB -> D (3)\nC -> B (1)\nC -> D (5)\nD -> E (2)\n"
                  "What is the shortest-path distance from A to E? Answer with just the number.",
        "max_new": 150,
    },
    {
        "id": "py_behavior_refactor",
        "category": "py_refactor",
        "prompt": "Refactor the following code into two functions: a helper `positive_sum(items)` "
                  "that returns the sum of the positive elements, and a `process(items)` that "
                  "uses the helper and keeps exactly the same behaviour as before. Do not change "
                  "what `process` returns for any input. Return only the refactored code in a "
                  "```python code block:\n"
                  "```python\ndef process(items):\n"
                  "    total = 0\n"
                  "    for x in items:\n"
                  "        if x > 0:\n"
                  "            total += x\n"
                  "    return total * 2\n```",
        "max_new": 400,
    },
]

ALL_PROMPTS: list[dict[str, Any]] = ORIG_PROMPTS + NEW_PROMPTS

# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class QRun:
    task_id: str
    category: str
    config: str
    artifact: str
    prompt_tokens: int
    generated_tokens: int
    prefill_seconds: float
    prefill_tps: float
    decode_tps: float
    acceptance_pct: float
    accept_length: float
    fallback: int
    completion: str
    raw_response: str


def build_command(binary: str, artifact: str, prompt: str, max_new: int,
                  config: str) -> list[str]:
    cmd = [
        binary, artifact, "--prompt", prompt,
        "--max-context", str(MAX_CONTEXT),
        "--max-new", str(max_new),
        "--kv-dtype", KV_DTYPE,
        "--prefill-chunk", "64",
        "--no-prefix-reuse",
        "--no-thinking",
        "--greedy",
    ]
    if config == "mtp3":
        cmd += ["--spec", "mtp", "--draft-tokens", "3"]
    return cmd


def run_one(binary: str, artifact: str, task: dict[str, Any], config: str,
            timeout: int, runs_dir: Path, tag: str, idx: int) -> QRun:
    runs_dir.mkdir(parents=True, exist_ok=True)
    safe_id = re.sub(r"[^A-Za-z0-9_-]", "_", task["id"])
    out_file = runs_dir / f"{safe_id}__{config}__{tag}_{idx:02d}.txt"
    cmd = build_command(binary, artifact, task["prompt"], task["max_new"], config)
    print(f"[run] {task['id']} {config}", flush=True)
    t0 = time.time()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        out = proc.stdout + ("\n[stderr]\n" + proc.stderr if proc.stderr.strip() else "")
        raw_response = proc.stdout
        completion = "ok" if proc.returncode == 0 else f"fail_rc{proc.returncode}"
    except subprocess.TimeoutExpired:
        out, raw_response, completion = "", "", "timeout"
    out_file.write_text(out)
    summary = parse_summary(out)
    prefill_t = 0.0
    for line in out.splitlines():
        if line.startswith("generate    text prefill"):
            try:
                prefill_t = float(line.split()[2])
            except (IndexError, ValueError):
                pass
    return QRun(
        task_id=task["id"],
        category=task["category"],
        config=config,
        artifact=artifact,
        prompt_tokens=to_int(summary.get("prompt tokens", "0")),
        generated_tokens=to_int(summary.get("generated tokens", "0")),
        prefill_seconds=prefill_t,
        prefill_tps=to_int(summary.get("prefill speed", "0")),
        decode_tps=to_int(summary.get("decode speed", "0")),
        acceptance_pct=to_int(summary.get("mtp acceptance rate", "0")),
        accept_length=to_int(summary.get("mtp acceptance length", "0")),
        fallback=to_int(summary.get("mtp fallback steps", "0")),
        completion=completion,
        raw_response=raw_response,
    )


def cmd_run(args: argparse.Namespace) -> int:
    tasks = ALL_PROMPTS
    if args.tasks == "orig":
        tasks = ORIG_PROMPTS
    elif args.tasks == "new":
        tasks = NEW_PROMPTS
    configs = [c.strip() for c in args.configs.split(",") if c.strip()]
    out_path = Path(args.results_dir) / f"{args.tag}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    runs_dir = Path(args.runs_dir) / args.tag
    with out_path.open("w") as fh:
        for idx, task in enumerate(tasks):
            for config in configs:
                r = run_one(args.binary, args.artifact, task, config,
                            args.timeout, runs_dir, args.tag, idx)
                fh.write(json.dumps(dataclasses.asdict(r)) + "\n")
                fh.flush()
                print(f"  -> {r.task_id:22s} {config:4s} gen={r.generated_tokens:4d} "
                      f"decode={r.decode_tps:5.1f} acc={r.acceptance_pct:5.1f}% "
                      f"fb={r.fallback} {r.completion}", flush=True)
    print(f"Results: {out_path}")
    return 0


# ---------------------------------------------------------------------------
# Validators
# ---------------------------------------------------------------------------

PY_BLOCK_RE = re.compile(r"```(?:python|py)\s*\n(.*?)```", re.DOTALL)
C_BLOCK_RE = re.compile(r"```c\s*\n(.*?)```", re.DOTALL)
CPP_BLOCK_RE = re.compile(r"```cpp\s*\n(.*?)```", re.DOTALL)
BASH_BLOCK_RE = re.compile(r"```(?:bash|sh|shell)\s*\n(.*?)```", re.DOTALL)
# greedy: JSDoc/code may contain nested fences; the block's real close is the last fence
TS_BLOCK_RE = re.compile(r"```(?:typescript|ts)\s*\n(.*)```", re.DOTALL)

# Opening fences per block regex, used only for the truncation fallback in _first.
_FENCE_OPENS = {
    PY_BLOCK_RE: re.compile(r"```(?:python|py)[ \t]*\n"),
    C_BLOCK_RE: re.compile(r"```c[ \t]*\n"),
    CPP_BLOCK_RE: re.compile(r"```cpp[ \t]*\n"),
    BASH_BLOCK_RE: re.compile(r"```(?:bash|sh|shell)[ \t]*\n"),
    TS_BLOCK_RE: re.compile(r"```(?:typescript|ts)[ \t]*\n"),
}


def _first(rx: re.Pattern, text: str) -> str | None:
    t = text or ""
    m = rx.search(t)
    if m and m.group(1).strip():
        return m.group(1).rstrip()
    # Truncation fallback: when the response ends inside a code fence because the
    # output budget cut the closing fence, the content after the last opening
    # fence is the intended block. It must still pass the functional validator.
    open_rx = _FENCE_OPENS.get(rx)
    if open_rx is not None:
        opens = list(re.finditer(open_rx, t))
        if opens:
            tail = t[opens[-1].end():]
            if tail.strip():
                return tail.rstrip()
    return None


def _run_python(model_code: str, driver: str, timeout: int = 15) -> tuple[bool, str]:
    with tempfile.TemporaryDirectory() as d:
        Path(d, "model.py").write_text(model_code)
        Path(d, "driver.py").write_text(driver)
        try:
            p = subprocess.run([sys.executable, "driver.py"], cwd=d,
                               capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return False, "timeout"
        if p.returncode == 0:
            return True, "ok"
        err = (p.stderr.strip().splitlines() or ["?"])[-1]
        return False, err[:120]


def _run_cc(model_code: str, driver: str, lang: str, timeout: int = 15) -> tuple[bool, str]:
    cc = "g++" if lang == "cpp" else "gcc"
    std = "-std=c++17" if lang == "cpp" else "-std=c11"
    with tempfile.TemporaryDirectory() as d:
        ext = "cpp" if lang == "cpp" else "c"
        if lang == "c" and "#include" not in model_code:
            model_code = "#include <stdio.h>\n" + model_code
        if lang == "c" and "#include" not in driver:
            driver = "#include <stdio.h>\n" + driver
        if lang == "cpp":
            for inc in ("vector", "deque", "cstdlib", "iostream", "string"):
                if f"#include <{inc}>" not in model_code:
                    model_code = f"#include <{inc}>\n" + model_code
        Path(d, f"model.{ext}").write_text(model_code)
        exe = Path(d, "run")
        if lang == "cpp":
            # single TU so the driver sees class/function definitions
            Path(d, f"driver.{ext}").write_text("/* combined */\n")
            combine = Path(d, "combined.cpp")
            combine.write_text(model_code + "\n" + driver)
            compile_args = [cc, std, "-O1", str(combine), "-o", str(exe)]
        else:
            Path(d, f"driver.{ext}").write_text(driver)
            compile_args = [cc, std, "-O1", f"model.{ext}", f"driver.{ext}", "-o", str(exe)]
        try:
            c = subprocess.run(compile_args, cwd=d,
                               capture_output=True, text=True, timeout=60)
            if c.returncode != 0:
                lines = c.stderr.strip().splitlines()
                err = next((l for l in lines if "error" in l.lower()),
                           lines[-1] if lines else "?")
                return False, "compile: " + err[:140]
            p = subprocess.run([str(exe)], cwd=d, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return False, "timeout"
        if p.returncode == 0 and p.stdout.strip() == "OK":
            return True, "ok"
        return False, f"run rc={p.returncode} out={p.stdout.strip()[:60]!r} err={(p.stderr.strip().splitlines() or ['?'])[-1][:80]}"


def _callables(module_ns: dict) -> list:
    out = []
    for name, obj in module_ns.items():
        if not name.startswith("_") and callable(obj) and name not in ("int", "str", "dict"):
            out.append((name, obj))
    return out


def _try_callable_with(ns: dict, arg: Any) -> Any | None:
    for _, obj in _callables(ns):
        try:
            return obj(arg)
        except Exception:
            continue
    return None


def _py_ns(code: str, tmp: str | None = None) -> dict:
    ns: dict = {}
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as fh:
        fh.write(code)
        path = fh.name
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("m", path)
        mod = importlib.util.module_from_spec(spec)
        import sys as _s
        _s.modules["m"] = mod
        spec.loader.exec_module(mod)
        ns.update(vars(mod))
    finally:
        Path(path).unlink(missing_ok=True)
    return ns


# ---------------- per-task validators ----------------
# Each returns (passed: bool, detail: str, strong: bool).
# strong=False means the check is a weak textual/mention check.


def v_small_add(raw: str) -> tuple[bool, str, bool]:
    code = _first(PY_BLOCK_RE, raw)
    if code is None:
        return False, "no python block", True
    ns = _py_ns(code)
    if "add" not in ns or not callable(ns["add"]):
        return False, "no add()", True
    try:
        ok = ns["add"](2, 3) == 5
    except Exception as e:
        return False, f"add(2,3) raised {e}", True
    return ok, "ok" if ok else "add(2,3) != 5", True


def v_small_factorial(raw: str) -> tuple[bool, str, bool]:
    code = _first(PY_BLOCK_RE, raw)
    if code is None:
        return False, "no python block", True
    ns = _py_ns(code)
    f = ns.get("factorial")
    if not callable(f):
        return False, "no factorial()", True
    try:
        ok1 = f(5) == 120 and f(0) == 1
        try:
            f(-1)
            raised = False
        except ValueError:
            raised = True
        except Exception:
            raised = False
        ok = ok1 and raised
        return ok, ("ok" if ok else
                    f"factorial(5)={f(5)}, factorial(0)={f(0)}, ValueError for -1: {raised}"), True
    except Exception as e:
        return False, f"raised {e}", True


def v_small_greet(raw: str) -> tuple[bool, str, bool]:
    code = _first(TS_BLOCK_RE, raw)
    if code is None:
        return False, "no typescript block", True
    with tempfile.TemporaryDirectory() as d:
        src = code + "\nconsole.log(greet('World'));\n"
        Path(d, "greet.ts").write_text(src)
        try:
            p = subprocess.run(["node", "greet.ts"], cwd=d,
                               capture_output=True, text=True, timeout=20)
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            return False, f"node unavailable: {e}", True
        out = p.stdout.strip()
        ok = p.returncode == 0 and "Hello, World!" in out
        if ok:
            return True, "ok", True
        # fallback: textual check for the template literal and function
        ok = ("greet" in code and re.search(r"Hello,\s*\$\{name\}!", code) is not None)
        return ok, f"node rc={p.returncode} out={out[:60]!r} err={p.stderr.strip()[:60]!r}", False


def v_medium_csv(raw: str) -> tuple[bool, str, bool]:
    code = _first(PY_BLOCK_RE, raw)
    if code is None:
        return False, "no python block", True
    with tempfile.TemporaryDirectory() as d:
        Path(d, "data.csv").write_text(
            "id,name,score\n1,alice,72\n2,bruce,48\n3,carol,90\n4,dora,41\n")
        Path(d, "script.py").write_text(code)
        try:
            p = subprocess.run([sys.executable, "script.py"], cwd=d,
                               capture_output=True, text=True, timeout=20)
        except subprocess.TimeoutExpired:
            return False, "timeout", True
        if p.returncode != 0:
            return False, (p.stderr.strip().splitlines() or ["?"])[-1][:100], True
        out = p.stdout
        # above-average scores: 72 and 90 -> alice and carol must appear; bruce/dora must not
        ok = ("alice" in out and "carol" in out
              and "bruce" not in out and "dora" not in out)
        return ok, f"stdout={out.strip()[:80]!r}", True


def v_medium_cli(raw: str) -> tuple[bool, str, bool]:
    code = _first(BASH_BLOCK_RE, raw)
    if code is None:
        return False, "no bash block", True
    with tempfile.TemporaryDirectory() as d:
        (Path(d, "a.log")).write_text("\n" * 1200)
        (Path(d, "b.log")).write_text("\n" * 5)
        (Path(d, "sub", )).mkdir()
        Path(d, "sub", "c.log").write_text("\n" * 1001)
        Path(d, "script.sh").write_text(code)
        try:
            p = subprocess.run(["bash", "script.sh"], cwd=d,
                               capture_output=True, text=True, timeout=30)
        except subprocess.TimeoutExpired:
            return False, "timeout", True
        # expected: a.log (1200) and sub/c.log (1001) flagged by non-zero exit;
        # all three paths reported with line counts
        out = p.stdout
        ok = (p.returncode != 0 and "a.log" in out and "c.log" in out
              and "1200" in out and "1001" in out)
        return ok, f"rc={p.returncode} out={out.strip()[:80]!r}", True


def v_medium_ratelimit(raw: str) -> tuple[bool, str, bool]:
    code = _first(PY_BLOCK_RE, raw)
    if code is None:
        return False, "no python block", True
    ns = _py_ns(code)
    rl = ns.get("RateLimiter")
    if not callable(rl):
        return False, "no RateLimiter", True
    try:
        r = rl(2, 1)
        first = r.acquire()
        second = r.acquire()
        ok = first is True and second is False
        return ok, f"first={first}, immediate second={second}", True
    except Exception as e:
        return False, f"raised {e}", True


def v_algo_twosum(raw: str) -> tuple[bool, str, bool]:
    code = _first(PY_BLOCK_RE, raw)
    if code is None:
        return False, "no python block", True
    ns = _py_ns(code)
    f = ns.get("two_sum")
    if not callable(f):
        return False, "no two_sum()", True
    try:
        r = f([2, 7, 11, 15], 9)
        ok = sorted(r) == [0, 1]
        return ok, f"two_sum -> {r}", True
    except Exception as e:
        return False, f"raised {e}", True


def v_algo_lis(raw: str) -> tuple[bool, str, bool]:
    code = _first(PY_BLOCK_RE, raw)
    if code is None:
        return False, "no python block", True
    ns = _py_ns(code)
    r = _try_callable_with(ns, [10, 9, 2, 5, 3, 7, 101, 18])
    ok = r == 4
    return ok, f"candidate call -> {r!r}", True


def v_algo_paren(raw: str) -> tuple[bool, str, bool]:
    code = _first(PY_BLOCK_RE, raw)
    if code is None:
        return False, "no python block", True
    ns = _py_ns(code)
    r1 = _try_callable_with(ns, "{[()]}")
    r2 = _try_callable_with(ns, "([)]")
    ok = r1 is True and r2 is False
    return ok, f"balanced ok={r1!r}, bad={r2!r}", True


def v_algo_dijkstra(raw: str) -> tuple[bool, str, bool]:
    code = _first(PY_BLOCK_RE, raw)
    if code is None:
        return False, "no python block", True
    ns = _py_ns(code)
    g = {1: [(2, 4), (3, 9)], 2: [(3, 1)], 3: []}
    for _, obj in _callables(ns):
        try:
            r = obj(g, 1)
        except Exception:
            try:
                r = obj(1, g)
            except Exception:
                continue
        if isinstance(r, dict):
            try:
                ok = r[3] == 5
                return ok, f"{obj.__name__}(graph,1)[3] = {r[3]!r} (expect 5)", True
            except Exception:
                continue
    return False, "no callable returning a distance dict", True


def v_debug_offbyone(raw: str) -> tuple[bool, str, bool]:
    code = _first(PY_BLOCK_RE, raw)
    if code is None:
        return False, "no python block", True
    ns = _py_ns(code)
    f = ns.get("fib")
    if not callable(f):
        return False, "no fib()", True
    try:
        vals = [f(i) for i in range(11)]
        ok = vals == [0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55]
        return ok, f"fib(0..10) = {vals}", True
    except Exception as e:
        return False, f"raised {e}", True


def v_debug_index(raw: str) -> tuple[bool, str, bool]:
    code = _first(PY_BLOCK_RE, raw)
    if code is None:
        return False, "no python block", True
    ns = _py_ns(code)
    f = ns.get("average")
    if not callable(f):
        return False, "no average()", True
    try:
        r_empty = f([])
        r = f([1, 2, 3])
        ok = r == 2.0
        return ok, f"empty -> {r_empty!r} (no crash), [1,2,3] -> {r!r}", True
    except Exception as e:
        return False, f"raised {e}", True


def v_debug_offbyone_list(raw: str) -> tuple[bool, str, bool]:
    code = _first(PY_BLOCK_RE, raw)
    if code is None:
        return False, "no python block", True
    ns = _py_ns(code)
    f = ns.get("rotate")
    if not callable(f):
        return False, "no rotate()", True
    try:
        r0 = f([1, 2, 3], 0)
        r3 = f([1, 2, 3], 3)
        r1 = f([1, 2, 3], 1)
        ok = r0 == [1, 2, 3] and r3 == [1, 2, 3] and r1 == [3, 1, 2]
        return ok, f"k=0: {r0!r}, k=3: {r3!r}, k=1: {r1!r}", True
    except Exception as e:
        return False, f"raised {e}", True


def v_modify_add_typehints(raw: str) -> tuple[bool, str, bool]:
    code = _first(PY_BLOCK_RE, raw)
    if code is None:
        return False, "no python block", True
    try:
        import ast
        tree = ast.parse(code)
        fns = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "add_item"]
        if not fns:
            return False, "add_item not defined", True
        hints = fns[0].args
        annotated = sum(1 for a in hints.args if a.annotation is not None)
        has_hints = annotated >= 2
    except SyntaxError as e:
        return False, f"syntax error {e}", True
    ns = _py_ns(code)
    f = ns.get("add_item")
    try:
        cart = []
        f(cart, "x")
        r1 = cart == ["x"]
        c2 = ["a"]
        f(c2, "a")
        r3 = c2 == ["a"]
        c3 = ["a"]
        f(c3, "b")
        r4 = c3 == ["a", "b"]
        ok = r1 and r3 and r4
        return ok and has_hints, f"behaviour={r1 and r3 and r4}, params_annotated={annotated}", True
    except Exception as e:
        return False, f"raised {e}", True


def v_modify_add_docstring(raw: str) -> tuple[bool, str, bool]:
    code = _first(PY_BLOCK_RE, raw)
    if code is None:
        return False, "no python block", True
    import ast
    try:
        tree = ast.parse(code)
        fns = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "clamp"]
        if not fns or ast.get_docstring(fns[0]) is None:
            return False, "clamp missing or has no docstring", True
    except SyntaxError as e:
        return False, f"syntax error {e}", True
    ns = _py_ns(code)
    f = ns.get("clamp")
    try:
        ok = f(5, 0, 10) == 5 and f(-1, 0, 10) == 0 and f(42, 0, 10) == 10
        return ok, "ok" if ok else "behaviour changed", True
    except Exception as e:
        return False, f"raised {e}", True


def v_refactor_split(raw: str) -> tuple[bool, str, bool]:
    code = _first(PY_BLOCK_RE, raw)
    if code is None:
        return False, "no python block", True
    import ast
    try:
        tree = ast.parse(code)
        names = [n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
        has_set_age = "set_age" in names
        has_helper = len([n for n in names if n != "set_age"]) >= 1
    except SyntaxError as e:
        return False, f"syntax error {e}", True
    ns = _py_ns(code)
    f = ns.get("set_age")
    try:
        class U:
            pass
        u = U()
        f(u, 100)
        r1 = u.age == 100
        try:
            f(u, -1)
            raised = False
        except ValueError:
            raised = True
        except Exception:
            raised = False
        ok = has_set_age and has_helper and r1 and raised
        return ok, f"set_age kept={has_set_age}, helper={has_helper}, " \
                   f"behaviour={r1 and raised}", True
    except Exception as e:
        return False, f"raised {e}", True


def v_explain_regex(raw: str) -> tuple[bool, str, bool]:
    import re as _re
    pat = _re.compile(r"^(?:https?://)?(?:www\.)?([a-z0-9-]+)\.(com|org|io)/?(?:\?([^#]*))?(?:#(.*))?$")
    # Candidate examples: inline quoted spans (backtick/quote delimited) and the
    # lines of fenced code blocks, so the check is independent of how the model
    # presents its examples. Every candidate is still evaluated strictly against
    # the regex: a "matching" example must actually match and a "non-matching"
    # one must actually fail.
    raw = raw or ""
    candidates = _re.findall(r"[`\"'“”]([^`\"'“”\n]{3,80})[`\"'“”]", raw)
    for block in _re.findall(r"```[^\n]*\n(.*?)```", raw, _re.DOTALL):
        candidates.extend(line.strip() for line in block.splitlines() if line.strip())
    matches = [s for s in candidates if pat.match(s.strip())]
    nonmatches = [s for s in candidates if s.strip() and not pat.match(s.strip())]
    ok = len(matches) >= 1 and len(nonmatches) >= 1
    return ok, f"examples matching: {len(matches)}, non-matching: {len(nonmatches)}", True


def v_reason_source_explain(raw: str) -> tuple[bool, str, bool]:
    t = (raw or "").lower()
    ok = ("merge" in t and "sorted_keys" in t) and ("sort" in t)
    return ok, "mentions merge/sorted_keys/sort", False


def v_multifile_mini_repo(raw: str) -> tuple[bool, str, bool]:
    blocks = PY_BLOCK_RE.findall(raw or "")
    if not blocks:
        return False, "no python block", True
    ns: dict = {}
    for b in blocks:
        try:
            ns.update(_py_ns(b, None))
        except Exception:
            continue
    counter = ns.get("Counter")
    if not callable(counter):
        return False, "Counter class not found in blocks", True
    try:
        c = counter()
        c.incr("a")
        c.incr("b", 2)
        c.incr("a")
        r1 = c.get("a") == 2
        s = str(c)
        ok = r1 and ("a" in s and "b" in s and s.index("a") < s.index("b"))
        return ok, f"get(a)={c.get('a')}, str={s!r}", True
    except Exception as e:
        return False, f"raised {e}", True


def v_shell_find_dup(raw: str) -> tuple[bool, str, bool]:
    t = raw or ""
    has_hash = ("sha256sum" in t or "md5sum" in t or "sha1sum" in t)
    has_sort = "sort" in t
    ok = has_hash and has_sort
    return ok, f"hash={has_hash}, sort={has_sort}", False


def v_shell_service(raw: str) -> tuple[bool, str, bool]:
    t = raw or ""
    checks = {
        "[Unit]": "[Unit]" in t,
        "network-online": "network-online.target" in t,
        "restart": "Restart=on-failure" in t,
        "delay5": "RestartSec=5" in t,
        "user": "User=myapp" in t,
        "execstart": "/usr/local/bin/myapp --port 8080" in t,
    }
    ok = all(checks.values())
    return ok, f"{checks}", True


def v_c_gcd(raw: str) -> tuple[bool, str, bool]:
    code = _first(C_BLOCK_RE, raw)
    if code is None:
        return False, "no c block", True
    driver = (
        "int gcd(int a, int b);\n"
        "int main(void){\n"
        "  if (gcd(48,18)!=6) return 1;\n"
        "  if (gcd(17,5)!=1) return 2;\n"
        "  if (gcd(0,5)!=5) return 3;\n"
        "  if (gcd(100,75)!=25) return 4;\n"
        "  if (gcd(1,1)!=1) return 5;\n"
        "  printf(\"OK\\n\"); return 0;\n"
        "}\n")
    ok, msg = _run_cc(code, driver, "c")
    return ok, msg, True


def v_c_debug_max(raw: str) -> tuple[bool, str, bool]:
    code = _first(C_BLOCK_RE, raw)
    if code is None:
        return False, "no c block", True
    driver = (
        "int array_max(int *a, int n);\n"
        "int main(void){\n"
        "  int v1[] = {-5,-2,-9};\n"
        "  int v2[] = {3,-1,7};\n"
        "  int v3[] = {-7};\n"
        "  if (array_max(v1,3)!=-2) return 1;\n"
        "  if (array_max(v2,3)!=7) return 2;\n"
        "  if (array_max(v3,1)!=-7) return 3;\n"
        "  printf(\"OK\\n\"); return 0;\n"
        "}\n")
    ok, msg = _run_cc(code, driver, "c")
    return ok, msg, True


def v_modify_c_api(raw: str) -> tuple[bool, str, bool]:
    code = _first(C_BLOCK_RE, raw)
    if code is None:
        return False, "no c block", True
    driver = (
        "int copy_n(int *dst, const int *src, int n);\n"
        "int main(void){\n"
        "  int dst[16], src[16];\n"
        "  for (int i=0;i<16;i++) src[i]=i+1;\n"
        "  int r1 = copy_n(dst, src, 10);\n"
        "  if (r1!=8) return 1;\n"
        "  if (dst[7]!=8) return 2;\n"
        "  int r2 = copy_n(dst, src, 3);\n"
        "  if (r2!=3 || dst[0]!=1 || dst[2]!=3) return 3;\n"
        "  if (copy_n(dst, src, 0)!=0) return 4;\n"
        "  if (copy_n(dst, src, -1)!=0) return 5;\n"
        "  printf(\"OK\\n\"); return 0;\n"
        "}\n")
    ok, msg = _run_cc(code, driver, "c")
    return ok, msg, True


def v_cpp_miniqueue(raw: str) -> tuple[bool, str, bool]:
    code = _first(CPP_BLOCK_RE, raw)
    if code is None:
        return False, "no cpp block", True
    driver = (
        "#include <iostream>\n"
        "int main(){\n"
        "  MiniQueue q;\n"
        "  if (q.size()!=0) return 1;\n"
        "  int x;\n"
        "  if (q.pop(x)) return 2;\n"
        "  q.push(1); q.push(2);\n"
        "  if (q.front()!=1) return 3;\n"
        "  if (!q.pop(x) || x!=1) return 4;\n"
        "  if (!q.pop(x) || x!=2) return 5;\n"
        "  if (q.pop(x)) return 6;\n"
        "  if (q.size()!=0) return 7;\n"
        "  std::cout << \"OK\" << std::endl;\n"
        "  return 0;\n"
        "}\n")
    ok, msg = _run_cc(code, driver, "cpp")
    return ok, msg, True


def v_cpp_debug_sorted(raw: str) -> tuple[bool, str, bool]:
    code = _first(CPP_BLOCK_RE, raw)
    if code is None:
        return False, "no cpp block", True
    driver = (
        "#include <vector>\n"
        "#include <iostream>\n"
        "int main(){\n"
        "  std::vector<int> a = {5,4,4,1};\n"
        "  std::vector<int> b = {1,2};\n"
        "  std::vector<int> c = {};\n"
        "  std::vector<int> d = {7};\n"
        "  std::vector<int> e = {3,3,2};\n"
        "  if (!is_desc(a)) return 1;\n"
        "  if (is_desc(b)) return 2;\n"
        "  if (!is_desc(c)) return 3;\n"
        "  if (!is_desc(d)) return 4;\n"
        "  if (!is_desc(e)) return 5;\n"
        "  std::cout << \"OK\" << std::endl;\n"
        "  return 0;\n"
        "}\n")
    ok, msg = _run_cc(code, driver, "cpp")
    return ok, msg, True


def v_cpp_debug_odds(raw: str) -> tuple[bool, str, bool]:
    code = _first(CPP_BLOCK_RE, raw)
    if code is None:
        return False, "no cpp block", True
    driver = (
        "#include <vector>\n"
        "#include <iostream>\n"
        "int main(){\n"
        "  std::vector<int> a = {1,-3,4,-5,6};\n"
        "  std::vector<int> b = {};\n"
        "  std::vector<int> c = {2,4,6};\n"
        "  if (sum_odds(a)!=-7) return 1;\n"
        "  if (sum_odds(b)!=0) return 2;\n"
        "  if (sum_odds(c)!=0) return 3;\n"
        "  std::cout << \"OK\" << std::endl;\n"
        "  return 0;\n"
        "}\n")
    ok, msg = _run_cc(code, driver, "cpp")
    return ok, msg, True


def v_py_lru(raw: str) -> tuple[bool, str, bool]:
    code = _first(PY_BLOCK_RE, raw)
    if code is None:
        return False, "no python block", True
    driver = (
        "from model import LRUCache\n"
        "c = LRUCache(2)\n"
        "c.put(1,'a'); c.put(2,'b')\n"
        "assert c.get(1) == 'a'\n"
        "c.put(3,'c')          # evicts 2\n"
        "assert c.get(2) == -1\n"
        "assert c.get(3) == 'c'\n"
        "c.put(4,'d')          # evicts 1 (3 is now most recent)\n"
        "assert c.get(1) == -1\n"
        "assert c.get(4) == 'd'\n"
        "c2 = LRUCache(1)\n"
        "c2.put('k','v')\n"
        "assert c2.get('k') == 'v'\n"
        "print('PYOK')\n")
    with tempfile.TemporaryDirectory() as d:
        Path(d, "model.py").write_text(code)
        Path(d, "driver.py").write_text(driver)
        try:
            p = subprocess.run([sys.executable, "driver.py"], cwd=d,
                               capture_output=True, text=True, timeout=15)
        except subprocess.TimeoutExpired:
            return False, "timeout", True
        if p.returncode == 0:
            return True, "ok", True
        return False, (p.stderr.strip().splitlines() or ["?"])[-1][:120], True


def v_if_json(raw: str) -> tuple[bool, str, bool]:
    t = (raw or "").strip()
    # The entire reply must be a single JSON object with one int key "answer"
    if not (t.startswith("{") and t.endswith("}")):
        return False, f"reply is not a bare JSON object: {t[:60]!r}", True
    try:
        obj = json.loads(t)
    except Exception as e:
        return False, f"invalid JSON: {e}", True
    if not isinstance(obj, dict) or set(obj.keys()) != {"answer"}:
        return False, f"keys = {sorted(obj.keys()) if isinstance(obj, dict) else type(obj)}", True
    v = obj["answer"]
    ok = isinstance(v, int) and not isinstance(v, bool) and v == 399
    return ok, f"answer={v!r} (expect 399)", True


def v_if_result_line(raw: str) -> tuple[bool, str, bool]:
    lines = [l for l in (raw or "").strip().splitlines() if l.strip()]
    if not lines:
        return False, "empty reply", True
    last = lines[-1].strip()
    m = re.fullmatch(r"RESULT:\s*(-?\d+)", last)
    if not m:
        return False, f"last line {last!r} is not RESULT: <int>", True
    ok = int(m.group(1)) == 8
    return ok, f"RESULT value {m.group(1)} (expect 8)", True


def v_understand_trace(raw: str) -> tuple[bool, str, bool]:
    t = raw or ""
    nums = re.findall(r"\b(\d+)\b", t)
    if not nums:
        return False, "no number in reply", True
    ok = int(nums[-1]) == 15
    return ok, f"last number {nums[-1]} (expect 15)", True


def v_understand_memo(raw: str) -> tuple[bool, str, bool]:
    t = raw or ""
    nums = re.findall(r"\b(\d+)\b", t)
    if not nums:
        return False, "no number in reply", True
    ok = int(nums[-1]) == 9
    return ok, f"last number {nums[-1]} (expect 9)", True


def v_reason_words(raw: str) -> tuple[bool, str, bool]:
    t = (raw or "").strip()
    ok = "11:06" in t
    return ok, f"reply {t[:80]!r} (expect 11:06)", True


def v_reason_graph(raw: str) -> tuple[bool, str, bool]:
    t = raw or ""
    nums = re.findall(r"\b(\d+)\b", t)
    if not nums:
        return False, "no number in reply", True
    ok = int(nums[-1]) == 8
    return ok, f"last number {nums[-1]} (expect 8)", True


def v_py_behavior_refactor(raw: str) -> tuple[bool, str, bool]:
    code = _first(PY_BLOCK_RE, raw)
    if code is None:
        return False, "no python block", True
    ns = _py_ns(code)
    ps = ns.get("positive_sum")
    proc = ns.get("process")
    if not callable(ps) or not callable(proc):
        return False, f"positive_sum={callable(ps)}, process={callable(proc)}", True
    try:
        ok = (proc([1, -2, 3, 0]) == 8 and ps([1, -2, 3]) == 4
              and proc([]) == 0 and proc([-1, -2]) == 0)
        return ok, "ok" if ok else "behaviour mismatch", True
    except Exception as e:
        return False, f"raised {e}", True


VALIDATORS: dict[str, tuple] = {
    "small_add": v_small_add,
    "small_factorial": v_small_factorial,
    "small_greet": v_small_greet,
    "medium_csv": v_medium_csv,
    "medium_cli": v_medium_cli,
    "medium_ratelimit": v_medium_ratelimit,
    "algo_twosum": v_algo_twosum,
    "algo_lis": v_algo_lis,
    "algo_paren": v_algo_paren,
    "algo_dijkstra": v_algo_dijkstra,
    "debug_offbyone": v_debug_offbyone,
    "debug_index": v_debug_index,
    "debug_offbyone_list": v_debug_offbyone_list,
    "modify_add_typehints": v_modify_add_typehints,
    "modify_add_docstring": v_modify_add_docstring,
    "refactor_split": v_refactor_split,
    "explain_regex": v_explain_regex,
    "reason_source_explain": v_reason_source_explain,
    "multifile_mini_repo": v_multifile_mini_repo,
    "shell_find_dup": v_shell_find_dup,
    "shell_service": v_shell_service,
    "c_gcd": v_c_gcd,
    "c_debug_max": v_c_debug_max,
    "cpp_miniqueue": v_cpp_miniqueue,
    "cpp_debug_sorted": v_cpp_debug_sorted,
    "cpp_debug_odds": v_cpp_debug_odds,
    "modify_c_api": v_modify_c_api,
    "py_lru": v_py_lru,
    "if_json": v_if_json,
    "if_result_line": v_if_result_line,
    "understand_trace": v_understand_trace,
    "understand_memo": v_understand_memo,
    "reason_words": v_reason_words,
    "reason_graph": v_reason_graph,
    "py_behavior_refactor": v_py_behavior_refactor,
}


def score_rows(rows: list[dict]) -> dict[str, Any]:
    per_task: dict[str, dict] = {}
    per_cfg: dict[str, dict] = {}
    detail = []
    for r in rows:
        tid = r["task_id"]
        cfg = r["config"]
        per_cfg.setdefault(cfg, {"n": 0, "passed": 0, "strong_n": 0, "strong_passed": 0,
                                 "runs_ok": 0})
        per_cfg[cfg]["n"] += 1
        if r.get("completion") == "ok":
            per_cfg[cfg]["runs_ok"] += 1
        v = VALIDATORS.get(tid)
        if v is None:
            detail.append({**r, "verdict": None, "detail": "no validator", "strong": None})
            continue
        passed, msg, strong = v(r.get("raw_response", ""))
        per_cfg[cfg]["passed"] += 1 if passed else 0
        if strong:
            per_cfg[cfg]["strong_n"] += 1
            per_cfg[cfg]["strong_passed"] += 1 if passed else 0
        per_task.setdefault(tid, {})[cfg] = {"passed": passed, "detail": msg, "strong": strong}
        detail.append({k: r.get(k) for k in
                       ("task_id", "category", "config", "artifact", "prompt_tokens",
                        "generated_tokens", "decode_tps", "acceptance_pct", "fallback",
                        "completion")} | {"verdict": passed, "detail": msg, "strong": strong})
    summary = {}
    for cfg, s in per_cfg.items():
        summary[cfg] = {
            "n": s["n"],
            "runs_ok": s["runs_ok"],
            "passed": s["passed"],
            "pass_pct": round(100.0 * s["passed"] / s["n"], 1) if s["n"] else None,
            "strong_n": s["strong_n"],
            "strong_passed": s["strong_passed"],
            "strong_pass_pct": round(100.0 * s["strong_passed"] / s["strong_n"], 1)
            if s["strong_n"] else None,
        }
    return {"summary": summary, "per_task": per_task, "detail": detail}


def cmd_score(args: argparse.Namespace) -> int:
    rows: list[dict] = []
    for path in args.results:
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    report = score_rows(rows)
    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        with open(args.report, "w") as fh:
            json.dump(report, fh, indent=2)
        print(f"Report: {args.report}")
    print(f"{'config':8s} {'n':>3s} {'ok':>3s} {'pass':>5s} {'pct':>6s} "
          f"{'strong_n':>8s} {'strong_pass':>11s} {'strong_pct':>10s}")
    for cfg, s in sorted(report["summary"].items()):
        print(f"{cfg:8s} {s['n']:>3d} {s['runs_ok']:>3d} {s['passed']:>5d} "
              f"{s['pass_pct'] or 0:>5.1f}% {s['strong_n']:>8d} "
              f"{s['strong_passed']:>11d} {s['strong_pass_pct'] or 0:>9.1f}%")
    print("\nPer-task verdicts (config -> pass/weak-detail):")
    for tid, bycfg in sorted(report["per_task"].items()):
        cells = []
        for cfg in sorted(bycfg):
            e = bycfg[cfg]
            mark = "PASS" if e["passed"] else "FAIL"
            w = "" if e["strong"] else "(weak)"
            cells.append(f"{cfg}={mark}{w}")
        print(f"  {tid:24s} " + "  ".join(cells))
    fails = [d for d in report["detail"] if d.get("verdict") is False]
    if fails:
        print(f"\n{len(fails)} failing entries; details in the report.")
    return 0


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    pr = sub.add_parser("run")
    pr.add_argument("--binary", default=BINARY_DEFAULT)
    pr.add_argument("--artifact", required=True)
    pr.add_argument("--tag", required=True)
    pr.add_argument("--tasks", choices=["all", "orig", "new"], default="all")
    pr.add_argument("--configs", default="mtp0,mtp3")
    pr.add_argument("--max-context", type=int, default=MAX_CONTEXT)
    pr.add_argument("--timeout", type=int, default=300)
    pr.add_argument("--results-dir", default="results/q4q5_quality")
    pr.add_argument("--runs-dir", default="results/q4q5_quality/runs")
    pr.set_defaults(func=cmd_run)
    ps = sub.add_parser("score")
    ps.add_argument("--results", nargs="+", required=True)
    ps.add_argument("--report")
    ps.set_defaults(func=cmd_score)
    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))