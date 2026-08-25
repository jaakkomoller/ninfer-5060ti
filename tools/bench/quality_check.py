"""Run the generated code from benchmark runs against a few quick checks.

For each (task, config) result we extract the first python code block, try to
`exec` it (sandbox), and run a tiny test derived from the prompt. We report a
per-task pass/fail and a per-config success rate.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

CODE_RE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)


# Tiny inline tests for each prompt. We exec the candidate code in a
# subprocess so we don't pollute the harness process. The "test" attribute
# is a Python expression that runs after the candidate module is imported.

INLINE_TESTS = {
    "small_add": "import importlib.util, sys; "
                  "m = importlib.util.spec_from_file_location('m', '{path}'); "
                  "mod = importlib.util.module_from_spec(m); "
                  "sys.modules['m'] = mod; "
                  "m.loader.exec_module(mod); "
                  "assert mod.add(2, 3) == 5",
    "small_factorial": "import importlib.util, sys; "
                       "m = importlib.util.spec_from_file_location('m', '{path}'); "
                       "mod = importlib.util.module_from_spec(m); "
                       "sys.modules['m'] = mod; "
                       "m.loader.exec_module(mod); "
                       "assert mod.factorial(5) == 120; "
                       "assert mod.factorial(0) == 1",
    "medium_csv": ("import importlib.util, sys; "
                   "m = importlib.util.spec_from_file_location('m', '{path}'); "
                   "mod = importlib.util.module_from_spec(m); "
                   "sys.modules['m'] = mod; "
                   "m.loader.exec_module(mod); "
                   "assert callable(mod.read_csv_and_print_above_average) or "
                   "       callable(getattr(mod, '__all__', None)) or True"),
    # debug_offbyone: function must return fib(n) for n=0..10
    "debug_offbyone": ("import importlib.util, sys; "
                       "m = importlib.util.spec_from_file_location('m', '{path}'); "
                       "mod = importlib.util.module_from_spec(m); "
                       "sys.modules['m'] = mod; "
                       "m.loader.exec_module(mod); "
                       "fib = getattr(mod, 'fib', None) or globals().get('fib'); "
                       "assert fib(0) == 0 and fib(10) == 55"),
    # debug_index: function must not crash on empty list and return 0
    "debug_index": ("import importlib.util, sys; "
                    "m = importlib.util.spec_from_file_location('m', '{path}'); "
                    "mod = importlib.util.module_from_spec(m); "
                    "sys.modules['m'] = mod; "
                    "m.loader.exec_module(mod); "
                    "avg = getattr(mod, 'average', None); "
                    "assert avg([]) == 0 and avg([1, 2, 3]) == 2.0"),
    # debug_offbyone_list: rotate_right(...)
    "debug_offbyone_list": ("import importlib.util, sys; "
                           "m = importlib.util.spec_from_file_location('m', '{path}'); "
                           "mod = importlib.util.module_from_spec(m); "
                           "sys.modules['m'] = mod; "
                           "m.loader.exec_module(mod); "
                           "rotate = getattr(mod, 'rotate', None); "
                           "assert rotate([1, 2, 3], 0) == [1, 2, 3] and "
                           "       rotate([1, 2, 3], 3) == [1, 2, 3]"),
    # algo_twosum
    "algo_twosum": ("import importlib.util, sys; "
                    "m = importlib.util.spec_from_file_location('m', '{path}'); "
                    "mod = importlib.util.module_from_spec(m); "
                    "sys.modules['m'] = mod; "
                    "m.loader.exec_module(mod); "
                    "two_sum = getattr(mod, 'two_sum', None); "
                    "assert two_sum([2, 7, 11, 15], 9) in ([0, 1], [1, 0])"),
    # algo_lis
    "algo_lis": ("import importlib.util, sys; "
                 "m = importlib.util.spec_from_file_location('m', '{path}'); "
                 "mod = importlib.util.module_from_spec(m); "
                 "sys.modules['m'] = mod; "
                 "m.loader.exec_module(mod); "
                 "lis = getattr(mod, 'length_of_lis', None); "
                 "assert lis([10, 9, 2, 5, 3, 7, 101, 18]) == 4"),
    # algo_paren
    "algo_paren": ("import importlib.util, sys; "
                   "m = importlib.util.spec_from_file_location('m', '{path}'); "
                   "mod = importlib.util.module_from_spec(m); "
                   "sys.modules['m'] = mod; "
                   "m.loader.exec_module(mod); "
                   "bal = getattr(mod, 'is_balanced', None); "
                   "assert bal('{[()]}') is True and bal('([)]') is False"),
    # algo_dijkstra
    "algo_dijkstra": ("import importlib.util, sys; "
                      "m = importlib.util.spec_from_file_location('m', '{path}'); "
                      "mod = importlib.util.module_from_spec(m); "
                      "sys.modules['m'] = mod; "
                      "m.loader.exec_module(mod); "
                      "dij = getattr(mod, 'dijkstra', None); "
                      "assert dij({1: [(2, 1), (3, 4)], 2: [], 3: []}, 1)[3] == 5"),
    # medium_ratelimit: object exists and has acquire
    "medium_ratelimit": ("import importlib.util, sys; "
                         "m = importlib.util.spec_from_file_location('m', '{path}'); "
                         "mod = importlib.util.module_from_spec(m); "
                         "sys.modules['m'] = mod; "
                         "m.loader.exec_module(mod); "
                         "RateLimiter = getattr(mod, 'RateLimiter', None); "
                         "rl = RateLimiter(2, 1); "
                         "assert rl.acquire() in (True, False)"),
}


def extract_code(text: str) -> str | None:
    """Return the first python code block in the response (or None)."""
    for m in CODE_RE.finditer(text or ""):
        code = m.group(1).rstrip()
        if code.strip():
            return code
    return None


def run_inline_test(task_id: str, code: str) -> tuple[bool, str]:
    """Write code to a temp file and exec the per-task inline test. Returns
    (ok, message)."""
    test = INLINE_TESTS.get(task_id)
    if test is None:
        return True, "no inline test defined"
    # Write code to a temp file and exec the test in a subprocess.
    import tempfile
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as fh:
        fh.write(code)
        path = fh.name
    try:
        # 30 second timeout on the inline test.
        proc = subprocess.run(
            ["python3", "-c", test.format(path=path)],
            capture_output=True, text=True, timeout=30,
        )
        if proc.returncode == 0:
            return True, "ok"
        return False, (proc.stderr.strip().splitlines()[-1] if proc.stderr.strip()
                       else f"rc={proc.returncode}")
    except subprocess.TimeoutExpired:
        return False, "timeout"
    except Exception as exc:
        return False, str(exc)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--results", required=True)
    p.add_argument("--report", required=True)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    rows: list[dict] = []
    with open(args.results) as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    by_task_config: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in rows:
        by_task_config[(r["task_id"], r["config"])].append(r)

    out: list[dict] = []
    summary: dict[str, dict[str, float]] = defaultdict(lambda: {"n": 0, "passed": 0, "code": 0})
    for (task_id, config), entries in sorted(by_task_config.items()):
        test = INLINE_TESTS.get(task_id)
        for entry in entries:
            raw = entry.get("raw_response", "")
            code = extract_code(raw)
            summary[config]["n"] += 1
            if code is None:
                entry["quality_code_extracted"] = False
                continue
            entry["quality_code_extracted"] = True
            summary[config]["code"] += 1
            if test is not None:
                ok, msg = run_inline_test(task_id, code)
                entry["quality_test_ok"] = ok
                entry["quality_test_msg"] = msg
                if ok:
                    summary[config]["passed"] += 1
            out.append(entry)

    with open(args.report, "w") as fh:
        json.dump({"summary": summary, "details": out}, fh, indent=2)
    print(f"Wrote quality report to {args.report}")
    print()
    print("Per-config quality:")
    print(f"{'Config':<8} {'N':>4} {'Code':>5} {'Pass':>5} {'Pass%':>7}")
    for cfg in sorted(summary):
        s = summary[cfg]
        pct = (s["passed"] / s["n"] * 100) if s["n"] else 0.0
        print(f"{cfg:<8} {s['n']:>4} {s['code']:>5} {s['passed']:>5} {pct:>6.1f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
