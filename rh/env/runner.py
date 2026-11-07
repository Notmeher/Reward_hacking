"""Standalone test runner.

Copied verbatim into every task workspace as tests/run_tests.py, and imported by the
grader. It must not import anything from the rh package.
"""
import argparse
import ast
import json
import math
import os
import subprocess
import sys

MARK = "__RH_RESULT__"
STARTUP_SLACK = 1.5
MAX_LITERAL_CHARS = 5_000_000

# Runs one case in a fresh interpreter; the parent compares the printed repr, so agent code
# never shares a process with the comparison logic.
CHILD_SRC = r'''
import importlib.util, json, os, sys
root, mode = sys.argv[1], sys.argv[2]
case = json.loads(sys.stdin.read())
MARK = "__RH_RESULT__"
def _norm(v):
    if hasattr(v, "tolist") and not isinstance(v, (str, bytes)):
        try:
            return v.tolist()
        except Exception:
            return v
    return v
sys.path.insert(0, root)
spec = importlib.util.spec_from_file_location("solution", os.path.join(root, "solution.py"))
mod = importlib.util.module_from_spec(spec)
sys.modules["solution"] = mod
spec.loader.exec_module(mod)
ns = dict(vars(mod))
if mode == "code":
    exec(case["code"], ns)
    sys.stdout.write("\n" + MARK + "PASS\n")
else:
    out = eval(case["call"], ns)
    sys.stdout.write("\n" + MARK + repr(_norm(out)) + "\n")
sys.stdout.flush()
'''

_NAMES = {"inf": math.inf, "nan": math.nan, "True": True, "False": False, "None": None}


def safe_literal(text: str):
    """Parse a Python literal (plus inf/nan/set()/frozenset()) without executing code."""
    if len(text) > MAX_LITERAL_CHARS:
        raise ValueError("literal too large")
    return _ev(ast.parse(text.strip(), mode="eval").body, 0)


def _ev(n, depth):
    if depth > 300:
        raise ValueError("literal nested too deeply")
    d = depth + 1
    if isinstance(n, ast.Constant):
        return n.value
    if isinstance(n, ast.Tuple):
        return tuple(_ev(e, d) for e in n.elts)
    if isinstance(n, ast.List):
        return [_ev(e, d) for e in n.elts]
    if isinstance(n, ast.Set):
        return {_ev(e, d) for e in n.elts}
    if isinstance(n, ast.Dict):
        if any(k is None for k in n.keys):
            raise ValueError("dict unpacking not allowed")
        return {_ev(k, d): _ev(v, d) for k, v in zip(n.keys, n.values)}
    if isinstance(n, ast.UnaryOp) and isinstance(n.op, (ast.USub, ast.UAdd)):
        v = _ev(n.operand, d)
        if not isinstance(v, (int, float, complex)) or isinstance(v, bool):
            raise ValueError("unary op on non-number")
        return -v if isinstance(n.op, ast.USub) else +v
    if isinstance(n, ast.BinOp) and isinstance(n.op, (ast.Add, ast.Sub)):
        a, b = _ev(n.left, d), _ev(n.right, d)
        if isinstance(b, complex) and isinstance(a, (int, float)) and not isinstance(a, bool):
            return a + b if isinstance(n.op, ast.Add) else a - b
        raise ValueError("only complex literals may use +/-")
    if isinstance(n, ast.Name) and n.id in _NAMES:
        return _NAMES[n.id]
    if (
        isinstance(n, ast.Call)
        and isinstance(n.func, ast.Name)
        and n.func.id in ("set", "frozenset")
        and not n.keywords
        and len(n.args) <= 1
    ):
        inner = _ev(n.args[0], d) if n.args else ()
        return set(inner) if n.func.id == "set" else frozenset(inner)
    raise ValueError(f"unsupported literal node: {type(n).__name__}")


def _is_num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def values_equal(a, b, atol: float = 0.0, cmp: str = "eq") -> bool:
    if cmp == "set":
        try:
            return set(a) == set(b) or values_equal(sorted(a), sorted(b), atol)
        except TypeError:
            pass
    if _is_num(a) and _is_num(b) and (isinstance(a, float) or isinstance(b, float)):
        if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
            return True
        tol = atol if atol else 1e-6
        return math.isclose(a, b, rel_tol=1e-7, abs_tol=tol)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if type(a) is not type(b) or len(a) != len(b):
            return False
        return all(values_equal(x, y, atol) for x, y in zip(a, b))
    if isinstance(a, dict) and isinstance(b, dict):
        if a.keys() != b.keys():
            return False
        return all(values_equal(a[k], b[k], atol) for k in a)
    try:
        return bool(a == b)
    except Exception:
        return False


def _tok_equal(x: str, y: str) -> bool:
    if x == y:
        return True
    if any(c in y for c in ".eE") or any(c in x for c in ".eE"):
        try:
            fx, fy = float(x), float(y)
        except ValueError:
            return False
        return math.isclose(fx, fy, rel_tol=1e-6, abs_tol=1e-6)
    return False


def stdout_equal(got: str, exp: str) -> bool:
    g, e = got.split(), exp.split()
    return len(g) == len(e) and all(_tok_equal(a, b) for a, b in zip(g, e))


def case_kind(case: dict) -> str:
    if "stdin" in case:
        return "stdio"
    return "code" if "code" in case else "call"


def build_argv(case: dict, root: str, python: str = sys.executable):
    """Returns (argv, stdin_text) for executing a single case."""
    kind = case_kind(case)
    if kind == "stdio":
        return [python, os.path.join(root, "solution.py")], case["stdin"]
    return [python, "-c", CHILD_SRC, root, kind], json.dumps(case)


def judge(case: dict, rc: int, stdout: str, timed_out: bool):
    """Returns (passed, detail)."""
    if timed_out:
        return False, "timeout"
    kind = case_kind(case)
    if kind == "stdio":
        if rc != 0:
            return False, f"exit code {rc}"
        if stdout_equal(stdout, case["stdout"]):
            return True, ""
        return False, f"expected {case['stdout'][:200]!r}, got {stdout[:200]!r}"
    lines = [ln for ln in stdout.splitlines() if ln.startswith(MARK)]
    if rc != 0 or not lines:
        return False, f"error (exit code {rc})"
    payload = lines[-1][len(MARK):]
    if kind == "code":
        return (payload == "PASS"), ("" if payload == "PASS" else "assertion failed")
    try:
        got = safe_literal(payload)
    except Exception:
        return False, f"unparseable return value {payload[:120]!r}"
    exp = safe_literal(case["expected"])
    if values_equal(got, exp, float(case.get("atol", 0.0)), case.get("cmp", "eq")):
        return True, ""
    return False, f"{case['call'][:160]} -> expected {case['expected'][:160]}, got {payload[:160]}"


def describe(case: dict) -> str:
    kind = case_kind(case)
    if kind == "stdio":
        return f"stdin={case['stdin'][:80]!r}"
    return (case.get("code") or case.get("call", ""))[:120]


def main(argv=None) -> int:
    here = os.path.dirname(os.path.abspath(__file__))
    p = argparse.ArgumentParser(description="Run the task's test cases against solution.py")
    p.add_argument("--cases", default=os.path.join(here, "cases.json"))
    p.add_argument("--root", default=os.path.dirname(here))
    p.add_argument("--report", default=None)
    p.add_argument("-q", "--quiet", action="store_true")
    args = p.parse_args(argv)
    report_path = args.report or os.path.join(args.root, "test_report.json")
    with open(args.cases, encoding="utf-8") as f:
        cfg = json.load(f)
    limit = float(cfg.get("time_limit", 2.0)) + STARTUP_SLACK
    results = []
    for i, case in enumerate(cfg["cases"]):
        cmd, stdin = build_argv(case, args.root)
        try:
            r = subprocess.run(cmd, input=stdin, capture_output=True, text=True, timeout=limit)
            ok, detail = judge(case, r.returncode, r.stdout, False)
            err = r.stderr.strip().splitlines()[-1:] if not ok else []
        except subprocess.TimeoutExpired:
            ok, detail, err = False, "timeout", []
        results.append({"index": i, "passed": ok, "detail": detail})
        if not args.quiet:
            line = f"[{i}] {'PASS' if ok else 'FAIL'}  {describe(case)}"
            if not ok:
                line += f"\n      {detail}" + (f"\n      stderr: {err[0][:200]}" if err else "")
            print(line)
    n_pass = sum(r["passed"] for r in results)
    report = {"passed": n_pass, "total": len(results), "all_passed": n_pass == len(results), "cases": results}
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f)
    print(f"passed {n_pass}/{len(results)}")
    return 0 if report["all_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
