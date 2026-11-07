"""Real-world coding tasks from HuggingFace datasets.

MBPP+ (evalplus/mbppplus): base asserts are visible, the EvalPlus extra inputs are split into
hidden and audit sets. CodeContests (deepmind/code_contests): public tests visible, private
tests hidden, generated tests audit; Codeforces rating is the difficulty.
"""
import ast
import random
import re

from rh.env.runner import _ev
from rh.tasks.base import Task, call_case, io_case

MAX_STDIN_CHARS = 200_000
PYTHON3 = 3


def _parse_plus_test(test_src: str):
    tree = ast.parse(test_src)
    found, atol = {}, 0.0
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name in ("inputs", "results"):
                found[name] = _ev(node.value, 0)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "assertion":
            if len(node.args) >= 3 and isinstance(node.args[2], ast.Constant):
                atol = float(node.args[2].value)
    cmp = "set" if re.search(r"^\s*out\s*=\s*set\(out\)", test_src, re.M) else "eq"
    return found.get("inputs"), found.get("results"), atol, cmp


def _entry_name(test_list, code):
    for t in test_list:
        m = re.search(r"assert\s+(?:\w+\()?\s*([A-Za-z_]\w*)\s*\(", t)
        if m and m.group(1) not in ("set", "math", "abs", "round", "sorted", "list", "tuple", "str", "isclose"):
            return m.group(1)
    m = re.search(r"^def\s+([A-Za-z_]\w*)\s*\(", code, re.M)
    return m.group(1) if m else None


def load_mbppplus(limit: int | None = None, seed: int = 0, n_hidden: int = 10, n_audit: int = 10) -> list[Task]:
    from datasets import load_dataset

    ds = load_dataset("evalplus/mbppplus", split="test")
    tasks = []
    for row in ds:
        entry = _entry_name(row["test_list"], row["code"])
        if not entry or f"def {entry}" not in row["code"]:
            continue
        try:
            inputs, results, atol, cmp = _parse_plus_test(row["test"])
        except (SyntaxError, ValueError):
            continue
        if not inputs or not results or len(inputs) != len(results):
            continue
        pairs = list(zip(inputs, results))[len(row["test_list"]):]
        rng = random.Random(f"mbpp:{seed}:{row['task_id']}")
        rng.shuffle(pairs)
        if len(pairs) < 6:
            continue
        cases = [call_case(f"{entry}(*{inp!r})", exp, atol=atol, cmp=cmp) for inp, exp in pairs[: n_hidden + n_audit]]
        half = min(n_hidden, len(cases) // 2)
        imports = "\n".join(row["test_imports"])
        visible = [{"code": (imports + "\n" + a).strip()} for a in row["test_list"]]
        sig = re.search(rf"^def\s+{entry}\s*\((.*?)\)\s*(->[^:]*)?:", row["code"], re.M)
        params = sig.group(1) if sig else "*args"
        prompt = (
            f"{row['prompt'].strip()}\n\nImplement it as a function named `{entry}` in `solution.py`. "
            f"Example test:\n\n    {row['test_list'][0]}"
        )
        tasks.append(
            Task(
                task_id=f"mbpp-{row['task_id']}",
                family="mbppplus",
                difficulty=1.0,
                kind="function",
                prompt=prompt,
                entry=entry,
                starter=f"def {entry}({params}):\n    raise NotImplementedError\n",
                visible=visible,
                hidden=cases[:half],
                audit=cases[half:],
                reference=row["code"],
                meta={"ref_lines": row["code"].strip().count("\n") + 1},
            )
        )
        if limit and len(tasks) >= limit:
            break
    return tasks


def _io_cases(block: dict, cap: int):
    out = []
    for i, o in zip(block.get("input") or [], block.get("output") or []):
        if len(i) <= MAX_STDIN_CHARS:
            out.append(io_case(i, o))
        if len(out) >= cap:
            break
    return out


def load_code_contests(
    split: str | list[str] = "test",
    limit: int | None = None,
    min_rating: int = 800,
    max_rating: int = 2400,
    n_hidden: int = 10,
    n_audit: int = 10,
) -> list[Task]:
    if not isinstance(split, str):
        out: list[Task] = []
        for s in split:
            out += load_code_contests(s, None if limit is None else limit - len(out), min_rating, max_rating,
                                      n_hidden, n_audit)
            if limit and len(out) >= limit:
                break
        return out
    from datasets import load_dataset

    ds = load_dataset("deepmind/code_contests", split=split, streaming=True)
    tasks = []
    for row in ds:
        rating = int(row.get("cf_rating") or 0)
        if not (min_rating <= rating <= max_rating):
            continue
        sols = row.get("solutions") or {}
        py = [s for lang, s in zip(sols.get("language", []), sols.get("solution", [])) if lang == PYTHON3]
        if not py:
            continue
        visible = _io_cases(row["public_tests"], 3)
        hidden = _io_cases(row["private_tests"], n_hidden)
        audit = _io_cases(row["generated_tests"], n_audit)
        if len(audit) < 3 and len(hidden) >= 6:
            hidden, audit = hidden[: len(hidden) // 2], hidden[len(hidden) // 2 :]
        if not visible or len(hidden) < 3 or len(audit) < 3:
            continue
        tl = row.get("time_limit") or {}
        seconds = float(tl.get("seconds") or 2) + float(tl.get("nanos") or 0) / 1e9
        name = re.sub(r"[^A-Za-z0-9]+", "_", row["name"]).strip("_")[:40]
        prompt = (
            f"{row['description'].strip()}\n\nWrite a complete Python 3 program in `solution.py` that reads the input "
            "from standard input and writes the answer to standard output."
        )
        tasks.append(
            Task(
                task_id=f"cc-{name}",
                family="code_contests",
                difficulty=float(rating),
                kind="stdio",
                prompt=prompt,
                starter="import sys\n\n\ndef main():\n    data = sys.stdin.read().split()\n\n\nmain()\n",
                visible=visible,
                hidden=hidden,
                audit=audit,
                reference=py[0],
                time_limit=min(max(seconds * 3, 4.0), 12.0),
                meta={"cf_rating": rating, "alt_references": py[1:4]},
            )
        )
        if limit and len(tasks) >= limit:
            break
    return tasks
