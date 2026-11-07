import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class Task:
    task_id: str
    family: str
    difficulty: float
    kind: str  # "function" | "stdio" | "swebench"
    prompt: str
    entry: str = ""
    starter: str = ""
    # Case formats: {"call": "f(1, 2)", "expected": "<repr>"} | {"code": "assert ..."} |
    # {"stdin": "...", "stdout": "..."}.  visible = shown to the agent, hidden = proxy grader
    # only, audit = never exposed (detects special-casing).
    visible: list = field(default_factory=list)
    hidden: list = field(default_factory=list)
    audit: list = field(default_factory=list)
    reference: str = ""
    impossible: bool = False
    time_limit: float = 2.0
    split: str = "train"
    meta: dict = field(default_factory=dict)


def call_case(call: str, expected, **extra) -> dict:
    return {"call": call, "expected": repr(expected), **extra}


def io_case(stdin: str, stdout: str) -> dict:
    return {"stdin": stdin, "stdout": stdout}


def save_tasks(tasks: list[Task], path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for t in tasks:
            f.write(json.dumps(asdict(t), ensure_ascii=False) + "\n")


def load_tasks(path) -> dict[str, Task]:
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                out[d["task_id"]] = Task(**d)
    return out


def split_for(task_id: str, eval_frac: float = 0.3) -> str:
    h = int(hashlib.sha1(task_id.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "eval" if h < eval_frac else "train"
