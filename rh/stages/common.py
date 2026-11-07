"""Helpers shared by pipeline stages."""
import json
import logging
import os
import random
import tempfile
from pathlib import Path

from rh.env.sandbox import LocalSandbox, is_root
from rh.io import read_jsonl
from rh.tasks.base import Task, load_tasks

log = logging.getLogger(__name__)


def protect(path: Path) -> None:
    """Keep run outputs (hidden tests, transcripts) unreadable by the sandbox user."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    if is_root():
        os.chmod(path, 0o700)


def run_dir(out_root: Path, name: str) -> Path:
    d = Path(out_root) / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def make_sandbox(cfg: dict) -> LocalSandbox:
    sc = cfg.get("sandbox", {})
    default = "/tmp/rh_sb" if is_root() else os.path.join(tempfile.gettempdir(), f"rh_sb_{os.getuid()}")
    return LocalSandbox(
        os.environ.get("RH_SANDBOX_DIR", sc.get("base_dir", default)),
        mem_mb=int(sc.get("mem_mb", 2048)),
        max_concurrency=int(sc.get("exec_concurrency", max(2, (os.cpu_count() or 4)))),
    )


def tasks_path(out_root: Path, cfg: dict) -> Path:
    return Path(out_root) / cfg.get("tasks_file", "tasks/tasks.jsonl")


def load_or_build_tasks(cfg: dict, out_root: Path) -> list[Task]:
    path = tasks_path(out_root, cfg)
    if not path.exists():
        if not cfg.get("tasks"):
            raise FileNotFoundError(f"{path} missing and no `tasks` spec in config; run `rh tasks` first")
        from rh.stages.tasks import build_and_save

        build_and_save(cfg["tasks"], path, cfg)
    return list(load_tasks(path).values())


def filter_tasks(tasks: list[Task], flt: dict | None) -> list[Task]:
    flt = {k: v for k, v in (flt or {}).items() if v is not None}
    out = []
    for t in tasks:
        if "split" in flt and t.split not in _as_list(flt["split"]):
            continue
        if "family" in flt and t.family not in _as_list(flt["family"]):
            continue
        if "impossible" in flt and bool(t.impossible) != bool(flt["impossible"]):
            continue
        if "min_difficulty" in flt and t.difficulty < float(flt["min_difficulty"]):
            continue
        if "max_difficulty" in flt and t.difficulty > float(flt["max_difficulty"]):
            continue
        if "task_ids" in flt and t.task_id not in set(flt["task_ids"]):
            continue
        out.append(t)
    cap = flt.get("max_per_family_level")
    if cap:
        rng = random.Random(int(flt.get("seed", 0)))
        rng.shuffle(out)
        counts: dict = {}
        kept = []
        for t in out:
            k = (t.family, t.difficulty, t.impossible)
            if counts.get(k, 0) < int(cap):
                counts[k] = counts.get(k, 0) + 1
                kept.append(t)
        out = sorted(kept, key=lambda t: t.task_id)
    return out


def _as_list(v):
    return v if isinstance(v, (list, tuple, set)) else [v]


def done_keys(path: Path) -> set:
    return {r["key"] for r in read_jsonl(path) if "key" in r and r.get("status") != "harness_error"}


def save_run_config(run: Path, cfg: dict) -> None:
    with open(run / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, default=str)
