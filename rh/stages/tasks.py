"""Stage 0: build the task suite and verify every reference solution in the sandbox."""
import asyncio
import collections
import logging
import os
import random
from pathlib import Path

from rh.env.code_env import run_cases
from rh.env.sandbox import is_root
from rh.tasks.base import save_tasks, split_for
from rh.tasks.build import build_tasks
from rh.tasks.impossible import make_impossible

log = logging.getLogger(__name__)


async def verify(tasks, sandbox, concurrency: int = 8):
    sem = asyncio.Semaphore(concurrency)

    async def check(t):
        if t.kind == "swebench":
            return t
        refs = [t.reference] + list(t.meta.get("alt_references", []))
        async with sem:
            for ref in refs:
                d = sandbox.new_dir("verify")
                (d / "solution.py").write_text(ref, encoding="utf-8")
                sandbox.give(d)
                res = await run_cases(sandbox, d, t.visible + t.hidden + t.audit, t.time_limit)
                sandbox.remove(d)
                if all(ok for ok, _ in res):
                    t.reference = ref
                    t.meta.pop("alt_references", None)
                    return t
        log.info("dropping %s: reference fails its own tests", t.task_id)
        return None

    return [t for t in await asyncio.gather(*(check(t) for t in tasks)) if t is not None]


def build_and_save(spec: dict, path: Path, cfg: dict | None = None) -> list:
    from rh.stages.common import make_sandbox

    spec = dict(spec)
    imp_frac = float(spec.pop("impossible_frac", 0.0))
    tasks = build_tasks({**spec, "impossible_frac": 0.0})
    log.info("built %d tasks; verifying references", len(tasks))
    sandbox = make_sandbox(cfg or {})
    tasks = asyncio.run(verify(tasks, sandbox))
    if imp_frac > 0:
        rng = random.Random(f"imp:{spec.get('seed', 0)}")
        extra = [make_impossible(t, int(spec.get("seed", 0))) for t in tasks if t.kind != "swebench" and rng.random() < imp_frac]
        tasks += [t for t in extra if t is not None]
    eval_frac = float(spec.get("eval_frac", 0.3))
    for t in tasks:
        t.split = split_for(t.meta.get("base_task", t.task_id), eval_frac)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_tasks(tasks, path)
    if is_root():
        os.chmod(path.parent, 0o700)
        os.chmod(path, 0o600)
    counts = collections.Counter((t.family, t.difficulty, t.impossible, t.split) for t in tasks)
    for k in sorted(counts):
        log.info("  %-14s difficulty=%-6g impossible=%-5s split=%-5s n=%d", *k, counts[k])
    log.info("saved %d tasks to %s", len(tasks), path)
    return tasks
