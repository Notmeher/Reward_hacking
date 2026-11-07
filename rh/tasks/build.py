import random

from rh.tasks import hf_code, procedural
from rh.tasks.base import Task, split_for
from rh.tasks.impossible import make_impossible


def build_tasks(cfg: dict) -> list[Task]:
    """cfg keys: seed, eval_frac, procedural, mbppplus, code_contests, swebench, impossible_frac."""
    seed = int(cfg.get("seed", 0))
    tasks: list[Task] = []
    if cfg.get("procedural"):
        tasks += procedural.generate(cfg["procedural"], seed)
    if cfg.get("mbppplus"):
        tasks += hf_code.load_mbppplus(seed=seed, **cfg["mbppplus"])
    if cfg.get("code_contests"):
        tasks += hf_code.load_code_contests(**cfg["code_contests"])
    if cfg.get("swebench"):
        from rh.swe.tasks import load_swebench

        tasks += load_swebench(**cfg["swebench"])
    frac = float(cfg.get("impossible_frac", 0.0))
    if frac > 0:
        rng = random.Random(f"imp:{seed}")
        extra = [make_impossible(t, seed) for t in tasks if t.kind != "swebench" and rng.random() < frac]
        tasks += [t for t in extra if t is not None]
    eval_frac = float(cfg.get("eval_frac", 0.3))
    for t in tasks:
        base = t.meta.get("base_task", t.task_id)
        t.split = split_for(base, eval_frac)
    return tasks
