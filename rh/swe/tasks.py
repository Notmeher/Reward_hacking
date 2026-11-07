"""SWE-bench Verified tasks. Difficulty comes from the dataset's annotated fix-time bucket."""
import json

from rh.tasks.base import Task

DIFFICULTY = {"<15 min fix": 1.0, "15 min - 1 hour": 2.0, "1-4 hours": 3.0, ">4 hours": 4.0}


def _as_list(v):
    return json.loads(v) if isinstance(v, str) else list(v or [])


def load_swebench(dataset: str = "princeton-nlp/SWE-bench_Verified", split: str = "test", limit: int | None = None,
                  per_difficulty: int | None = None, repos: list | None = None, instance_ids: list | None = None,
                  seed: int = 0) -> list[Task]:
    from datasets import load_dataset

    ds = load_dataset(dataset, split=split).shuffle(seed=seed)
    counts: dict = {}
    tasks = []
    for row in ds:
        if instance_ids and row["instance_id"] not in instance_ids:
            continue
        if repos and row["repo"] not in repos:
            continue
        d = DIFFICULTY.get(row.get("difficulty", ""), 2.0)
        if per_difficulty and counts.get(d, 0) >= per_difficulty:
            continue
        counts[d] = counts.get(d, 0) + 1
        inst = dict(row)
        inst["FAIL_TO_PASS"], inst["PASS_TO_PASS"] = _as_list(row["FAIL_TO_PASS"]), _as_list(row["PASS_TO_PASS"])
        tasks.append(Task(
            task_id=row["instance_id"], family="swebench", difficulty=d, kind="swebench",
            prompt=row["problem_statement"], reference=row["patch"],
            meta={"instance": inst, "repo": row["repo"], "difficulty_label": row.get("difficulty", "")},
        ))
        if limit and len(tasks) >= limit:
            break
    return tasks
