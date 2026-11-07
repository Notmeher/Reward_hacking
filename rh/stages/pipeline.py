"""Run every stage in one session. Each step is a `rh` subprocess; a list of steps runs in parallel.

Steps that finish inside their time limit are recorded in <out>/pipeline_state.json and skipped
when the pipeline is resumed; unfinished stages resume from their own saved outputs.
"""
import logging
import subprocess
import sys
import threading
import time
from pathlib import Path

from rh.io import read_json, write_json

log = logging.getLogger(__name__)


def _name(step: dict) -> str:
    return step.get("name") or f"{step['stage']}:{Path(step['config']).stem}"


def _run(step: dict, out_root: Path, hours: float, results: dict):
    name = _name(step)
    cmd = [sys.executable, "-m", "rh.cli", step["stage"], "--config", step["config"], "--out", str(out_root),
           "--max-hours", f"{hours:.3f}"]
    for o in step.get("overrides", []):
        cmd += ["-o", o]
    log.info("[%s] start (limit %.2f h): %s", name, hours, " ".join(cmd[3:]))
    t0 = time.time()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in proc.stdout:
        print(f"[{name}] {line}", end="", flush=True)
    rc = proc.wait()
    elapsed = time.time() - t0
    results[name] = {"rc": rc, "hours": round(elapsed / 3600, 3),
                     "finished": rc == 0 and elapsed < hours * 3600 - 120}
    log.info("[%s] exit %d after %.2f h", name, rc, elapsed / 3600)


def main(cfg: dict, out_root: Path, max_hours: float) -> dict:
    deadline = time.time() + max_hours * 3600
    state_path = Path(out_root) / "pipeline_state.json"
    state = read_json(state_path, {"done": []})
    results: dict = {}
    for group in cfg["steps"]:
        group = group if isinstance(group, list) else [group]
        todo = [s for s in group if _name(s) not in state["done"]]
        if not todo:
            log.info("skipping finished: %s", ", ".join(_name(s) for s in group))
            continue
        left = (deadline - time.time()) / 3600
        if left < 0.1:
            log.warning("pipeline time budget used up; rerun with the saved outputs to continue")
            break
        threads = [threading.Thread(target=_run, args=(s, out_root, min(float(s.get("hours", left)), left), results))
                   for s in todo]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        for s in todo:
            if results[_name(s)]["finished"]:
                state["done"].append(_name(s))
        write_json(state_path, state)
    log.info("pipeline summary:")
    for name, r in results.items():
        log.info("  %-34s exit=%d  %.2f h  %s", name, r["rc"], r["hours"], "done" if r["finished"] else "INCOMPLETE")
    return results
