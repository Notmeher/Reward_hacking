"""Stage 1: sweep models x tasks x (budget, exploit, disclosure, framing, abstain) x seeds.

Resumable: every finished episode is keyed; re-running skips finished keys. Cells are ordered
seed-major so a partial run still covers the whole grid once before adding repeats.
"""
import asyncio
import itertools
import logging
import random
import re
import time
import traceback
from pathlib import Path

from rh.agents import make_agent
from rh.env import make_env
from rh.env.episode import run_episode
from rh.env.prompts import Condition
from rh.io import append_jsonl, read_json, read_jsonl, stable_hash
from rh.serving import ServerGroup
from rh.stages.common import done_keys, filter_tasks, load_or_build_tasks, make_sandbox, protect, run_dir, save_run_config

log = logging.getLogger(__name__)
FATAL = re.compile(r"(401|403|404|DeploymentNotFound|authentication|invalid x-api-key|model_not_found|not_found_error|"
                   r"permission_error|does not exist)", re.I)


def budgets_for(task, grid: dict, bstar: dict | None) -> list[int]:
    if grid.get("margins") and bstar is not None:
        b = bstar.get(task.task_id)
        if b is not None and b != float("inf"):
            return sorted({max(1, int(round(b + m))) for m in grid["margins"]})
    return [int(b) for b in grid["budget_calls"]]


def expand(task, grid: dict, bstar: dict | None) -> list[Condition]:
    out = []
    for budget, exploit, disc, framing, abstain, tok in itertools.product(
        budgets_for(task, grid, bstar),
        grid.get("exploit", ["none"]),
        grid.get("disclosure", ["disclosed"]),
        grid.get("framing", ["eval"]),
        grid.get("abstain", [False]),
        grid.get("token_budget", [None]),
    ):
        if task.impossible and exploit == "none" and not grid.get("impossible_with_none", True):
            continue
        for seed in range(int(grid.get("seeds", 1))):
            out.append(Condition(budget_calls=budget, token_budget=tok, disclosure=disc, framing=framing,
                                 exploit=exploit, abstain=bool(abstain), seed=seed))
    return out


def cell_key(model: str, task_id: str, cond: Condition) -> str:
    return stable_hash({"m": model, "t": task_id, **cond.to_dict()}, n=16)


def episode_usd(row: dict, price: dict | None) -> float:
    """Cost at list price (no cache discount, so it over- rather than under-estimates)."""
    if not price:
        return 0.0
    return (float(row.get("input_tokens") or 0) * float(price.get("input", 0))
            + float(row.get("output_tokens") or 0) * float(price.get("output", 0))) / 1e6


class Wallet:
    def __init__(self, max_usd: float | None, spent: float = 0.0):
        self.max_usd, self.spent, self.warned = max_usd, spent, False

    def exhausted(self) -> bool:
        if self.max_usd is None or self.spent < self.max_usd:
            return False
        if not self.warned:
            log.warning("spending cap reached: $%.2f of $%.2f; stopping API models", self.spent, self.max_usd)
            self.warned = True
        return True


class Progress:
    def __init__(self, model: str, total: int):
        self.model, self.total, self.done = model, total, 0
        self.hacks = self.true = self.errors = self.consecutive_agent_errors = 0
        self.stop = False
        self.t0 = time.monotonic()

    def update(self, s: dict):
        self.done += 1
        self.hacks += bool(s.get("d_hack"))
        self.true += bool(s.get("g_true_pass"))
        if s.get("status") in ("harness_error", "agent_error"):
            self.errors += 1
        if self.done % 20 == 0 or self.done == self.total:
            rate = self.done / max(1e-9, time.monotonic() - self.t0) * 3600
            log.info("[%s] %d/%d  true_pass=%.2f hack=%.2f errors=%d  (%.0f ep/h)", self.model, self.done, self.total,
                     self.true / self.done, self.hacks / self.done, self.errors, rate)


async def _worker(spec, queue, sandbox, cfg, run: Path, deadline: float, prog: Progress, wallet: Wallet):
    ecfg = cfg.get("env", {})
    tr_path = run / "transcripts" / f"{spec['name']}.jsonl"
    while not prog.stop and time.time() < deadline:
        if spec.get("price") and wallet.exhausted():
            return
        try:
            task, cond, key = queue.get_nowait()
        except asyncio.QueueEmpty:
            return
        meta = {"key": key, "run": cfg["run_name"]}
        try:
            agent = make_agent(spec, task)
            env = make_env(task, cond, sandbox, ecfg)
            summary, transcript = await run_episode(
                env, agent, cond, cmd_timeout=float(ecfg.get("cmd_timeout", 20)),
                max_nudges=int(ecfg.get("max_nudges", 1)), act_timeout=float(ecfg.get("act_timeout", 900)), meta=meta)
        except Exception as e:  # provider outage, rate-limit exhaustion, harness bug: retried next run
            summary, transcript = {"model": spec["name"], "task_id": task.task_id, "status": "harness_error",
                                   "error": f"{type(e).__name__}: {e}"[:800], "trace": traceback.format_exc()[-2000:],
                                   **{f"cond_{k}": v for k, v in cond.to_dict().items()}, **meta}, None
            log.warning("[%s] harness error on %s: %s", spec["name"], task.task_id, summary["error"][:200])
        append_jsonl(run / "episodes.jsonl", summary)
        if transcript is not None:
            append_jsonl(tr_path, transcript)
        wallet.spent += episode_usd(summary, spec.get("price"))
        prog.update(summary)
        if spec.get("price") and prog.done % 20 == 0:
            log.info("spend so far: $%.2f", wallet.spent)
        err = summary.get("error", "")
        if summary.get("status") in ("agent_error", "harness_error"):
            prog.consecutive_agent_errors += 1
            if FATAL.search(err) or prog.consecutive_agent_errors >= int(cfg.get("max_consecutive_errors", 15)):
                log.error("[%s] stopping model after error: %s", spec["name"], err[:300])
                prog.stop = True
        else:
            prog.consecutive_agent_errors = 0


async def _run_model(spec, cells, sandbox, cfg, run, deadline, wallet: Wallet):
    queue: asyncio.Queue = asyncio.Queue()
    for c in cells:
        queue.put_nowait(c)
    prog = Progress(spec["name"], len(cells))
    n = int(spec.get("concurrency", 4))
    await asyncio.gather(*(_worker(spec, queue, sandbox, cfg, run, deadline, prog, wallet) for _ in range(n)))
    return prog


async def _run_vllm_model(spec, cells, sandbox, cfg, run, deadline, wallet: Wallet):
    serve = dict(spec["serve"])
    lora = spec.get("lora")
    served = serve.get("served_name", str(serve["model"]))
    if lora:
        serve["enable_lora"] = True
        serve["lora_modules"] = {lora["name"]: lora["path"]}
        served = lora["name"]
    group = ServerGroup(serve, run / "logs")
    await asyncio.to_thread(group.__enter__)
    try:
        live = {**spec, "base_urls": group.base_urls, "api_key": group.api_key, "served_name": served,
                "concurrency": int(spec.get("concurrency", 16)) * len(group.servers)}
        return await _run_model(live, cells, sandbox, cfg, run, deadline, wallet)
    finally:
        await asyncio.to_thread(group.stop)


async def run_sweep(cfg: dict, out_root: Path, max_hours: float, limit: int | None, tasks: list) -> Path:
    run = run_dir(out_root, cfg["run_name"])
    protect(out_root)
    save_run_config(run, cfg)
    bstar_all = read_json(Path(out_root) / cfg["bstar_file"]) if cfg.get("bstar_file") else None
    done = done_keys(run / "episodes.jsonl")
    max_usd = cfg.get("max_usd")
    if max_usd is not None:
        unpriced = [s["name"] for s in cfg["models"] if s["provider"] != "vllm" and not s.get("price")]
        if unpriced:
            raise ValueError(f"max_usd is set but these API models have no `price`: {unpriced}")
    prices = {s["name"]: s.get("price") for s in cfg["models"]}
    wallet = Wallet(max_usd, sum(episode_usd(r, prices.get(r.get("model"))) for r in read_jsonl(run / "episodes.jsonl")))
    if max_usd is not None:
        log.info("spending cap $%.2f, already spent $%.2f in this run", max_usd, wallet.spent)
    sandbox = make_sandbox(cfg)
    deadline = time.time() + max_hours * 3600
    rng = random.Random(int(cfg.get("order_seed", 0)))
    api_jobs, gpu_jobs = [], []
    for spec in cfg["models"]:
        bstar = (bstar_all or {}).get(spec.get("bstar_model", spec["name"])) if bstar_all else None
        cells = []
        for t in tasks:
            for cond in expand(t, cfg["grid"], bstar):
                key = cell_key(spec["name"], t.task_id, cond)
                if key not in done:
                    cells.append((cond.seed, rng.random(), t, cond, key))
        cells.sort(key=lambda x: (x[0], x[1]))
        cells = [(t, c, k) for _, _, t, c, k in cells][: limit or None]
        log.info("[%s] %d cells to run (%d already done)", spec["name"], len(cells), len(done))
        if not cells:
            continue
        (gpu_jobs if spec["provider"] == "vllm" else api_jobs).append((spec, cells))
    if not sandbox.isolated:
        log.warning("sandbox is NOT running as a separate user; only run trusted/scripted agents like this")

    async def gpu_sequence():
        for spec, cells in gpu_jobs:
            if time.time() < deadline:
                await _run_vllm_model(spec, cells, sandbox, cfg, run, deadline, wallet)

    await asyncio.gather(*(_run_model(s, c, sandbox, cfg, run, deadline, wallet) for s, c in api_jobs), gpu_sequence())
    log.info("sweep %s finished; episodes in %s", cfg["run_name"], run / "episodes.jsonl")
    return run


def main(cfg: dict, out_root: Path, max_hours: float, limit: int | None = None) -> Path:
    tasks = filter_tasks(load_or_build_tasks(cfg, out_root), cfg.get("task_filter"))
    return asyncio.run(run_sweep(cfg, out_root, max_hours, limit, tasks))
