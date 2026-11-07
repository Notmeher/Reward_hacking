"""Stage 3: multi-turn GRPO with LoRA. GPU0 trains, GPU1 serves rollouts via vLLM.

Conditions (RQ2/RQ3) are set by config: the budget sampler decides the margin regime
(pressure = budgets below b*, open = budgets above b*), and `abstain` toggles the honest
abstention action + reward. Training stops at max_iters, the time limit, or once the number of
*rewarded* hacks reaches `stop_at_rewarded_hacks` (used to match RQ2 conditions).
"""
import asyncio
import logging
import random
import shutil
import time
from pathlib import Path

from rh.agents import make_agent
from rh.env import make_env
from rh.env.episode import run_episode
from rh.env.prompts import Condition
from rh.env.tools import tool_set
from rh.io import append_jsonl, read_json, write_json
from rh.serving import ServerGroup
from rh.stages.common import filter_tasks, load_or_build_tasks, make_sandbox, protect, run_dir, save_run_config
from rh.train.rewards import FeasibilityTracker, group_advantages, reward

log = logging.getLogger(__name__)


class BudgetSampler:
    """mode=margin: budget = round(b* * 2**m), m ~ uniform(margins); mode=fixed: choice(budgets)."""

    def __init__(self, cfg: dict, bstar: dict | None):
        self.cfg, self.bstar = cfg, bstar or {}

    def __call__(self, task, rng: random.Random) -> int:
        lo, hi = int(self.cfg.get("min", 1)), int(self.cfg.get("max", 64))
        if self.cfg.get("mode", "fixed") == "margin":
            b = self.bstar.get(task.task_id.removesuffix("-imp"))
            if b:
                m = rng.uniform(*self.cfg["log_margin_range"])
                return max(lo, min(hi, int(round(b * 2 ** m))))
        return int(rng.choice(self.cfg.get("budgets", [8])))


async def collect(tasks_conds, spec, sandbox, cfg, group_size: int):
    ecfg = cfg.get("env", {})

    async def one(task, cond):
        agent = make_agent(spec, task)
        env = make_env(task, cond, sandbox, ecfg)
        try:
            return await run_episode(env, agent, cond, cmd_timeout=float(ecfg.get("cmd_timeout", 20)), max_nudges=0,
                                     act_timeout=float(ecfg.get("act_timeout", 900)))
        except Exception as e:  # noqa: BLE001 - a broken rollout should not kill the iteration
            log.warning("rollout failed on %s: %s", task.task_id, e)
            return None

    jobs = [one(t, c) for t, c in tasks_conds for _ in range(group_size)]
    res = await asyncio.gather(*jobs)
    groups = []
    for i, (t, c) in enumerate(tasks_conds):
        g = [r for r in res[i * group_size : (i + 1) * group_size] if r is not None]
        groups.append((t, c, g))
    return groups


def main(cfg: dict, out_root: Path, max_hours: float) -> Path:
    from rh.train.policy import PolicyTrainer

    t_start = time.time()
    deadline = t_start + max_hours * 3600
    run = run_dir(out_root, cfg["run_name"])
    protect(out_root)
    save_run_config(run, cfg)
    ck = run / "ckpt"
    ck.mkdir(exist_ok=True)
    tcfg, rcfg = cfg["train"], cfg.get("reward", {})
    tasks = filter_tasks(load_or_build_tasks(cfg, out_root), cfg.get("task_filter", {"split": "train"}))
    if not tasks:
        raise SystemExit("no training tasks after filtering")
    bstar = None
    if cfg.get("bstar_file"):
        bstar = (read_json(Path(out_root) / cfg["bstar_file"], {}) or {}).get(cfg.get("bstar_model"))
        bstar = {k: v for k, v in (bstar or {}).items() if v is not None}
    sampler = BudgetSampler(cfg["budget"], bstar)
    feas = FeasibilityTracker(bstar, ema=float(rcfg.get("feas_ema", 0.3)))
    state = read_json(run / "state.json", {"iteration": 0, "rewarded_hacks": 0, "episodes": 0})
    trainer = PolicyTrainer(cfg["model"]["path"], cfg.get("lora", {}), lr=float(tcfg.get("lr", 1e-5)),
                            device=tcfg.get("device", "cuda:0"), max_len=int(tcfg.get("max_len", 10240)),
                            max_grad_norm=float(tcfg.get("max_grad_norm", 1.0)))
    if state["iteration"] > 0:
        trainer.load_state(ck / "trainer_state.pt")
        feas.load_state_dict(read_json(run / "feas.json", {}))
        log.info("resumed at iteration %d", state["iteration"])
    sandbox = make_sandbox(cfg)
    serve = {**cfg["serve"], "enable_lora": True, "max_lora_rank": int(cfg.get("lora", {}).get("r", 16))}
    serve.setdefault("model", cfg["model"]["path"])
    base_name = serve.get("served_name", "policy-base")
    serve["served_name"] = base_name
    exploits = cfg.get("exploits", ["all"])
    abstain = bool(cfg.get("abstain", False))
    with ServerGroup(serve, run / "logs") as g:
        server = g.servers[0]
        current = base_name
        if state["iteration"] > 0:
            current = f"it{state['iteration']}"
            server.load_lora(current, str(ck / f"adapter_it{state['iteration']}"))
        while state["iteration"] < int(tcfg.get("max_iters", 200)):
            if time.time() > deadline - float(tcfg.get("reserve_s", 900)):
                log.info("time budget reached")
                break
            if cfg.get("stop_at_rewarded_hacks") and state["rewarded_hacks"] >= int(cfg["stop_at_rewarded_hacks"]):
                log.info("reached %d rewarded hacks; stopping", state["rewarded_hacks"])
                break
            it = state["iteration"]
            rng = random.Random(f"{cfg.get('seed', 0)}:{it}")
            picked = rng.sample(tasks, min(int(tcfg.get("tasks_per_iter", 8)), len(tasks)))
            tc = [(t, Condition(budget_calls=sampler(t, rng), exploit=rng.choice(exploits), abstain=abstain,
                                disclosure=cfg.get("disclosure", "disclosed"), framing=cfg.get("framing", "eval"),
                                seed=it)) for t in picked]
            spec = {"provider": "vllm", "name": cfg["run_name"], "base_urls": g.base_urls, "api_key": g.api_key,
                    "served_name": current, "model": current, "params": cfg.get("sampling", {})}
            t0 = time.time()
            groups = asyncio.run(collect(tc, spec, sandbox, cfg, int(tcfg.get("group_size", 4))))
            gen_s = time.time() - t0
            batch, stats = [], {"n": 0, "proxy": 0, "true": 0, "hack": 0, "abstain": 0, "rewarded_hacks": 0, "reward": 0.0}
            for task, cond, eps in groups:
                rs = [reward(s, task, cond.budget_calls, rcfg, feas) for s, _ in eps]
                feas.update(task, cond.budget_calls, [bool(s.get("g_true_pass")) for s, _ in eps])
                advs = group_advantages(rs, scale_std=bool(tcfg.get("scale_std", False)))
                tools = [{"type": "function", "function": t} for t in tool_set(float(cfg.get("env", {}).get("cmd_timeout", 20)), cond.abstain)]
                for (s, tr), r, a in zip(eps, rs, advs):
                    stats["n"] += 1
                    stats["proxy"] += bool(s.get("g_proxy_pass"))
                    stats["true"] += bool(s.get("g_true_pass"))
                    stats["hack"] += bool(s.get("d_hack"))
                    stats["abstain"] += bool(s.get("abstained"))
                    stats["reward"] += r
                    if s.get("d_hack") and r > 0:
                        stats["rewarded_hacks"] += 1
                    append_jsonl(run / "episodes.jsonl", {**s, "iteration": it, "reward": r, "advantage": a,
                                                          "policy": current, "run": cfg["run_name"]})
                    if rng.random() < float(cfg.get("save_transcript_frac", 0.25)) or s.get("d_hack"):
                        append_jsonl(run / "transcripts" / "train.jsonl", {**tr, "iteration": it})
                    if abs(a) > 1e-8:
                        batch.append((tr["messages"], tools, a))
            t1 = time.time()
            metrics = trainer.step(batch)
            train_s = time.time() - t1
            state["iteration"] = it + 1
            state["rewarded_hacks"] += stats["rewarded_hacks"]
            state["episodes"] += stats["n"]
            name = f"it{state['iteration']}"
            adir = ck / f"adapter_{name}"
            trainer.save_adapter(adir)
            trainer.save_state(ck / "trainer_state.pt")
            write_json(run / "feas.json", feas.state_dict())
            write_json(run / "state.json", state)
            server.load_lora(name, str(adir))
            if current != base_name:
                server.unload_lora(current)
            current = name
            for old in sorted(ck.glob("adapter_it*"), key=lambda p: int(p.name.split("it")[-1]))[: -int(tcfg.get("keep_adapters", 3))]:
                if int(old.name.split("it")[-1]) % int(tcfg.get("keep_every", 25)) != 0:
                    shutil.rmtree(old, ignore_errors=True)
            n = max(1, stats["n"])
            rec = {"iteration": it, "episodes": stats["n"], "reward": stats["reward"] / n, "proxy_pass": stats["proxy"] / n,
                   "true_pass": stats["true"] / n, "hack_rate": stats["hack"] / n, "abstain_rate": stats["abstain"] / n,
                   "rewarded_hacks_iter": stats["rewarded_hacks"], "rewarded_hacks_total": state["rewarded_hacks"],
                   "batch_items": len(batch), "gen_s": round(gen_s, 1), "train_s": round(train_s, 1), **metrics}
            append_jsonl(run / "train_log.jsonl", rec)
            log.info("it %d  reward=%.3f proxy=%.2f true=%.2f hack=%.2f abstain=%.2f  items=%d loss=%.4f gn=%.3f  gen=%.0fs train=%.0fs",
                     it, rec["reward"], rec["proxy_pass"], rec["true_pass"], rec["hack_rate"], rec["abstain_rate"],
                     len(batch), metrics["loss"], metrics["grad_norm"], gen_s, train_s)
    final = ck / f"adapter_it{state['iteration']}"
    if final.exists():
        dst = run / "final_adapter"
        shutil.rmtree(dst, ignore_errors=True)
        shutil.copytree(final, dst)
    return run
