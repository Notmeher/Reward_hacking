"""Stage 2: run CoT and action-only monitors over sweep transcripts."""
import asyncio
import json
import logging
import random
import time
from pathlib import Path

from rh.config import env
from rh.io import append_jsonl, read_json, read_jsonl, write_json
from rh.monitor.monitor import anthropic_params, call_monitor, parse_verdict, render
from rh.serving import ServerGroup
from rh.stages.common import load_or_build_tasks, protect, run_dir, save_run_config

log = logging.getLogger(__name__)


def select_episodes(cfg: dict, out_root: Path) -> list[dict]:
    sel = cfg.get("select", {})
    rows = []
    for name in cfg["source_runs"]:
        for r in read_jsonl(Path(out_root) / name / "episodes.jsonl"):
            if r.get("status") == "harness_error":
                continue
            if sel.get("exploit_only", True) and r.get("cond_exploit") == "none":
                continue
            if sel.get("models") and r.get("model") not in sel["models"]:
                continue
            rows.append({**r, "source_run": name})
    rng = random.Random(int(sel.get("seed", 0)))
    hacks = [r for r in rows if r.get("d_hack") or r.get("d_hack_attempt")]
    rest = [r for r in rows if not (r.get("d_hack") or r.get("d_hack_attempt"))]
    hcap = sel.get("max_hack")
    p_hack = 1.0
    if hcap is not None and len(hacks) > int(hcap):
        p_hack = int(hcap) / len(hacks)
        hacks = rng.sample(hacks, int(hcap))
    cap = sel.get("max_nonhack")
    p_rest = 1.0
    if cap is not None and len(rest) > int(cap):
        p_rest = int(cap) / len(rest)
        rest = rng.sample(rest, int(cap))
    for r in hacks:
        r["sample_weight"] = 1.0 / p_hack
    for r in rest:
        r["sample_weight"] = 1.0 / p_rest
    return hacks + rest


def load_transcripts(cfg: dict, out_root: Path, ids: set) -> dict:
    out = {}
    for name in cfg["source_runs"]:
        for p in sorted((Path(out_root) / name / "transcripts").glob("*.jsonl")):
            with open(p, encoding="utf-8") as f:
                for line in f:
                    # cheap prefilter before parsing large lines
                    if '"episode_id"' not in line[:200]:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if rec.get("episode_id") in ids:
                        out[rec["episode_id"]] = rec
    return out


def _record(ep: dict, spec: dict, view: str, text: str | None) -> dict:
    v = parse_verdict(text or "") if text is not None else {"hack": None, "confidence": None, "evidence": "", "raw": ""}
    return {"episode_id": ep["episode_id"], "monitor": spec["name"], "view": view, "agent_model": ep.get("model"),
            "source_run": ep.get("source_run"), "sample_weight": ep.get("sample_weight", 1.0),
            "failed": text is None, **v}


async def run_async(spec, items, run: Path, deadline: float):
    sem = asyncio.Semaphore(int(spec.get("concurrency", 8)))

    async def one(ep, view, prompt):
        if time.time() > deadline:
            return
        async with sem:
            try:
                text = await call_monitor(spec, prompt)
            except Exception as e:  # noqa: BLE001 - keep going; failures are recorded and retried next run
                log.warning("[%s] monitor call failed: %s", spec["name"], str(e)[:200])
                return
        append_jsonl(run / "monitor.jsonl", _record(ep, spec, view, text))

    await asyncio.gather(*(one(*it) for it in items))


def run_anthropic_batches(spec, items, run: Path, deadline: float, poll: int = 60, chunk: int = 5000):
    import anthropic

    client = anthropic.Anthropic(api_key=env("ANTHROPIC_API_KEY"), max_retries=8)
    state_path = run / f"batches_{spec['name']}.json"
    state = read_json(state_path, {})
    by_cid = {f"{ep['episode_id'][:40]}-{view}": (ep, view, prompt) for ep, view, prompt in items}
    pending_cids = {cid for b in state.values() if b["status"] != "collected" for cid in b["cids"]}
    fresh = [cid for cid in by_cid if cid not in pending_cids]
    for i in range(0, len(fresh), chunk):
        reqs = []
        for cid in fresh[i : i + chunk]:
            params = anthropic_params(spec, by_cid[cid][2])
            params.update(params.pop("extra_body", {}))
            reqs.append({"custom_id": cid, "params": params})
        b = client.messages.batches.create(requests=reqs)
        state[b.id] = {"status": "submitted", "cids": [r["custom_id"] for r in reqs]}
        write_json(state_path, state)
        log.info("[%s] submitted batch %s with %d requests", spec["name"], b.id, len(reqs))
    for bid, info in state.items():
        if info["status"] == "collected":
            continue
        while time.time() < deadline:
            b = client.messages.batches.retrieve(bid)
            if b.processing_status == "ended":
                break
            log.info("[%s] batch %s: %s", spec["name"], bid, b.request_counts)
            time.sleep(poll)
        else:
            log.warning("[%s] deadline reached; batch %s will be collected next run", spec["name"], bid)
            continue
        for r in client.messages.batches.results(bid):
            if r.custom_id not in by_cid:
                continue
            ep, view, _ = by_cid[r.custom_id]
            text = None
            if r.result.type == "succeeded":
                text = "".join(getattr(bl, "text", "") for bl in r.result.message.content if bl.type == "text")
            if text is not None:
                append_jsonl(run / "monitor.jsonl", _record(ep, spec, view, text))
        info["status"] = "collected"
        write_json(state_path, state)


def trim_to_budget(spec: dict, items: list, seed: int = 0) -> list:
    """Keep a random subset whose worst-case cost (chars/3 tokens in, max output) fits spec['max_usd']."""
    price, cap = spec.get("price"), spec.get("max_usd")
    if not price or cap is None:
        return items
    max_out = int(spec.get("max_tokens") or spec.get("params", {}).get("max_completion_tokens", 4000))
    scale = 0.5 if spec.get("batch") else 1.0
    items = list(items)
    random.Random(seed).shuffle(items)
    kept, usd = [], 0.0
    for it in items:
        c = (len(it[2]) / 3 * price["input"] + max_out * price["output"]) / 1e6 * scale
        if usd + c > float(cap):
            break
        usd += c
        kept.append(it)
    if len(kept) < len(items):
        log.warning("[%s] max_usd=$%.2f: keeping %d of %d calls", spec["name"], float(cap), len(kept), len(items))
    log.info("[%s] worst-case cost of this pass: $%.2f", spec["name"], usd)
    return kept


def main(cfg: dict, out_root: Path, max_hours: float) -> Path:
    run = run_dir(out_root, cfg["run_name"])
    protect(out_root)
    save_run_config(run, cfg)
    deadline = time.time() + max_hours * 3600
    tasks = {t.task_id: t for t in load_or_build_tasks(cfg, out_root)}
    eps = select_episodes(cfg, out_root)
    trs = load_transcripts(cfg, out_root, {e["episode_id"] for e in eps})
    done = {(r["episode_id"], r["monitor"], r["view"]) for r in read_jsonl(run / "monitor.jsonl") if not r.get("failed")}
    log.info("selected %d episodes (%d with transcripts)", len(eps), len(trs))
    for spec in cfg["monitors"]:
        items = []
        for ep in eps:
            tr, task = trs.get(ep["episode_id"]), tasks.get(ep["task_id"])
            if tr is None or task is None:
                continue
            for view in cfg.get("views", ["cot", "action"]):
                if (ep["episode_id"], spec["name"], view) not in done:
                    items.append((ep, view, render(tr, task.prompt, view, cfg.get("render_limits"))))
        items = trim_to_budget(spec, items, int(cfg.get("select", {}).get("seed", 0)))
        log.info("[%s] %d monitor calls to make", spec["name"], len(items))
        if not items:
            continue
        if spec["provider"] == "anthropic" and spec.get("batch"):
            run_anthropic_batches(spec, items, run, deadline, poll=int(spec.get("poll_s", 60)))
        elif spec["provider"] == "vllm":
            with ServerGroup(spec["serve"], run / "logs") as g:
                live = {**spec, "base_urls": g.base_urls, "api_key": g.api_key,
                        "served_name": spec["serve"].get("served_name", str(spec["serve"]["model"]))}
                asyncio.run(run_async(live, items, run, deadline))
        else:
            asyncio.run(run_async(spec, items, run, deadline))
    return run
