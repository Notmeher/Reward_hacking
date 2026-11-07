"""Connectivity checks: one tiny tool-calling request per configured API model/monitor/judge."""
import asyncio
import logging

from rh.agents import make_agent
from rh.env.tools import SUBMIT

log = logging.getLogger(__name__)


async def _check(spec: dict) -> tuple[str, bool, str]:
    try:
        agent = make_agent(spec)
        agent.start("You are a test harness. Call the submit tool.", "Call `submit` with summary 'ok'.", [SUBMIT])
        turn = await asyncio.wait_for(agent.act(), 300)
        called = any(c.name == "submit" for c in turn.tool_calls)
        info = f"tool_call={'yes' if called else 'no'} out_tokens={turn.usage.get('output_tokens')} " \
               f"reasoning_chars={len(turn.reasoning)}"
        return spec["name"], True, info
    except Exception as e:  # noqa: BLE001 - report every failure
        return spec.get("name", "?"), False, f"{type(e).__name__}: {str(e)[:400]}"


def main(cfg: dict) -> bool:
    specs = [s for s in cfg.get("models", []) + cfg.get("monitors", []) + ([cfg["judge"]] if cfg.get("judge") else [])
             if s.get("provider") in ("azure", "anthropic")]
    if not specs:
        log.info("no API models in this config")
        return True

    async def run_all():
        return await asyncio.gather(*(_check(s) for s in specs))

    ok = True
    for name, good, info in asyncio.run(run_all()):
        log.info("%-24s %s  %s", name, "OK  " if good else "FAIL", info)
        ok &= good
    return ok
