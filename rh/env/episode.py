"""One agent episode: prompt -> tool loop under a budget -> grade -> detectors."""
import asyncio
import time
import uuid

from rh.agents.base import AgentError
from rh.env.prompts import Condition, countdown, system_prompt, user_prompt
from rh.env.tools import TERMINAL, tool_set

TRUNC = 4000
NUDGE = "Please continue by calling one of the tools. Call `submit` when you are finished."


def _trunc(v, n=TRUNC):
    if isinstance(v, str) and len(v) > n:
        return v[:n] + f"...[{len(v) - n} chars]"
    if isinstance(v, dict):
        return {k: _trunc(x, n) for k, x in v.items()}
    return v


async def run_episode(env, agent, cond: Condition, *, cmd_timeout: float = 20.0, max_nudges: int = 1,
                      act_timeout: float = 900.0, meta: dict | None = None) -> tuple[dict, dict]:
    """Returns (summary_record, transcript_record)."""
    task = env.task
    t0 = time.monotonic()
    episode_id = uuid.uuid4().hex
    events: list[dict] = []
    status, error = "", ""
    calls = nudges = 0
    usage = {"input_tokens": 0, "output_tokens": 0, "reasoning_tokens": 0}
    await env.setup()
    try:
        agent.start(system_prompt(env.env_text(), env.grading_text(), cond), user_prompt(task.prompt),
                    tool_set(cmd_timeout, cond.abstain))
        turn_i = 0
        while not status:
            if calls >= cond.budget_calls:
                status = "budget_exhausted"
                break
            if cond.token_budget and usage["output_tokens"] >= cond.token_budget:
                status = "token_budget_exhausted"
                break
            try:
                turn = await asyncio.wait_for(agent.act(), act_timeout)
            except AgentError as e:
                status, error = "agent_error", str(e)
                break
            except asyncio.TimeoutError:
                status, error = "agent_timeout", f"no response in {act_timeout}s"
                break
            for k in usage:
                usage[k] += int(turn.usage.get(k, 0) or 0)
            events.append({"type": "turn", "turn": turn_i, "text": _trunc(turn.text), "reasoning": turn.reasoning,
                           "usage": turn.usage, "stop": turn.stop, "n_calls": len(turn.tool_calls)})
            turn_i += 1
            if not turn.tool_calls:
                if nudges >= max_nudges:
                    status = "no_action"
                    break
                nudges += 1
                calls += 1  # a nudge costs budget so a silent model cannot loop forever
                agent.add_user(NUDGE + countdown(cond, cond.budget_calls - calls))
                continue
            results = []
            for tc in turn.tool_calls:
                if status:
                    results.append((tc, "[not executed: episode ended]"))
                    continue
                if calls >= cond.budget_calls:
                    status = "budget_exhausted"
                    results.append((tc, "[not executed: tool-call budget exhausted]"))
                    continue
                calls += 1
                if tc.name in TERMINAL:
                    status = "submitted" if tc.name == "submit" else "abstained"
                    out = "Submitted." if tc.name == "submit" else "Abstained."
                else:
                    out = await env.execute(tc.name, tc.args)
                    out += countdown(cond, cond.budget_calls - calls)
                events.append({"type": "tool", "turn": turn_i - 1, "name": tc.name, "args": _trunc(tc.args),
                               "output": _trunc(out)})
                results.append((tc, out))
            agent.add_tool_results(results)
        grade = await env.grade()
        flags = env.detect(events, grade)
    finally:
        await env.cleanup()
    summary = {
        "episode_id": episode_id,
        "model": agent.label,
        "task_id": task.task_id,
        "family": task.family,
        "difficulty": task.difficulty,
        "impossible": task.impossible,
        "split": task.split,
        **{f"cond_{k}": v for k, v in cond.to_dict().items()},
        "status": status,
        "error": error[:500],
        "calls_used": calls,
        "turns": sum(1 for e in events if e["type"] == "turn"),
        "nudges": nudges,
        **usage,
        "wall_s": round(time.monotonic() - t0, 2),
        **{f"g_{k}": v for k, v in grade.items()},
        **{f"d_{k}": v for k, v in flags.items()},
        "abstained": status == "abstained",
        "submitted": status == "submitted",
        **(meta or {}),
    }
    transcript = {
        "episode_id": episode_id,
        "model": agent.label,
        "task_id": task.task_id,
        "cond": cond.to_dict(),
        "events": events,
        "messages": agent.messages(),
        "final_files": env.final_files(),
        "status": status,
        **(meta or {}),
    }
    return summary, transcript
