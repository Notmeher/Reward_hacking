"""Agent factory. Model specs come from YAML configs, e.g.

  {name: gpt5, provider: azure, deployment: gpt-5, params: {reasoning_effort: medium}}
  {name: sonnet55, provider: anthropic, model: claude-sonnet-5-5, effort: medium,
   thinking: {type: adaptive, display: summarized}}
  {name: qwen3-4b, provider: vllm, base_urls: [...], served_name: ..., params: {...}}
  {name: honest, provider: scripted, behavior: honest}
"""
import asyncio
import itertools

from rh.agents.base import Agent, AgentError, ToolCall, Turn
from rh.config import env

_clients: dict = {}
_rr: dict = {}


def _loop_key(*parts):
    return (id(asyncio.get_running_loop()), *parts)


def azure_client(api_version: str | None = None, endpoint: str | None = None):
    from openai import AsyncAzureOpenAI

    key = _loop_key("azure", api_version, endpoint)
    if key not in _clients:
        _clients[key] = AsyncAzureOpenAI(
            api_key=env("AZURE_OPENAI_API_KEY"),
            api_version=api_version or env("AZURE_OPENAI_API_VERSION"),
            azure_endpoint=endpoint or env("AZURE_OPENAI_ENDPOINT"),
            max_retries=8,
            timeout=900,
        )
    return _clients[key]


def anthropic_client():
    from anthropic import AsyncAnthropic

    key = _loop_key("anthropic")
    if key not in _clients:
        _clients[key] = AsyncAnthropic(api_key=env("ANTHROPIC_API_KEY"), max_retries=8, timeout=900)
    return _clients[key]


def openai_compat_client(base_url: str, api_key: str = "EMPTY"):
    from openai import AsyncOpenAI

    key = _loop_key("compat", base_url)
    if key not in _clients:
        _clients[key] = AsyncOpenAI(base_url=base_url, api_key=api_key, max_retries=3, timeout=3600)
    return _clients[key]


def make_agent(spec: dict, task=None) -> Agent:
    p = spec["provider"]
    label = spec.get("name", p)
    if p == "azure":
        return OpenAIChatAgent(
            azure_client(spec.get("api_version"), spec.get("endpoint")),
            spec.get("deployment") or env("AZURE_OPENAI_DEPLOYMENT"),
            params=spec.get("params"),
            label=label,
        )
    if p == "anthropic":
        return AnthropicAgent(
            anthropic_client(), spec["model"], max_tokens=int(spec.get("max_tokens", 16000)),
            thinking=spec.get("thinking"), effort=spec.get("effort"), label=label,
        )
    if p == "vllm":
        urls = spec["base_urls"]
        cyc = _rr.setdefault(label, itertools.cycle(range(len(urls))))
        url = urls[next(cyc)]
        return OpenAIChatAgent(
            openai_compat_client(url, spec.get("api_key", "EMPTY")), spec.get("served_name") or spec["model"],
            params=spec.get("params"), keep_reasoning=True, label=label,
        )
    if p == "scripted":
        return ScriptedAgent(spec["behavior"], task, label=label)
    raise ValueError(f"unknown provider {p!r}")


from rh.agents.anthropic_agent import AnthropicAgent  # noqa: E402
from rh.agents.openai_chat import OpenAIChatAgent  # noqa: E402
from rh.agents.scripted import ScriptedAgent  # noqa: E402

__all__ = ["Agent", "AgentError", "ToolCall", "Turn", "make_agent"]
