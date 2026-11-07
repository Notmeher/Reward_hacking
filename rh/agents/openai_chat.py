"""Chat-completions agent for Azure OpenAI (gpt-5) and OpenAI-compatible vLLM servers."""
import json
import re
import uuid

from rh.agents.base import Agent, AgentError, ToolCall, Turn

HERMES = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)


def _json_args(raw) -> dict:
    if isinstance(raw, dict):
        return raw
    try:
        val = json.loads(raw or "{}")
        return val if isinstance(val, dict) else {"_raw": raw}
    except (json.JSONDecodeError, TypeError):
        return {"_raw": raw}


def _reasoning(msg) -> str:
    for attr in ("reasoning_content", "reasoning"):
        val = getattr(msg, attr, None)
        if isinstance(val, str) and val:
            return val
    extra = getattr(msg, "model_extra", None) or {}
    for key in ("reasoning_content", "reasoning"):
        if isinstance(extra.get(key), str) and extra[key]:
            return extra[key]
    return ""


def split_think(text: str) -> tuple[str, str]:
    """Split inline <think>...</think> when the server has no reasoning parser."""
    if "</think>" not in text:
        return "", text
    head, _, tail = text.partition("</think>")
    return head.replace("<think>", "").strip(), tail.strip()


def parse_hermes(text: str) -> list[ToolCall]:
    calls = []
    for m in HERMES.finditer(text):
        try:
            obj = json.loads(m.group(1))
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and "name" in obj:
            calls.append(ToolCall(f"call_{uuid.uuid4().hex[:12]}", obj["name"], _json_args(obj.get("arguments", {}))))
    return calls


class OpenAIChatAgent(Agent):
    def __init__(self, client, model: str, params: dict | None = None, keep_reasoning: bool = False, label: str = "openai"):
        self.client, self.model, self.params = client, model, dict(params or {})
        self.keep_reasoning, self.label = keep_reasoning, label
        self.msgs: list = []
        self.tools: list = []

    def start(self, system, user, tools):
        self.msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        self.tools = [{"type": "function", "function": t} for t in tools]

    async def act(self) -> Turn:
        import openai

        try:
            resp = await self.client.chat.completions.create(
                model=self.model, messages=self.msgs, tools=self.tools, tool_choice="auto", **self.params
            )
        except (openai.BadRequestError, openai.UnprocessableEntityError) as e:
            raise AgentError(f"{type(e).__name__}: {str(e)[:500]}") from e
        if not resp.choices:
            raise AgentError("empty response (no choices)")
        choice = resp.choices[0]
        msg = choice.message
        text = msg.content or ""
        reasoning = _reasoning(msg)
        if not reasoning:
            reasoning, text = split_think(text)
        calls = [ToolCall(tc.id, tc.function.name, _json_args(tc.function.arguments)) for tc in (msg.tool_calls or [])]
        if not calls and "<tool_call>" in text:
            calls = parse_hermes(text)
            text = HERMES.sub("", text).strip()
        # vLLM/HF chat templates render None literally, so open models get "" instead
        assistant: dict = {"role": "assistant", "content": text or ("" if self.keep_reasoning else None)}
        if calls:
            assistant["tool_calls"] = [
                {"id": c.id, "type": "function", "function": {"name": c.name, "arguments": json.dumps(c.args)}}
                for c in calls
            ]
        elif not text:
            assistant["content"] = ""
        if self.keep_reasoning and reasoning:
            assistant["reasoning_content"] = reasoning
        self.msgs.append(assistant)
        u = resp.usage
        details = getattr(u, "completion_tokens_details", None) if u else None
        usage = {
            "input_tokens": getattr(u, "prompt_tokens", 0) or 0,
            "output_tokens": getattr(u, "completion_tokens", 0) or 0,
            "reasoning_tokens": (getattr(details, "reasoning_tokens", 0) or 0) if details else 0,
        }
        return Turn(text=text, reasoning=reasoning, tool_calls=calls, usage=usage, stop=choice.finish_reason or "")

    def add_tool_results(self, results):
        for tc, out in results:
            self.msgs.append({"role": "tool", "tool_call_id": tc.id, "content": out})

    def add_user(self, text):
        self.msgs.append({"role": "user", "content": text})

    def messages(self):
        return self.msgs
