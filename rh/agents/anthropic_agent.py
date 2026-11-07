"""Anthropic Messages API agent (Claude) with adaptive thinking and summarized display."""
from rh.agents.base import Agent, AgentError, ToolCall, Turn


class AnthropicAgent(Agent):
    def __init__(self, client, model: str, max_tokens: int = 16000, thinking: dict | None = None,
                 effort: str | None = None, label: str = "anthropic"):
        self.client, self.model, self.max_tokens = client, model, max_tokens
        self.thinking, self.effort, self.label = thinking, effort, label
        self.system = ""
        self.msgs: list = []
        self.tools: list = []

    def start(self, system, user, tools):
        self.system = system
        self.msgs = [{"role": "user", "content": user}]
        self.tools = [{"name": t["name"], "description": t["description"], "input_schema": t["parameters"]} for t in tools]

    async def act(self) -> Turn:
        import anthropic

        kwargs = dict(model=self.model, max_tokens=self.max_tokens, system=self.system, messages=self.msgs, tools=self.tools)
        extra = {}
        if self.thinking:
            extra["thinking"] = self.thinking
        if self.effort:
            extra["output_config"] = {"effort": self.effort}
        if extra:
            kwargs["extra_body"] = extra
        try:
            resp = await self.client.messages.create(**kwargs)
        except anthropic.BadRequestError as e:
            raise AgentError(f"BadRequestError: {str(e)[:500]}") from e
        blocks = [b.model_dump(exclude_none=True) for b in resp.content]
        text, reasoning, calls = [], [], []
        for b in blocks:
            t = b.get("type")
            if t == "text":
                text.append(b.get("text", ""))
            elif t == "thinking":
                reasoning.append(b.get("thinking", ""))
            elif t == "redacted_thinking":
                reasoning.append("[redacted thinking]")
            elif t == "tool_use":
                calls.append(ToolCall(b["id"], b["name"], b.get("input") or {}))
        self.msgs.append({"role": "assistant", "content": blocks})
        u = resp.usage.model_dump() if resp.usage is not None else {}
        details = u.get("output_tokens_details") or {}
        usage = {
            "input_tokens": (u.get("input_tokens") or 0) + (u.get("cache_read_input_tokens") or 0)
            + (u.get("cache_creation_input_tokens") or 0),
            "output_tokens": u.get("output_tokens") or 0,
            "reasoning_tokens": details.get("thinking_tokens") or 0,
        }
        stop = resp.stop_reason or ""
        if stop == "refusal" and not calls:
            raise AgentError("refusal")
        return Turn(text="\n".join(text).strip(), reasoning="\n".join(r for r in reasoning if r).strip(),
                    tool_calls=calls, usage=usage, stop=stop)

    def add_tool_results(self, results):
        content = [{"type": "tool_result", "tool_use_id": tc.id, "content": out} for tc, out in results]
        self.msgs.append({"role": "user", "content": content})

    def add_user(self, text):
        last = self.msgs[-1] if self.msgs else None
        if last and last["role"] == "user" and isinstance(last["content"], list):
            last["content"].append({"type": "text", "text": text})
        else:
            self.msgs.append({"role": "user", "content": text})

    def messages(self):
        return [{"role": "system", "content": self.system}] + self.msgs
