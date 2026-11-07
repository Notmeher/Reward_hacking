from abc import ABC, abstractmethod
from dataclasses import dataclass, field


class AgentError(Exception):
    """Unrecoverable provider error for this episode (bad request, refusal, context overflow)."""


@dataclass
class ToolCall:
    id: str
    name: str
    args: dict


@dataclass
class Turn:
    text: str = ""
    reasoning: str = ""
    tool_calls: list = field(default_factory=list)
    usage: dict = field(default_factory=dict)
    stop: str = ""


class Agent(ABC):
    label: str = "agent"

    @abstractmethod
    def start(self, system: str, user: str, tools: list[dict]) -> None: ...

    @abstractmethod
    async def act(self) -> Turn: ...

    @abstractmethod
    def add_tool_results(self, results: list[tuple[ToolCall, str]]) -> None: ...

    @abstractmethod
    def add_user(self, text: str) -> None: ...

    @abstractmethod
    def messages(self) -> list: ...
