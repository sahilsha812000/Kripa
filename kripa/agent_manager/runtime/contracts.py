"""Provider-neutral contracts for one agent execution."""

from dataclasses import dataclass
from typing import Any, Optional, Protocol

from kripa.agent_manager.contracts import AgentRun, validate_id


class AgentProvider(Protocol):
    """Existing provider boundary used by configured and adaptive providers."""

    def reply(self, messages):
        """Return one response for the supplied conversation messages."""
        ...


class AgentRuntimeRepository(Protocol):
    """Persistence and definition lookup required by the single-agent runtime."""

    def get_agent(self, agent_id):
        ...

    def get_skill(self, skill_id):
        ...

    def create_run(self, run):
        ...

    def get_run(self, run_id):
        ...

    def save_run(self, run):
        ...


@dataclass(frozen=True)
class ToolRequest:
    """One strict model proposal; policy validates its arguments and grant."""

    tool_id: str
    arguments: Any

    def __post_init__(self):
        validate_id(self.tool_id, "tool_id")


@dataclass(frozen=True)
class AgentExecutionResult:
    """Persisted lifecycle record and the response for this invocation."""

    run: AgentRun
    response: Optional[str]
    approval_id: Optional[str] = None
    pending_tool_request: Optional[ToolRequest] = None
