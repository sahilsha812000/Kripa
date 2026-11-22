"""Validated lifecycle operations for locally stored agent definitions."""

from dataclasses import replace

from kripa.agent_manager.contracts import AgentDefinition
from kripa.agent_manager.store import AgentManagerSnapshot, AgentManagerStore


class AgentAlreadyExistsError(ValueError):
    """Raised when creating an agent with an ID already in use."""


class AgentNotFoundError(LookupError):
    """Raised when an operation references an unknown agent ID."""


class AgentRegistry:
    """Create, inspect, update, and retire definitions without executing them."""

    _UPDATABLE_FIELDS = frozenset({
        "name", "description", "instructions", "model_preferences", "skill_ids",
        "tool_ids", "parent_agent_id", "max_delegation_depth", "metadata",
    })

    def __init__(self, store=None):
        self.store = store or AgentManagerStore()

    def create_agent(self, definition):
        if not isinstance(definition, AgentDefinition):
            raise TypeError("definition must be an AgentDefinition.")
        snapshot = self._snapshot()
        if any(agent.agent_id == definition.agent_id for agent in snapshot.agents):
            raise AgentAlreadyExistsError("An agent with ID '{}' already exists.".format(definition.agent_id))
        # Reconstruct through the contract to ensure persisted values are validated
        # at this boundary too, even if callers pass a subclass.
        validated = AgentDefinition(**_agent_values(definition))
        self._save_agents(snapshot, (*snapshot.agents, validated))
        return validated

    def list_agents(self, include_retired=True):
        if not isinstance(include_retired, bool):
            raise ValueError("include_retired must be a boolean.")
        agents = self._snapshot().agents
        result = agents if include_retired else tuple(item for item in agents if item.enabled)
        return tuple(sorted(result, key=lambda item: item.agent_id.lower()))

    def get_agent(self, agent_id):
        snapshot = self._snapshot()
        for agent in snapshot.agents:
            if agent.agent_id == agent_id:
                return agent
        raise AgentNotFoundError("Agent '{}' was not found.".format(agent_id))

    def update_agent(self, agent_id, **changes):
        unknown = set(changes).difference(self._UPDATABLE_FIELDS)
        if unknown:
            raise ValueError("Unsupported agent field(s): {}.".format(", ".join(sorted(unknown))))
        if not changes:
            raise ValueError("At least one agent field must be provided for update.")
        snapshot = self._snapshot()
        current = self._find(snapshot, agent_id)
        updated = replace(current, **changes)
        updated = AgentDefinition(**_agent_values(updated))
        self._save_agents(snapshot, tuple(updated if item.agent_id == agent_id else item for item in snapshot.agents))
        return updated

    def retire_agent(self, agent_id):
        snapshot = self._snapshot()
        current = self._find(snapshot, agent_id)
        if not current.enabled:
            return current
        retired = replace(current, enabled=False)
        self._save_agents(snapshot, tuple(retired if item.agent_id == agent_id else item for item in snapshot.agents))
        return retired

    def _snapshot(self):
        snapshot = self.store.load()
        if snapshot.issues:
            raise ValueError("Agent Manager storage is incomplete or unsupported; no changes were made.")
        ids = [agent.agent_id for agent in snapshot.agents]
        if len(ids) != len(set(ids)):
            raise ValueError("Agent Manager storage contains duplicate agent IDs; no changes were made.")
        return snapshot

    @staticmethod
    def _find(snapshot, agent_id):
        for agent in snapshot.agents:
            if agent.agent_id == agent_id:
                return agent
        raise AgentNotFoundError("Agent '{}' was not found.".format(agent_id))

    def _save_agents(self, snapshot, agents):
        self.store.save(AgentManagerSnapshot(
            agents=tuple(agents), runs=snapshot.runs, skills=snapshot.skills,
            tools=snapshot.tools, approvals=snapshot.approvals,
            workflows=snapshot.workflows,
        ))


def _agent_values(agent):
    return {name: getattr(agent, name) for name in AgentDefinition.__dataclass_fields__}
