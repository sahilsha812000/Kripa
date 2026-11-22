"""Definition-only registry; it never accepts or invokes callables."""

import json

from kripa.agent_manager.contracts import ToolDefinition
from kripa.agent_manager.store import AgentManagerStore
from kripa.agent_manager.tools.repositories import AgentManagerToolRepository
from kripa.agent_manager.tools.schema import validate_tool_schema


class ToolRegistry:
    def __init__(self, repository=None):
        self.repository = repository or AgentManagerToolRepository(AgentManagerStore())

    def register(self, definition):
        # Exact type prevents subclasses from smuggling executable attributes.
        if type(definition) is not ToolDefinition:
            raise TypeError("Only a ToolDefinition record can be registered.")
        _validate_definition(definition)
        try:
            self.repository.get_tool(definition.tool_id)
        except LookupError:
            pass
        else:
            raise ValueError("Tool ID is already registered.")
        stored = _clone_tool(definition)
        self.repository.save_tool(stored)
        return _clone_tool(stored)

    def inspect(self, tool_id):
        definition = self.repository.get_tool(tool_id)
        _validate_definition(definition)
        return _clone_tool(definition)

    def list_tools(self, include_disabled=True):
        if not isinstance(include_disabled, bool):
            raise ValueError("include_disabled must be a boolean.")
        tools = self.repository.list_tools()
        for definition in tools:
            _validate_definition(definition)
        return tuple(_clone_tool(item) for item in tools if include_disabled or item.enabled)


def _validate_definition(definition):
    if type(definition) is not ToolDefinition:
        raise ValueError("Tool repository returned an invalid definition.")
    validate_tool_schema(definition.input_schema)


def _clone_tool(definition):
    return ToolDefinition(
        tool_id=definition.tool_id,
        name=definition.name,
        description=definition.description,
        input_schema=json.loads(json.dumps(definition.input_schema, allow_nan=False)),
        risk_level=definition.risk_level,
        requires_approval=definition.requires_approval,
        enabled=definition.enabled,
        metadata=json.loads(json.dumps(definition.metadata, allow_nan=False)),
    )
