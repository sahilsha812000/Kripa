"""Small, explicit, read-only tools compiled into Kripa."""

from datetime import datetime, timezone

from kripa.agent_manager.contracts import ToolDefinition


UTC_TIME_TOOL_ID = "utc_time"


def utc_time_definition():
    return ToolDefinition(
        tool_id=UTC_TIME_TOOL_ID,
        name="Current UTC time",
        description="Return the current UTC timestamp. This tool only reads the system clock.",
        input_schema={
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
        risk_level="low",
        requires_approval=False,
        enabled=True,
    )


class UtcTimeTool:
    """Read the system clock; it has no filesystem or network access."""

    def __init__(self, clock=None):
        self.clock = clock or (lambda: datetime.now(timezone.utc))

    def can_execute(self, definition):
        return definition == utc_time_definition()

    def execute(self, arguments):
        if arguments != {}:
            raise ValueError("UTC time tool accepts no arguments.")
        value = self.clock()
        if not isinstance(value, datetime) or value.tzinfo is None:
            raise ValueError("UTC time source returned an invalid timestamp.")
        return value.astimezone(timezone.utc).isoformat()
