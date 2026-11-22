"""Validated, provider-neutral contracts for the Agent Manager foundation."""

import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional, Tuple


_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_SECRET_KEYS = {"api_key", "apikey", "access_token", "auth_token", "client_secret", "credential", "credentials", "password", "secret", "token"}


def validate_id(value, name="id"):
    if not isinstance(value, str) or not _ID.match(value):
        raise ValueError("{} must be 1-128 safe identifier characters.".format(name))
    return value


def validate_text(value, name, allow_empty=False):
    if not isinstance(value, str) or (not allow_empty and not value.strip()):
        raise ValueError("{} must be non-empty text.".format(name))
    return value


def validate_timestamp(value, name):
    validate_text(value, name)
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        raise ValueError("{} must be an ISO-8601 timestamp.".format(name))
    if parsed.tzinfo is None:
        raise ValueError("{} must include a timezone.".format(name))
    return value


def validate_json_value(value, name="metadata"):
    """Allow JSON-safe metadata while rejecting credential-shaped keys."""
    def check(item, path):
        if item is None or isinstance(item, (str, bool, int, float)):
            return
        if isinstance(item, list):
            for index, child in enumerate(item):
                check(child, "{}[{}]".format(path, index))
            return
        if isinstance(item, dict):
            for key, child in item.items():
                if not isinstance(key, str):
                    raise ValueError("{} keys must be strings.".format(path))
                normalized = re.sub(r"[^a-z]", "", key.lower())
                if normalized in _SECRET_KEYS or any(part in normalized for part in ("apikey", "password", "secret", "credential", "accesstoken")):
                    raise ValueError("{} cannot contain credential fields.".format(path))
                check(child, "{}.{}".format(path, key))
            return
        raise ValueError("{} must contain JSON-compatible values.".format(path))

    check(value, name)


def _string_tuple(values, name):
    if not isinstance(values, (tuple, list)):
        raise ValueError("{} must be a list of non-empty strings.".format(name))
    result = tuple(validate_text(item, name) for item in values)
    if len(result) != len(set(result)):
        raise ValueError("{} values must be unique.".format(name))
    return result


@dataclass(frozen=True)
class AgentDefinition:
    agent_id: str
    name: str
    description: str
    instructions: str
    model_preferences: Tuple[str, ...] = ()
    skill_ids: Tuple[str, ...] = ()
    tool_ids: Tuple[str, ...] = ()
    parent_agent_id: Optional[str] = None
    max_delegation_depth: int = 0
    enabled: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        validate_id(self.agent_id, "agent_id")
        validate_text(self.name, "name")
        validate_text(self.description, "description", allow_empty=True)
        validate_text(self.instructions, "instructions")
        if self.parent_agent_id is not None:
            validate_id(self.parent_agent_id, "parent_agent_id")
        for name in ("model_preferences", "skill_ids", "tool_ids"):
            object.__setattr__(self, name, _string_tuple(getattr(self, name), name))
        if not isinstance(self.max_delegation_depth, int) or isinstance(self.max_delegation_depth, bool) or self.max_delegation_depth < 0:
            raise ValueError("max_delegation_depth must be a nonnegative integer.")
        if not isinstance(self.enabled, bool):
            raise ValueError("enabled must be a boolean.")
        validate_json_value(self.metadata)


@dataclass(frozen=True)
class AgentRun:
    run_id: str
    agent_id: str
    task: str
    status: str
    started_at: Optional[str]
    completed_at: Optional[str] = None
    parent_run_id: Optional[str] = None
    result_summary: str = ""
    error_summary: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: Optional[str] = None

    def __post_init__(self):
        validate_id(self.run_id, "run_id")
        validate_id(self.agent_id, "agent_id")
        if self.parent_run_id is not None:
            validate_id(self.parent_run_id, "parent_run_id")
        validate_text(self.task, "task")
        # Keep the original persisted states readable for compatibility while
        # the single-agent runtime uses the explicit Phase 5 lifecycle names.
        if self.status not in {
            "pending", "running", "completed", "failed", "cancelled",
            "queued", "waiting_approval", "succeeded",
        }:
            raise ValueError("status is not a supported agent run status.")
        if self.started_at is not None:
            validate_timestamp(self.started_at, "started_at")
        if self.created_at is not None:
            validate_timestamp(self.created_at, "created_at")
        if self.started_at is None and (self.status not in {"pending", "cancelled"} or self.created_at is None):
            raise ValueError("started_at may be empty only for a created pending or cancelled run.")
        if self.completed_at is not None:
            validate_timestamp(self.completed_at, "completed_at")
        validate_text(self.result_summary, "result_summary", allow_empty=True)
        validate_text(self.error_summary, "error_summary", allow_empty=True)
        validate_json_value(self.metadata)


@dataclass(frozen=True)
class SkillDefinition:
    skill_id: str
    name: str
    description: str
    version: str
    instructions: str
    source: str = "local"
    enabled: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        validate_id(self.skill_id, "skill_id")
        for key in ("name", "version", "instructions", "source"):
            validate_text(getattr(self, key), key)
        validate_text(self.description, "description", allow_empty=True)
        if not isinstance(self.enabled, bool):
            raise ValueError("enabled must be a boolean.")
        validate_json_value(self.metadata)


@dataclass(frozen=True)
class ToolDefinition:
    tool_id: str
    name: str
    description: str
    input_schema: Dict[str, Any]
    risk_level: str = "low"
    requires_approval: bool = False
    enabled: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        validate_id(self.tool_id, "tool_id")
        validate_text(self.name, "name")
        validate_text(self.description, "description", allow_empty=True)
        if self.risk_level not in {"low", "medium", "high", "critical"}:
            raise ValueError("risk_level must be low, medium, high, or critical.")
        if self.risk_level in {"high", "critical"} and not self.requires_approval:
            raise ValueError("High-risk and critical tools must require approval.")
        if not isinstance(self.input_schema, dict):
            raise ValueError("input_schema must be a JSON object.")
        validate_json_value(self.input_schema, "input_schema")
        validate_json_value(self.metadata)
        if not isinstance(self.requires_approval, bool) or not isinstance(self.enabled, bool):
            raise ValueError("requires_approval and enabled must be booleans.")


@dataclass(frozen=True)
class Approval:
    approval_id: str
    run_id: str
    action: str
    action_digest: str
    status: str
    requested_at: str
    expires_at: str
    decided_at: Optional[str] = None
    agent_id: Optional[str] = None
    tool_id: Optional[str] = None
    arguments_digest: Optional[str] = None
    rationale: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        for key in ("approval_id", "run_id"):
            validate_id(getattr(self, key), key)
        if self.agent_id is not None:
            validate_id(self.agent_id, "agent_id")
        if self.tool_id is not None:
            validate_id(self.tool_id, "tool_id")
        binding_values = (self.agent_id, self.tool_id, self.arguments_digest)
        if any(value is not None for value in binding_values) and any(value is None for value in binding_values):
            raise ValueError("Approval agent, tool, and argument digest bindings must be supplied together.")
        if self.arguments_digest is not None and not re.match(r"^[a-f0-9]{64}$", self.arguments_digest):
            raise ValueError("arguments_digest must be a lowercase SHA-256 hex digest.")
        if self.arguments_digest is not None and not re.match(r"^[a-f0-9]{64}$", self.action_digest):
            raise ValueError("Bound approval action_digest must be a lowercase SHA-256 hex digest.")
        for key in ("action", "action_digest"):
            validate_text(getattr(self, key), key)
        if self.status not in {"pending", "approved", "rejected", "expired", "cancelled"}:
            raise ValueError("status is not a supported approval status.")
        for key in ("requested_at", "expires_at"):
            validate_timestamp(getattr(self, key), key)
        requested_at = datetime.fromisoformat(self.requested_at.replace("Z", "+00:00"))
        expires_at = datetime.fromisoformat(self.expires_at.replace("Z", "+00:00"))
        if expires_at <= requested_at:
            raise ValueError("expires_at must follow requested_at.")
        if self.decided_at is not None:
            validate_timestamp(self.decided_at, "decided_at")
        validate_text(self.rationale, "rationale", allow_empty=True)
        validate_json_value(self.metadata)


@dataclass(frozen=True)
class WorkflowDefinition:
    workflow_id: str
    name: str
    description: str
    steps: Tuple[Dict[str, Any], ...]
    version: int = 1
    enabled: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        validate_id(self.workflow_id, "workflow_id")
        validate_text(self.name, "name")
        validate_text(self.description, "description", allow_empty=True)
        if not isinstance(self.version, int) or isinstance(self.version, bool) or self.version < 1:
            raise ValueError("version must be a positive integer.")
        if not isinstance(self.steps, (tuple, list)) or not self.steps:
            raise ValueError("steps must contain at least one step.")
        steps = tuple(dict(step) if isinstance(step, dict) else step for step in self.steps)
        if any(not isinstance(step, dict) for step in steps):
            raise ValueError("Each workflow step must be an object.")
        step_ids = []
        for step in steps:
            validate_id(step.get("step_id"), "step_id")
            validate_text(step.get("agent_id"), "agent_id")
            step_ids.append(step["step_id"])
            validate_json_value(step)
        if len(step_ids) != len(set(step_ids)):
            raise ValueError("workflow step IDs must be unique.")
        object.__setattr__(self, "steps", steps)
        if not isinstance(self.enabled, bool):
            raise ValueError("enabled must be a boolean.")
        validate_json_value(self.metadata)
