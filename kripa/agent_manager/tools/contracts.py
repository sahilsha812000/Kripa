"""Typed decisions and safe audit records for tool authorization."""

import re
from dataclasses import dataclass
from enum import Enum

from kripa.agent_manager.contracts import validate_id, validate_timestamp, validate_text


_SAFE_EVENT_CODE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_OPAQUE_REFERENCE = re.compile(r"^[a-f0-9]{24}$")


class PolicyOutcome(str, Enum):
    ALLOW = "ALLOW"
    DENY = "DENY"
    REQUIRES_APPROVAL = "REQUIRES_APPROVAL"


class PolicyReason(str, Enum):
    AUTHORIZED = "authorized"
    UNKNOWN_AGENT = "unknown_agent"
    DISABLED_AGENT = "disabled_agent"
    UNKNOWN_TOOL = "unknown_tool"
    DISABLED_TOOL = "disabled_tool"
    GRANT_MISSING = "grant_missing"
    INVALID_ARGUMENTS = "invalid_arguments"
    APPROVAL_REQUIRED = "approval_required"
    APPROVAL_UNKNOWN = "approval_unknown"
    APPROVAL_MISMATCH = "approval_mismatch"
    APPROVAL_PENDING = "approval_pending"
    APPROVAL_REJECTED = "approval_rejected"
    APPROVAL_EXPIRED = "approval_expired"
    APPROVAL_NOT_PENDING = "approval_not_pending"
    AUDIT_UNAVAILABLE = "audit_unavailable"
    INVALID_REQUEST = "invalid_request"


class ToolRiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ToolExecutionOutcome(str, Enum):
    AUTHORIZED = "AUTHORIZED"
    EXECUTED = "EXECUTED"
    DENIED = "DENIED"
    REQUIRES_APPROVAL = "REQUIRES_APPROVAL"
    FAILED = "FAILED"


@dataclass(frozen=True)
class PolicyDecision:
    outcome: PolicyOutcome
    reason: PolicyReason
    risk_level: ToolRiskLevel
    action_digest: str = ""
    arguments_digest: str = ""
    approval_id: str = ""


@dataclass(frozen=True)
class ToolAuditRecord:
    """Audit event contains hashes and classifications, never raw arguments."""

    event_id: str
    recorded_at: str
    outcome: PolicyOutcome
    reason: PolicyReason
    risk_level: ToolRiskLevel
    agent_ref: str
    run_ref: str
    tool_ref: str
    action_digest: str = ""
    approval_id: str = ""

    def __post_init__(self):
        validate_id(self.event_id, "event_id")
        validate_timestamp(self.recorded_at, "recorded_at")
        for name in ("outcome", "reason", "risk_level"):
            value = getattr(self, name)
            enum_type = {"outcome": PolicyOutcome, "reason": PolicyReason,
                         "risk_level": ToolRiskLevel}[name]
            if not isinstance(value, enum_type):
                object.__setattr__(self, name, enum_type(value))
        for name in ("agent_ref", "run_ref", "tool_ref"):
            validate_text(getattr(self, name), name)
        for name in ("action_digest", "approval_id"):
            value = getattr(self, name)
            if not isinstance(value, str):
                raise ValueError("{} must be text.".format(name))
        if self.action_digest and not re.match(r"^[a-f0-9]{64}$", self.action_digest):
            raise ValueError("action_digest must be a lowercase SHA-256 digest.")
        if self.approval_id:
            validate_id(self.approval_id, "approval_id")


@dataclass(frozen=True)
class ToolApprovalAuditRecord:
    """Redacted approval transition event; principal and object IDs are hashed."""

    event_id: str
    recorded_at: str
    transition: str
    agent_ref: str
    run_ref: str
    tool_ref: str
    approval_id: str
    action_digest: str
    actor_ref: str

    def __post_init__(self):
        validate_id(self.event_id, "event_id")
        validate_timestamp(self.recorded_at, "recorded_at")
        if self.transition not in {"approved", "rejected", "expired"}:
            raise ValueError("Unsupported approval audit transition.")
        for name in ("agent_ref", "run_ref", "tool_ref", "actor_ref"):
            validate_text(getattr(self, name), name)
        validate_id(self.approval_id, "approval_id")
        if not re.match(r"^[a-f0-9]{64}$", self.action_digest):
            raise ValueError("action_digest must be a lowercase SHA-256 digest.")


@dataclass(frozen=True)
class ToolExecutionResult:
    """Safe result returned to the application; arguments are never included."""

    outcome: ToolExecutionOutcome
    reason: str
    risk_level: ToolRiskLevel
    action_digest: str = ""
    approval_id: str = ""
    result: object = None
    arguments_digest: str = ""

    def __post_init__(self):
        if not isinstance(self.outcome, ToolExecutionOutcome):
            object.__setattr__(self, "outcome", ToolExecutionOutcome(self.outcome))
        if not isinstance(self.risk_level, ToolRiskLevel):
            object.__setattr__(self, "risk_level", ToolRiskLevel(self.risk_level))
        if not isinstance(self.reason, str) or not _SAFE_EVENT_CODE.match(self.reason):
            raise ValueError("reason must be a safe event code.")
        if self.action_digest and not re.match(r"^[a-f0-9]{64}$", self.action_digest):
            raise ValueError("action_digest must be a lowercase SHA-256 digest.")
        if self.arguments_digest and not re.match(r"^[a-f0-9]{64}$", self.arguments_digest):
            raise ValueError("arguments_digest must be a lowercase SHA-256 digest.")
        if self.approval_id:
            validate_id(self.approval_id, "approval_id")


@dataclass(frozen=True)
class ToolExecutionAuditRecord:
    """Execution outcome with opaque IDs and digests, never arguments/results."""

    event_id: str
    recorded_at: str
    outcome: ToolExecutionOutcome
    reason: str
    risk_level: ToolRiskLevel
    agent_ref: str
    run_ref: str
    tool_ref: str
    action_digest: str = ""
    approval_ref: str = ""
    result_digest: str = ""
    error_code: str = ""

    def __post_init__(self):
        validate_id(self.event_id, "event_id")
        validate_timestamp(self.recorded_at, "recorded_at")
        if not isinstance(self.outcome, ToolExecutionOutcome):
            object.__setattr__(self, "outcome", ToolExecutionOutcome(self.outcome))
        if not isinstance(self.risk_level, ToolRiskLevel):
            object.__setattr__(self, "risk_level", ToolRiskLevel(self.risk_level))
        if not isinstance(self.reason, str) or not _SAFE_EVENT_CODE.match(self.reason):
            raise ValueError("reason must be a safe event code.")
        for name in ("agent_ref", "run_ref", "tool_ref"):
            if not isinstance(getattr(self, name), str) or not _OPAQUE_REFERENCE.match(getattr(self, name)):
                raise ValueError("{} must be an opaque SHA-256 reference.".format(name))
        for name in ("action_digest", "approval_ref", "result_digest", "error_code"):
            if not isinstance(getattr(self, name), str):
                raise ValueError("{} must be text.".format(name))
        if self.approval_ref and not _OPAQUE_REFERENCE.match(self.approval_ref):
            raise ValueError("approval_ref must be an opaque SHA-256 reference.")
        if self.error_code and not _SAFE_EVENT_CODE.match(self.error_code):
            raise ValueError("error_code must be a safe event code.")
        for name in ("action_digest", "result_digest"):
            value = getattr(self, name)
            if value and not re.match(r"^[a-f0-9]{64}$", value):
                raise ValueError("{} must be a lowercase SHA-256 digest.".format(name))
