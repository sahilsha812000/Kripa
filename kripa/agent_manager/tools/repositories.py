"""Repository interfaces and adapters for tool policy state."""

import hashlib
import json
import os
from typing import Protocol

from kripa.agent_manager.registry import AgentRegistry
from kripa.agent_manager.contracts import Approval, ToolDefinition
from kripa.agent_manager.store import AgentManagerStore
from kripa.agent_manager.tools.contracts import ToolApprovalAuditRecord, ToolAuditRecord


class ToolRepository(Protocol):
    def get_tool(self, tool_id): ...
    def list_tools(self): ...
    def save_tool(self, definition): ...


class AgentGrantRepository(Protocol):
    def get_agent(self, agent_id): ...
    def has_grant(self, agent_id, tool_id): ...
    def grant(self, agent_id, tool_id): ...
    def revoke(self, agent_id, tool_id): ...


class ApprovalRepository(Protocol):
    def get_approval(self, approval_id): ...
    def list_approvals(self): ...
    def save_approval(self, approval): ...


class AuditRepository(Protocol):
    def record(self, event): ...


class AgentManagerToolRepository:
    """Tool definitions stored in the existing AgentManagerStore collection."""

    def __init__(self, store=None):
        self.store = store or AgentManagerStore()

    def get_tool(self, tool_id):
        snapshot = self._snapshot()
        matches = [item for item in snapshot.tools if item.tool_id == tool_id]
        if len(matches) > 1:
            raise ValueError("Tool storage contains duplicate IDs.")
        if not matches:
            raise LookupError("Tool was not found.")
        return matches[0]

    def list_tools(self):
        return tuple(sorted(self._snapshot().tools, key=lambda item: item.tool_id.lower()))

    def save_tool(self, definition):
        if not isinstance(definition, ToolDefinition):
            raise TypeError("definition must be a ToolDefinition.")
        self.store.upsert("tools", definition)

    def _snapshot(self):
        snapshot = self.store.load()
        if snapshot.issues:
            raise ValueError("Agent Manager storage is incomplete or unsupported.")
        ids = [item.tool_id for item in snapshot.tools]
        if len(ids) != len(set(ids)):
            raise ValueError("Tool storage contains duplicate IDs.")
        return snapshot


class AgentManagerGrantRepository:
    """Use AgentDefinition.tool_ids as the explicit tool allowlist."""

    def __init__(self, registry=None):
        self.registry = registry or AgentRegistry()

    def get_agent(self, agent_id):
        return self.registry.get_agent(agent_id)

    def has_grant(self, agent_id, tool_id):
        agent = self.get_agent(agent_id)
        return tool_id in agent.tool_ids

    def grant(self, agent_id, tool_id):
        agent = self.get_agent(agent_id)
        if not agent.enabled:
            raise ValueError("Disabled agents cannot receive tool grants.")
        if tool_id in agent.tool_ids:
            return agent
        return self.registry.update_agent(agent_id, tool_ids=agent.tool_ids + (tool_id,))

    def revoke(self, agent_id, tool_id):
        agent = self.get_agent(agent_id)
        if tool_id not in agent.tool_ids:
            return agent
        return self.registry.update_agent(
            agent_id, tool_ids=tuple(item for item in agent.tool_ids if item != tool_id),
        )


class AgentManagerApprovalRepository:
    """Approval records stored in the existing AgentManagerStore collection."""

    def __init__(self, store=None):
        self.store = store or AgentManagerStore()

    def get_approval(self, approval_id):
        matches = [item for item in self._snapshot().approvals if item.approval_id == approval_id]
        if len(matches) > 1:
            raise ValueError("Approval storage contains duplicate IDs.")
        if not matches:
            raise LookupError("Approval was not found.")
        return matches[0]

    def list_approvals(self):
        return self._snapshot().approvals

    def save_approval(self, approval):
        if not isinstance(approval, Approval):
            raise TypeError("approval must be an Approval.")
        self.store.upsert("approvals", approval)

    def _snapshot(self):
        snapshot = self.store.load()
        if snapshot.issues:
            raise ValueError("Agent Manager storage is incomplete or unsupported.")
        ids = [item.approval_id for item in snapshot.approvals]
        if len(ids) != len(set(ids)):
            raise ValueError("Approval storage contains duplicate IDs.")
        return snapshot


class JsonLinesToolAuditRepository:
    """Append redacted tool-policy audit records with restrictive file modes."""

    def __init__(self, path=None):
        root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
        self.path = path or os.path.join(root, ".kripa", "agents", "tool_audit.jsonl")

    def record(self, event):
        if isinstance(event, ToolAuditRecord):
            payload = {
                "event_type": "policy_decision",
                "event_id": event.event_id,
                "recorded_at": event.recorded_at,
                "outcome": event.outcome.value,
                "reason": event.reason.value,
                "risk_level": event.risk_level.value,
                "agent_ref": event.agent_ref,
                "run_ref": event.run_ref,
                "tool_ref": event.tool_ref,
                "action_digest": event.action_digest,
                "approval_id": event.approval_id,
            }
        elif isinstance(event, ToolApprovalAuditRecord):
            payload = {
                "event_type": "approval_transition",
                "event_id": event.event_id,
                "recorded_at": event.recorded_at,
                "transition": event.transition,
                "agent_ref": event.agent_ref,
                "run_ref": event.run_ref,
                "tool_ref": event.tool_ref,
                "approval_id": event.approval_id,
                "action_digest": event.action_digest,
                "actor_ref": event.actor_ref,
            }
        else:
            raise TypeError("event must be a tool policy audit record.")
        line = (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
        directory = os.path.dirname(os.path.abspath(self.path))
        os.makedirs(directory, mode=0o700, exist_ok=True)
        os.chmod(directory, 0o700)
        flags = os.O_WRONLY | os.O_APPEND | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(self.path, flags, 0o600)
        try:
            os.fchmod(descriptor, 0o600)
            with os.fdopen(descriptor, "ab") as handle:
                handle.write(line)
                handle.flush()
                os.fsync(handle.fileno())
        except Exception:
            try:
                os.close(descriptor)
            except OSError:
                pass
            raise


def opaque_reference(value):
    """Hash identifiers before audit storage so malformed input is never echoed."""
    safe = value if isinstance(value, str) else ""
    return hashlib.sha256(safe.encode("utf-8", errors="replace")).hexdigest()[:24]
