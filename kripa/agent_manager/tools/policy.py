"""Fail-closed, non-executing tool authorization and approval service."""

import hashlib
import uuid
from datetime import datetime, timedelta, timezone

from kripa.agent_manager.contracts import Approval, validate_id
from kripa.agent_manager.registry import AgentRegistry
from kripa.agent_manager.store import AgentManagerStore
from kripa.agent_manager.tools.contracts import (
    PolicyDecision, PolicyOutcome, PolicyReason, ToolApprovalAuditRecord,
    ToolAuditRecord, ToolRiskLevel,
)
from kripa.agent_manager.tools.registry import ToolRegistry
from kripa.agent_manager.tools.repositories import (
    AgentManagerApprovalRepository, AgentManagerGrantRepository,
    JsonLinesToolAuditRepository, opaque_reference,
)
from kripa.agent_manager.tools.schema import canonical_json, normalize_arguments


class ToolPolicyService:
    """Authorize tool proposals; this class has no tool execution facility."""

    def __init__(self, registry=None, grant_repository=None, approval_repository=None,
                 audit_repository=None, approval_ttl_seconds=300):
        if not isinstance(approval_ttl_seconds, int) or isinstance(approval_ttl_seconds, bool) or approval_ttl_seconds <= 0:
            raise ValueError("approval_ttl_seconds must be a positive integer.")
        store = AgentManagerStore()
        agent_registry = AgentRegistry(store)
        self.registry = registry or ToolRegistry()
        self.grants = grant_repository or AgentManagerGrantRepository(agent_registry)
        self.approvals = approval_repository or AgentManagerApprovalRepository(store)
        self.audit = audit_repository or JsonLinesToolAuditRepository()
        self.approval_ttl_seconds = approval_ttl_seconds

    def grant_tool(self, agent_id, tool_id):
        """Record an explicit agent grant using AgentDefinition.tool_ids."""
        try:
            validate_id(agent_id, "agent_id")
            validate_id(tool_id, "tool_id")
            tool = self.registry.inspect(tool_id)
            agent = self.grants.get_agent(agent_id)
        except Exception:
            raise LookupError("Agent or tool was not found or could not be inspected.")
        if not tool.enabled:
            raise ValueError("Disabled tools cannot be granted.")
        if not agent.enabled:
            raise ValueError("Disabled agents cannot receive tool grants.")
        return self.grants.grant(agent_id, tool_id)

    def revoke_tool(self, agent_id, tool_id):
        try:
            validate_id(agent_id, "agent_id")
            validate_id(tool_id, "tool_id")
            return self.grants.revoke(agent_id, tool_id)
        except Exception:
            raise LookupError("Agent tool grant could not be updated.")

    def authorize(self, agent_id, run_id, tool_id, arguments, approval_id=None, now=None):
        """Return a policy decision for a proposed action; never call a tool."""
        current = _now(now)
        risk = ToolRiskLevel.LOW
        action_digest = ""
        arguments_digest = ""
        normalized = None
        try:
            validate_id(agent_id, "agent_id")
            validate_id(run_id, "run_id")
            validate_id(tool_id, "tool_id")
            if approval_id is not None:
                validate_id(approval_id, "approval_id")
        except (TypeError, ValueError):
            return self._finish(
                PolicyDecision(PolicyOutcome.DENY, PolicyReason.INVALID_REQUEST, risk),
                agent_id, run_id, tool_id, current,
            )

        try:
            agent = self.grants.get_agent(agent_id)
        except Exception:
            return self._finish(
                PolicyDecision(PolicyOutcome.DENY, PolicyReason.UNKNOWN_AGENT, risk),
                agent_id, run_id, tool_id, current,
            )
        if not agent.enabled:
            return self._finish(
                PolicyDecision(PolicyOutcome.DENY, PolicyReason.DISABLED_AGENT, risk),
                agent_id, run_id, tool_id, current,
            )

        try:
            tool = self.registry.inspect(tool_id)
        except Exception:
            return self._finish(
                PolicyDecision(PolicyOutcome.DENY, PolicyReason.UNKNOWN_TOOL, risk),
                agent_id, run_id, tool_id, current,
            )
        risk = ToolRiskLevel(tool.risk_level)
        if not tool.enabled:
            return self._finish(
                PolicyDecision(PolicyOutcome.DENY, PolicyReason.DISABLED_TOOL, risk),
                agent_id, run_id, tool_id, current,
            )
        try:
            granted = self.grants.has_grant(agent_id, tool_id)
        except Exception:
            granted = False
        if not granted:
            return self._finish(
                PolicyDecision(PolicyOutcome.DENY, PolicyReason.GRANT_MISSING, risk),
                agent_id, run_id, tool_id, current,
            )
        try:
            normalized = normalize_arguments(tool.input_schema, arguments)
            arguments_digest = _digest(normalized)
            action_digest = _action_digest(agent_id, run_id, tool_id, arguments_digest, None)
        except (TypeError, ValueError, OverflowError):
            return self._finish(
                PolicyDecision(PolicyOutcome.DENY, PolicyReason.INVALID_ARGUMENTS, risk),
                agent_id, run_id, tool_id, current,
            )

        if approval_id is not None:
            decision = self._check_approval(
                approval_id, agent_id, run_id, tool_id, arguments_digest,
                action_digest, risk, current,
            )
            return self._finish(decision, agent_id, run_id, tool_id, current)

        if not _requires_approval(tool, risk):
            return self._finish(
                PolicyDecision(PolicyOutcome.ALLOW, PolicyReason.AUTHORIZED, risk,
                               action_digest, arguments_digest),
                agent_id, run_id, tool_id, current,
            )

        approval_id = "approval-{}".format(uuid.uuid4().hex)
        expires_at = (current + timedelta(seconds=self.approval_ttl_seconds)).isoformat()
        approval = Approval(
            approval_id=approval_id,
            agent_id=agent_id,
            run_id=run_id,
            tool_id=tool_id,
            arguments_digest=arguments_digest,
            action="tool_action",
            action_digest=_action_digest(
                agent_id, run_id, tool_id, arguments_digest, expires_at,
            ),
            status="pending",
            requested_at=current.isoformat(),
            expires_at=expires_at,
        )
        action_digest = approval.action_digest
        decision = PolicyDecision(
            PolicyOutcome.REQUIRES_APPROVAL, PolicyReason.APPROVAL_REQUIRED, risk,
            action_digest, arguments_digest, approval.approval_id,
        )
        try:
            self.approvals.save_approval(approval)
        except Exception:
            return self._finish(
                PolicyDecision(PolicyOutcome.DENY, PolicyReason.AUDIT_UNAVAILABLE, risk,
                               action_digest, arguments_digest),
                agent_id, run_id, tool_id, current,
            )
        if not self._write_audit(decision, agent_id, run_id, tool_id, current):
            try:
                self.approvals.save_approval(_transition(approval, "rejected", current, ""))
            except Exception:
                pass
            return PolicyDecision(PolicyOutcome.DENY, PolicyReason.AUDIT_UNAVAILABLE,
                                  risk, action_digest, arguments_digest)
        return decision

    def decide_approval(self, approval_id, approved, decided_by, now=None):
        """Record an explicit human decision; caller must authenticate that human."""
        current = _now(now)
        validate_id(approval_id, "approval_id")
        if not isinstance(approved, bool):
            raise ValueError("approved must be a boolean.")
        if not isinstance(decided_by, str) or not decided_by.strip():
            raise ValueError("decided_by must identify the approving principal.")
        try:
            approval = self.approvals.get_approval(approval_id)
        except Exception:
            raise LookupError("Approval was not found.")
        if (approval.agent_id is None or approval.tool_id is None
                or approval.arguments_digest is None):
            raise ValueError("Approval is not bound to a tool action.")
        if approval.status == "expired":
            raise ValueError("Approval has expired.")
        if approval.status != "pending":
            raise ValueError("Only pending approvals can be decided.")
        if current >= _parse_time(approval.expires_at):
            expired = _transition(approval, "expired", current, decided_by)
            self.approvals.save_approval(expired)
            self._write_approval_audit(expired, "expired", decided_by, current)
            raise ValueError("Approval has expired.")
        status = "approved" if approved else "rejected"
        updated = _transition(approval, status, current, decided_by)
        self.approvals.save_approval(updated)
        if not self._write_approval_audit(updated, status, decided_by, current):
            if status == "approved":
                self.approvals.save_approval(_transition(updated, "rejected", current, ""))
            raise ValueError("Approval transition could not be audited.")
        return updated

    def _check_approval(self, approval_id, agent_id, run_id, tool_id,
                        arguments_digest, action_digest, risk, now):
        try:
            approval = self.approvals.get_approval(approval_id)
        except Exception:
            return PolicyDecision(PolicyOutcome.DENY, PolicyReason.APPROVAL_UNKNOWN, risk,
                                  action_digest, arguments_digest)
        expected_digest = _action_digest(
            agent_id, run_id, tool_id, arguments_digest, approval.expires_at,
        )
        if (approval.agent_id != agent_id or approval.run_id != run_id
                or approval.tool_id != tool_id
                or approval.arguments_digest != arguments_digest
                or approval.action_digest != expected_digest):
            return PolicyDecision(PolicyOutcome.DENY, PolicyReason.APPROVAL_MISMATCH, risk,
                                  expected_digest, arguments_digest)
        action_digest = expected_digest
        if now >= _parse_time(approval.expires_at):
            if approval.status == "pending":
                try:
                    expired = _transition(approval, "expired", now, "")
                    self.approvals.save_approval(expired)
                    self._write_approval_audit(expired, "expired", "system-expiry", now)
                except Exception:
                    pass
            return PolicyDecision(PolicyOutcome.DENY, PolicyReason.APPROVAL_EXPIRED, risk,
                                  action_digest, arguments_digest)
        if approval.status == "rejected":
            return PolicyDecision(PolicyOutcome.DENY, PolicyReason.APPROVAL_REJECTED, risk,
                                  action_digest, arguments_digest)
        if approval.status == "expired":
            return PolicyDecision(PolicyOutcome.DENY, PolicyReason.APPROVAL_EXPIRED, risk,
                                  action_digest, arguments_digest)
        if approval.status == "pending":
            return PolicyDecision(PolicyOutcome.REQUIRES_APPROVAL, PolicyReason.APPROVAL_PENDING,
                                  risk, action_digest, arguments_digest, approval.approval_id)
        if approval.status == "approved":
            return PolicyDecision(PolicyOutcome.ALLOW, PolicyReason.AUTHORIZED, risk,
                                  action_digest, arguments_digest, approval.approval_id)
        return PolicyDecision(PolicyOutcome.DENY, PolicyReason.APPROVAL_REJECTED, risk,
                              action_digest, arguments_digest)

    def _finish(self, decision, agent_id, run_id, tool_id, now):
        if not self._write_audit(decision, agent_id, run_id, tool_id, now):
            if decision.outcome is not PolicyOutcome.DENY:
                return PolicyDecision(
                    PolicyOutcome.DENY, PolicyReason.AUDIT_UNAVAILABLE, decision.risk_level,
                    decision.action_digest, decision.arguments_digest,
                )
        return decision

    def _write_audit(self, decision, agent_id, run_id, tool_id, now):
        event = ToolAuditRecord(
            event_id="audit-{}".format(uuid.uuid4().hex),
            recorded_at=now.isoformat(),
            outcome=decision.outcome,
            reason=decision.reason,
            risk_level=decision.risk_level,
            agent_ref=opaque_reference(agent_id),
            run_ref=opaque_reference(run_id),
            tool_ref=opaque_reference(tool_id),
            action_digest=decision.action_digest,
            approval_id=decision.approval_id,
        )
        try:
            self.audit.record(event)
            return True
        except Exception:
            return False

    def _write_approval_audit(self, approval, transition, actor, now):
        event = ToolApprovalAuditRecord(
            event_id="audit-{}".format(uuid.uuid4().hex),
            recorded_at=now.isoformat(),
            transition=transition,
            agent_ref=opaque_reference(approval.agent_id),
            run_ref=opaque_reference(approval.run_id),
            tool_ref=opaque_reference(approval.tool_id),
            approval_id=approval.approval_id,
            action_digest=approval.action_digest,
            actor_ref=opaque_reference(actor),
        )
        try:
            self.audit.record(event)
            return True
        except Exception:
            return False


def _requires_approval(tool, risk):
    return tool.requires_approval or risk in {ToolRiskLevel.HIGH, ToolRiskLevel.CRITICAL}


def _transition(approval, status, now, decided_by):
    import dataclasses

    metadata = dict(approval.metadata)
    if decided_by:
        metadata["decided_by_ref"] = opaque_reference(decided_by)
    return dataclasses.replace(
        approval, status=status, decided_at=now.isoformat(), metadata=metadata,
    )


def _digest(value):
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _action_digest(agent_id, run_id, tool_id, arguments_digest, expires_at):
    return _digest({
        "agent_id": agent_id,
        "run_id": run_id,
        "tool_id": tool_id,
        "arguments_digest": arguments_digest,
        "expires_at": expires_at,
    })


def _now(value):
    current = value or datetime.now(timezone.utc)
    if not isinstance(current, datetime) or current.tzinfo is None:
        raise ValueError("now must be a timezone-aware datetime.")
    return current.astimezone(timezone.utc)


def _parse_time(value):
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return parsed.astimezone(timezone.utc)
