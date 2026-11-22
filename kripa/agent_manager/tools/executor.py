"""Deny-by-default tool execution boundary, separate from provider adapters."""

import hashlib
import json
import math
import re
import uuid
from datetime import datetime, timezone

from kripa.agent_manager.contracts import validate_id
from kripa.agent_manager.skills.security import secret_findings
from kripa.agent_manager.tools.builtins import UtcTimeTool
from kripa.agent_manager.tools.contracts import (
    PolicyOutcome, ToolExecutionAuditRecord, ToolExecutionOutcome,
    ToolExecutionResult,
)
from kripa.agent_manager.tools.execution_repositories import default_execution_storage
from kripa.agent_manager.tools.repositories import opaque_reference
from kripa.agent_manager.tools.schema import canonical_json, normalize_arguments


_MAX_RESULT_BYTES = 64 * 1024
_SENSITIVE_KEY_PARTS = (
    "apikey", "token", "secret", "credential", "password", "authorization",
)
_SECRET_WORDS = re.compile(r"[^a-z]")


class ToolExecutor:
    """Invoke only explicitly composed handlers after policy and approval checks.

    Registry data never supplies an executable. Handlers are fixed application
    code passed at composition time and must affirm that they implement the
    exact registered definition.
    """

    def __init__(self, registry, policy, handlers, approval_consumption_repository,
                 audit_repository, clock=None):
        self.registry = registry
        self.policy = policy
        self.handlers = dict(handlers or {})
        self.consumptions = approval_consumption_repository
        self.audit = audit_repository
        self.clock = clock or (lambda: datetime.now(timezone.utc))

    def available_tools(self, agent_id):
        """Describe only registered, enabled, granted tools with known handlers."""
        try:
            agent = self.policy.grants.get_agent(agent_id)
        except Exception:
            return ()
        if not agent.enabled:
            return ()
        result = []
        for tool_id in agent.tool_ids:
            try:
                definition = self.registry.inspect(tool_id)
            except Exception:
                continue
            handler = self.handlers.get(tool_id)
            try:
                compatible = (
                    definition.enabled and handler is not None
                    and callable(getattr(handler, "can_execute", None))
                    and handler.can_execute(definition)
                )
                definition_text = "{} {} {}".format(
                    definition.name, definition.description,
                    canonical_json(definition.input_schema),
                )
                if secret_findings(definition_text):
                    compatible = False
            except Exception:
                compatible = False
            if compatible:
                result.append({
                    "tool_id": definition.tool_id,
                    "name": definition.name,
                    "description": definition.description,
                    "input_schema": definition.input_schema,
                    "risk_level": definition.risk_level,
                    "requires_approval": definition.requires_approval,
                })
        return tuple(result)

    def execute(self, agent_id, run_id, tool_id, arguments, approval_id=None,
                expected_action_digest=None):
        """Authorize, optionally pause, consume approval once, then call handler."""
        decision = self.policy.authorize(
            agent_id, run_id, tool_id, arguments, approval_id=approval_id,
            now=self.clock(),
        )
        if decision.outcome is PolicyOutcome.REQUIRES_APPROVAL:
            self._record(decision, agent_id, run_id, tool_id, approval_id,
                         ToolExecutionOutcome.REQUIRES_APPROVAL, "approval_required")
            return ToolExecutionResult(
                ToolExecutionOutcome.REQUIRES_APPROVAL,
                decision.reason.value,
                decision.risk_level,
                decision.action_digest,
                decision.approval_id,
                arguments_digest=decision.arguments_digest,
            )
        if decision.outcome is PolicyOutcome.DENY:
            self._record(decision, agent_id, run_id, tool_id, approval_id,
                         ToolExecutionOutcome.DENIED, decision.reason.value)
            return ToolExecutionResult(
                ToolExecutionOutcome.DENIED, decision.reason.value,
                decision.risk_level, decision.action_digest,
                _safe_approval_id(approval_id),
                arguments_digest=decision.arguments_digest,
            )

        if (expected_action_digest is not None
                and (not isinstance(expected_action_digest, str)
                     or decision.action_digest != expected_action_digest)):
            return self._deny(
                decision, agent_id, run_id, tool_id, approval_id,
                "pending_action_mismatch",
            )

        try:
            definition = self.registry.inspect(tool_id)
            handler = self.handlers.get(tool_id)
            if (handler is None or not callable(getattr(handler, "can_execute", None))
                    or not handler.can_execute(definition)):
                return self._deny(
                    decision, agent_id, run_id, tool_id, approval_id,
                    "handler_unavailable",
                )
            normalized = normalize_arguments(definition.input_schema, arguments)
        except Exception:
            return self._deny(
                decision, agent_id, run_id, tool_id, approval_id,
                "definition_or_arguments_invalid",
            )

        if approval_id is not None:
            try:
                consumed = self.consumptions.consume(approval_id, decision.action_digest)
            except Exception:
                consumed = False
            if not consumed:
                return self._deny(
                    decision, agent_id, run_id, tool_id, approval_id,
                    "approval_replayed_or_consumption_unavailable",
                )

        # Record authorization before the final recheck. Audit failure is
        # fail-closed, and no handler is invoked without this durable record.
        if not self._record(decision, agent_id, run_id, tool_id, approval_id,
                            ToolExecutionOutcome.AUTHORIZED, "execution_preflight"):
            return ToolExecutionResult(
                ToolExecutionOutcome.DENIED, "execution_audit_unavailable",
                decision.risk_level, decision.action_digest,
                _safe_approval_id(approval_id),
                arguments_digest=decision.arguments_digest,
            )

        # Final re-read of the grant, tool state, normalized arguments,
        # approval binding/status and expiry immediately before the handler.
        final = self.policy.authorize(
            agent_id, run_id, tool_id, normalized, approval_id=approval_id,
            now=self.clock(),
        )
        if final.outcome is PolicyOutcome.REQUIRES_APPROVAL:
            self._record(final, agent_id, run_id, tool_id, approval_id,
                         ToolExecutionOutcome.REQUIRES_APPROVAL, "approval_required")
            return ToolExecutionResult(
                ToolExecutionOutcome.REQUIRES_APPROVAL, final.reason.value,
                final.risk_level, final.action_digest, final.approval_id,
                arguments_digest=final.arguments_digest,
            )
        if (final.outcome is not PolicyOutcome.ALLOW
                or final.action_digest != decision.action_digest):
            return self._deny(
                final, agent_id, run_id, tool_id, approval_id,
                "final_policy_check_failed",
            )

        try:
            raw_result = handler.execute(normalized)
            safe_result = _sanitize_result(raw_result)
            result_digest = hashlib.sha256(canonical_json(safe_result).encode("utf-8")).hexdigest()
        except Exception:
            self._record(final, agent_id, run_id, tool_id, approval_id,
                         ToolExecutionOutcome.FAILED, "handler_failed", error_code="handler_error")
            return ToolExecutionResult(
                ToolExecutionOutcome.FAILED, "handler_failed", final.risk_level,
                final.action_digest, approval_id or "",
                arguments_digest=final.arguments_digest,
            )

        if not self._record(
            final, agent_id, run_id, tool_id, approval_id,
            ToolExecutionOutcome.EXECUTED, "execution_completed",
            result_digest=result_digest,
        ):
            return ToolExecutionResult(
                ToolExecutionOutcome.EXECUTED, "execution_completed_audit_unavailable",
                final.risk_level, final.action_digest, _safe_approval_id(approval_id),
                safe_result, final.arguments_digest,
            )
        return ToolExecutionResult(
            ToolExecutionOutcome.EXECUTED, "executed", final.risk_level,
            final.action_digest, _safe_approval_id(approval_id), safe_result,
            final.arguments_digest,
        )

    def decide_approval(self, approval_id, approved, decided_by):
        """Record a human decision; caller is responsible for authenticating them."""
        return self.policy.decide_approval(
            approval_id, approved, decided_by, now=self.clock(),
        )

    def _deny(self, decision, agent_id, run_id, tool_id, approval_id, reason):
        self._record(decision, agent_id, run_id, tool_id, approval_id,
                     ToolExecutionOutcome.DENIED, reason)
        return ToolExecutionResult(
            ToolExecutionOutcome.DENIED, reason, decision.risk_level,
            decision.action_digest, _safe_approval_id(approval_id),
            arguments_digest=decision.arguments_digest,
        )

    def _record(self, decision, agent_id, run_id, tool_id, approval_id,
                outcome, reason, result_digest="", error_code=""):
        event = ToolExecutionAuditRecord(
            event_id="exec-{}".format(uuid.uuid4().hex),
            recorded_at=_timestamp(self.clock()),
            outcome=outcome,
            reason=reason,
            risk_level=decision.risk_level,
            agent_ref=opaque_reference(agent_id),
            run_ref=opaque_reference(run_id),
            tool_ref=opaque_reference(tool_id),
            action_digest=decision.action_digest,
            approval_ref=opaque_reference(approval_id) if approval_id else "",
            result_digest=result_digest,
            error_code=error_code,
        )
        try:
            self.audit.record_execution(event)
        except Exception:
            return False
        return True


def build_default_tool_executor(store=None):
    """Compose the single built-in tool with existing policy repositories."""
    import os

    from kripa.agent_manager.registry import AgentRegistry
    from kripa.agent_manager.store import AgentManagerStore
    from kripa.agent_manager.tools.builtins import UTC_TIME_TOOL_ID, utc_time_definition
    from kripa.agent_manager.tools.policy import ToolPolicyService
    from kripa.agent_manager.tools.registry import ToolRegistry
    from kripa.agent_manager.tools.repositories import (
        AgentManagerApprovalRepository, AgentManagerGrantRepository,
        AgentManagerToolRepository, JsonLinesToolAuditRepository,
    )

    store = store or AgentManagerStore()
    registry = ToolRegistry(AgentManagerToolRepository(store))
    try:
        registry.inspect(UTC_TIME_TOOL_ID)
    except LookupError:
        registry.register(utc_time_definition())
    grants = AgentManagerGrantRepository(AgentRegistry(store))
    approvals = AgentManagerApprovalRepository(store)
    audit_path = os.path.join(os.path.dirname(os.path.abspath(store.path)), "tool_audit.jsonl")
    policy = ToolPolicyService(
        registry=registry, grant_repository=grants,
        approval_repository=approvals,
        audit_repository=JsonLinesToolAuditRepository(audit_path),
    )
    consumption, execution_audit = default_execution_storage(store)
    return ToolExecutor(
        registry=registry,
        policy=policy,
        handlers={"utc_time": UtcTimeTool()},
        approval_consumption_repository=consumption,
        audit_repository=execution_audit,
    )


def _sanitize_result(value, depth=0):
    if depth > 20:
        raise ValueError("Tool result is too deeply nested.")
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Tool result must contain finite numbers.")
        return value
    if isinstance(value, str):
        if secret_findings(value):
            return "[REDACTED]"
        if len(value.encode("utf-8")) > _MAX_RESULT_BYTES:
            raise ValueError("Tool result exceeds the supported size limit.")
        return value
    if isinstance(value, list):
        result = [_sanitize_result(item, depth + 1) for item in value]
    elif isinstance(value, dict):
        result = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError("Tool result object keys must be strings.")
            normalized = _SECRET_WORDS.sub("", key.lower())
            if any(part in normalized for part in _SENSITIVE_KEY_PARTS):
                result[key] = "[REDACTED]"
            else:
                result[key] = _sanitize_result(item, depth + 1)
    else:
        raise ValueError("Tool results must contain JSON values only.")
    if len(canonical_json(result).encode("utf-8")) > _MAX_RESULT_BYTES:
        raise ValueError("Tool result exceeds the supported size limit.")
    return result


def _timestamp(value):
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise ValueError("Tool executor clock must return a timezone-aware datetime.")
    return value.astimezone(timezone.utc).isoformat()


def _safe_approval_id(value):
    if not isinstance(value, str):
        return ""
    try:
        return validate_id(value, "approval_id")
    except (TypeError, ValueError):
        return ""
