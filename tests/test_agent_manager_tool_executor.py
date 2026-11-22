"""Safety tests for the Phase 6 tool execution boundary."""

import json
import os
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

from kripa.agent_manager import AgentDefinition, ToolDefinition
from kripa.agent_manager.tools.builtins import UtcTimeTool, utc_time_definition
from kripa.agent_manager.tools.contracts import (
    ToolExecutionOutcome,
)
from kripa.agent_manager.tools.execution_repositories import (
    ExclusiveFileApprovalConsumptionRepository,
    JsonLinesToolExecutionAuditRepository,
)
from kripa.agent_manager.tools.executor import ToolExecutor
from kripa.agent_manager.tools.policy import ToolPolicyService
from kripa.agent_manager.tools.registry import ToolRegistry


NOW = datetime(2026, 11, 22, 12, 0, 0, tzinfo=timezone.utc)
SCHEMA = {
    "type": "object",
    "properties": {"query": {"type": "string", "maxLength": 200}},
    "required": ["query"],
    "additionalProperties": False,
}


class Clock:
    def __init__(self, value=NOW):
        self.value = value

    def __call__(self):
        return self.value


class FakeToolRepository:
    def __init__(self, tools=()):
        self.tools = {item.tool_id: item for item in tools}

    def get_tool(self, tool_id):
        if tool_id not in self.tools:
            raise LookupError("missing")
        return self.tools[tool_id]

    def list_tools(self):
        return tuple(self.tools.values())

    def save_tool(self, definition):
        self.tools[definition.tool_id] = definition


class FakeGrantRepository:
    def __init__(self, agent):
        self.agent = agent

    def get_agent(self, agent_id):
        if agent_id != self.agent.agent_id:
            raise LookupError("missing")
        return self.agent

    def has_grant(self, agent_id, tool_id):
        return tool_id in self.get_agent(agent_id).tool_ids


class FakeApprovalRepository:
    def __init__(self):
        self.approvals = {}

    def get_approval(self, approval_id):
        if approval_id not in self.approvals:
            raise LookupError("missing")
        return self.approvals[approval_id]

    def list_approvals(self):
        return tuple(self.approvals.values())

    def save_approval(self, approval):
        self.approvals[approval.approval_id] = approval


class FakePolicyAuditRepository:
    def __init__(self):
        self.events = []

    def record(self, event):
        self.events.append(event)


class FakeConsumptionRepository:
    def __init__(self):
        self.used = set()
        self.calls = []

    def consume(self, approval_id, action_digest):
        self.calls.append((approval_id, action_digest))
        if approval_id in self.used:
            return False
        self.used.add(approval_id)
        return True


class FakeExecutionAuditRepository:
    def __init__(self, fail=False):
        self.events = []
        self.fail = fail

    def record_execution(self, event):
        if self.fail:
            raise OSError("private audit path detail")
        self.events.append(event)


class FailCompletionAuditRepository(FakeExecutionAuditRepository):
    def record_execution(self, event):
        if event.outcome is ToolExecutionOutcome.EXECUTED:
            raise OSError("private audit path detail")
        self.events.append(event)


class FakeHandler:
    def __init__(self, tool_id, response=None, error=None):
        self.tool_id = tool_id
        self.response = response
        self.error = error
        self.calls = []

    def can_execute(self, definition):
        return definition.tool_id == self.tool_id

    def execute(self, arguments):
        self.calls.append(arguments)
        if self.error:
            raise self.error
        return self.response if self.response is not None else arguments


def sensitive_definition(enabled=True):
    return ToolDefinition(
        "send_note", "Send note", "Send a note to an external recipient.",
        SCHEMA, risk_level="high", requires_approval=True, enabled=enabled,
    )


def make_executor(tools, grants, handlers, clock=None, audit=None, ttl=300):
    repository = FakeToolRepository(tools)
    registry = ToolRegistry(repository)
    approvals = FakeApprovalRepository()
    policy_audit = FakePolicyAuditRepository()
    policy = ToolPolicyService(
        registry=registry,
        grant_repository=FakeGrantRepository(AgentDefinition(
            "agent", "Agent", "", "Perform the requested task.", tool_ids=tuple(grants),
        )),
        approval_repository=approvals,
        audit_repository=policy_audit,
        approval_ttl_seconds=ttl,
    )
    consumption = FakeConsumptionRepository()
    execution_audit = audit or FakeExecutionAuditRepository()
    executor = ToolExecutor(
        registry, policy, handlers, consumption, execution_audit,
        clock=clock or Clock(),
    )
    return executor, approvals, consumption, execution_audit


class ToolExecutorTests(unittest.TestCase):
    def test_authorized_read_only_utc_execution(self):
        definition = utc_time_definition()
        handler = UtcTimeTool(clock=lambda: NOW)
        executor, _, consumption, audit = make_executor(
            (definition,), (definition.tool_id,), {definition.tool_id: handler},
        )

        result = executor.execute("agent", "run-1", "utc_time", {})

        self.assertEqual(result.outcome, ToolExecutionOutcome.EXECUTED)
        self.assertEqual(result.result, NOW.isoformat())
        self.assertEqual(consumption.calls, [])
        self.assertEqual([item.outcome for item in audit.events], [
            ToolExecutionOutcome.AUTHORIZED, ToolExecutionOutcome.EXECUTED,
        ])

    def test_model_sees_only_explicitly_granted_enabled_tools_with_handlers(self):
        granted = ToolDefinition("read", "Read", "Read data", SCHEMA)
        disabled = ToolDefinition("disabled", "Disabled", "", SCHEMA, enabled=False)
        handler = FakeHandler("read")
        executor, _, _, _ = make_executor(
            (granted, disabled), ("read", "disabled"), {"read": handler},
        )
        self.assertEqual([item["tool_id"] for item in executor.available_tools("agent")], ["read"])

        no_grant, _, _, _ = make_executor((granted,), (), {"read": handler})
        self.assertEqual(no_grant.available_tools("agent"), ())

    def test_unknown_tool_is_denied_without_handler_call(self):
        handler = FakeHandler("known")
        executor, _, _, _ = make_executor((), (), {"known": handler})
        result = executor.execute("agent", "run-1", "missing", {})
        self.assertEqual(result.outcome, ToolExecutionOutcome.DENIED)
        self.assertEqual(handler.calls, [])

    def test_disabled_tool_is_denied(self):
        definition = ToolDefinition("read", "Read", "", SCHEMA, enabled=False)
        handler = FakeHandler("read")
        executor, _, _, _ = make_executor((definition,), ("read",), {"read": handler})
        result = executor.execute("agent", "run-1", "read", {"query": "x"})
        self.assertEqual(result.outcome, ToolExecutionOutcome.DENIED)
        self.assertEqual(handler.calls, [])

    def test_missing_agent_grant_is_denied(self):
        definition = ToolDefinition("read", "Read", "", SCHEMA)
        handler = FakeHandler("read")
        executor, _, _, _ = make_executor((definition,), (), {"read": handler})
        result = executor.execute("agent", "run-1", "read", {"query": "x"})
        self.assertEqual(result.outcome, ToolExecutionOutcome.DENIED)
        self.assertEqual(handler.calls, [])

    def test_invalid_arguments_are_denied(self):
        definition = ToolDefinition("read", "Read", "", SCHEMA)
        handler = FakeHandler("read")
        executor, _, _, _ = make_executor((definition,), ("read",), {"read": handler})
        result = executor.execute("agent", "run-1", "read", {"query": "x", "extra": "y"})
        self.assertEqual(result.outcome, ToolExecutionOutcome.DENIED)
        self.assertEqual(handler.calls, [])

    def test_sensitive_action_pauses_until_explicit_approval(self):
        definition = sensitive_definition()
        handler = FakeHandler("send_note")
        executor, approvals, _, _ = make_executor(
            (definition,), ("send_note",), {"send_note": handler},
        )
        result = executor.execute("agent", "run-1", "send_note", {"query": "hello"})

        self.assertEqual(result.outcome, ToolExecutionOutcome.REQUIRES_APPROVAL)
        self.assertEqual(approvals.get_approval(result.approval_id).status, "pending")
        self.assertEqual(handler.calls, [])

    def test_rejected_approval_cannot_execute(self):
        definition = sensitive_definition()
        handler = FakeHandler("send_note")
        executor, _, _, _ = make_executor(
            (definition,), ("send_note",), {"send_note": handler},
        )
        pending = executor.execute("agent", "run-1", "send_note", {"query": "hello"})
        executor.decide_approval(pending.approval_id, False, "human")
        denied = executor.execute(
            "agent", "run-1", "send_note", {"query": "hello"}, pending.approval_id,
        )
        self.assertEqual(denied.outcome, ToolExecutionOutcome.DENIED)
        self.assertEqual(handler.calls, [])

    def test_expired_approval_is_rejected(self):
        definition = sensitive_definition()
        handler = FakeHandler("send_note")
        clock = Clock()
        executor, approvals, _, _ = make_executor(
            (definition,), ("send_note",), {"send_note": handler}, clock=clock, ttl=30,
        )
        pending = executor.execute("agent", "run-1", "send_note", {"query": "hello"})
        clock.value = NOW + timedelta(seconds=31)
        denied = executor.execute(
            "agent", "run-1", "send_note", {"query": "hello"}, pending.approval_id,
        )
        self.assertEqual(denied.outcome, ToolExecutionOutcome.DENIED)
        self.assertEqual(approvals.get_approval(pending.approval_id).status, "expired")
        self.assertEqual(handler.calls, [])

    def test_changed_arguments_are_denied_before_consumption(self):
        definition = sensitive_definition()
        handler = FakeHandler("send_note")
        executor, _, consumption, _ = make_executor(
            (definition,), ("send_note",), {"send_note": handler},
        )
        pending = executor.execute("agent", "run-1", "send_note", {"query": "original"})
        executor.decide_approval(pending.approval_id, True, "human")
        changed = executor.execute(
            "agent", "run-1", "send_note", {"query": "changed"}, pending.approval_id,
        )
        self.assertEqual(changed.outcome, ToolExecutionOutcome.DENIED)
        self.assertEqual(consumption.calls, [])
        self.assertEqual(handler.calls, [])

    def test_approval_replay_is_rejected_after_one_execution(self):
        definition = sensitive_definition()
        handler = FakeHandler("send_note", response={"sent": True})
        with tempfile.TemporaryDirectory() as directory:
            executor, _, _, _ = make_executor(
                (definition,), ("send_note",), {"send_note": handler},
            )
            executor.consumptions = ExclusiveFileApprovalConsumptionRepository(directory)
            args = {"query": "hello"}
            pending = executor.execute("agent", "run-1", "send_note", args)
            executor.decide_approval(pending.approval_id, True, "human")
            first = executor.execute("agent", "run-1", "send_note", args, pending.approval_id)
            replay = executor.execute("agent", "run-1", "send_note", args, pending.approval_id)
        self.assertEqual(first.outcome, ToolExecutionOutcome.EXECUTED)
        self.assertEqual(replay.outcome, ToolExecutionOutcome.DENIED)
        self.assertEqual(len(handler.calls), 1)

    def test_expected_action_digest_is_checked_before_execution(self):
        definition = sensitive_definition()
        handler = FakeHandler("send_note")
        executor, _, consumption, _ = make_executor(
            (definition,), ("send_note",), {"send_note": handler},
        )
        args = {"query": "hello"}
        pending = executor.execute("agent", "run-1", "send_note", args)
        executor.decide_approval(pending.approval_id, True, "human")
        denied = executor.execute(
            "agent", "run-1", "send_note", args, pending.approval_id,
            expected_action_digest="0" * 64,
        )
        self.assertEqual(denied.outcome, ToolExecutionOutcome.DENIED)
        self.assertEqual(consumption.calls, [])
        self.assertEqual(handler.calls, [])

    def test_handler_failure_is_sanitized_and_audited(self):
        definition = ToolDefinition("read", "Read", "", SCHEMA)
        handler = FakeHandler("read", error=RuntimeError("api_key=private-credential-value"))
        executor, _, _, audit = make_executor(
            (definition,), ("read",), {"read": handler},
        )
        result = executor.execute("agent", "run-1", "read", {"query": "x"})
        serialized = json.dumps([event.__dict__ for event in audit.events], default=str)
        self.assertEqual(result.outcome, ToolExecutionOutcome.FAILED)
        self.assertEqual(result.reason, "handler_failed")
        self.assertNotIn("private-credential-value", serialized)
        self.assertNotIn("api_key", serialized)
        self.assertEqual(audit.events[-1].error_code, "handler_error")

    def test_execution_results_and_audit_redact_secrets(self):
        definition = ToolDefinition("read", "Read", "", SCHEMA)
        secret = "password=private-value-123"
        handler = FakeHandler("read", response=secret)
        with tempfile.TemporaryDirectory() as directory:
            audit_path = os.path.join(directory, "execution.jsonl")
            executor, _, _, _ = make_executor(
                (definition,), ("read",), {"read": handler},
                audit=JsonLinesToolExecutionAuditRepository(audit_path),
            )
            result = executor.execute("agent", "run-1", "read", {"query": secret})
            with open(audit_path, "r", encoding="utf-8") as stream:
                contents = stream.read()
            self.assertEqual(result.result, "[REDACTED]")
            self.assertNotIn(secret, contents)
            self.assertNotIn("private-value-123", contents)
            self.assertNotIn("query", contents)
            self.assertEqual(os.stat(audit_path).st_mode & 0o777, 0o600)

    def test_policy_deny_never_invokes_handler(self):
        definition = ToolDefinition("read", "Read", "", SCHEMA)
        handler = FakeHandler("read")
        executor, _, _, _ = make_executor((definition,), (), {"read": handler})
        denied = executor.execute("agent", "run-1", "read", {"query": "x"})
        self.assertEqual(denied.outcome, ToolExecutionOutcome.DENIED)
        self.assertEqual(handler.calls, [])

    def test_preflight_audit_failure_fails_closed_before_handler(self):
        definition = ToolDefinition("read", "Read", "", SCHEMA)
        handler = FakeHandler("read")
        audit = FakeExecutionAuditRepository(fail=True)
        executor, _, _, _ = make_executor(
            (definition,), ("read",), {"read": handler}, audit=audit,
        )
        result = executor.execute("agent", "run-1", "read", {"query": "x"})
        self.assertEqual(result.outcome, ToolExecutionOutcome.DENIED)
        self.assertEqual(result.reason, "execution_audit_unavailable")
        self.assertEqual(handler.calls, [])

    def test_completion_audit_failure_does_not_claim_execution_failed(self):
        definition = ToolDefinition("read", "Read", "", SCHEMA)
        handler = FakeHandler("read", response={"safe": True})
        audit = FailCompletionAuditRepository()
        executor, _, _, _ = make_executor(
            (definition,), ("read",), {"read": handler}, audit=audit,
        )
        result = executor.execute("agent", "run-1", "read", {"query": "x"})
        self.assertEqual(result.outcome, ToolExecutionOutcome.EXECUTED)
        self.assertEqual(result.reason, "execution_completed_audit_unavailable")
        self.assertEqual(result.result, {"safe": True})
        self.assertEqual(len(handler.calls), 1)
        self.assertEqual([event.outcome for event in audit.events], [ToolExecutionOutcome.AUTHORIZED])


class ApprovalConsumptionRepositoryTests(unittest.TestCase):
    def test_exclusive_file_claim_allows_only_one_concurrent_consumer(self):
        with tempfile.TemporaryDirectory() as directory:
            repository = ExclusiveFileApprovalConsumptionRepository(
                os.path.join(directory, "consumed"),
            )
            digest = "a" * 64
            with ThreadPoolExecutor(max_workers=12) as pool:
                results = list(pool.map(
                    lambda _: repository.consume("approval-once", digest), range(24),
                ))
            self.assertEqual(sum(bool(value) for value in results), 1)
            self.assertTrue(repository.is_consumed("approval-once"))
            marker = os.path.join(
                directory, "consumed",
                __import__("hashlib").sha256(b"approval-once").hexdigest()[:24] + ".json",
            )
            self.assertEqual(os.stat(marker).st_mode & 0o777, 0o600)
            with open(marker, "r", encoding="utf-8") as stream:
                content = stream.read()
            self.assertNotIn("approval-once", content)
            self.assertIn(digest, content)


if __name__ == "__main__":
    unittest.main()
