"""Unit tests for non-executing tool registration and policy decisions."""

import json
import os
import tempfile
import unittest
from dataclasses import replace
from datetime import datetime, timedelta, timezone

from kripa.agent_manager import AgentDefinition, AgentManagerStore, AgentRegistry, ToolDefinition
from kripa.agent_manager.tools import PolicyOutcome, PolicyReason, ToolPolicyService, ToolRegistry
from kripa.agent_manager.tools.repositories import (
    AgentManagerApprovalRepository, AgentManagerGrantRepository,
    AgentManagerToolRepository, JsonLinesToolAuditRepository,
)


NOW = datetime(2026, 11, 22, 12, 0, 0, tzinfo=timezone.utc)
SCHEMA = {
    "type": "object",
    "properties": {
        "message": {"type": "string", "minLength": 1, "maxLength": 1000},
        "count": {"type": "integer", "minimum": 0, "maximum": 10},
    },
    "required": ["message"],
    "additionalProperties": False,
}


class FakeToolRepository:
    def __init__(self, tools=()):
        self.tools = {tool.tool_id: tool for tool in tools}

    def get_tool(self, tool_id):
        if tool_id not in self.tools:
            raise LookupError("missing")
        return self.tools[tool_id]

    def list_tools(self):
        return tuple(self.tools.values())

    def save_tool(self, definition):
        self.tools[definition.tool_id] = definition


class FakeGrantRepository:
    def __init__(self, agents=()):
        self.agents = {agent.agent_id: agent for agent in agents}

    def get_agent(self, agent_id):
        if agent_id not in self.agents:
            raise LookupError("missing")
        return self.agents[agent_id]

    def has_grant(self, agent_id, tool_id):
        return tool_id in self.get_agent(agent_id).tool_ids

    def grant(self, agent_id, tool_id):
        agent = self.get_agent(agent_id)
        self.agents[agent_id] = replace(agent, tool_ids=agent.tool_ids + (tool_id,))
        return self.agents[agent_id]

    def revoke(self, agent_id, tool_id):
        agent = self.get_agent(agent_id)
        self.agents[agent_id] = replace(agent, tool_ids=tuple(x for x in agent.tool_ids if x != tool_id))
        return self.agents[agent_id]


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


class FakeAuditRepository:
    def __init__(self, fail=False):
        self.events = []
        self.fail = fail

    def record(self, event):
        if self.fail:
            raise OSError("private audit storage detail")
        self.events.append(event)


class ToolPolicyTests(unittest.TestCase):
    def setUp(self):
        self.read_tool = ToolDefinition("read_note", "Read note", "Read a note", SCHEMA)
        self.sensitive_tool = ToolDefinition(
            "send_note", "Send note", "Send an external note", SCHEMA,
            risk_level="high", requires_approval=True,
        )
        self.disabled_tool = ToolDefinition("disabled", "Disabled", "", SCHEMA, enabled=False)
        agent = AgentDefinition(
            "writer", "Writer", "", "Prepare notes.",
            tool_ids=("read_note", "send_note", "disabled"),
        )
        self.tool_repo = FakeToolRepository((self.read_tool, self.sensitive_tool, self.disabled_tool))
        self.grants = FakeGrantRepository((agent,))
        self.approvals = FakeApprovalRepository()
        self.audit = FakeAuditRepository()
        self.registry = ToolRegistry(self.tool_repo)
        self.policy = ToolPolicyService(
            registry=self.registry,
            grant_repository=self.grants,
            approval_repository=self.approvals,
            audit_repository=self.audit,
            approval_ttl_seconds=60,
        )
        self.arguments = {"message": "hello", "count": 2}

    def authorize(self, tool_id="read_note", arguments=None, **kwargs):
        return self.policy.authorize(
            "writer", "run-1", tool_id,
            self.arguments if arguments is None else arguments,
            now=NOW, **kwargs
        )

    def test_unknown_tool_is_denied(self):
        decision = self.authorize("missing")
        self.assertEqual(decision.outcome, PolicyOutcome.DENY)
        self.assertEqual(decision.reason, PolicyReason.UNKNOWN_TOOL)

    def test_disabled_tool_is_denied(self):
        decision = self.authorize("disabled")
        self.assertEqual(decision.outcome, PolicyOutcome.DENY)
        self.assertEqual(decision.reason, PolicyReason.DISABLED_TOOL)

    def test_missing_agent_tool_grant_is_denied(self):
        self.grants.revoke("writer", "read_note")
        decision = self.authorize()
        self.assertEqual(decision.outcome, PolicyOutcome.DENY)
        self.assertEqual(decision.reason, PolicyReason.GRANT_MISSING)

    def test_invalid_arguments_and_unknown_fields_are_denied(self):
        missing = self.authorize(arguments={"count": 1})
        extra = self.authorize(arguments={"message": "hello", "secret_extra": "value"})
        bad_type = self.authorize(arguments={"message": "hello", "count": True})
        self.assertEqual(missing.reason, PolicyReason.INVALID_ARGUMENTS)
        self.assertEqual(extra.reason, PolicyReason.INVALID_ARGUMENTS)
        self.assertEqual(bad_type.reason, PolicyReason.INVALID_ARGUMENTS)
        self.assertEqual(missing.outcome, PolicyOutcome.DENY)
        self.assertEqual(extra.outcome, PolicyOutcome.DENY)
        self.assertEqual(bad_type.outcome, PolicyOutcome.DENY)

    def test_circular_and_non_string_object_keys_are_denied(self):
        circular = {"message": "hello"}
        circular["nested"] = circular
        cyclic_decision = self.authorize(arguments=circular)
        key_decision = self.authorize(arguments={"message": "hello", 4: "bad"})
        self.assertEqual(cyclic_decision.reason, PolicyReason.INVALID_ARGUMENTS)
        self.assertEqual(key_decision.reason, PolicyReason.INVALID_ARGUMENTS)

    def test_canonical_digest_is_stable_for_equivalent_object_key_order(self):
        first = self.authorize(arguments={"message": "hello", "count": 2})
        second = self.authorize(arguments={"count": 2, "message": "hello"})
        self.assertEqual(first.outcome, PolicyOutcome.ALLOW)
        self.assertEqual(first.arguments_digest, second.arguments_digest)
        self.assertEqual(first.action_digest, second.action_digest)
        self.assertEqual(len(first.action_digest), 64)

    def test_sensitive_action_requires_approval_then_allows_exact_action(self):
        request = self.authorize("send_note")
        self.assertEqual(request.outcome, PolicyOutcome.REQUIRES_APPROVAL)
        self.assertEqual(request.reason, PolicyReason.APPROVAL_REQUIRED)
        approval = self.approvals.get_approval(request.approval_id)
        self.assertEqual(approval.status, "pending")
        self.assertEqual(approval.agent_id, "writer")
        self.assertEqual(approval.run_id, "run-1")
        self.assertEqual(approval.tool_id, "send_note")
        self.assertEqual(approval.arguments_digest, request.arguments_digest)
        self.policy.decide_approval(request.approval_id, True, "nitin", now=NOW + timedelta(seconds=1))
        allowed = self.authorize("send_note", approval_id=request.approval_id)
        self.assertEqual(allowed.outcome, PolicyOutcome.ALLOW)
        self.assertEqual(allowed.action_digest, request.action_digest)

    def test_approval_digest_binds_its_expiry(self):
        request = self.authorize("send_note")
        pending = self.approvals.get_approval(request.approval_id)
        self.approvals.save_approval(replace(
            pending, expires_at=(NOW + timedelta(hours=1)).isoformat(),
        ))
        decision = self.authorize("send_note", approval_id=request.approval_id)
        self.assertEqual(decision.outcome, PolicyOutcome.DENY)
        self.assertEqual(decision.reason, PolicyReason.APPROVAL_MISMATCH)

    def test_approval_cannot_be_reused_with_changed_arguments_or_context(self):
        request = self.authorize("send_note")
        self.policy.decide_approval(request.approval_id, True, "nitin", now=NOW)
        changed = self.authorize(
            "send_note", arguments={"message": "changed", "count": 2},
            approval_id=request.approval_id,
        )
        self.assertEqual(changed.outcome, PolicyOutcome.DENY)
        self.assertEqual(changed.reason, PolicyReason.APPROVAL_MISMATCH)
        other_run = self.policy.authorize(
            "writer", "run-2", "send_note", self.arguments,
            approval_id=request.approval_id, now=NOW,
        )
        self.assertEqual(other_run.reason, PolicyReason.APPROVAL_MISMATCH)

    def test_expired_approval_is_rejected_and_marked_expired(self):
        request = self.authorize("send_note")
        expired_at = NOW + timedelta(seconds=60)
        decision = self.policy.authorize(
            "writer", "run-1", "send_note", self.arguments,
            approval_id=request.approval_id, now=expired_at,
        )
        self.assertEqual(decision.outcome, PolicyOutcome.DENY)
        self.assertEqual(decision.reason, PolicyReason.APPROVAL_EXPIRED)
        self.assertEqual(self.approvals.get_approval(request.approval_id).status, "expired")
        with self.assertRaisesRegex(ValueError, "expired"):
            self.policy.decide_approval(request.approval_id, True, "nitin", now=expired_at)

    def test_rejected_approval_cannot_be_authorized(self):
        request = self.authorize("send_note")
        self.policy.decide_approval(request.approval_id, False, "nitin", now=NOW)
        decision = self.authorize("send_note", approval_id=request.approval_id)
        self.assertEqual(decision.outcome, PolicyOutcome.DENY)
        self.assertEqual(decision.reason, PolicyReason.APPROVAL_REJECTED)

    def test_approval_is_single_transition_and_cannot_be_reapproved(self):
        request = self.authorize("send_note")
        self.policy.decide_approval(request.approval_id, False, "nitin", now=NOW)
        with self.assertRaisesRegex(ValueError, "pending"):
            self.policy.decide_approval(request.approval_id, True, "nitin", now=NOW)

    def test_audit_records_have_hashed_refs_and_no_raw_arguments_or_secrets(self):
        sensitive_value = "do-not-log-secret-value"
        decision = self.authorize("send_note", arguments={"message": sensitive_value})
        self.assertEqual(decision.outcome, PolicyOutcome.REQUIRES_APPROVAL)
        serialized = json.dumps([event.__dict__ for event in self.audit.events])
        self.assertNotIn(sensitive_value, serialized)
        self.assertNotIn("writer", serialized)
        self.assertNotIn("run-1", serialized)
        self.assertIn(decision.action_digest, serialized)

    def test_audit_failure_fails_closed_for_allow_and_approval(self):
        self.policy.audit = FakeAuditRepository(fail=True)
        allowed = self.authorize()
        approval = self.authorize("send_note")
        self.assertEqual(allowed.outcome, PolicyOutcome.DENY)
        self.assertEqual(allowed.reason, PolicyReason.AUDIT_UNAVAILABLE)
        self.assertEqual(approval.outcome, PolicyOutcome.DENY)
        self.assertEqual(approval.reason, PolicyReason.AUDIT_UNAVAILABLE)
        pending = self.approvals.list_approvals()
        self.assertTrue(all(record.status == "rejected" for record in pending))

    def test_no_tool_callable_or_provider_can_be_invoked(self):
        invocations = []

        class CallableDefinition(ToolDefinition):
            def __call__(self, *args, **kwargs):
                invocations.append((args, kwargs))

        malicious = CallableDefinition("callable", "Callable", "", SCHEMA)
        with self.assertRaises(TypeError):
            self.registry.register(malicious)
        decision = self.authorize()
        self.assertEqual(decision.outcome, PolicyOutcome.ALLOW)
        self.assertEqual(invocations, [])
        self.assertFalse(hasattr(self.policy, "execute"))


class ToolRegistryTests(unittest.TestCase):
    def test_register_inspect_list_and_reject_duplicate_or_non_strict_schema(self):
        repo = FakeToolRepository()
        registry = ToolRegistry(repo)
        definition = ToolDefinition("lookup", "Lookup", "Read-only lookup", SCHEMA)
        self.assertEqual(registry.register(definition), definition)
        self.assertEqual(registry.inspect("lookup"), definition)
        self.assertEqual(registry.list_tools(), (definition,))
        with self.assertRaisesRegex(ValueError, "already registered"):
            registry.register(definition)
        loose_schema = ToolDefinition("loose", "Loose", "", {"type": "object"})
        with self.assertRaisesRegex(ValueError, "additionalProperties"):
            registry.register(loose_schema)

    def test_registry_detaches_mutable_schema_from_caller(self):
        repo = FakeToolRepository()
        registry = ToolRegistry(repo)
        definition = ToolDefinition("lookup", "Lookup", "Read-only lookup", SCHEMA)
        registry.register(definition)
        definition.input_schema["properties"]["message"]["maxLength"] = 0
        fetched = registry.inspect("lookup")
        self.assertEqual(fetched.input_schema["properties"]["message"]["maxLength"], 1000)

    def test_persistent_repositories_use_temp_storage_and_existing_agent_allowlist(self):
        with tempfile.TemporaryDirectory() as directory:
            store = AgentManagerStore(os.path.join(directory, "agents.json"))
            agents = AgentRegistry(store)
            agents.create_agent(AgentDefinition("agent", "Agent", "", "Work"))
            registry = ToolRegistry(AgentManagerToolRepository(store))
            registry.register(ToolDefinition("read", "Read", "", SCHEMA))
            grants = AgentManagerGrantRepository(agents)
            approvals = AgentManagerApprovalRepository(store)
            audit = JsonLinesToolAuditRepository(os.path.join(directory, "audit.jsonl"))
            policy = ToolPolicyService(registry, grants, approvals, audit)
            self.assertEqual(policy.authorize("agent", "run", "read", {"message": "ok"}, now=NOW).outcome,
                             PolicyOutcome.DENY)
            policy.grant_tool("agent", "read")
            decision = policy.authorize("agent", "run", "read", {"message": "ok"}, now=NOW)
            self.assertEqual(decision.outcome, PolicyOutcome.ALLOW)
            with open(os.path.join(directory, "audit.jsonl"), encoding="utf-8") as handle:
                event = json.loads(handle.readline())
            self.assertEqual(event["outcome"], "DENY")
            self.assertNotIn("message", event)
            self.assertEqual(os.stat(os.path.join(directory, "audit.jsonl")).st_mode & 0o777, 0o600)

    def test_persisted_approval_round_trip_is_bound_before_authorization(self):
        with tempfile.TemporaryDirectory() as directory:
            store = AgentManagerStore(os.path.join(directory, "agents.json"))
            agents = AgentRegistry(store)
            agents.create_agent(AgentDefinition("agent", "Agent", "", "Work", tool_ids=("send",)))
            registry = ToolRegistry(AgentManagerToolRepository(store))
            registry.register(ToolDefinition(
                "send", "Send", "Sensitive", SCHEMA,
                risk_level="critical", requires_approval=True,
            ))
            approvals = AgentManagerApprovalRepository(store)
            policy = ToolPolicyService(
                registry, AgentManagerGrantRepository(agents), approvals,
                FakeAuditRepository(),
            )
            pending = policy.authorize("agent", "run", "send", {"message": "confirm"}, now=NOW)
            stored = AgentManagerApprovalRepository(AgentManagerStore(os.path.join(directory, "agents.json")))
            self.assertEqual(stored.get_approval(pending.approval_id).status, "pending")
            policy.decide_approval(pending.approval_id, True, "human-user", now=NOW)
            allowed = policy.authorize(
                "agent", "run", "send", {"message": "confirm"},
                approval_id=pending.approval_id, now=NOW,
            )
            self.assertEqual(allowed.outcome, PolicyOutcome.ALLOW)
            self.assertEqual(stored.get_approval(pending.approval_id).agent_id, "agent")


if __name__ == "__main__":
    unittest.main()
