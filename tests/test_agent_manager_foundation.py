"""Offline tests for Agent Manager contracts and versioned persistence."""

import json
import os
import tempfile
import unittest
from datetime import datetime, timezone

from kripa.agent_manager import (
    AgentDefinition, AgentManagerSnapshot, AgentManagerStore, AgentRun,
    Approval, SkillDefinition, ToolDefinition, WorkflowDefinition,
)


def timestamp():
    return datetime.now(timezone.utc).isoformat()


class AgentManagerContractTests(unittest.TestCase):
    def test_valid_records_cover_all_contract_types(self):
        now = timestamp()
        agent = AgentDefinition("researcher", "Researcher", "Finds facts", "Use cited sources.",
                                model_preferences=("gemini:flash",), skill_ids=("research",),
                                tool_ids=("search",), max_delegation_depth=2)
        run = AgentRun("run-1", "researcher", "Research a topic", "running", now)
        skill = SkillDefinition("research", "Research", "Research skill", "1.0", "Check sources.")
        tool = ToolDefinition("search", "Search", "Search the web", {"type": "object"})
        approval = Approval("approval-1", "run-1", "send_message", "sha256:abc", "pending", now,
                            "2030-01-01T00:00:00+00:00")
        workflow = WorkflowDefinition("flow", "Research flow", "One step",
                                      ({"step_id": "step1", "agent_id": "researcher"},))
        self.assertEqual(agent.max_delegation_depth, 2)
        self.assertEqual(run.status, "running")
        self.assertEqual(skill.version, "1.0")
        self.assertEqual(tool.risk_level, "low")
        self.assertEqual(approval.status, "pending")
        self.assertEqual(workflow.steps[0]["agent_id"], "researcher")

    def test_contracts_reject_invalid_ids_states_and_risk_policy(self):
        with self.assertRaises(ValueError):
            AgentDefinition("../bad", "A", "", "Instructions")
        with self.assertRaises(ValueError):
            AgentRun("run", "agent", "task", "unknown", timestamp())
        with self.assertRaises(ValueError):
            ToolDefinition("delete", "Delete", "", {}, risk_level="critical")
        with self.assertRaises(ValueError):
            WorkflowDefinition("wf", "Workflow", "", ())

    def test_credentials_rejected_from_metadata(self):
        with self.assertRaisesRegex(ValueError, "credential"):
            AgentDefinition("agent", "Agent", "", "Do work", metadata={"nested": {"OPENAI_API_KEY": "secret"}})


class AgentManagerStoreTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.directory.name, "agents", "store.json")
        self.store = AgentManagerStore(self.path)

    def tearDown(self):
        self.directory.cleanup()

    def test_round_trip_all_record_collections_and_version(self):
        now = timestamp()
        snapshot = AgentManagerSnapshot(
            agents=(AgentDefinition("agent", "Agent", "", "Instructions", skill_ids=("skill",)),),
            runs=(AgentRun("run", "agent", "Task", "succeeded", now, completed_at=now),),
            skills=(SkillDefinition("skill", "Skill", "", "1", "Instructions"),),
            tools=(ToolDefinition("tool", "Tool", "", {"type": "object"}),),
            approvals=(Approval("approval", "run", "action", "digest", "approved", now,
                                "2030-01-01T00:00:00+00:00", decided_at=now),),
            workflows=(WorkflowDefinition("workflow", "Workflow", "", (
                {"step_id": "one", "agent_id": "agent"},)),),
        )
        self.store.save(snapshot)
        loaded = self.store.load()
        self.assertFalse(loaded.issues)
        self.assertEqual(loaded, snapshot)
        with open(self.path, encoding="utf-8") as handle:
            payload = json.load(handle)
        self.assertEqual(payload["schema_version"], 1)
        self.assertEqual(set(payload) - {"schema_version"}, set(AgentManagerStore.COLLECTION_TYPES))
        self.assertEqual(os.stat(self.path).st_mode & 0o777, 0o600)

    def test_upsert_preserves_other_records_and_replaces_same_id(self):
        self.store.upsert("agents", AgentDefinition("agent", "First", "", "Do it"))
        self.store.upsert("skills", SkillDefinition("skill", "Skill", "", "1", "Instructions"))
        self.store.upsert("agents", AgentDefinition("agent", "Updated", "", "Do it"))
        loaded = self.store.load()
        self.assertEqual(len(loaded.agents), 1)
        self.assertEqual(loaded.agents[0].name, "Updated")
        self.assertEqual(len(loaded.skills), 1)

    def test_rejects_credentials_and_refuses_to_overwrite_malformed_store(self):
        with self.assertRaisesRegex(ValueError, "credential"):
            self.store.save(AgentManagerSnapshot(agents=(
                AgentDefinition("agent", "Agent", "", "Instructions", metadata={"api_key": "nope"}),
            )))
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as handle:
            handle.write("not json")
        with self.assertRaisesRegex(ValueError, "incomplete"):
            self.store.upsert("agents", AgentDefinition("agent", "Agent", "", "Instructions"))
        with open(self.path, encoding="utf-8") as handle:
            self.assertEqual(handle.read(), "not json")

    def test_unknown_schema_is_reported_and_preserved(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as handle:
            json.dump({"schema_version": 99}, handle)
        loaded = self.store.load()
        self.assertTrue(loaded.issues)
        with self.assertRaises(ValueError):
            self.store.save(loaded)


if __name__ == "__main__":
    unittest.main()
