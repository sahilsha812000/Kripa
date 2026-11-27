"""Tests for Agent Registry lifecycle and CLI metadata management."""

import io
import os
import tempfile
import threading
import unittest

from kripa.agent_manager import (
    AgentAlreadyExistsError, AgentDefinition, AgentManagerStore,
    AgentNotFoundError, AgentRegistry,
)
from kripa.agent_manager.cli import run_agents


class AgentRegistryTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.registry = AgentRegistry(AgentManagerStore(os.path.join(self.directory.name, "agents.json")))
        self.definition = AgentDefinition(
            "writer", "Writer", "Drafts clear copy", "Write concise, accurate prose.",
            model_preferences=("gemini:configured",), skill_ids=("style",), tool_ids=("read_file",),
        )

    def tearDown(self):
        self.directory.cleanup()

    def test_create_and_list_agent(self):
        created = self.registry.create_agent(self.definition)
        self.assertEqual(created, self.definition)
        self.assertEqual(self.registry.list_agents(), (self.definition,))
        self.assertEqual(self.registry.list_agents(include_retired=False), (self.definition,))

    def test_create_rejects_duplicate_id(self):
        self.registry.create_agent(self.definition)
        with self.assertRaises(AgentAlreadyExistsError):
            self.registry.create_agent(AgentDefinition("writer", "Other", "", "Instructions"))

    def test_get_returns_details_and_unknown_id_is_rejected(self):
        self.registry.create_agent(self.definition)
        self.assertEqual(self.registry.get_agent("writer"), self.definition)
        with self.assertRaises(AgentNotFoundError):
            self.registry.get_agent("missing")

    def test_update_validates_and_persists_all_supported_fields(self):
        self.registry.create_agent(self.definition)
        updated = self.registry.update_agent(
            "writer", name="Senior Writer", description="Edits drafts",
            instructions="Edit for clarity.", model_preferences=("ollama:local",),
            skill_ids=("editing",), tool_ids=(), parent_agent_id="lead",
            max_delegation_depth=1,
        )
        self.assertEqual(updated.name, "Senior Writer")
        self.assertEqual(self.registry.get_agent("writer"), updated)
        with self.assertRaises(ValueError):
            self.registry.update_agent("writer", name="")
        with self.assertRaises(TypeError):
            self.registry.update_agent("writer", agent_id="different")
        with self.assertRaises(ValueError):
            self.registry.update_agent("writer")

    def test_retire_disables_but_preserves_definition(self):
        self.registry.create_agent(self.definition)
        retired = self.registry.retire_agent("writer")
        self.assertFalse(retired.enabled)
        self.assertEqual(self.registry.list_agents(), (retired,))
        self.assertEqual(self.registry.list_agents(include_retired=False), ())
        self.assertEqual(self.registry.retire_agent("writer"), retired)

    def test_registry_rejects_invalid_definition_type(self):
        with self.assertRaises(TypeError):
            self.registry.create_agent({"agent_id": "writer"})

    def test_concurrent_create_agent_preserves_every_agent(self):
        """Concurrent AgentRegistry.create_agent calls must preserve every agent without lost updates."""
        agents_to_create = [
            AgentDefinition("agent-{}".format(i), "Agent {}".format(i), "", "Instructions")
            for i in range(8)
        ]
        errors = []

        def worker(agent_def):
            try:
                self.registry.create_agent(agent_def)
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(a,)) for a in agents_to_create]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(errors, [])
        persisted_ids = {a.agent_id for a in self.registry.list_agents()}
        expected_ids = {a.agent_id for a in agents_to_create}
        self.assertEqual(persisted_ids, expected_ids)

    def test_concurrent_create_same_agent_allows_only_one(self):
        """Only one concurrent creator succeeds for the same agent ID; others get AgentAlreadyExistsError."""
        definition = AgentDefinition("same-agent", "Same Agent", "", "Instructions")
        successes = []
        already_exists = []
        other_errors = []

        def worker():
            try:
                self.registry.create_agent(definition)
                successes.append(True)
            except AgentAlreadyExistsError:
                already_exists.append(True)
            except Exception as exc:
                other_errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(other_errors, [])
        self.assertEqual(len(successes), 1)
        self.assertEqual(len(already_exists), 7)
        self.assertEqual(len(self.registry.list_agents()), 1)

    def test_concurrent_update_and_retire_do_not_lose_updates(self):
        """Concurrent update and retire operations do not overwrite each other."""
        self.registry.create_agent(AgentDefinition("agent-a", "Agent A", "", "Instructions"))
        self.registry.create_agent(AgentDefinition("agent-b", "Agent B", "", "Instructions"))
        errors = []

        def updater():
            try:
                self.registry.update_agent("agent-a", name="Updated Agent A")
            except Exception as exc:
                errors.append(exc)

        def retirer():
            try:
                self.registry.retire_agent("agent-b")
            except Exception as exc:
                errors.append(exc)

        t1 = threading.Thread(target=updater)
        t2 = threading.Thread(target=retirer)
        t1.start(); t2.start()
        t1.join(); t2.join()

        self.assertEqual(errors, [])
        agent_a = self.registry.get_agent("agent-a")
        agent_b = self.registry.get_agent("agent-b")
        self.assertEqual(agent_a.name, "Updated Agent A")
        self.assertFalse(agent_b.enabled)


class AgentRegistryCliTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.registry = AgentRegistry(AgentManagerStore(os.path.join(self.directory.name, "agents.json")))
        self.stdout = io.StringIO()
        self.stderr = io.StringIO()

    def tearDown(self):
        self.directory.cleanup()

    def run_cli(self, args):
        return run_agents(args, registry=self.registry, stdout=self.stdout, stderr=self.stderr)

    def test_cli_create_list_show_update_retire(self):
        self.assertEqual(self.run_cli(["create", "writer", "--name", "Writer", "--instructions", "Write clearly."]), 0)
        self.assertEqual(self.run_cli(["list"]), 0)
        self.assertIn("writer\tactive\tWriter", self.stdout.getvalue())
        self.assertEqual(self.run_cli(["show", "writer"]), 0)
        self.assertIn('"instructions_configured": true', self.stdout.getvalue())
        self.assertNotIn("Write clearly.", self.stdout.getvalue())
        self.assertEqual(self.run_cli(["update", "writer", "--name", "Editor"]), 0)
        self.assertEqual(self.registry.get_agent("writer").name, "Editor")
        self.assertEqual(self.run_cli(["retire", "writer"]), 0)
        self.assertFalse(self.registry.get_agent("writer").enabled)

    def test_cli_duplicate_and_missing_agent_errors_are_safe(self):
        self.registry.create_agent(AgentDefinition("writer", "Writer", "", "Sensitive key should not print"))
        self.assertEqual(self.run_cli(["create", "writer", "--name", "Writer", "--instructions", "Do work"]), 2)
        self.assertEqual(self.run_cli(["show", "missing"]), 2)
        self.assertNotIn("Sensitive key", self.stderr.getvalue())
        self.assertNotIn("missing", self.stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
