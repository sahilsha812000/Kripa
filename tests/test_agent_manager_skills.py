"""Offline tests for local skill discovery and explicitly approved lifecycle."""

import io
import json
import os
import tempfile
import threading
import unittest

from kripa.agent_manager import AgentDefinition, AgentManagerStore, AgentRegistry
from kripa.agent_manager.skills import SkillManager
from kripa.agent_manager.skills.cli import run_skills


class SkillManagerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = os.path.join(self.temporary.name, "catalog")
        os.makedirs(self.root)
        self.package = os.path.join(self.root, "writing-helper")
        self.manager = SkillManager(AgentManagerStore(os.path.join(self.temporary.name, "store.json")))
        self.write_package()

    def tearDown(self):
        self.temporary.cleanup()

    def write_package(self, version="1.2.3", instructions="# Writing helper\n\nUse clear language.\n", compatibility=None):
        os.makedirs(self.package, exist_ok=True)
        manifest = {
            "schema_version": 1,
            "skill_id": "writing-helper",
            "name": "Writing Helper",
            "description": "Helps write clear prose.",
            "version": version,
            "entrypoint": "SKILL.md",
            "compatibility": compatibility or {"kripa_min_version": "0.1.0"},
        }
        with open(os.path.join(self.package, "skill.json"), "w", encoding="utf-8") as handle:
            json.dump(manifest, handle)
        with open(os.path.join(self.package, "SKILL.md"), "w", encoding="utf-8") as handle:
            handle.write(instructions)

    def test_discovery_and_inspection_read_metadata_and_digest(self):
        results = self.manager.discover(self.root)
        self.assertEqual(len(results), 1)
        inspection = results[0]
        self.assertEqual(inspection.skill_id, "writing-helper")
        self.assertEqual(inspection.version, "1.2.3")
        self.assertTrue(inspection.safe_to_consider)
        self.assertEqual(len(inspection.sha256), 64)
        self.assertIn("Use clear language", inspection.instructions)

    def test_invalid_manifest_extra_files_and_symlink_packages_are_not_discovered(self):
        broken = os.path.join(self.root, "broken")
        os.makedirs(broken)
        with open(os.path.join(broken, "skill.json"), "w", encoding="utf-8") as handle:
            handle.write("{}")
        with open(os.path.join(broken, "SKILL.md"), "w", encoding="utf-8") as handle:
            handle.write("Instructions")
        self.assertEqual([item.skill_id for item in self.manager.discover(self.root)], ["writing-helper"])
        with self.assertRaisesRegex(ValueError, "manifest"):
            self.manager.inspect(broken)
        with open(os.path.join(self.package, "helper.py"), "w", encoding="utf-8") as handle:
            handle.write("raise RuntimeError('must not run')")
        with self.assertRaisesRegex(ValueError, "unsupported files"):
            self.manager.inspect(self.package)
        os.unlink(os.path.join(self.package, "helper.py"))
        link = os.path.join(self.root, "linked")
        os.symlink(self.package, link)
        with self.assertRaisesRegex(ValueError, "symlink"):
            self.manager.inspect(link)

    def test_install_requires_matching_explicit_approval_digest(self):
        inspection = self.manager.inspect(self.package)
        with self.assertRaisesRegex(ValueError, "approval"):
            self.manager.install(self.package)
        with self.assertRaisesRegex(ValueError, "digest"):
            self.manager.install(self.package, approval_digest="0" * 64)
        installed = self.manager.install(self.package, approval_digest=inspection.sha256)
        self.assertEqual(installed.skill_id, "writing-helper")
        self.assertEqual(installed.version, "1.2.3")
        self.assertEqual(installed.metadata["content_sha256"], inspection.sha256)
        self.assertEqual(installed.metadata["approval"], "human_digest_confirmation")
        self.assertEqual(self.manager.list_skills(), (installed,))

    def test_lifecycle_enable_disable_and_remove_preserves_agent_references(self):
        inspection = self.manager.inspect(self.package)
        installed = self.manager.install(self.package, approval_digest=inspection.sha256)
        disabled = self.manager.disable(installed.skill_id)
        self.assertFalse(disabled.enabled)
        self.assertEqual(self.manager.list_skills(include_disabled=False), ())
        self.assertTrue(self.manager.enable(installed.skill_id).enabled)
        registry = AgentRegistry(self.manager.store)
        registry.create_agent(AgentDefinition("writer", "Writer", "", "Instructions", skill_ids=(installed.skill_id,)))
        with self.assertRaisesRegex(ValueError, "active agent"):
            self.manager.remove(installed.skill_id)
        registry.update_agent("writer", skill_ids=())
        self.manager.remove(installed.skill_id)
        self.assertEqual(self.manager.list_skills(), ())

    def test_upgrade_requires_newer_version_and_tracks_installed_version(self):
        inspection = self.manager.inspect(self.package)
        self.manager.install(self.package, approval_digest=inspection.sha256)
        with self.assertRaisesRegex(ValueError, "newer"):
            self.manager.install(self.package, approval_digest=inspection.sha256)
        self.write_package(version="1.3.0")
        upgraded = self.manager.inspect(self.package)
        result = self.manager.install(self.package, approval_digest=upgraded.sha256)
        self.assertEqual(result.version, "1.3.0")

    def test_security_and_compatibility_block_install(self):
        self.write_package(instructions="Set OPENAI_API_KEY=sk-abcdefghijk1234567890")
        inspection = self.manager.inspect(self.package)
        self.assertFalse(inspection.safe_to_consider)
        self.assertTrue(inspection.safety_issues)
        with self.assertRaisesRegex(ValueError, "security"):
            self.manager.install(self.package, approval_digest=inspection.sha256)
        self.write_package(instructions="Safe instructions.", compatibility={"kripa_max_version": "0.0.1"})
        incompatible = self.manager.inspect(self.package)
        self.assertTrue(incompatible.compatibility_issues)
        with self.assertRaisesRegex(ValueError, "incompatible"):
            self.manager.install(self.package, approval_digest=incompatible.sha256)

    def test_concurrent_skill_operations_preserve_state(self):
        """Concurrent skill enable/disable operations use the protected transaction path and do not lose state."""
        inspection = self.manager.inspect(self.package)
        self.manager.install(self.package, approval_digest=inspection.sha256)
        pkg2 = os.path.join(self.root, "second-helper")
        os.makedirs(pkg2, exist_ok=True)
        manifest = {
            "schema_version": 1, "skill_id": "second-helper", "name": "Second Helper",
            "description": "Second", "version": "1.0.0", "entrypoint": "SKILL.md",
            "compatibility": {"kripa_min_version": "0.1.0"},
        }
        with open(os.path.join(pkg2, "skill.json"), "w", encoding="utf-8") as f:
            json.dump(manifest, f)
        with open(os.path.join(pkg2, "SKILL.md"), "w", encoding="utf-8") as f:
            f.write("# Second helper\n")
        insp2 = self.manager.inspect(pkg2)
        self.manager.install(pkg2, approval_digest=insp2.sha256)

        errors = []
        def op1():
            try:
                self.manager.disable("writing-helper")
            except Exception as e:
                errors.append(e)

        def op2():
            try:
                self.manager.enable("second-helper")
            except Exception as e:
                errors.append(e)

        t1 = threading.Thread(target=op1)
        t2 = threading.Thread(target=op2)
        t1.start(); t2.start()
        t1.join(); t2.join()

        self.assertEqual(errors, [])
        self.assertFalse(self.manager.get_skill("writing-helper").enabled)
        self.assertTrue(self.manager.get_skill("second-helper").enabled)

    def test_cli_inspection_redacts_possible_credentials_and_install_is_approved(self):
        self.write_package(instructions="OPENAI_API_KEY=sk-abcdefghijk1234567890")
        output = io.StringIO()
        errors = io.StringIO()
        self.assertEqual(run_skills(["inspect", self.package], manager=self.manager, stdout=output, stderr=errors), 0)
        self.assertNotIn("sk-abcdefghijk1234567890", output.getvalue())
        self.assertIn("hidden: possible credential", output.getvalue())
        self.assertEqual(run_skills(["install", self.package, "--approve-sha256", "0" * 64], manager=self.manager,
                                    stdout=output, stderr=errors), 2)
        self.assertNotIn("sk-abcdefghijk1234567890", errors.getvalue())


if __name__ == "__main__":
    unittest.main()
