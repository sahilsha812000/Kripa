"""Offline tests for Agent Manager contracts and versioned persistence."""

import errno
import json
import os
import stat
import sys
import tempfile
import threading
import time
import unittest
from datetime import datetime, timezone
from unittest.mock import patch

from kripa.agent_manager import (
    AgentDefinition, AgentManagerSnapshot, AgentManagerStore, AgentRun,
    Approval, SkillDefinition, ToolDefinition, WorkflowDefinition,
)
from kripa.agent_manager.store import (
    _LOCK_RETRY_INTERVAL, _LOCK_TIMEOUT_SECONDS, _StoreWriteLock,
)

# Sentinel used to distinguish "key absent" from "key mapped to None" in sys.modules.
_SENTINEL = object()


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


class StoreWriteLockTests(unittest.TestCase):
    """Step 3 security regression: concurrent write protection and lock safety."""

    # ------------------------------------------------------------------ helpers

    def _store(self, directory, timeout=_LOCK_TIMEOUT_SECONDS,
               retry=_LOCK_RETRY_INTERVAL):
        path = os.path.join(directory, "agents", "store.json")
        return AgentManagerStore(path, lock_timeout=timeout,
                                 lock_retry_interval=retry)

    @staticmethod
    def _agent(agent_id, name="Agent"):
        return AgentDefinition(agent_id, name, "", "Instructions")

    # ------------------------------------------------------------------ tests

    def test_lock_file_is_created_with_restrictive_permissions(self):
        with tempfile.TemporaryDirectory() as d:
            store = self._store(d)
            store.upsert("agents", self._agent("alpha"))
            lock_path = store._lock_path
            self.assertTrue(os.path.exists(lock_path))
            mode = os.stat(lock_path).st_mode & 0o777
            self.assertEqual(mode, 0o600, "Lock file must be owner-only (0o600)")

    def test_concurrent_upserts_produce_correct_merged_state(self):
        """Two threads each upsert a distinct agent; both must survive."""
        with tempfile.TemporaryDirectory() as d:
            store = self._store(d)
            errors = []

            def writer(agent_id):
                try:
                    store.upsert("agents", self._agent(agent_id))
                except Exception as exc:
                    errors.append(exc)

            t1 = threading.Thread(target=writer, args=("agent-one",))
            t2 = threading.Thread(target=writer, args=("agent-two",))
            t1.start(); t2.start()
            t1.join(); t2.join()

            self.assertEqual(errors, [], "No thread should raise: {}".format(errors))
            loaded = store.load()
            ids = {a.agent_id for a in loaded.agents}
            self.assertEqual(ids, {"agent-one", "agent-two"},
                             "Both agents must be persisted; a lost update occurred")

    def test_lock_timeout_raises_timeout_error(self):
        """Holding the lock in a background thread must block a concurrent upsert."""
        try:
            import fcntl
        except ImportError:
            self.skipTest("fcntl not available on this platform")

        with tempfile.TemporaryDirectory() as d:
            store = self._store(d, timeout=0.15, retry=0.01)
            lock_path = store._lock_path
            # Ensure lock file exists before the background thread opens it.
            os.makedirs(os.path.dirname(lock_path), mode=0o700, exist_ok=True)
            lock_fd = os.open(
                lock_path,
                os.O_WRONLY | os.O_CREAT | os.O_NOFOLLOW | os.O_NONBLOCK,
                0o600,
            )
            fcntl.flock(lock_fd, fcntl.LOCK_EX)
            try:
                with self.assertRaises(TimeoutError) as ctx:
                    store.upsert("agents", self._agent("blocked"))
                self.assertIn("could not be acquired", str(ctx.exception))
            finally:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
                os.close(lock_fd)

    def test_lock_is_released_after_exception_in_write_body(self):
        """An exception during the locked section must still release the lock."""
        with tempfile.TemporaryDirectory() as d:
            store = self._store(d, timeout=0.15, retry=0.01)
            # An AgentManagerSnapshot with issues is refused by _write(); this
            # simulates a failure that occurs while the lock is held.
            bad_snapshot = AgentManagerSnapshot(issues=("forced error",))
            with self.assertRaises(Exception):
                store.save(bad_snapshot)
            # The lock must now be acquirable again.
            store.upsert("agents", self._agent("recovery-agent"))
            self.assertEqual(len(store.load().agents), 1)

    def test_lock_is_released_after_exception_during_upsert(self):
        """A validation error during upsert must still release the lock."""
        with tempfile.TemporaryDirectory() as d:
            store = self._store(d, timeout=0.15, retry=0.01)
            # First write populates the store with a valid record.
            store.upsert("agents", self._agent("first"))
            # Inject a corrupt store file; `_write` will be called but
            # `load()` inside the locked section will return issues and raise.
            with open(store.path, "w", encoding="utf-8") as handle:
                handle.write("not json")
            with self.assertRaises(Exception):
                store.upsert("agents", self._agent("second"))
            # Restore the store and verify the lock is not stuck.
            store.save(AgentManagerSnapshot(agents=(self._agent("restored"),)))
            self.assertEqual(store.load().agents[0].agent_id, "restored")

    def test_save_and_upsert_hold_lock_in_same_lock_path(self):
        """Both write entry points use the same lock file."""
        with tempfile.TemporaryDirectory() as d:
            store = self._store(d)
            store.save(AgentManagerSnapshot())
            store.upsert("agents", self._agent("x"))
            # Both must have created/used the same lock path.
            self.assertTrue(os.path.exists(store._lock_path))

    def test_fcntl_unavailable_raises_os_error_not_silent_no_op(self):
        """When fcntl cannot be imported, writes must fail closed, not silently."""
        with tempfile.TemporaryDirectory() as d:
            store = self._store(d)
            # Temporarily hide fcntl from the import system.
            original = sys.modules.get("fcntl", _SENTINEL)
            sys.modules["fcntl"] = None  # causes ImportError on `import fcntl`
            try:
                with self.assertRaises(OSError) as ctx:
                    store.upsert("agents", self._agent("shouldfail"))
                self.assertIn("fcntl", str(ctx.exception))
            finally:
                if original is _SENTINEL:
                    del sys.modules["fcntl"]
                else:
                    sys.modules["fcntl"] = original

    def test_read_load_does_not_acquire_lock(self):
        """load() must succeed even when the write lock is held externally."""
        try:
            import fcntl
        except ImportError:
            self.skipTest("fcntl not available on this platform")

        with tempfile.TemporaryDirectory() as d:
            store = self._store(d)
            store.save(AgentManagerSnapshot(agents=(self._agent("readable"),)))
            lock_path = store._lock_path
            lock_fd = os.open(
                lock_path,
                os.O_WRONLY | os.O_CREAT | os.O_NOFOLLOW | os.O_NONBLOCK,
                0o600,
            )
            fcntl.flock(lock_fd, fcntl.LOCK_EX)
            try:
                # load() must not block or raise while LOCK_EX is held.
                loaded = store.load()
                self.assertEqual(len(loaded.agents), 1)
            finally:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
                os.close(lock_fd)

    def test_unexpected_flock_error_fails_immediately(self):
        """Unexpected OS errors from flock (e.g. EINVAL) must fail immediately without retrying."""
        try:
            import fcntl
        except ImportError:
            self.skipTest("fcntl not available on this platform")

        with tempfile.TemporaryDirectory() as d:
            store = self._store(d, timeout=2.0, retry=0.05)
            with patch("fcntl.flock", side_effect=OSError(errno.EINVAL, "Invalid argument")):
                start = time.monotonic()
                with self.assertRaises(OSError) as ctx:
                    with _StoreWriteLock(store._lock_path, store._lock_timeout, store._lock_retry_interval):
                        pass
                elapsed = time.monotonic() - start
                self.assertLess(elapsed, 0.5, "Unexpected error must not retry until timeout")
                self.assertEqual(ctx.exception.errno, errno.EINVAL)

    def test_file_descriptor_closed_on_unexpected_flock_error(self):
        """When flock raises an unexpected error, the opened file descriptor must be closed."""
        try:
            import fcntl
        except ImportError:
            self.skipTest("fcntl not available on this platform")

        with tempfile.TemporaryDirectory() as d:
            store = self._store(d)
            opened_fds = []
            orig_open = os.open

            def tracking_open(*args, **kwargs):
                fd = orig_open(*args, **kwargs)
                opened_fds.append(fd)
                return fd

            with patch("os.open", side_effect=tracking_open), \
                 patch("fcntl.flock", side_effect=OSError(errno.EBADF, "Bad file descriptor")):
                with self.assertRaises(OSError):
                    with _StoreWriteLock(store._lock_path, store._lock_timeout, store._lock_retry_interval):
                        pass

            self.assertEqual(len(opened_fds), 1)
            with self.assertRaises(OSError) as ctx:
                os.fstat(opened_fds[0])
            self.assertEqual(ctx.exception.errno, errno.EBADF, "File descriptor must be closed")

    def test_file_descriptor_closed_on_lock_timeout(self):
        """When lock acquisition times out, the opened file descriptor must be closed."""
        try:
            import fcntl
        except ImportError:
            self.skipTest("fcntl not available on this platform")

        with tempfile.TemporaryDirectory() as d:
            store = self._store(d, timeout=0.1, retry=0.02)
            opened_fds = []
            orig_open = os.open

            def tracking_open(*args, **kwargs):
                fd = orig_open(*args, **kwargs)
                opened_fds.append(fd)
                return fd

            lock_path = store._lock_path
            os.makedirs(os.path.dirname(lock_path), mode=0o700, exist_ok=True)
            holder_fd = os.open(lock_path, os.O_WRONLY | os.O_CREAT | os.O_NOFOLLOW | os.O_NONBLOCK, 0o600)
            fcntl.flock(holder_fd, fcntl.LOCK_EX)
            try:
                with patch("os.open", side_effect=tracking_open):
                    with self.assertRaises(TimeoutError):
                        with _StoreWriteLock(lock_path, store._lock_timeout, store._lock_retry_interval):
                            pass
            finally:
                fcntl.flock(holder_fd, fcntl.LOCK_UN)
                os.close(holder_fd)

            self.assertEqual(len(opened_fds), 1)
            with self.assertRaises(OSError) as ctx:
                os.fstat(opened_fds[0])
            self.assertEqual(ctx.exception.errno, errno.EBADF, "File descriptor must be closed")

    def test_fchmod_failure_causes_fail_closed_and_closes_fd(self):
        """Any fchmod failure must fail closed with an OSError and close the file descriptor."""
        with tempfile.TemporaryDirectory() as d:
            store = self._store(d)
            opened_fds = []
            orig_open = os.open

            def tracking_open(*args, **kwargs):
                fd = orig_open(*args, **kwargs)
                opened_fds.append(fd)
                return fd

            with patch("os.open", side_effect=tracking_open), \
                 patch("os.fchmod", side_effect=OSError(errno.EPERM, "Operation not permitted")):
                with self.assertRaises(OSError) as ctx:
                    with _StoreWriteLock(store._lock_path, store._lock_timeout, store._lock_retry_interval):
                        pass

            self.assertEqual(ctx.exception.errno, errno.EPERM)
            self.assertEqual(len(opened_fds), 1)
            with self.assertRaises(OSError) as ctx_close:
                os.fstat(opened_fds[0])
            self.assertEqual(ctx_close.exception.errno, errno.EBADF, "File descriptor must be closed")

    def test_symlink_lock_file_is_rejected(self):
        """A symlink lock file must be rejected via O_NOFOLLOW and fail closed."""
        with tempfile.TemporaryDirectory() as d:
            store = self._store(d)
            real_target = os.path.join(d, "target_file")
            with open(real_target, "w") as f:
                f.write("target")
            os.makedirs(os.path.dirname(store._lock_path), mode=0o700, exist_ok=True)
            os.symlink(real_target, store._lock_path)
            with self.assertRaises(OSError):
                with _StoreWriteLock(store._lock_path, store._lock_timeout, store._lock_retry_interval):
                    pass

    def test_fifo_lock_file_is_rejected_without_hanging(self):
        """A FIFO at the lock path must be rejected immediately without hanging."""
        if not hasattr(os, "mkfifo"):
            self.skipTest("mkfifo not available on this platform")

        with tempfile.TemporaryDirectory() as d:
            store = self._store(d)
            os.makedirs(os.path.dirname(store._lock_path), mode=0o700, exist_ok=True)
            os.mkfifo(store._lock_path)
            start = time.monotonic()
            with self.assertRaises(OSError):
                with _StoreWriteLock(store._lock_path, store._lock_timeout, store._lock_retry_interval):
                    pass
            elapsed = time.monotonic() - start
            self.assertLess(elapsed, 0.5, "FIFO must be rejected immediately without blocking")

    def test_transaction_aborts_and_writes_nothing_on_error(self):
        """When a transaction mutator raises, no data is written to the store."""
        with tempfile.TemporaryDirectory() as d:
            store = self._store(d)
            store.upsert("agents", self._agent("existing"))

            def failing_mutator(snapshot):
                raise RuntimeError("mutation failure")

            with self.assertRaises(RuntimeError):
                store.transaction(failing_mutator)

            loaded = store.load()
            self.assertEqual(len(loaded.agents), 1)
            self.assertEqual(loaded.agents[0].agent_id, "existing")

    def test_transaction_commits_mutation_atomically(self):
        """A transaction mutator returns (new_snapshot, return_val) and persists atomically."""
        with tempfile.TemporaryDirectory() as d:
            store = self._store(d)

            def add_agent(snapshot):
                new_agent = self._agent("trans-1", name="Transactional Agent")
                new_snapshot = AgentManagerSnapshot(
                    agents=(*snapshot.agents, new_agent),
                    runs=snapshot.runs,
                    skills=snapshot.skills,
                    tools=snapshot.tools,
                    approvals=snapshot.approvals,
                    workflows=snapshot.workflows,
                )
                return new_snapshot, new_agent

            result = store.transaction(add_agent)
            self.assertEqual(result.agent_id, "trans-1")
            loaded = store.load()
            self.assertEqual(len(loaded.agents), 1)
            self.assertEqual(loaded.agents[0].name, "Transactional Agent")

    def test_transaction_requires_callable(self):
        """store.transaction rejects non-callable mutator argument."""
        with tempfile.TemporaryDirectory() as d:
            store = self._store(d)
            with self.assertRaises(TypeError):
                store.transaction("not a callable")

    def test_o_nonblock_unavailable_fails_closed(self):
        """When O_NONBLOCK is missing from os, locking must fail closed with an OSError."""
        with tempfile.TemporaryDirectory() as d:
            store = self._store(d)
            orig_nonblock = getattr(os, "O_NONBLOCK", _SENTINEL)
            if orig_nonblock is not _SENTINEL:
                del os.O_NONBLOCK
            try:
                with self.assertRaises(OSError) as ctx:
                    with _StoreWriteLock(store._lock_path, store._lock_timeout, store._lock_retry_interval):
                        pass
                self.assertIn("O_NONBLOCK", str(ctx.exception))
            finally:
                if orig_nonblock is not _SENTINEL:
                    os.O_NONBLOCK = orig_nonblock

    def test_fdopen_failure_closes_opened_descriptor(self):
        """When os.fdopen() fails during _write(), the mkstemp descriptor must be closed."""
        with tempfile.TemporaryDirectory() as d:
            store = self._store(d)
            opened_descriptors = []
            orig_mkstemp = tempfile.mkstemp

            def tracking_mkstemp(*args, **kwargs):
                fd, path = orig_mkstemp(*args, **kwargs)
                opened_descriptors.append(fd)
                return fd, path

            with patch("tempfile.mkstemp", side_effect=tracking_mkstemp), \
                 patch("os.fdopen", side_effect=OSError(errno.EMFILE, "Too many open files")):
                with self.assertRaises(OSError):
                    store.save(AgentManagerSnapshot())

            self.assertEqual(len(opened_descriptors), 1)
            with self.assertRaises(OSError) as ctx:
                os.fstat(opened_descriptors[0])
            self.assertEqual(ctx.exception.errno, errno.EBADF, "Descriptor must be closed")

    def test_no_redundant_chmod_after_replace_preserves_file_mode(self):
        """File permissions are set on the temporary file before replacement; no trailing chmod on store.path."""
        with tempfile.TemporaryDirectory() as d:
            store = self._store(d)
            store.save(AgentManagerSnapshot())
            self.assertTrue(os.path.exists(store.path))
            mode = os.stat(store.path).st_mode & 0o777
            self.assertEqual(mode, 0o600)

            chmod_calls = []
            orig_chmod = os.chmod

            def tracking_chmod(path, mode):
                chmod_calls.append((path, mode))
                return orig_chmod(path, mode)

            with patch("os.chmod", side_effect=tracking_chmod):
                store.save(AgentManagerSnapshot())

            target_path = os.path.abspath(store.path)
            for path, _ in chmod_calls:
                self.assertNotEqual(
                    os.path.abspath(path), target_path,
                    "Redundant chmod on store.path after atomic replacement must not occur",
                )


if __name__ == "__main__":
    unittest.main()
