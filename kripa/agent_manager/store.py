"""Versioned, atomic local JSON storage for Agent Manager records."""

import errno
import json
import os
import stat
import tempfile
import time
from dataclasses import dataclass
from typing import Tuple

from kripa.agent_manager.contracts import (
    AgentDefinition, AgentRun, Approval, SkillDefinition, ToolDefinition,
    WorkflowDefinition, validate_json_value,
)


_LOCK_TIMEOUT_SECONDS = 2.0
_LOCK_RETRY_INTERVAL = 0.05  # 50 ms between attempts


class _StoreWriteLock:
    """Exclusive advisory file lock protecting AgentManagerStore write operations.

    Uses ``fcntl.flock(LOCK_EX | LOCK_NB)`` on POSIX/macOS with a retry loop
    so that short-lived concurrent writers queue rather than fail immediately.
    If the lock cannot be acquired within *timeout* seconds a ``TimeoutError``
    is raised and the caller's write does not proceed.

    On platforms where ``fcntl`` or ``os.O_NOFOLLOW`` is unavailable, the lock
    cannot be provided safely. The context manager raises ``OSError`` rather
    than falling back to an unsafe no-op; failing closed prevents silent data
    loss or symlink attacks.

    Implementation notes
    --------------------
    * The lock file is opened with ``O_NOFOLLOW`` and ``O_NONBLOCK`` (where
      supported) so that symlinks fail closed immediately and FIFOs do not block
      indefinitely on open.
    * ``os.fstat()`` and ``stat.S_ISREG()`` verify that the opened lock path is
      a regular file.
    * ``os.fchmod(fd, 0o600)`` ensures restrictive permissions and fails closed
      if setting mode fails.
    * The descriptor is closed on every acquisition failure (timeout, unexpected
      OS errors, interrupts, non-regular files).
    * Only genuine lock-contention errors (EAGAIN, EWOULDBLOCK, and EACCES)
      trigger retries; unexpected OS errors fail immediately and preserve the
      original exception.
    * The lock file is separate from ``store.json`` so the exclusive descriptor
      remains open across the full READ -> MODIFY -> WRITE transaction.
    """

    def __init__(self, lock_path, timeout=_LOCK_TIMEOUT_SECONDS,
                 retry_interval=_LOCK_RETRY_INTERVAL):
        self._lock_path = lock_path
        self._timeout = timeout
        self._retry_interval = retry_interval
        self._fd = None

    def __enter__(self):
        try:
            import fcntl as _fcntl
        except ImportError:
            raise OSError(
                "Advisory file locking requires fcntl (POSIX/macOS). "
                "AgentManagerStore write operations cannot proceed safely on "
                "this platform. Failing closed to prevent silent data loss."
            )

        if not hasattr(os, "O_NOFOLLOW"):
            raise OSError(
                "O_NOFOLLOW is required for secure file locking but is unavailable "
                "on this platform. Failing closed to prevent symlink attacks."
            )
        if not hasattr(os, "O_NONBLOCK"):
            raise OSError(
                "O_NONBLOCK is required for secure file locking but is unavailable "
                "on this platform. Failing closed to prevent blocking on non-regular files."
            )

        directory = os.path.dirname(os.path.abspath(self._lock_path))
        os.makedirs(directory, mode=0o700, exist_ok=True)

        flags = os.O_WRONLY | os.O_CREAT | os.O_NOFOLLOW | os.O_NONBLOCK
        fd = os.open(self._lock_path, flags, 0o600)
        acquired = False
        try:
            os.fchmod(fd, 0o600)

            info = os.fstat(fd)
            if not stat.S_ISREG(info.st_mode):
                raise OSError(
                    "Lock path '{}' is not a regular file.".format(self._lock_path)
                )

            deadline = time.monotonic() + self._timeout
            while True:
                try:
                    _fcntl.flock(fd, _fcntl.LOCK_EX | _fcntl.LOCK_NB)
                    self._fd = fd
                    acquired = True
                    return self
                except (BlockingIOError, OSError) as exc:
                    is_contention = isinstance(exc, BlockingIOError) or (
                        isinstance(exc, OSError)
                        and exc.errno in (errno.EAGAIN, errno.EWOULDBLOCK, errno.EACCES)
                    )
                    if not is_contention:
                        raise

                    if time.monotonic() >= deadline:
                        raise TimeoutError(
                            "AgentManagerStore write lock could not be acquired within "
                            "{:.1f} seconds; another writer may hold it. "
                            "Failing closed to prevent lost updates.".format(self._timeout)
                        )
                    time.sleep(self._retry_interval)
        finally:
            if not acquired:
                try:
                    os.close(fd)
                except OSError:
                    pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        fd = self._fd
        self._fd = None
        if fd is not None:
            try:
                import fcntl as _fcntl
                _fcntl.flock(fd, _fcntl.LOCK_UN)
            except OSError:
                pass
            finally:
                try:
                    os.close(fd)
                except OSError:
                    pass
        return False  # never suppress caller exceptions


SCHEMA_VERSION = 1


@dataclass(frozen=True)
class AgentManagerSnapshot:
    agents: Tuple[AgentDefinition, ...] = ()
    runs: Tuple[AgentRun, ...] = ()
    skills: Tuple[SkillDefinition, ...] = ()
    tools: Tuple[ToolDefinition, ...] = ()
    approvals: Tuple[Approval, ...] = ()
    workflows: Tuple[WorkflowDefinition, ...] = ()
    issues: Tuple[str, ...] = ()


class AgentManagerStore:
    """Store records below project `.kripa/agents/`; credentials are rejected.

    Write operations are protected by an advisory file lock (via ``fcntl.flock`` on
    POSIX/macOS) to prevent concurrent writers from overwriting each other's updates.
    Read-modify-write workflows should use ``transaction()`` or ``upsert()`` to ensure
    the lock is held across the entire operation. Each write persists records atomically
    using a temporary file and atomic replace.
    """

    COLLECTION_TYPES = {
        "agents": AgentDefinition,
        "runs": AgentRun,
        "skills": SkillDefinition,
        "tools": ToolDefinition,
        "approvals": Approval,
        "workflows": WorkflowDefinition,
    }

    def __init__(self, path=None, lock_timeout=_LOCK_TIMEOUT_SECONDS,
                 lock_retry_interval=_LOCK_RETRY_INTERVAL):
        root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        self.path = path or os.path.join(root, ".kripa", "agents", "store.json")
        self._lock_path = os.path.join(
            os.path.dirname(os.path.abspath(self.path)), "store.lock"
        )
        self._lock_timeout = lock_timeout
        self._lock_retry_interval = lock_retry_interval

    def load(self):
        if not os.path.exists(self.path):
            return AgentManagerSnapshot()
        try:
            with open(self.path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, ValueError, UnicodeDecodeError):
            return AgentManagerSnapshot(issues=("Agent Manager data could not be read as valid JSON.",))
        if not isinstance(payload, dict) or payload.get("schema_version") != SCHEMA_VERSION:
            return AgentManagerSnapshot(issues=("Agent Manager data has an unsupported schema version.",))

        issues = []
        collections = {}
        for name, record_type in self.COLLECTION_TYPES.items():
            values = payload.get(name, [])
            parsed = []
            if not isinstance(values, list):
                issues.append("{} collection is invalid and was ignored.".format(name.capitalize()))
                collections[name] = ()
                continue
            for index, value in enumerate(values):
                try:
                    if not isinstance(value, dict):
                        raise TypeError("record must be an object")
                    validate_json_value(value, name)
                    if name == "agents":
                        value["model_preferences"] = tuple(value.get("model_preferences", ()))
                        value["skill_ids"] = tuple(value.get("skill_ids", ()))
                        value["tool_ids"] = tuple(value.get("tool_ids", ()))
                    elif name == "workflows":
                        value["steps"] = tuple(value["steps"])
                    parsed.append(record_type(**value))
                except (KeyError, TypeError, ValueError):
                    issues.append("Invalid {} record at index {}; it was ignored.".format(name[:-1].capitalize(), index))
            collections[name] = tuple(parsed)
        return AgentManagerSnapshot(**collections, issues=tuple(issues))

    def transaction(self, mutator):
        """Execute ``mutator(snapshot)`` under an exclusive store write lock.

        Holds the lock across the complete READ -> MODIFY -> WRITE transaction.
        The mutator receives the freshly loaded snapshot and must return either:
        - an updated ``AgentManagerSnapshot``, or
        - a 2-tuple ``(updated_snapshot, result)`` where ``updated_snapshot`` is
          persisted atomically and ``result`` is returned to the caller.

        If ``mutator`` raises an exception, the transaction is aborted and no
        write occurs. If the loaded snapshot contains issues, ValueError is
        raised before invoking the mutator.
        """
        if not callable(mutator):
            raise TypeError("mutator must be callable.")
        with _StoreWriteLock(self._lock_path, self._lock_timeout,
                             self._lock_retry_interval):
            snapshot = self.load()
            if snapshot.issues:
                raise ValueError("Refusing to update incomplete or unsupported Agent Manager data.")
            mutation_result = mutator(snapshot)
            if isinstance(mutation_result, tuple) and len(mutation_result) == 2 and isinstance(mutation_result[0], AgentManagerSnapshot):
                new_snapshot, return_val = mutation_result
            elif isinstance(mutation_result, AgentManagerSnapshot):
                new_snapshot, return_val = mutation_result, mutation_result
            else:
                raise TypeError("transaction mutator must return an AgentManagerSnapshot or (AgentManagerSnapshot, result).")
            self._write(new_snapshot)
            return return_val

    def save(self, snapshot):
        if not isinstance(snapshot, AgentManagerSnapshot):
            raise TypeError("snapshot must be an AgentManagerSnapshot.")
        with _StoreWriteLock(self._lock_path, self._lock_timeout,
                             self._lock_retry_interval):
            self._write(snapshot)

    def _write(self, snapshot):
        """Validate and persist a snapshot atomically; caller must hold the write lock."""
        if not isinstance(snapshot, AgentManagerSnapshot):
            raise TypeError("snapshot must be an AgentManagerSnapshot.")
        if snapshot.issues:
            raise ValueError("Refusing to overwrite incomplete or unsupported Agent Manager data.")
        payload = {"schema_version": SCHEMA_VERSION}
        for name, record_type in self.COLLECTION_TYPES.items():
            records = getattr(snapshot, name)
            if not isinstance(records, (tuple, list)) or any(not isinstance(item, record_type) for item in records):
                raise ValueError("{} must contain validated {} records.".format(name, record_type.__name__))
            payload[name] = [_record_to_json(record) for record in records]
        validate_json_value(payload)
        directory = os.path.dirname(os.path.abspath(self.path))
        os.makedirs(directory, mode=0o700, exist_ok=True)
        descriptor, temporary_path = tempfile.mkstemp(prefix=".agents-", suffix=".tmp", dir=directory)
        handle = None
        try:
            os.chmod(temporary_path, 0o600)
            handle = os.fdopen(descriptor, "w", encoding="utf-8")
            with handle:
                json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=False)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_path, self.path)
        finally:
            if handle is None:
                try:
                    os.close(descriptor)
                except OSError:
                    pass
            if os.path.exists(temporary_path):
                try:
                    os.unlink(temporary_path)
                except OSError:
                    pass

    def upsert(self, collection, record):
        """Replace by stable record ID and save; holds the write lock for the full
        read-modify-write cycle so concurrent callers cannot overwrite each other."""
        if collection not in self.COLLECTION_TYPES:
            raise ValueError("Unknown Agent Manager collection.")
        record_type = self.COLLECTION_TYPES[collection]
        if not isinstance(record, record_type):
            raise TypeError("record must be a {}.".format(record_type.__name__))
        key_name = {
            "agents": "agent_id", "runs": "run_id", "skills": "skill_id",
            "tools": "tool_id", "approvals": "approval_id",
            "workflows": "workflow_id",
        }[collection]

        def _apply(snapshot):
            records = [
                item for item in getattr(snapshot, collection)
                if getattr(item, key_name) != getattr(record, key_name)
            ]
            records.append(record)
            return AgentManagerSnapshot(**{
                name: tuple(records) if name == collection else getattr(snapshot, name)
                for name in self.COLLECTION_TYPES
            })

        self.transaction(_apply)


def _record_to_json(record):
    if isinstance(record, AgentDefinition):
        return {"agent_id": record.agent_id, "name": record.name, "description": record.description,
                "instructions": record.instructions, "model_preferences": list(record.model_preferences),
                "skill_ids": list(record.skill_ids), "tool_ids": list(record.tool_ids),
                "parent_agent_id": record.parent_agent_id, "max_delegation_depth": record.max_delegation_depth,
                "enabled": record.enabled, "metadata": record.metadata}
    if isinstance(record, WorkflowDefinition):
        return {"workflow_id": record.workflow_id, "name": record.name, "description": record.description,
                "steps": list(record.steps), "version": record.version, "enabled": record.enabled,
                "metadata": record.metadata}
    fields = record.__dataclass_fields__
    return {name: getattr(record, name) for name in fields}
