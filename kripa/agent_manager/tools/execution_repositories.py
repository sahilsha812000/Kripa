"""Execution audit and one-time approval-consumption repositories."""

import json
import os
import re
import stat
from datetime import datetime, timezone
from typing import Protocol

from kripa.agent_manager.contracts import validate_id
from kripa.agent_manager.tools.contracts import ToolExecutionAuditRecord
from kripa.agent_manager.tools.repositories import opaque_reference


_SHA256 = re.compile(r"^[a-f0-9]{64}$")


class ApprovalConsumptionRepository(Protocol):
    def consume(self, approval_id, action_digest):
        """Atomically claim an approval once; return false after first claim."""
        ...


class ToolExecutionAuditRepository(Protocol):
    def record_execution(self, event):
        ...


class ExclusiveFileApprovalConsumptionRepository:
    """Use atomic exclusive file creation as an immutable one-use ledger.

    Unlike an AgentManagerStore read/modify/write, ``O_EXCL`` makes the claim
    race-safe across processes on the local filesystem. Markers are retained;
    deleting one would re-enable its old approval.
    """

    def __init__(self, directory):
        self.directory = os.path.abspath(directory)

    def consume(self, approval_id, action_digest):
        validate_id(approval_id, "approval_id")
        if not isinstance(action_digest, str) or not _SHA256.match(action_digest):
            raise ValueError("action_digest must be a lowercase SHA-256 digest.")
        filename = opaque_reference(approval_id) + ".json"
        payload = json.dumps({
            "schema_version": 1,
            "approval_ref": opaque_reference(approval_id),
            "action_digest": action_digest,
            "consumed_at": datetime.now(timezone.utc).isoformat(),
        }, sort_keys=True, separators=(",", ":")).encode("utf-8") + b"\n"
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
        directory_fd = self._open_directory()
        try:
            try:
                descriptor = os.open(filename, flags, 0o600, dir_fd=directory_fd)
            except FileExistsError:
                return False
            try:
                os.fchmod(descriptor, 0o600)
                with os.fdopen(descriptor, "wb") as handle:
                    handle.write(payload)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.fsync(directory_fd)
            except Exception:
                try:
                    os.close(descriptor)
                except OSError:
                    pass
                # Leave any created marker in place: an incomplete consume is
                # still considered consumed, preventing replay after a crash.
                raise
        finally:
            os.close(directory_fd)
        return True

    def is_consumed(self, approval_id):
        validate_id(approval_id, "approval_id")
        directory_fd = self._open_directory()
        try:
            try:
                os.stat(
                    opaque_reference(approval_id) + ".json",
                    dir_fd=directory_fd, follow_symlinks=False,
                )
                return True
            except FileNotFoundError:
                return False
        finally:
            os.close(directory_fd)

    def _open_directory(self):
        os.makedirs(self.directory, mode=0o700, exist_ok=True)
        flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(self.directory, flags)
        info = os.fstat(descriptor)
        if not stat.S_ISDIR(info.st_mode):
            os.close(descriptor)
            raise OSError("Approval consumption storage is not a real directory.")
        os.fchmod(descriptor, 0o700)
        return descriptor


class JsonLinesToolExecutionAuditRepository:
    """Append execution outcomes without arguments or raw tool results."""

    def __init__(self, path):
        self.path = os.path.abspath(path)

    def record_execution(self, event):
        if not isinstance(event, ToolExecutionAuditRecord):
            raise TypeError("event must be a ToolExecutionAuditRecord.")
        payload = {
            "event_type": "tool_execution",
            "event_id": event.event_id,
            "recorded_at": event.recorded_at,
            "outcome": event.outcome.value,
            "reason": event.reason,
            "risk_level": event.risk_level.value,
            "agent_ref": event.agent_ref,
            "run_ref": event.run_ref,
            "tool_ref": event.tool_ref,
            "action_digest": event.action_digest,
            "approval_ref": event.approval_ref,
            "result_digest": event.result_digest,
            "error_code": event.error_code,
        }
        line = (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
        directory = os.path.dirname(self.path)
        os.makedirs(directory, mode=0o700, exist_ok=True)
        directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
        directory_fd = os.open(directory, directory_flags)
        if not stat.S_ISDIR(os.fstat(directory_fd).st_mode):
            os.close(directory_fd)
            raise OSError("Tool execution audit storage is not a real directory.")
        os.fchmod(directory_fd, 0o700)
        flags = os.O_WRONLY | os.O_APPEND | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(os.path.basename(self.path), flags, 0o600, dir_fd=directory_fd)
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
        finally:
            os.close(directory_fd)


def default_execution_storage(store):
    """Create the local marker and audit repositories beside Agent Manager data."""
    directory = os.path.dirname(os.path.abspath(store.path))
    return (
        ExclusiveFileApprovalConsumptionRepository(
            os.path.join(directory, "consumed_approvals"),
        ),
        JsonLinesToolExecutionAuditRepository(
            os.path.join(directory, "tool_execution_audit.jsonl"),
        ),
    )
