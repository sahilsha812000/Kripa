"""Versioned, atomic local JSON storage for Agent Manager records."""

import json
import os
import tempfile
from dataclasses import dataclass
from typing import Tuple

from kripa.agent_manager.contracts import (
    AgentDefinition, AgentRun, Approval, SkillDefinition, ToolDefinition,
    WorkflowDefinition, validate_json_value,
)


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

    Each save atomically replaces the complete snapshot. Callers should load,
    update, and save promptly; this store is intentionally single-process and
    does not provide concurrent-writer locking.
    """

    COLLECTION_TYPES = {
        "agents": AgentDefinition,
        "runs": AgentRun,
        "skills": SkillDefinition,
        "tools": ToolDefinition,
        "approvals": Approval,
        "workflows": WorkflowDefinition,
    }

    def __init__(self, path=None):
        root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        self.path = path or os.path.join(root, ".kripa", "agents", "store.json")

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

    def save(self, snapshot):
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
        try:
            os.chmod(temporary_path, 0o600)
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=False)
                handle.write("\n")
            os.replace(temporary_path, self.path)
            os.chmod(self.path, 0o600)
        except Exception:
            try:
                os.close(descriptor)
            except OSError:
                pass
            raise
        finally:
            if os.path.exists(temporary_path):
                os.unlink(temporary_path)

    def upsert(self, collection, record):
        """Replace by stable record ID and save; does not alter other collections."""
        if collection not in self.COLLECTION_TYPES:
            raise ValueError("Unknown Agent Manager collection.")
        record_type = self.COLLECTION_TYPES[collection]
        if not isinstance(record, record_type):
            raise TypeError("record must be a {}.".format(record_type.__name__))
        snapshot = self.load()
        if snapshot.issues:
            raise ValueError("Refusing to update incomplete or unsupported Agent Manager data.")
        key_name = {"agents": "agent_id", "runs": "run_id", "skills": "skill_id",
                    "tools": "tool_id", "approvals": "approval_id", "workflows": "workflow_id"}[collection]
        records = [item for item in getattr(snapshot, collection) if getattr(item, key_name) != getattr(record, key_name)]
        records.append(record)
        self.save(AgentManagerSnapshot(**dict(
            (name, tuple(records) if name == collection else getattr(snapshot, name))
            for name in self.COLLECTION_TYPES
        )))


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
