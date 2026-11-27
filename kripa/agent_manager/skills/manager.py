"""Inspect and manage local instruction-only skill packages."""

import hashlib
import hmac
import json
import os
import re
from dataclasses import dataclass, replace
from datetime import datetime, timezone

from kripa.agent_manager.contracts import SkillDefinition, validate_id, validate_json_value, validate_text
from kripa.agent_manager.store import AgentManagerStore
from kripa.agent_manager.skills.security import secret_findings, validate_package_directory
from kripa.agent_manager.skills.sources import LocalDirectorySkillSource


_VERSION = re.compile(r"^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)$")
_MANIFEST_FIELDS = frozenset({"schema_version", "skill_id", "name", "description", "version", "entrypoint", "compatibility"})


@dataclass(frozen=True)
class SkillInspection:
    path: str
    skill_id: str
    name: str
    description: str
    version: str
    entrypoint: str
    compatibility: dict
    compatibility_issues: tuple
    instructions: str
    sha256: str
    safety_issues: tuple

    @property
    def safe_to_consider(self):
        return not self.safety_issues and not self.compatibility_issues


class SkillManager:
    """Discover, inspect, approve, install, enable, disable, and remove skills.

    A skill is instruction text only. No discovered or installed file is
    imported, evaluated, or executed.
    """

    def __init__(self, store=None, discovery_source=None):
        self.store = store or AgentManagerStore()
        self.discovery_source = discovery_source or LocalDirectorySkillSource()

    def discover(self, directory):
        return self.discovery_source.discover(self, directory)

    def inspect(self, path):
        package_path = os.path.abspath(path)
        contents = validate_package_directory(package_path)
        try:
            manifest = json.loads(contents["skill.json"])
        except (ValueError, TypeError):
            raise ValueError("skill.json must contain a valid JSON object.")
        if not isinstance(manifest, dict):
            raise ValueError("skill.json must contain a JSON object.")
        unknown = set(manifest).difference(_MANIFEST_FIELDS)
        missing = _MANIFEST_FIELDS.difference(manifest)
        if unknown or missing:
            raise ValueError("Skill manifest has missing or unsupported fields.")
        if isinstance(manifest.get("schema_version"), bool) or manifest.get("schema_version") != 1:
            raise ValueError("Skill manifest schema_version must be 1.")
        skill_id = validate_id(manifest["skill_id"], "skill_id")
        name = validate_text(manifest["name"], "name")
        description = validate_text(manifest["description"], "description", allow_empty=True)
        version = validate_text(manifest["version"], "version")
        if not _VERSION.match(version):
            raise ValueError("Skill version must use MAJOR.MINOR.PATCH format.")
        if manifest["entrypoint"] != "SKILL.md":
            raise ValueError("Skill entrypoint must be SKILL.md.")
        compatibility = manifest["compatibility"]
        if not isinstance(compatibility, dict):
            raise ValueError("compatibility must be a JSON object.")
        allowed_compatibility = {"kripa_min_version", "kripa_max_version"}
        if set(compatibility).difference(allowed_compatibility):
            raise ValueError("compatibility contains unsupported fields.")
        for value in compatibility.values():
            if not isinstance(value, str) or not _VERSION.match(value):
                raise ValueError("Compatibility versions must use MAJOR.MINOR.PATCH format.")
        if (compatibility.get("kripa_min_version") and compatibility.get("kripa_max_version")
                and _version_tuple(compatibility["kripa_min_version"])
                > _version_tuple(compatibility["kripa_max_version"])):
            raise ValueError("Skill compatibility minimum version cannot exceed its maximum version.")
        from kripa import __version__ as current_version
        current = _version_tuple(current_version)
        compatibility_issues = []
        minimum = compatibility.get("kripa_min_version")
        maximum = compatibility.get("kripa_max_version")
        if minimum and current < _version_tuple(minimum):
            compatibility_issues.append("Requires Kripa {} or newer.".format(minimum))
        if maximum and current > _version_tuple(maximum):
            compatibility_issues.append("Supports Kripa only through {}.".format(maximum))
        instructions = contents["SKILL.md"]
        if not instructions.strip():
            raise ValueError("SKILL.md must contain instructions.")
        validate_json_value(manifest, "skill manifest")
        issues = tuple(secret_findings(contents["skill.json"] + "\n" + instructions))
        digest = hashlib.sha256()
        for name_key in ("skill.json", "SKILL.md"):
            digest.update(name_key.encode("utf-8"))
            digest.update(b"\0")
            digest.update(contents[name_key].encode("utf-8"))
            digest.update(b"\0")
        return SkillInspection(
            path=package_path, skill_id=skill_id, name=name, description=description,
            version=version, entrypoint="SKILL.md", compatibility=dict(compatibility),
            compatibility_issues=tuple(compatibility_issues), instructions=instructions,
            sha256=digest.hexdigest(), safety_issues=issues,
        )

    def install(self, path, approval_digest=None):
        inspection = self.inspect(path)
        if inspection.safety_issues:
            raise ValueError("Skill package failed security checks and cannot be installed.")
        if inspection.compatibility_issues:
            raise ValueError("Skill package is incompatible with this Kripa version.")
        if not approval_digest or not isinstance(approval_digest, str):
            raise ValueError("Explicit human approval of the inspected package digest is required.")
        if not hmac.compare_digest(approval_digest.lower(), inspection.sha256):
            raise ValueError("Approval digest does not match the inspected skill package.")
        skill = SkillDefinition(
            skill_id=inspection.skill_id, name=inspection.name, description=inspection.description,
            version=inspection.version, instructions=inspection.instructions, source="local",
            enabled=True,
            metadata={
                "source_path": inspection.path,
                "content_sha256": inspection.sha256,
                "installed_at": datetime.now(timezone.utc).isoformat(),
                "approval": "human_digest_confirmation",
                "compatibility": inspection.compatibility,
                "discovery_source": "local_directory",
            },
        )

        def _mutate(snapshot):
            self._validate_snapshot_integrity(snapshot)
            prior = next((item for item in snapshot.skills if item.skill_id == inspection.skill_id), None)
            if prior is not None and _version_tuple(inspection.version) <= _version_tuple(prior.version):
                raise ValueError("Installed skill version must be newer than the registered version.")
            records = [
                item for item in snapshot.skills if item.skill_id != skill.skill_id
            ]
            records.append(skill)
            new_snapshot = type(snapshot)(
                agents=snapshot.agents, runs=snapshot.runs,
                skills=tuple(records),
                tools=snapshot.tools, approvals=snapshot.approvals,
                workflows=snapshot.workflows,
            )
            return new_snapshot, skill

        return self.store.transaction(_mutate)

    def list_skills(self, include_disabled=True):
        if not isinstance(include_disabled, bool):
            raise ValueError("include_disabled must be a boolean.")
        snapshot = self._snapshot()
        skills = snapshot.skills if include_disabled else tuple(item for item in snapshot.skills if item.enabled)
        return tuple(sorted(skills, key=lambda item: item.skill_id.lower()))

    def get_skill(self, skill_id):
        for skill in self._snapshot().skills:
            if skill.skill_id == skill_id:
                return skill
        raise LookupError("Skill was not found.")

    def enable(self, skill_id):
        return self._set_enabled(skill_id, True)

    def disable(self, skill_id):
        return self._set_enabled(skill_id, False)

    def remove(self, skill_id):
        def _mutate(snapshot):
            self._validate_snapshot_integrity(snapshot)
            target = next((item for item in snapshot.skills if item.skill_id == skill_id), None)
            if target is None:
                raise LookupError("Skill was not found.")
            if any(skill_id in agent.skill_ids and agent.enabled for agent in snapshot.agents):
                raise ValueError("Skill is assigned to an active agent; disable or update that agent first.")
            new_snapshot = type(snapshot)(
                agents=snapshot.agents, runs=snapshot.runs,
                skills=tuple(item for item in snapshot.skills if item.skill_id != skill_id),
                tools=snapshot.tools, approvals=snapshot.approvals,
                workflows=snapshot.workflows,
            )
            return new_snapshot, target

        return self.store.transaction(_mutate)

    def _set_enabled(self, skill_id, enabled):
        def _mutate(snapshot):
            self._validate_snapshot_integrity(snapshot)
            skill = next((item for item in snapshot.skills if item.skill_id == skill_id), None)
            if skill is None:
                raise LookupError("Skill was not found.")
            updated = replace(skill, enabled=enabled)
            records = [
                item for item in snapshot.skills if item.skill_id != skill_id
            ]
            records.append(updated)
            new_snapshot = type(snapshot)(
                agents=snapshot.agents, runs=snapshot.runs,
                skills=tuple(records),
                tools=snapshot.tools, approvals=snapshot.approvals,
                workflows=snapshot.workflows,
            )
            return new_snapshot, updated

        return self.store.transaction(_mutate)

    def _validate_snapshot_integrity(self, snapshot):
        if snapshot.issues:
            raise ValueError("Agent Manager storage is incomplete or unsupported; no changes were made.")
        ids = [skill.skill_id for skill in snapshot.skills]
        if len(ids) != len(set(ids)):
            raise ValueError("Agent Manager storage contains duplicate skill IDs; no changes were made.")

    def _snapshot(self):
        snapshot = self.store.load()
        self._validate_snapshot_integrity(snapshot)
        return snapshot


def _version_tuple(version):
    parts = tuple(int(part) for part in version.split("."))
    if not 1 <= len(parts) <= 3:
        raise ValueError("Invalid version value.")
    return parts + (0,) * (3 - len(parts))
