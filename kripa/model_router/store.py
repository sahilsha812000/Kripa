"""Versioned local JSON persistence for routing evidence and provider status."""

import json
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Tuple

from kripa.model_router.contracts import (
    Availability,
    BenchmarkResult,
    ModelProvider,
    TaskCategory,
)


SCHEMA_VERSION = 1
DEFAULT_BENCHMARK_TTL_DAYS = 30


@dataclass(frozen=True)
class StoredBenchmark:
    result: BenchmarkResult
    recorded_at: str
    expires_at: str


@dataclass(frozen=True)
class ProviderStatus:
    provider: ModelProvider
    model_id: str
    availability: Availability
    checked_at: str
    detail: str


@dataclass(frozen=True)
class VerifiedCapability:
    provider: ModelProvider
    model_id: str
    capability: TaskCategory
    verified_at: str
    verification_source: str


@dataclass(frozen=True)
class RouterDataSnapshot:
    benchmarks: Tuple[StoredBenchmark, ...] = ()
    availability: Tuple[ProviderStatus, ...] = ()
    capabilities: Tuple[VerifiedCapability, ...] = ()
    issues: Tuple[str, ...] = ()

    def current_benchmarks(self, now=None):
        now = now or utc_now()
        return tuple(entry.result for entry in self.benchmarks if _parse_time(entry.expires_at) > now)

    def expired_benchmarks(self, now=None):
        now = now or utc_now()
        return tuple(entry.result for entry in self.benchmarks if _parse_time(entry.expires_at) <= now)

    def fresh_statuses(self, now=None, ttl_seconds=300):
        now = now or utc_now()
        fresh = []
        for status in self.availability:
            checked_at = _parse_time(status.checked_at)
            if checked_at <= now and (now - checked_at).total_seconds() < ttl_seconds:
                fresh.append(status)
        return tuple(fresh)


class ModelRouterStore:
    """Read/write router metadata under the project-local .kripa directory.

    API credentials are not part of the schema. Writes use a same-directory
    temporary file and atomic replacement so interrupted updates leave the
    previous snapshot intact.
    """

    def __init__(self, path=None, benchmark_ttl_days=DEFAULT_BENCHMARK_TTL_DAYS):
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        self.path = path or os.path.join(project_root, ".kripa", "model_router.json")
        self.benchmark_ttl_days = benchmark_ttl_days

    def load(self):
        if not os.path.exists(self.path):
            return RouterDataSnapshot()
        try:
            with open(self.path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, ValueError, UnicodeDecodeError):
            return RouterDataSnapshot(issues=("Router data file could not be read as valid JSON.",))
        if not isinstance(payload, dict) or payload.get("schema_version") != SCHEMA_VERSION:
            return RouterDataSnapshot(issues=("Router data file has an unsupported schema version.",))

        issues = []
        benchmarks = self._load_collection(payload, "benchmarks", _parse_benchmark, issues)
        availability = self._load_collection(payload, "availability", _parse_status, issues)
        capabilities = self._load_collection(payload, "verified_capabilities", _parse_capability, issues)
        return RouterDataSnapshot(
            tuple(benchmarks), tuple(availability), tuple(capabilities), tuple(issues),
        )

    @staticmethod
    def _load_collection(payload, name, parser, issues):
        values = payload.get(name, [])
        if not isinstance(values, list):
            issues.append("{} data is incomplete and was ignored.".format(name.capitalize()))
            return []
        parsed = []
        for index, value in enumerate(values):
            try:
                parsed.append(parser(value))
            except (KeyError, TypeError, ValueError, OverflowError):
                issues.append("Incomplete {} record at index {}; it was ignored.".format(name, index))
        return parsed

    def save_arena_report(self, report, now=None):
        now = now or utc_now()
        existing = self.load()
        self._ensure_writable(existing)
        entries = list(existing.benchmarks)
        new_records = []
        for summary in report.summaries:
            result = BenchmarkResult.from_arena_summary(summary)
            recorded = now.isoformat()
            expires = (now + timedelta(days=self.benchmark_ttl_days)).isoformat()
            new_records.append(StoredBenchmark(result, recorded, expires))

        replaced_keys = {
            (entry.result.task_category, entry.result.provider, entry.result.model_id)
            for entry in new_records
        }
        entries = [
            entry for entry in entries
            if (entry.result.task_category, entry.result.provider, entry.result.model_id) not in replaced_keys
        ]
        entries.extend(new_records)
        self._write(entries, existing.availability, existing.capabilities)
        return len(new_records)

    def save_availability(self, statuses):
        existing = self.load()
        self._ensure_writable(existing)
        updates = {(item.provider, item.model_id): item for item in statuses}
        records = [
            item for item in existing.availability
            if (item.provider, item.model_id) not in updates
        ]
        records.extend(updates.values())
        self._write(existing.benchmarks, records, existing.capabilities)

    def register_verified_capability(
        self, provider, model_id, capability, verification_source, verified_at=None,
    ):
        """Persist a capability only when caller supplies explicit verification evidence."""
        if not verification_source or not verification_source.strip():
            raise ValueError("An explicit verification source is required.")
        record = VerifiedCapability(
            provider=_as_provider(provider),
            model_id=_required_text(model_id, "model_id"),
            capability=_as_task(capability),
            verified_at=(verified_at or utc_now()).isoformat(),
            verification_source=verification_source.strip(),
        )
        existing = self.load()
        self._ensure_writable(existing)
        records = [
            item for item in existing.capabilities
            if (item.provider, item.model_id, item.capability)
            != (record.provider, record.model_id, record.capability)
        ]
        records.append(record)
        self._write(existing.benchmarks, existing.availability, records)
        return record

    @staticmethod
    def _ensure_writable(snapshot):
        if snapshot.issues:
            raise ValueError("Existing router data is incomplete or unsupported; refusing to overwrite it.")

    def _write(self, benchmarks, availability, capabilities):
        directory = os.path.dirname(self.path)
        os.makedirs(directory, mode=0o700, exist_ok=True)
        payload = {
            "schema_version": SCHEMA_VERSION,
            "benchmarks": [_benchmark_to_json(item) for item in benchmarks],
            "availability": [_status_to_json(item) for item in availability],
            "verified_capabilities": [_capability_to_json(item) for item in capabilities],
        }
        descriptor, temporary_path = tempfile.mkstemp(prefix=".model-router-", suffix=".tmp", dir=directory)
        try:
            try:
                os.chmod(temporary_path, 0o600)
                with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                    json.dump(payload, handle, indent=2, sort_keys=True)
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


def utc_now():
    return datetime.now(timezone.utc)


def _parse_time(value):
    if not isinstance(value, str):
        raise ValueError("Timestamp must be a string.")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("Timestamp must have a timezone.")
    return parsed.astimezone(timezone.utc)


def _as_provider(value):
    return value if isinstance(value, ModelProvider) else ModelProvider(value)


def _as_task(value):
    return value if isinstance(value, TaskCategory) else TaskCategory(value)


def _required_text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError("{} must be non-empty text.".format(name))
    return value.strip()


def _parse_benchmark(value):
    if not isinstance(value, dict):
        raise TypeError("Benchmark entry must be an object.")
    result_data = value["result"]
    result = BenchmarkResult(
        task_category=result_data["task_category"],
        provider=result_data["provider"],
        model_id=result_data["model_id"],
        sample_count=result_data["sample_count"],
        success_count=result_data["success_count"],
        failure_count=result_data["failure_count"],
        accuracy=result_data.get("accuracy"),
        mean_latency_seconds=result_data.get("mean_latency_seconds"),
        median_latency_seconds=result_data.get("median_latency_seconds"),
        consistency=result_data.get("consistency"),
        input_tokens=result_data.get("input_tokens"),
        output_tokens=result_data.get("output_tokens"),
        provider_cost_usd=result_data.get("provider_cost_usd"),
        usage_reports=result_data.get("usage_reports", 0),
    )
    recorded_at = _required_text(value["recorded_at"], "recorded_at")
    expires_at = _required_text(value["expires_at"], "expires_at")
    if _parse_time(expires_at) <= _parse_time(recorded_at):
        raise ValueError("Benchmark expiry must follow its creation time.")
    return StoredBenchmark(result, recorded_at, expires_at)


def _parse_status(value):
    if not isinstance(value, dict):
        raise TypeError("Availability entry must be an object.")
    return ProviderStatus(
        provider=_as_provider(value["provider"]),
        model_id=_required_text(value["model_id"], "model_id"),
        availability=Availability(value["availability"]),
        checked_at=_required_text(value["checked_at"], "checked_at"),
        detail=_required_text(value.get("detail", "Status checked."), "detail"),
    )


def _parse_capability(value):
    if not isinstance(value, dict):
        raise TypeError("Capability entry must be an object.")
    return VerifiedCapability(
        provider=_as_provider(value["provider"]),
        model_id=_required_text(value["model_id"], "model_id"),
        capability=_as_task(value["capability"]),
        verified_at=_required_text(value["verified_at"], "verified_at"),
        verification_source=_required_text(value["verification_source"], "verification_source"),
    )


def _benchmark_to_json(entry):
    result = entry.result
    return {
        "recorded_at": entry.recorded_at,
        "expires_at": entry.expires_at,
        "result": {
            "task_category": result.task_category.value,
            "provider": result.provider.value,
            "model_id": result.model_id,
            "sample_count": result.sample_count,
            "success_count": result.success_count,
            "failure_count": result.failure_count,
            "accuracy": result.accuracy,
            "mean_latency_seconds": result.mean_latency_seconds,
            "median_latency_seconds": result.median_latency_seconds,
            "consistency": result.consistency,
            "input_tokens": result.input_tokens,
            "output_tokens": result.output_tokens,
            "provider_cost_usd": result.provider_cost_usd,
            "usage_reports": result.usage_reports,
        },
    }


def _status_to_json(item):
    return {
        "provider": item.provider.value,
        "model_id": item.model_id,
        "availability": item.availability.value,
        "checked_at": item.checked_at,
        "detail": item.detail,
    }


def _capability_to_json(item):
    return {
        "provider": item.provider.value,
        "model_id": item.model_id,
        "capability": item.capability.value,
        "verified_at": item.verified_at,
        "verification_source": item.verification_source,
    }
