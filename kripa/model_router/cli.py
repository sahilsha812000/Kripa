"""Read-only status command for shadow-router evidence."""

from kripa.config import get_settings
from kripa.model_router.availability import AvailabilityRefresher
from kripa.model_router.contracts import Availability, TaskCategory
from kripa.model_router.registry import build_configured_registry
from kripa.model_router.store import ModelRouterStore


def format_status(registry, snapshot, statuses, now=None):
    """Format a safe evidence report; never includes credentials or raw errors."""
    from datetime import datetime, timezone

    now = now or datetime.now(timezone.utc)
    status_map = {(s.provider, s.model_id): s for s in statuses}
    fresh_benchmarks = snapshot.current_benchmarks(now)
    expired = snapshot.expired_benchmarks(now)
    capability_map = {}
    for item in snapshot.capabilities:
        capability_map.setdefault((item.provider, item.model_id), set()).add(item.capability)

    providers = sorted({
        candidate.provider.value for candidate in registry.candidates
        if status_map.get((candidate.provider, candidate.model_id)) is not None
        and status_map[(candidate.provider, candidate.model_id)].availability is Availability.AVAILABLE
    })
    lines = ["# Kripa Shadow Router Status", "", "Available providers: {}".format(
        ", ".join(providers) if providers else "none confirmed",
    ), "", "## Registered models"]
    missing = []
    for candidate in registry.candidates:
        status = status_map.get((candidate.provider, candidate.model_id))
        availability = status.availability.value if status else "unknown (stale or not checked)"
        caps = capability_map.get((candidate.provider, candidate.model_id), set())
        missing_caps = [task for task in TaskCategory if task not in caps]
        missing.append((candidate, missing_caps))
        lines.append("- {} / {} — {}; verified capabilities: {}".format(
            candidate.provider.value, candidate.model_id, availability,
            ", ".join(sorted(item.value for item in caps)) if caps else "none",
        ))
    lines.extend(["", "## Benchmark coverage"])
    if not fresh_benchmarks:
        lines.append("- No current benchmark results.")
    for task in TaskCategory:
        task_count = sum(1 for item in fresh_benchmarks if item.task_category is task)
        coverage = "{} current record(s).".format(task_count) if task_count else "No Arena benchmark exists."
        lines.append("- {}: {}".format(task.value, coverage))
    lines.append("\nBenchmark records by provider and model:")
    for item in fresh_benchmarks:
        accuracy = "N/A" if item.accuracy is None else "{:.0%}".format(item.accuracy)
        latency = "N/A" if item.median_latency_seconds is None else "{:.3f}s median".format(item.median_latency_seconds)
        sample_state = "complete" if item.success_count + item.failure_count == item.sample_count else "incomplete"
        lines.append("- {} / {} / {} — {} samples ({} success, {} failure), accuracy {}, {}".format(
            item.provider.value, item.model_id, item.task_category.value, item.sample_count,
            item.success_count, item.failure_count, accuracy, latency,
        ))
        if sample_state == "incomplete":
            lines.append("  Incomplete: some sample outcomes are unaccounted for.")
    lines.append("- Expired benchmark records: {}.".format(len(expired)))
    lines.extend(["", "## Registered model capability gaps"])
    for candidate, missing_caps in missing:
        lines.append("- {} / {}: {}".format(
            candidate.provider.value, candidate.model_id,
            ", ".join(item.value for item in missing_caps) if missing_caps else "none",
        ))
    ready = []
    readiness_details = []
    for task in TaskCategory:
        task_benchmarks = [b for b in fresh_benchmarks if b.task_category is task]
        candidate_benchmarks = [
            b for b in task_benchmarks
            if registry.find(b.provider, b.model_id) is not None
        ]
        capable_benchmarks = [
            b for b in candidate_benchmarks
            if task in capability_map.get((b.provider, b.model_id), set())
        ]
        available_benchmarks = [
            b for b in capable_benchmarks
            if status_map.get((b.provider, b.model_id)) is not None
            and status_map[(b.provider, b.model_id)].availability is Availability.AVAILABLE
        ]
        has_candidate = any(
            b.sample_count > 0 and b.success_count > 0
            and b.success_count + b.failure_count == b.sample_count
            and b.accuracy is not None
            for b in available_benchmarks
        )
        if has_candidate:
            ready.append(task)
            readiness_details.append((task, "READY", "All required evidence is present."))
            continue

        blockers = []
        if not task_benchmarks:
            blockers.append("No Arena benchmark exists.")
        elif not candidate_benchmarks:
            blockers.append("Arena benchmark exists, but it does not match a registered model.")
        else:
            if not capable_benchmarks:
                blockers.append("Benchmark exists, capability not verified.")
            if capable_benchmarks and not available_benchmarks:
                blockers.append("Provider availability is not confirmed.")
            if available_benchmarks and not any(
                b.sample_count > 0 and b.success_count > 0
                and b.success_count + b.failure_count == b.sample_count
                for b in available_benchmarks
            ):
                blockers.append("Benchmark is incomplete or has no successful samples.")
            if task is TaskCategory.WRITING and any(
                b.accuracy is None for b in candidate_benchmarks
            ):
                blockers.append("Writing requires manual accuracy review.")
            elif available_benchmarks and not any(
                b.accuracy is not None for b in available_benchmarks
            ):
                blockers.append("Benchmark accuracy is unavailable.")
        readiness_details.append((task, "NOT READY", " ".join(blockers)))

    missing_ready = [task.value for task in TaskCategory if task not in ready]
    lines.extend(["", "## Routing readiness", "Shadow mode remains enabled; active provider selection is unchanged."])
    if not missing_ready:
        lines.append("Ready: current availability, explicit capability verification, and successful current benchmark evidence exist for all task categories.")
    else:
        lines.append("Not ready for all task categories. Missing evidence for: {}.".format(", ".join(missing_ready)))
    for task, state, detail in readiness_details:
        lines.append("- {} — {}: {}".format(task.value, state, detail))
    if snapshot.issues:
        lines.append("Local data issues: {} incomplete or unreadable item(s); affected records were ignored.".format(len(snapshot.issues)))
    return "\n".join(lines)


def run_status(arguments=None, settings=None, store=None, refresher=None):
    arguments = list(arguments or [])
    if any(arg != "--refresh" for arg in arguments) or arguments.count("--refresh") > 1:
        print("Usage: python -m kripa router status [--refresh]")
        return 2
    force = "--refresh" in arguments
    settings = settings or get_settings()
    store = store or ModelRouterStore()
    registry = build_configured_registry(settings)
    snapshot = store.load()
    try:
        result = (refresher or AvailabilityRefresher()).refresh(
            settings, registry, snapshot.availability, force=force,
        )
        store.save_availability(result.statuses)
        snapshot = store.load()
        fresh = snapshot.fresh_statuses()
        # Stale records are deliberately not printed as current status.
        print(format_status(registry, snapshot, fresh))
        return 0
    except (OSError, ValueError):
        print("Router status could not be refreshed or saved. Existing chat behavior is unchanged.")
        return 1
