"""Compare configured models with provider catalogs without changing settings."""

import re
from datetime import datetime, timezone

from kripa.model_updates.discoverers import (
    discover_gemini,
    discover_groq,
    discover_ollama,
    discover_openrouter,
)
from kripa.model_updates.models import ModelUpdateReport, ModelUpdateRow


class ModelUpdateManager:
    """Discover available models and build a read-only comparison report."""

    def __init__(self, discoverers=None):
        self._discoverers = discoverers or {
            "Gemini": discover_gemini,
            "Groq": discover_groq,
            "OpenRouter": discover_openrouter,
            "Ollama": discover_ollama,
        }

    def generate_report(self, settings):
        snapshots = {
            name: discoverer(settings)
            for name, discoverer in self._discoverers.items()
        }
        configured = {
            "Gemini": settings.gemini_model,
            "Groq": settings.groq_model,
            "OpenRouter": settings.openrouter_model,
            "Ollama": settings.ollama_model,
        }
        rows = []
        for provider in ("Gemini", "Groq", "OpenRouter", "Ollama"):
            snapshot = snapshots.get(provider)
            current = configured[provider]
            if snapshot is None:
                rows.append(_unknown_row(provider, current, "No catalog discoverer configured."))
            elif provider == "Ollama":
                rows.append(_ollama_row(current, snapshot))
            else:
                rows.append(_cloud_row(provider, current, snapshot))
        ollama_snapshot = snapshots.get("Ollama")
        installed = ollama_snapshot.installed_models if ollama_snapshot else []
        return ModelUpdateReport(rows=rows, ollama_installed_models=installed)


def _cloud_row(provider, current, snapshot):
    if snapshot.error:
        return _unknown_row(provider, current, snapshot.error)
    if not snapshot.records:
        note = snapshot.note or "The provider returned no compatible models."
        return _unknown_row(provider, current, note)

    current_record = next(
        (record for record in snapshot.records if _same_id(record.model_id, current)),
        None,
    )
    latest_record = _newest_record(snapshot.records)
    latest = latest_record.model_id if latest_record else "Not supplied by provider catalog"
    if current_record is None:
        recommendation = _replacement(provider, current, snapshot.records, latest_record)
        return ModelUpdateRow(
            provider=provider,
            current_model=current,
            latest_available_model=latest,
            deprecation_status="not listed",
            recommended_replacement=recommendation or "None identified",
            compatibility="Configured model was not returned by the provider catalog.",
            note=(snapshot.note + " Configured ID may be retired or unavailable to this account."),
        )

    expiry = _parse_time(current_record.expires_at)
    now = datetime.now(timezone.utc).timestamp()
    if current_record.active is False or (expiry is not None and expiry <= now):
        status = "retired"
    elif current_record.deprecated or expiry is not None:
        status = "deprecated"
    else:
        successor = _replacement(provider, current, snapshot.records, latest_record)
        current_rank = _record_rank(current_record)
        successor_record = next(
            (record for record in snapshot.records if _same_id(record.model_id, successor)),
            None,
        ) if successor else None
        if successor_record and _record_rank(successor_record) > current_rank:
            status = "outdated"
        else:
            status = "active"
    recommendation = (
        _replacement(provider, current, snapshot.records, latest_record)
        if status in {"retired", "deprecated", "outdated"}
        else "None; configured model is available"
    )
    return ModelUpdateRow(
        provider=provider,
        current_model=current,
        latest_available_model=latest,
        deprecation_status=status,
        recommended_replacement=recommendation or "None identified",
        compatibility=current_record.compatibility or "Listed in provider catalog.",
        note=snapshot.note,
    )


def _ollama_row(current, snapshot):
    if snapshot.error:
        return _unknown_row("Ollama", current, snapshot.error)
    check = next(
        (item for item in snapshot.installed_models if _same_id(item["model"], current)),
        None,
    )
    if check is None:
        return _unknown_row("Ollama", current, snapshot.note or "Ollama model status is unknown.")
    if not check.get("installed"):
        latest = _digest_model_label(current, check.get("latest_digest"))
        return ModelUpdateRow(
            provider="Ollama",
            current_model=current,
            latest_available_model=latest,
            deprecation_status="not installed",
            recommended_replacement="None; configured model is not installed",
            compatibility="Model is not present in the local Ollama store.",
            note=check.get("note") or "No model was installed or downloaded.",
        )
    if check.get("update_available"):
        status = "update available"
        recommendation = current
    elif check.get("latest_digest"):
        status = "current"
        recommendation = "None; installed digest matches the registry"
    else:
        status = "unknown"
        recommendation = "None; registry comparison unavailable"
    record = next((item for item in snapshot.records if _same_id(item.model_id, current)), None)
    compatibility = record.compatibility if record else "Installed in local Ollama store."
    return ModelUpdateRow(
        provider="Ollama",
        current_model=current,
        latest_available_model=_digest_model_label(current, check.get("latest_digest")),
        deprecation_status=status,
        recommended_replacement=recommendation,
        compatibility=compatibility,
        note=check.get("note") or "Registry check reads manifest metadata only; no model files were downloaded.",
    )


def _unknown_row(provider, current, note):
    return ModelUpdateRow(
        provider=provider,
        current_model=current,
        latest_available_model="Unknown",
        deprecation_status="not checked",
        recommended_replacement="None",
        compatibility="Not verified",
        note=note,
    )


def _replacement(provider, current, records, latest_record):
    same_family = [
        record for record in records
        if not _same_id(record.model_id, current)
        and _family(provider, record.model_id) == _family(provider, current)
        and record.active is not False
    ]
    if same_family:
        return _newest_record(same_family).model_id
    return None


def _family(provider, model_id):
    value = _normalize_id(model_id).lower()
    if provider == "OpenRouter":
        value = value.split("/", 1)[-1]
        for family in ("gpt-oss", "gpt", "claude", "gemini", "llama", "qwen", "mistral", "deepseek"):
            if family in value:
                return family
    if provider == "Gemini":
        for family in ("flash-lite", "flash", "pro", "live", "embedding"):
            if family in value:
                return family
    if provider == "Groq":
        for family in ("gpt-oss", "llama", "qwen", "kimi", "gemma", "whisper", "compound"):
            if family in value:
                return family
    return re.sub(r"[0-9.]+", "", value)


def _newest_record(records):
    ranked = [(record, _record_rank(record)) for record in records]
    ranked = [item for item in ranked if item[1] > (0,)]
    if not ranked:
        return None
    return max(ranked, key=lambda item: item[1])[0]


def _record_rank(record):
    created = record.created_at
    if isinstance(created, (int, float)):
        return (float(created),)
    parsed = _parse_time(created)
    if parsed is not None:
        return (parsed,)
    version = record.metadata.get("version")
    numeric = re.findall(r"\d+", str(version or record.model_id))
    return tuple(int(value) for value in numeric) or (0,)


def _parse_time(value):
    if isinstance(value, (int, float)):
        return float(value)
    if not isinstance(value, str) or not value.strip():
        if isinstance(value, datetime):
            if value.tzinfo is None:
                value = value.replace(tzinfo=timezone.utc)
            return value.timestamp()
        return None
    try:
        normalized = value.strip().replace("Z", "+00:00")
        parsed = datetime.fromisoformat(normalized)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.timestamp()
    except ValueError:
        return None


def _normalize_id(value):
    value = (value or "").strip()
    return value[len("models/"):] if value.startswith("models/") else value


def _same_id(left, right):
    return _normalize_id(left).casefold() == _normalize_id(right).casefold()


def _digest_model_label(model, digest):
    return "{} @ {}".format(model, digest[:12]) if digest else "Not available from registry"
