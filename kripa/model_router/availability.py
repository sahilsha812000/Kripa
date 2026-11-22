"""Cache-aware, read-only provider availability checks."""

from dataclasses import dataclass
from datetime import datetime, timezone
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

from kripa.model_router.contracts import Availability, ModelCandidateRegistry, ModelProvider
from kripa.model_router.store import ProviderStatus, _parse_time


@dataclass(frozen=True)
class RefreshResult:
    statuses: tuple
    checked: int


class AvailabilityRefresher:
    """Refresh stale candidate statuses through injectable read-only checkers."""

    def __init__(self, checkers=None, ttl_seconds=300):
        self.checkers = checkers or DEFAULT_CHECKERS
        self.ttl_seconds = ttl_seconds

    def refresh(self, settings, registry, cached=(), force=False, now=None):
        now = now or datetime.now(timezone.utc)
        fresh = {}
        for status in cached:
            try:
                age = (now - _parse_time(status.checked_at)).total_seconds()
            except (TypeError, ValueError):
                continue
            if 0 <= age < self.ttl_seconds:
                fresh[(status.provider, status.model_id)] = status
        checked = 0
        output = []
        for candidate in registry.candidates:
            key = (candidate.provider, candidate.model_id)
            if key in fresh and not force:
                output.append(fresh[key])
                continue
            checker = self.checkers.get(candidate.provider)
            if checker is None:
                status = ProviderStatus(candidate.provider, candidate.model_id, Availability.UNKNOWN,
                                        now.isoformat(), "No status checker is configured.")
            else:
                status = checker(settings, candidate, now)
                checked += 1
            output.append(status)
        return RefreshResult(tuple(output), checked)


def _catalog_checker(discoverer):
    def check(settings, candidate, now):
        key_field = {
            ModelProvider.GEMINI: "gemini_api_key",
            ModelProvider.GROQ: "groq_api_key",
        }.get(candidate.provider)
        if key_field and not getattr(settings, key_field, None):
            return ProviderStatus(candidate.provider, candidate.model_id, Availability.UNKNOWN,
                                  now.isoformat(), "Credential is not configured; check skipped.")
        try:
            snapshot = discoverer(settings)
            if snapshot.error:
                return ProviderStatus(candidate.provider, candidate.model_id, Availability.UNKNOWN,
                                      now.isoformat(), "Provider status check failed.")
            match = any(_same(record.model_id, candidate.model_id) for record in snapshot.records)
            available = match
            if candidate.provider is ModelProvider.GROQ and match:
                record = next(r for r in snapshot.records if _same(r.model_id, candidate.model_id))
                if record.active is False:
                    available = False
            return ProviderStatus(candidate.provider, candidate.model_id,
                                  Availability.AVAILABLE if available else Availability.UNAVAILABLE,
                                  now.isoformat(), "Configured model was listed." if available else
                                  "Configured model was not listed.")
        except Exception:
            return ProviderStatus(candidate.provider, candidate.model_id, Availability.UNKNOWN,
                                  now.isoformat(), "Provider status check failed.")
    return check


def _ollama_check(settings, candidate, now):
    try:
        from kripa.model_updates.discoverers import _get_json
        models = _get_json(settings.ollama_base_url + "/api/tags", timeout=4).get("models", [])
        names = [item.get("name") or item.get("model") for item in models if isinstance(item, dict)]
        found = any(name and _same(name, candidate.model_id) for name in names)
        return ProviderStatus(candidate.provider, candidate.model_id,
                              Availability.AVAILABLE if found else Availability.UNAVAILABLE,
                              now.isoformat(), "Configured model is installed." if found else
                              "Configured model is not installed locally.")
    except Exception:
        return ProviderStatus(candidate.provider, candidate.model_id, Availability.UNKNOWN,
                              now.isoformat(), "Local Ollama status check failed.")


def _keyed_model_check(provider, key_name, endpoint, headers):
    def check(settings, candidate, now):
        api_key = getattr(settings, key_name, None)
        if not api_key:
            return ProviderStatus(provider, candidate.model_id, Availability.UNKNOWN,
                                  now.isoformat(), "Credential is not configured; check skipped.")
        request_headers = headers(api_key)
        request = Request(endpoint(candidate.model_id), headers=request_headers, method="GET")
        try:
            with urlopen(request, timeout=5) as response:
                available = 200 <= response.status < 300
            status, detail = (Availability.AVAILABLE, "Configured model endpoint responded.") if available else (
                Availability.UNKNOWN, "Provider status check returned an unexpected response.")
        except HTTPError as error:
            if error.code == 404:
                status, detail = Availability.UNAVAILABLE, "Configured model was not found."
            else:
                status, detail = Availability.UNKNOWN, "Provider status check was rejected or unavailable."
        except (URLError, TimeoutError, OSError):
            status, detail = Availability.UNKNOWN, "Provider status check failed."
        except Exception:
            status, detail = Availability.UNKNOWN, "Provider status check failed."
        return ProviderStatus(provider, candidate.model_id, status, now.isoformat(), detail)
    return check


def _same(left, right):
    return left.removeprefix("models/").casefold() == right.removeprefix("models/").casefold()


from kripa.model_updates.discoverers import discover_gemini, discover_groq, discover_openrouter

DEFAULT_CHECKERS = {
    ModelProvider.GEMINI: _catalog_checker(discover_gemini),
    ModelProvider.GROQ: _catalog_checker(discover_groq),
    ModelProvider.OPENROUTER: _catalog_checker(discover_openrouter),
    ModelProvider.OLLAMA: _ollama_check,
    ModelProvider.OPENAI: _keyed_model_check(
        ModelProvider.OPENAI, "openai_api_key",
        lambda model: "https://api.openai.com/v1/models/" + quote(model, safe=""),
        lambda key: {"Authorization": "Bearer " + key},
    ),
    ModelProvider.ANTHROPIC: _keyed_model_check(
        ModelProvider.ANTHROPIC, "anthropic_api_key",
        lambda model: "https://api.anthropic.com/v1/models/" + quote(model, safe=""),
        lambda key: {"x-api-key": key, "anthropic-version": "2023-06-01"},
    ),
}
