"""Provider-specific, read-only model catalog discovery."""

import hashlib
import json
import re
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from kripa.engine.providers.openrouter_provider import OPENROUTER_BASE_URL
from kripa.model_updates.models import CatalogSnapshot, ModelRecord


def discover_gemini(settings):
    """List models exposed by the configured Gemini API key."""
    if not settings.gemini_api_key:
        return CatalogSnapshot("Gemini", note="Not checked: GEMINI_API_KEY is not configured.")
    try:
        url = "https://generativelanguage.googleapis.com/v1beta/models"
        page_token = None
        records = []
        while True:
            query = {"pageSize": "1000"}
            if page_token:
                query["pageToken"] = page_token
            request = Request(
                url + "?" + urlencode(query),
                headers={"x-goog-api-key": settings.gemini_api_key},
                method="GET",
            )
            page = _get_json(request)
            for model in page.get("models", []):
                actions = model.get("supportedActions") or []
                if actions and not any("generate" in str(action).lower() for action in actions):
                    continue
                model_id = _normalize_model_id(model.get("name"))
                if not model_id or _is_non_chat_model(model_id):
                    continue
                records.append(ModelRecord(
                    model_id=model_id,
                    created_at=model.get("createTime"),
                    deprecated=_bool_or_none(model.get("deprecated")),
                    expires_at=_string_or_none(
                        model.get("deprecationDate") or model.get("expirationDate")
                    ),
                    compatibility=_gemini_compatibility(model, actions),
                    metadata={
                        "version": model.get("version"),
                        "base_model_id": model.get("baseModelId"),
                    },
                ))
            page_token = page.get("nextPageToken")
            if not page_token:
                break
        return CatalogSnapshot("Gemini", records, note="Gemini models.list catalog.")
    except Exception as error:
        return _failed_snapshot("Gemini", error)


def discover_groq(settings):
    """List Groq models available to the configured account."""
    if not settings.groq_api_key:
        return CatalogSnapshot("Groq", note="Not checked: GROQ_API_KEY is not configured.")
    try:
        from groq import Groq

        response = Groq(api_key=settings.groq_api_key).models.list()
        records = []
        for model in getattr(response, "data", []) or []:
            model_id = _string_or_none(getattr(model, "id", None))
            if not model_id:
                continue
            if _is_non_chat_model(model_id):
                continue
            active = _bool_or_none(getattr(model, "active", None))
            created = getattr(model, "created", None)
            records.append(ModelRecord(
                model_id=model_id,
                created_at=created,
                active=active,
                compatibility=_groq_compatibility(model),
                metadata={
                    "context_window": getattr(model, "context_window", None),
                    "max_completion_tokens": getattr(model, "max_completion_tokens", None),
                    "owned_by": getattr(model, "owned_by", None),
                },
            ))
        return CatalogSnapshot("Groq", records, note="Groq models.list catalog.")
    except Exception as error:
        return _failed_snapshot("Groq", error)


def discover_openrouter(settings):
    """Read the public OpenRouter model catalog (no key is sent)."""
    try:
        url = OPENROUTER_BASE_URL + "/models?sort=newest"
        payload = _get_json(url)
        records = []
        for model in payload.get("data", []):
            model_id = _string_or_none(model.get("id"))
            if not model_id:
                continue
            if _is_non_chat_model(model_id):
                continue
            architecture = model.get("architecture") or {}
            outputs = architecture.get("output_modalities") or []
            if outputs and "text" not in outputs:
                continue
            records.append(ModelRecord(
                model_id=model_id,
                created_at=model.get("created"),
                expires_at=_string_or_none(model.get("expiration_date")),
                compatibility=_openrouter_compatibility(model),
                metadata={
                    "canonical_slug": model.get("canonical_slug"),
                    "context_length": model.get("context_length"),
                    "pricing": model.get("pricing"),
                    "supported_parameters": model.get("supported_parameters"),
                    "architecture": architecture,
                },
            ))
        return CatalogSnapshot("OpenRouter", records, note="OpenRouter public models catalog.")
    except Exception as error:
        return _failed_snapshot("OpenRouter", error)


def discover_ollama(settings, timeout=5):
    """Inspect local Ollama models and compare their registry manifests read-only."""
    provider = "Ollama"
    try:
        payload = _get_json(settings.ollama_base_url + "/api/tags", timeout=timeout)
        local_models = payload.get("models", [])
        inventory = []
        records = []
        configured_model = settings.ollama_model
        configured_seen = False
        for model in local_models:
            name = _string_or_none(model.get("name") or model.get("model"))
            if not name:
                continue
            configured_seen = configured_seen or _same_model_id(name, configured_model)
            local_digest = _normalize_digest(model.get("digest"))
            remote_digest, remote_error = _ollama_registry_digest(name, timeout=timeout)
            update_available = bool(
                local_digest and remote_digest and local_digest != remote_digest
            )
            status = (
                "update available" if update_available
                else "current" if local_digest and remote_digest
                else "update status unknown"
            )
            inventory.append({
                "model": name,
                "installed": True,
                "local_digest": local_digest,
                "latest_digest": remote_digest,
                "update_available": update_available,
                "status": status,
                "note": remote_error,
            })
            details = model.get("details") or {}
            records.append(ModelRecord(
                model_id=name,
                active=True,
                compatibility=_ollama_compatibility(details),
                metadata={"digest": local_digest, "details": details},
            ))

        if configured_model and not configured_seen:
            remote_digest, remote_error = _ollama_registry_digest(
                configured_model, timeout=timeout
            )
            inventory.append({
                "model": configured_model,
                "installed": False,
                "local_digest": None,
                "latest_digest": remote_digest,
                "update_available": False,
                "status": "available remotely; not installed" if remote_digest else "not installed",
                "note": remote_error,
            })
        return CatalogSnapshot(
            provider,
            records,
            note="Local Ollama /api/tags and read-only remote registry manifest checks.",
            installed_models=inventory,
        )
    except Exception as error:
        return _failed_snapshot(provider, error)


def _ollama_registry_digest(model_name, timeout=5):
    """Read a model tag's registry manifest digest without pulling model layers."""
    try:
        repository, tag = _ollama_repository_and_tag(model_name)
        manifest_url = "https://registry.ollama.ai/v2/{}/manifests/{}".format(
            repository, tag
        )
        accept = ", ".join((
            "application/vnd.oci.image.index.v1+json",
            "application/vnd.oci.image.manifest.v1+json",
            "application/vnd.docker.distribution.manifest.v2+json",
            "application/vnd.docker.distribution.manifest.list.v2+json",
        ))
        headers = {"Accept": accept}
        try:
            response = _open_url(manifest_url, headers=headers, timeout=timeout)
        except HTTPError as error:
            challenge = error.headers.get("WWW-Authenticate", "")
            if error.code != 401 or not challenge.lower().startswith("bearer "):
                raise
            token = _registry_token(challenge, timeout)
            headers["Authorization"] = "Bearer " + token
            response = _open_url(manifest_url, headers=headers, timeout=timeout)
        try:
            digest = _normalize_digest(response.headers.get("Docker-Content-Digest"))
            if digest:
                return digest, ""
            # Some registry frontends omit Docker-Content-Digest. Hashing the
            # small OCI manifest body checks the tag without fetching any layers.
            manifest = response.read()
            return hashlib.sha256(manifest).hexdigest(), ""
        finally:
            response.close()
    except Exception as error:
        return None, "Remote registry check unavailable ({})".format(type(error).__name__)


def _registry_token(challenge, timeout):
    """Request an anonymous pull token from the registry's advertised realm."""
    values = dict(re.findall(r'(\w+)="([^"]*)"', challenge))
    realm = values.get("realm")
    if not realm:
        raise ValueError("registry did not provide a token realm")
    query = {key: values[key] for key in ("service", "scope") if values.get(key)}
    token_url = realm + ("&" if "?" in realm else "?") + urlencode(query)
    token_response = _get_json(token_url, timeout=timeout)
    token = token_response.get("token") or token_response.get("access_token")
    if not token:
        raise ValueError("registry token response was empty")
    return token


def _ollama_repository_and_tag(model_name):
    """Translate Ollama's model:tag name to an Ollama library repository path."""
    value = model_name.strip()
    if value.startswith("registry.ollama.ai/"):
        value = value[len("registry.ollama.ai/"):]
    elif "://" in value or value.startswith("hf.co/"):
        raise ValueError("model does not use the public Ollama registry")
    if ":" in value:
        repository_name, tag = value.rsplit(":", 1)
    else:
        repository_name, tag = value, "latest"
    if "/" not in repository_name:
        repository_name = "library/" + repository_name
    return repository_name, tag


def _get_json(url, headers=None, timeout=10):
    request = (
        url if isinstance(url, Request)
        else Request(url, headers=headers or {}, method="GET")
    )
    response = _open_url(request, timeout=timeout)
    try:
        return json.loads(response.read().decode("utf-8"))
    finally:
        response.close()


def _open_url(url_or_request, headers=None, timeout=10):
    request = url_or_request
    if isinstance(url_or_request, str):
        request = Request(url_or_request, headers=headers or {}, method="GET")
    return urlopen(request, timeout=timeout)


def _failed_snapshot(provider, error):
    return CatalogSnapshot(
        provider,
        error="Catalog check failed ({})".format(type(error).__name__),
    )


def _normalize_model_id(value):
    value = _string_or_none(value)
    if value and value.startswith("models/"):
        return value[len("models/"):]
    return value


def _normalize_digest(value):
    value = _string_or_none(value)
    if value and value.startswith("sha256:"):
        return value[len("sha256:"):]
    return value


def _same_model_id(left, right):
    return _normalize_model_id(left).casefold() == _normalize_model_id(right).casefold()


def _string_or_none(value):
    return value.strip() if isinstance(value, str) and value.strip() else None


def _bool_or_none(value):
    return value if isinstance(value, bool) else None


def _gemini_compatibility(model, actions):
    details = []
    if actions:
        details.append("actions: {}".format(", ".join(str(action) for action in actions)))
    for field, label in (("input_token_limit", "input tokens"), ("output_token_limit", "output tokens")):
        camel_case = "inputTokenLimit" if field == "input_token_limit" else "outputTokenLimit"
        value = model.get(camel_case) if isinstance(model, dict) else getattr(model, field, None)
        if value:
            details.append("{} {}".format(label, value))
    return "; ".join(details) or "Listed in Gemini API catalog."


def _is_non_chat_model(model_id):
    lowered = model_id.lower()
    return any(
        marker in lowered
        for marker in (
            "whisper",
            "transcribe",
            "speech-to",
            "text-to-speech",
            "native-audio",
            "-tts",
            "safeguard",
            "prompt-guard",
            "embedding",
        )
    )


def _groq_compatibility(model):
    details = ["available through Groq account catalog"]
    context = getattr(model, "context_window", None)
    if context:
        details.append("context window {} tokens".format(context))
    completion = getattr(model, "max_completion_tokens", None)
    if completion:
        details.append("max completion {} tokens".format(completion))
    return "; ".join(details)


def _openrouter_compatibility(model):
    architecture = model.get("architecture") or {}
    details = []
    modalities = architecture.get("input_modalities") or []
    outputs = architecture.get("output_modalities") or []
    if modalities or outputs:
        details.append("modalities {} -> {}".format(
            ",".join(modalities) or "unknown", ",".join(outputs) or "unknown"
        ))
    context = model.get("context_length")
    if context:
        details.append("context window {} tokens".format(context))
    return "; ".join(details) or "Listed in OpenRouter catalog."


def _ollama_compatibility(details):
    values = []
    for key, label in (("family", "family"), ("parameter_size", "parameters"), ("quantization_level", "quantization")):
        value = details.get(key)
        if value:
            values.append("{} {}".format(label, value))
    return "; ".join(values) or "Installed in local Ollama model store."
