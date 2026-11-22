"""Build a metadata-only registry from configured model IDs."""

from kripa.model_router.contracts import (
    Availability,
    ModelCandidate,
    ModelCandidateRegistry,
    ModelProvider,
    TaskCategory,
)


MODEL_SETTING_NAMES = {
    ModelProvider.GEMINI: "gemini_model",
    ModelProvider.GROQ: "groq_model",
    ModelProvider.OPENROUTER: "openrouter_model",
    ModelProvider.OLLAMA: "ollama_model",
    ModelProvider.OPENAI: "openai_model",
    ModelProvider.ANTHROPIC: "anthropic_model",
}


def build_configured_registry(settings, availability=None, capabilities=None):
    """Return configured model candidates without probing or selecting them.

    Availability defaults to UNKNOWN. Capabilities default to None (not
    declared), avoiding unsupported claims about model abilities. Optional
    metadata mappings may use provider enums or their string values as keys.
    """
    availability = availability or {}
    capabilities = capabilities or {}
    candidates = []
    for provider, setting_name in MODEL_SETTING_NAMES.items():
        model_id = getattr(settings, setting_name, None)
        if not model_id:
            continue
        candidates.append(ModelCandidate(
            provider=provider,
            model_id=model_id,
            availability=_lookup(availability, provider, Availability.UNKNOWN),
            supported_capabilities=_lookup(capabilities, provider, None),
        ))
    return ModelCandidateRegistry(tuple(candidates))


def _lookup(values, provider, default):
    if provider in values:
        return values[provider]
    return values.get(provider.value, default)
