"""Construct the configured provider adapter."""

from kripa.engine.errors import ProviderError
from kripa.engine.adaptive_provider import AdaptiveProvider
from kripa.engine.providers.anthropic_provider import AnthropicProvider
from kripa.engine.providers.gemini_provider import GeminiProvider
from kripa.engine.providers.groq_provider import GroqProvider
from kripa.engine.providers.openai_provider import OpenAIProvider
from kripa.engine.providers.openrouter_provider import OpenRouterProvider
from kripa.engine.providers.ollama_provider import OllamaProvider


def create_provider(settings):
    """Create the selected provider without exposing credentials."""
    try:
        if settings.provider == "gemini":
            fallback = OllamaProvider(
                getattr(settings, "ollama_base_url", "http://localhost:11434"),
                getattr(settings, "ollama_model", "llama3.2:3b"),
            )
            return AdaptiveProvider(
                lambda: GeminiProvider(settings.gemini_api_key, settings.gemini_model),
                fallback,
                additional_fallbacks=[
                    lambda: GroqProvider(
                        getattr(settings, "groq_api_key", None),
                        getattr(settings, "groq_model", "openai/gpt-oss-120b"),
                    ),
                    lambda: OpenRouterProvider(
                        getattr(settings, "openrouter_api_key", None),
                        getattr(settings, "openrouter_model", "openai/gpt-oss-120b"),
                    ),
                ],
            )
        if settings.provider == "groq":
            return GroqProvider(settings.groq_api_key, settings.groq_model)
        if settings.provider == "openrouter":
            return OpenRouterProvider(settings.openrouter_api_key, settings.openrouter_model)
        if settings.provider == "ollama":
            return OllamaProvider(
                getattr(settings, "ollama_base_url", "http://localhost:11434"),
                getattr(settings, "ollama_model", "llama3.2:3b"),
            )
        if settings.provider == "openai":
            return OpenAIProvider(settings.openai_api_key, settings.openai_model)
        if settings.provider in {"anthropic", "claude"}:
            return AnthropicProvider(settings.anthropic_api_key, settings.anthropic_model)
        raise ProviderError(
            "KRIPA_PROVIDER must be set to 'gemini', 'groq', 'openrouter', 'ollama', 'openai', or 'anthropic'."
        )
    except ProviderError:
        raise
    except Exception:
        raise ProviderError("Could not initialize the selected AI provider.") from None
