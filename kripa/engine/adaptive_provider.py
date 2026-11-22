"""Gemini-first provider that falls back to Ollama and retries Gemini each turn."""

import socket

from kripa.engine.errors import ProviderError


def gemini_connectivity_available(timeout=0.75):
    """Return whether Google's Gemini API host is reachable over HTTPS."""
    try:
        with socket.create_connection(
            ("generativelanguage.googleapis.com", 443), timeout=timeout
        ):
            return True
    except OSError:
        return False


class AdaptiveProvider:
    """Try Gemini, then cloud fallbacks, then Ollama; retry Gemini each turn."""

    def __init__(
        self,
        primary_factory,
        fallback_provider,
        connectivity_check=None,
        additional_fallbacks=None,
    ):
        self._primary_factory = primary_factory
        self._fallback = fallback_provider
        self._additional_fallbacks = list(additional_fallbacks or [])
        self._connectivity_check = connectivity_check or gemini_connectivity_available
        self._primary = None

    def reply(self, messages):
        if self._connectivity_check():
            try:
                if self._primary is None:
                    self._primary = self._primary_factory()
                return self._primary.reply(messages)
            except ProviderError:
                # Treat authentication, service, and request failures as unavailable
                # for this turn; the next turn checks Gemini again for recovery.
                pass
            except Exception:
                # Do not expose unexpected SDK or network exception details.
                pass

        fallback_chain = self._additional_fallbacks + [self._fallback]
        for fallback in fallback_chain:
            try:
                provider = fallback() if callable(fallback) and not hasattr(fallback, "reply") else fallback
                return provider.reply(messages)
            except ProviderError:
                continue
            except Exception:
                # Keep walking the chain without exposing SDK or transport details.
                continue

        raise ProviderError(
            "Gemini, Groq, and OpenRouter are unavailable, and Ollama could not reply. "
            "Check provider configuration and make sure Ollama is running locally."
        ) from None
