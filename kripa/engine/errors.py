"""Safe errors surfaced by provider adapters."""


class ProviderError(Exception):
    """A user-safe provider failure message with no raw SDK details."""
