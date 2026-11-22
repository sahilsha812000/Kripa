"""Anthropic Messages API adapter."""

from kripa.engine.errors import ProviderError
from kripa.system_instruction import system_instruction_for


class AnthropicProvider:
    def __init__(self, api_key: str, model: str, client=None):
        if not api_key:
            raise ProviderError("ANTHROPIC_API_KEY is missing. Add it to your local .env file.")
        if client is None:
            try:
                from anthropic import Anthropic
            except ImportError:
                raise ProviderError("Anthropic SDK is missing. Install requirements.txt.") from None
            client = Anthropic(api_key=api_key)
        self.client = client
        self.model = model

    def reply(self, messages) -> str:
        system = system_instruction_for("Anthropic", self.model)
        extra_system = "\n\n".join(message.content for message in messages if message.role == "system")
        if extra_system:
            system = "{}\n\n{}".format(system, extra_system)
        turns = [
            {"role": message.role, "content": message.content}
            for message in messages
            if message.role in {"user", "assistant"}
        ]
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=1024,
                system=system,
                messages=turns,
            )
            return "".join(block.text for block in response.content if getattr(block, "type", None) == "text")
        except Exception as error:
            raise ProviderError(_safe_error_message(error)) from None


def _safe_error_message(error: Exception) -> str:
    name = type(error).__name__
    if name == "AuthenticationError":
        return "Anthropic authentication failed. Check ANTHROPIC_API_KEY in your .env file."
    if name == "RateLimitError":
        return "Anthropic rate limit reached. Wait a moment and try again."
    if name == "APIConnectionError":
        return "Could not connect to Anthropic. Check your network and try again."
    return "Anthropic request failed. Check your configuration and try again."
