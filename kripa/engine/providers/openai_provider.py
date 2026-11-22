"""OpenAI Responses API adapter."""

from kripa.engine.errors import ProviderError
from kripa.system_instruction import system_instruction_for


class OpenAIProvider:
    def __init__(self, api_key: str, model: str, client=None):
        if not api_key:
            raise ProviderError("OPENAI_API_KEY is missing. Add it to your local .env file.")
        if client is None:
            try:
                from openai import OpenAI
            except ImportError:
                raise ProviderError("OpenAI SDK is missing. Install requirements.txt.") from None
            client = OpenAI(api_key=api_key)
        self.client = client
        self.model = model

    def reply(self, messages) -> str:
        try:
            response = self.client.responses.create(
                model=self.model,
                instructions=system_instruction_for("OpenAI", self.model),
                input=[{"role": message.role, "content": message.content} for message in messages],
            )
            return response.output_text
        except Exception as error:
            raise ProviderError(_safe_error_message(error)) from None


def _safe_error_message(error: Exception) -> str:
    name = type(error).__name__
    if name == "AuthenticationError":
        return "OpenAI authentication failed. Check OPENAI_API_KEY in your .env file."
    if name == "RateLimitError":
        return "OpenAI rate limit reached. Wait a moment and try again."
    if name == "APIConnectionError":
        return "Could not connect to OpenAI. Check your network and try again."
    return "OpenAI request failed. Check your configuration and try again."
