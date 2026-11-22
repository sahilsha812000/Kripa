"""OpenRouter Chat Completions adapter using the existing OpenAI SDK."""

from kripa.engine.errors import ProviderError
from kripa.engine.usage import record_provider_usage
from kripa.system_instruction import system_instruction_for


OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


class OpenRouterProvider:
    """Send conversation history to a model selected through OpenRouter."""

    def __init__(self, api_key, model, client=None):
        if not api_key:
            raise ProviderError(
                "OPENROUTER_API_KEY is missing. Add it to your local .env file."
            )
        if client is None:
            try:
                from openai import OpenAI
            except ImportError:
                raise ProviderError("OpenAI SDK is missing. Install requirements.txt.") from None
            try:
                client = OpenAI(api_key=api_key, base_url=OPENROUTER_BASE_URL)
            except Exception:
                raise ProviderError(
                    "Could not initialize OpenRouter. Check your configuration."
                ) from None
        self.client = client
        self.model = model
        self.last_usage = None
        self.last_provider_cost_usd = None

    def reply(self, messages):
        """Return a chat completion with Kripa's shared identity instructions."""
        chat_messages = [{
            "role": "system",
            "content": system_instruction_for("OpenRouter", self.model),
        }]
        chat_messages.extend(
            {"role": message.role, "content": message.content}
            for message in messages
            if message.role in {"user", "assistant"}
        )
        self.last_usage = None
        self.last_provider_cost_usd = None
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=chat_messages,
            )
            answer = response.choices[0].message.content
            record_provider_usage(
                self, getattr(response, "usage", None),
                ("prompt_tokens", "input_tokens"),
                ("completion_tokens", "output_tokens"),
                ("cost", "total_cost_usd"),
            )
        except Exception as error:
            raise ProviderError(_safe_error_message(error)) from None
        if not isinstance(answer, str) or not answer.strip():
            raise ProviderError("OpenRouter returned an empty response. Please try again.")
        return answer


def _safe_error_message(error):
    """Map SDK/network failures to safe messages without raw exception data."""
    name = type(error).__name__.lower()
    status = getattr(error, "status_code", None) or getattr(error, "status", None)
    if status in (401, 403) or "authentication" in name or "permission" in name:
        return "OpenRouter authentication failed. Check OPENROUTER_API_KEY in your .env file."
    if status == 429 or "ratelimit" in name:
        return "OpenRouter rate limit reached. Wait a moment and try again."
    if "timeout" in name:
        return "OpenRouter request timed out. Try again shortly."
    if "connection" in name or "network" in name:
        return "Could not connect to OpenRouter. Check your internet connection and try again."
    return "OpenRouter request failed. Check your configuration and try again."
