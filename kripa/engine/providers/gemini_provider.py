"""Google Gemini Interactions API adapter."""

from kripa.engine.errors import ProviderError
from kripa.engine.usage import record_provider_usage
from kripa.system_instruction import system_instruction_for


class GeminiProvider:
    """Use a Gemini model while keeping provider SDK details isolated."""

    def __init__(self, api_key, model, client=None):
        if not api_key:
            raise ProviderError("GEMINI_API_KEY is missing. Add it to your local .env file.")
        if client is None:
            try:
                from google import genai
            except ImportError:
                raise ProviderError("Google GenAI SDK is missing. Install requirements.txt.") from None
            try:
                client = genai.Client(api_key=api_key)
            except Exception:
                raise ProviderError("Could not initialize Gemini. Check your configuration.") from None
        self.client = client
        self.model = model
        self.previous_interaction_id = None
        self._supports_interactions = hasattr(client, "interactions")
        self.last_usage = None
        self.last_provider_cost_usd = None

    def reply(self, messages):
        """Send the latest user turn, continuing this Gemini chat session."""
        system_instruction = system_instruction_for("Google Gemini", self.model)
        latest_user_message = next(
            (message.content for message in reversed(messages) if message.role == "user"),
            None,
        )
        if not latest_user_message:
            raise ProviderError("Gemini needs a user message to reply.")

        self.last_usage = None
        self.last_provider_cost_usd = None
        try:
            if self._supports_interactions:
                request = {
                    "model": self.model,
                    "input": latest_user_message,
                    "system_instruction": system_instruction,
                }
                if self.previous_interaction_id:
                    request["previous_interaction_id"] = self.previous_interaction_id
                interaction = self.client.interactions.create(**request)
                answer = getattr(interaction, "output_text", None)
                self.previous_interaction_id = getattr(interaction, "id", None)
                usage = getattr(interaction, "usage", None)
            else:
                contents = [
                    {
                        "role": "model" if message.role == "assistant" else "user",
                        "parts": [{"text": message.content}],
                    }
                    for message in messages
                    if message.role in {"user", "assistant"}
                ]
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=contents,
                    config={"system_instruction": system_instruction},
                )
                answer = getattr(response, "text", None)
                usage = getattr(response, "usage_metadata", None)
            record_provider_usage(
                self,
                usage,
                ("input_tokens", "prompt_tokens", "prompt_token_count"),
                ("output_tokens", "completion_tokens", "candidates_token_count"),
            )
        except Exception as error:
            raise ProviderError(_safe_error_message(error)) from None

        if not isinstance(answer, str) or not answer.strip():
            raise ProviderError("Gemini returned an empty response. Please try again.")
        return answer


def _safe_error_message(error):
    """Convert SDK/network errors to safe messages without raw details."""
    name = type(error).__name__.lower()
    status = getattr(error, "code", None) or getattr(error, "status_code", None)
    if status in (401, 403) or "unauthorized" in name or "forbidden" in name:
        return "Gemini authentication failed. Check GEMINI_API_KEY in your .env file."
    if status == 429 or "ratelimit" in name or "resourceexhausted" in name:
        return "Gemini rate limit reached. Wait a moment and try again."
    if status == 504 or "timeout" in name:
        return "Gemini request timed out. Try again or select a lower-latency model."
    if status in (500, 502, 503) or "servererror" in name:
        return "Gemini is temporarily overloaded or unavailable. Try again shortly."
    if any(token in name for token in ("connection", "timeout", "connecterror", "network")):
        return "Could not connect to Gemini. Check your internet connection and try again."
    return "Gemini request failed. Check your configuration and try again."
