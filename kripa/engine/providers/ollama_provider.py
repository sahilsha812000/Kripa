"""Ollama's local chat API adapter, implemented with Python's standard library."""

import json
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from kripa.engine.errors import ProviderError
from kripa.engine.usage import record_provider_usage
from kripa.system_instruction import system_instruction_for


class OllamaProvider:
    """Send chat turns to an Ollama server running on the local machine."""

    def __init__(self, base_url="http://localhost:11434", model="llama3.2:3b", timeout=120):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.last_usage = None
        self.last_provider_cost_usd = None

    def reply(self, messages):
        self.last_usage = None
        self.last_provider_cost_usd = None
        chat_messages = [{
            "role": "system",
            "content": system_instruction_for("Ollama", self.model),
        }]
        chat_messages.extend(
            {"role": message.role, "content": message.content}
            for message in messages
            if message.role in {"user", "assistant"}
        )
        payload = json.dumps({
            "model": self.model,
            "messages": chat_messages,
            "stream": False,
        }).encode("utf-8")
        request = Request(
            self.base_url + "/api/chat",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout) as response:
                result = json.loads(response.read().decode("utf-8"))
        except HTTPError:
            raise ProviderError(
                "Ollama could not complete the request. Check that the configured model is available."
            ) from None
        except (URLError, TimeoutError, OSError):
            raise ProviderError(
                "Could not connect to Ollama. Make sure Ollama is running locally."
            ) from None
        except (ValueError, UnicodeDecodeError):
            raise ProviderError("Ollama returned an invalid response. Please try again.") from None

        answer = result.get("message", {}).get("content") if isinstance(result, dict) else None
        record_provider_usage(
            self, result,
            ("prompt_eval_count",),
            ("eval_count",),
        )
        if not isinstance(answer, str) or not answer.strip():
            raise ProviderError("Ollama returned an empty response. Please try again.")
        return answer
