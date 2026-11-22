"""Basic offline tests for conversation and provider adapters."""

import unittest
import json
import os
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from kripa.config import get_settings
from kripa.engine import Conversation, Message
from kripa.engine.adaptive_provider import AdaptiveProvider
from kripa.engine.errors import ProviderError
from kripa.engine.factory import create_provider
from kripa.engine.providers.anthropic_provider import AnthropicProvider
from kripa.engine.providers.gemini_provider import GeminiProvider
from kripa.engine.providers.groq_provider import GroqProvider
from kripa.engine.providers.openai_provider import OpenAIProvider
from kripa.engine.providers.ollama_provider import OllamaProvider
from kripa.engine.providers.openrouter_provider import OpenRouterProvider
from kripa.system_instruction import KRIPA_SYSTEM_INSTRUCTION, system_instruction_for


class FakeProvider:
    def __init__(self, answer="Hello"):
        self.answer = answer
        self.seen = []

    def reply(self, messages):
        self.seen.append(messages)
        return self.answer


class ConversationTests(unittest.TestCase):
    def test_keeps_multi_turn_history(self):
        provider = FakeProvider()
        conversation = Conversation(provider)
        self.assertEqual(conversation.send("Hi"), "Hello")
        conversation.send("Again")
        self.assertEqual(
            [(m.role, m.content) for m in provider.seen[1]],
            [("user", "Hi"), ("assistant", "Hello"), ("user", "Again")],
        )

    def test_failed_turn_does_not_enter_history(self):
        class BrokenProvider:
            def reply(self, messages):
                raise ProviderError("Safe error")

        conversation = Conversation(BrokenProvider())
        with self.assertRaises(ProviderError):
            conversation.send("Hi")
        self.assertEqual(conversation.messages, [])


class ProviderAdapterTests(unittest.TestCase):
    def test_factory_keeps_gemini_default_and_wraps_with_ollama_fallback(self):
        settings = SimpleNamespace(
            provider="gemini",
            gemini_api_key="secret-placeholder",
            gemini_model="test-model",
        )
        provider = create_provider(settings)
        self.assertIsInstance(provider, AdaptiveProvider)
        self.assertEqual(provider._fallback.model, "llama3.2:3b")
        self.assertEqual(len(provider._additional_fallbacks), 2)

    def test_openrouter_environment_settings_are_read_without_loading_dotenv(self):
        environment = {
            "OPENROUTER_API_KEY": "placeholder-test-key",
            "OPENROUTER_MODEL": "provider/test-model",
        }
        with patch("kripa.config.load_dotenv"), patch.dict(os.environ, environment, clear=True):
            settings = get_settings()
        self.assertEqual(settings.openrouter_api_key, "placeholder-test-key")
        self.assertEqual(settings.openrouter_model, "provider/test-model")

    def test_factory_supports_direct_openrouter_selection(self):
        settings = SimpleNamespace(
            provider="openrouter",
            openrouter_api_key="secret-placeholder",
            openrouter_model="test-openrouter-model",
        )
        with patch("kripa.engine.factory.OpenRouterProvider") as provider_class:
            create_provider(settings)
        provider_class.assert_called_once_with("secret-placeholder", "test-openrouter-model")

    def test_factory_supports_direct_groq_selection(self):
        settings = SimpleNamespace(
            provider="groq",
            groq_api_key="secret-placeholder",
            groq_model="test-groq-model",
        )
        with patch("kripa.engine.factory.GroqProvider") as provider_class:
            create_provider(settings)
        provider_class.assert_called_once_with("secret-placeholder", "test-groq-model")

    def test_factory_supports_direct_ollama_selection(self):
        settings = SimpleNamespace(provider="ollama")
        provider = create_provider(settings)
        self.assertIsInstance(provider, OllamaProvider)
        self.assertEqual(provider.base_url, "http://localhost:11434")
        self.assertEqual(provider.model, "llama3.2:3b")

    def test_adaptive_provider_uses_gemini_when_online(self):
        primary = FakeProvider("Gemini answer")
        fallback = FakeProvider("Ollama answer")
        provider = AdaptiveProvider(lambda: primary, fallback, lambda: True)
        self.assertEqual(provider.reply([Message("user", "Hi")]), "Gemini answer")
        self.assertEqual(len(primary.seen), 1)
        self.assertEqual(fallback.seen, [])

    def test_adaptive_provider_falls_back_when_offline(self):
        primary = FakeProvider()
        fallback = FakeProvider("Ollama answer")
        provider = AdaptiveProvider(lambda: primary, fallback, lambda: False)
        self.assertEqual(provider.reply([Message("user", "Hi")]), "Ollama answer")
        self.assertEqual(primary.seen, [])
        self.assertEqual(len(fallback.seen), 1)

    def test_adaptive_provider_returns_to_gemini_after_recovery(self):
        primary = FakeProvider("Gemini answer")
        fallback = FakeProvider("Ollama answer")
        states = iter([False, True])
        provider = AdaptiveProvider(lambda: primary, fallback, lambda: next(states))
        self.assertEqual(provider.reply([Message("user", "First")]), "Ollama answer")
        self.assertEqual(provider.reply([Message("user", "Second")]), "Gemini answer")
        self.assertEqual(len(fallback.seen), 1)
        self.assertEqual(len(primary.seen), 1)

    def test_adaptive_provider_falls_back_on_gemini_provider_failure(self):
        class BrokenProvider:
            def reply(self, messages):
                raise ProviderError("safe primary error")

        fallback = FakeProvider("Ollama answer")
        provider = AdaptiveProvider(BrokenProvider, fallback, lambda: True)
        self.assertEqual(provider.reply([Message("user", "Hi")]), "Ollama answer")

    def test_adaptive_provider_uses_groq_before_ollama(self):
        class BrokenGemini:
            def reply(self, messages):
                raise ProviderError("Gemini unavailable")

        groq = FakeProvider("Groq answer")
        ollama = FakeProvider("Ollama answer")
        provider = AdaptiveProvider(
            BrokenGemini,
            ollama,
            lambda: True,
            additional_fallbacks=[lambda: groq],
        )
        self.assertEqual(provider.reply([Message("user", "Hi")]), "Groq answer")
        self.assertEqual(len(groq.seen), 1)
        self.assertEqual(ollama.seen, [])

    def test_groq_failure_falls_back_to_ollama(self):
        class BrokenProvider:
            def reply(self, messages):
                raise ProviderError("Groq safely unavailable")

        ollama = FakeProvider("Ollama answer")
        provider = AdaptiveProvider(
            lambda: (_ for _ in ()).throw(ProviderError("Gemini unavailable")),
            ollama,
            lambda: True,
            additional_fallbacks=[BrokenProvider()],
        )
        self.assertEqual(provider.reply([Message("user", "Hi")]), "Ollama answer")

    def test_adaptive_provider_uses_openrouter_after_groq_failure(self):
        class BrokenProvider:
            def reply(self, messages):
                raise ProviderError("cloud provider failed")

        openrouter = FakeProvider("OpenRouter answer")
        ollama = FakeProvider("Ollama answer")
        provider = AdaptiveProvider(
            lambda: (_ for _ in ()).throw(ProviderError("Gemini unavailable")),
            ollama,
            lambda: True,
            additional_fallbacks=[BrokenProvider(), lambda: openrouter],
        )
        self.assertEqual(provider.reply([Message("user", "Hi")]), "OpenRouter answer")
        self.assertEqual(len(openrouter.seen), 1)
        self.assertEqual(ollama.seen, [])

    def test_openrouter_failure_falls_through_to_ollama(self):
        class BrokenProvider:
            def reply(self, messages):
                raise ProviderError("OpenRouter safely unavailable")

        ollama = FakeProvider("Ollama answer")
        provider = AdaptiveProvider(
            lambda: (_ for _ in ()).throw(ProviderError("Gemini unavailable")),
            ollama,
            lambda: True,
            additional_fallbacks=[BrokenProvider()],
        )
        self.assertEqual(provider.reply([Message("user", "Hi")]), "Ollama answer")

    def test_openrouter_uses_chat_completions_api_and_shared_identity(self):
        class Client:
            def __init__(self):
                self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))
                self.call = None

            def create(self, **kwargs):
                self.call = kwargs
                return SimpleNamespace(
                    choices=[SimpleNamespace(message=SimpleNamespace(content="Router answer"))],
                    usage=SimpleNamespace(prompt_tokens=11, completion_tokens=4, cost=0.0002),
                )

        client = Client()
        provider = OpenRouterProvider("secret-placeholder", "provider/test-model", client=client)
        history = [Message("user", "Hi"), Message("assistant", "Hello"), Message("user", "Again")]
        self.assertEqual(provider.reply(history), "Router answer")
        self.assertEqual(client.call["model"], "provider/test-model")
        self.assertEqual(
            client.call["messages"][1:],
            [
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Hello"},
                {"role": "user", "content": "Again"},
            ],
        )
        self.assertTrue(
            client.call["messages"][0]["content"].startswith(KRIPA_SYSTEM_INSTRUCTION)
        )
        self.assertEqual(provider.last_usage, {"input_tokens": 11, "output_tokens": 4})
        self.assertEqual(provider.last_provider_cost_usd, 0.0002)
    def test_openrouter_authentication_failure_is_sanitized(self):
        class AuthenticationError(Exception):
            status_code = 401

        class Client:
            def __init__(self):
                self.chat = SimpleNamespace(
                    completions=SimpleNamespace(create=self.create)
                )

            def create(self, **kwargs):
                raise AuthenticationError("secret-placeholder and raw headers")

        provider = OpenRouterProvider("secret-placeholder", "test-model", client=Client())
        with self.assertRaises(ProviderError) as caught:
            provider.reply([Message("user", "Hi")])
        self.assertIn("OPENROUTER_API_KEY", str(caught.exception))
        self.assertNotIn("secret-placeholder", str(caught.exception))
        self.assertNotIn("raw headers", str(caught.exception))

    def test_groq_uses_official_chat_completions_shape_and_identity(self):
        class Client:
            def __init__(self):
                self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))
                self.call = None

            def create(self, **kwargs):
                self.call = kwargs
                return SimpleNamespace(
                    choices=[SimpleNamespace(message=SimpleNamespace(content="Groq says hello"))],
                    usage=SimpleNamespace(prompt_tokens=9, completion_tokens=6),
                )

        client = Client()
        provider = GroqProvider("secret-placeholder", "test-model", client=client)
        history = [Message("user", "Hi"), Message("assistant", "Hello"), Message("user", "Again")]
        self.assertEqual(provider.reply(history), "Groq says hello")
        self.assertEqual(client.call["model"], "test-model")
        self.assertEqual(
            client.call["messages"][1:],
            [
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Hello"},
                {"role": "user", "content": "Again"},
            ],
        )
        self.assertTrue(client.call["messages"][0]["content"].startswith(KRIPA_SYSTEM_INSTRUCTION))
        self.assertEqual(provider.last_usage, {"input_tokens": 9, "output_tokens": 6})
        self.assertIsNone(provider.last_provider_cost_usd)

    def test_groq_provider_failure_is_safe(self):
        class AuthenticationError(Exception):
            status_code = 401

        class Client:
            def __init__(self):
                self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

            def create(self, **kwargs):
                raise AuthenticationError("secret-placeholder raw header details")

        provider = GroqProvider("secret-placeholder", "test-model", client=Client())
        with self.assertRaises(ProviderError) as caught:
            provider.reply([Message("user", "Hi")])
        self.assertIn("GROQ_API_KEY", str(caught.exception))
        self.assertNotIn("secret-placeholder", str(caught.exception))
        self.assertNotIn("raw header details", str(caught.exception))

    def test_ollama_sends_shared_identity_and_parses_chat_response(self):
        response = MagicMock()
        response.__enter__.return_value = response
        response.read.return_value = json.dumps(
            {"message": {"content": "Hello from Ollama"}, "prompt_eval_count": 13, "eval_count": 5}
        ).encode()

        def fake_urlopen(request, timeout):
            self.assertEqual(request.full_url, "http://localhost:11434/api/chat")
            self.assertEqual(timeout, 120)
            payload = json.loads(request.data.decode())
            self.assertEqual(payload["model"], "llama3.2:3b")
            self.assertFalse(payload["stream"])
            self.assertIn("Kripa, Nitin's personal AI assistant", payload["messages"][0]["content"])
            self.assertEqual(payload["messages"][1], {"role": "user", "content": "Hi"})
            return response

        with patch("kripa.engine.providers.ollama_provider.urlopen", fake_urlopen):
            provider = OllamaProvider()
            self.assertEqual(provider.reply([Message("user", "Hi")]), "Hello from Ollama")
            self.assertEqual(provider.last_usage, {"input_tokens": 13, "output_tokens": 5})
            self.assertIsNone(provider.last_provider_cost_usd)

    def test_gemini_uses_interactions_api_and_continues_conversation(self):
        class Client:
            def __init__(self):
                self.interactions = SimpleNamespace(create=self.create)
                self.calls = []

            def create(self, **kwargs):
                self.calls.append(kwargs)
                return SimpleNamespace(
                    output_text="Hello", id="interaction-1",
                    usage=SimpleNamespace(input_tokens=12, output_tokens=8),
                )

        client = Client()
        provider = GeminiProvider("secret-placeholder", "test-model", client=client)
        history = [Message("user", "Hi"), Message("assistant", "Hello"), Message("user", "Again")]
        self.assertEqual(provider.reply(history), "Hello")
        self.assertEqual(client.calls[0]["model"], "test-model")
        self.assertEqual(client.calls[0]["input"], "Again")
        self.assertTrue(client.calls[0]["system_instruction"].startswith(KRIPA_SYSTEM_INSTRUCTION))
        self.assertEqual(provider.previous_interaction_id, "interaction-1")
        self.assertEqual(provider.last_usage, {"input_tokens": 12, "output_tokens": 8})
        provider.reply(history)
        self.assertEqual(client.calls[1]["previous_interaction_id"], "interaction-1")

    def test_gemini_connection_error_is_safe(self):
        class ConnectionError(Exception):
            pass

        class Client:
            def __init__(self):
                self.interactions = SimpleNamespace(create=lambda **kwargs: (_ for _ in ()).throw(
                    ConnectionError("secret-placeholder and raw request details")
                ))

        provider = GeminiProvider("secret-placeholder", "test-model", client=Client())
        with self.assertRaises(ProviderError) as caught:
            provider.reply([Message("user", "Hi")])
        self.assertIn("internet connection", str(caught.exception))
        self.assertNotIn("secret-placeholder", str(caught.exception))
        self.assertNotIn("raw request details", str(caught.exception))

    def test_gemini_server_timeouts_are_reported_as_transient(self):
        class ServerError(Exception):
            code = 504

        class Client:
            def __init__(self):
                self.models = SimpleNamespace(generate_content=lambda **kwargs: (_ for _ in ()).throw(
                    ServerError("internal provider details")
                ))

        provider = GeminiProvider("secret-placeholder", "test-model", client=Client())
        with self.assertRaises(ProviderError) as caught:
            provider.reply([Message("user", "Hi")])
        self.assertIn("timed out", str(caught.exception))
        self.assertNotIn("internal provider details", str(caught.exception))

    def test_gemini_generate_content_fallback_keeps_chat_history(self):
        class Models:
            def __init__(self):
                self.call = None

            def generate_content(self, **kwargs):
                self.call = kwargs
                return SimpleNamespace(text="Hello")

        client = SimpleNamespace(models=Models())
        provider = GeminiProvider("secret-placeholder", "test-model", client=client)
        messages = [Message("user", "Hi"), Message("assistant", "Hello"), Message("user", "Again")]
        self.assertEqual(provider.reply(messages), "Hello")
        self.assertEqual(client.models.call["model"], "test-model")
        self.assertEqual(
            client.models.call["contents"],
            [
                {"role": "user", "parts": [{"text": "Hi"}]},
                {"role": "model", "parts": [{"text": "Hello"}]},
                {"role": "user", "parts": [{"text": "Again"}]},
            ],
        )
        self.assertTrue(
            client.models.call["config"]["system_instruction"].startswith(KRIPA_SYSTEM_INSTRUCTION)
        )

    def test_openai_uses_responses_api(self):
        class Client:
            def __init__(self):
                self.responses = SimpleNamespace(create=self.create)
                self.call = None

            def create(self, **kwargs):
                self.call = kwargs
                return SimpleNamespace(output_text="Hello")

        client = Client()
        provider = OpenAIProvider("secret-placeholder", "test-model", client=client)
        self.assertEqual(provider.reply([Message("user", "Hi")]), "Hello")
        self.assertEqual(client.call["model"], "test-model")
        self.assertEqual(client.call["input"][0]["content"], "Hi")
        self.assertTrue(client.call["instructions"].startswith(KRIPA_SYSTEM_INSTRUCTION))

    def test_anthropic_uses_messages_api(self):
        class Client:
            def __init__(self):
                self.messages = SimpleNamespace(create=self.create)
                self.call = None

            def create(self, **kwargs):
                self.call = kwargs
                return SimpleNamespace(content=[SimpleNamespace(type="text", text="Hello")])

        client = Client()
        provider = AnthropicProvider("secret-placeholder", "test-model", client=client)
        answer = provider.reply([Message("user", "Hi")])
        self.assertEqual(answer, "Hello")
        self.assertEqual(client.call["model"], "test-model")
        self.assertEqual(client.call["messages"][0]["content"], "Hi")
        self.assertTrue(client.call["system"].startswith(KRIPA_SYSTEM_INSTRUCTION))

    def test_identity_policy_is_shared_and_allows_truthful_model_disclosure_on_request(self):
        prompts = [
            system_instruction_for("Google Gemini", "gemini-test"),
            system_instruction_for("Ollama", "llama3.2:3b"),
            system_instruction_for("Groq", "llama-3.3-70b-versatile"),
            system_instruction_for("OpenRouter", "openai/gpt-oss-120b"),
            system_instruction_for("OpenAI", "openai-test"),
            system_instruction_for("Anthropic", "claude-test"),
        ]
        for prompt in prompts:
            self.assertTrue(prompt.startswith(KRIPA_SYSTEM_INSTRUCTION))
            self.assertIn("Kripa, Nitin's personal AI assistant", prompt)
            self.assertIn("Only discuss the provider or model", prompt)
            self.assertIn("concise, professional, and helpful", prompt)
        self.assertNotEqual(prompts[0], prompts[1])
        self.assertIn("model=gemini-test", prompts[0])
        self.assertIn("model=llama3.2:3b", prompts[1])
        self.assertIn("model=llama-3.3-70b-versatile", prompts[2])
        self.assertIn("model=openai/gpt-oss-120b", prompts[3])

    def test_provider_failure_message_does_not_include_sdk_details(self):
        class AuthenticationError(Exception):
            pass

        class Client:
            def __init__(self):
                self.responses = SimpleNamespace(create=lambda **kwargs: (_ for _ in ()).throw(
                    AuthenticationError("sensitive request details")
                ))

        provider = OpenAIProvider("secret-placeholder", "test-model", client=Client())
        with self.assertRaises(ProviderError) as caught:
            provider.reply([Message("user", "Hi")])
        self.assertNotIn("sensitive request details", str(caught.exception))
        self.assertNotIn("secret-placeholder", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
