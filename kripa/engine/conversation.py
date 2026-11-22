"""Simple in-memory multi-turn conversation support."""

import logging
from dataclasses import dataclass
from typing import List

from kripa.engine.errors import ProviderError


_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class Message:
    role: str
    content: str


class Conversation:
    """Maintain conversation history and append successful replies."""

    def __init__(self, provider, shadow_router=None):
        self.provider = provider
        self.shadow_router = shadow_router
        self.last_shadow_decision = None
        self.messages: List[Message] = []

    def send(self, text: str) -> str:
        """Send one user turn and return the assistant response."""
        prompt = text.strip()
        if not prompt:
            raise ValueError("Message cannot be empty.")
        self.last_shadow_decision = None
        if self.shadow_router is not None:
            try:
                self.last_shadow_decision = self.shadow_router.evaluate(prompt)
                _LOGGER.debug(
                    "Shadow routing task=%s recommendation=%s model=%s confidence=%s reason=%s",
                    getattr(self.last_shadow_decision.task_category, "value", None),
                    getattr(self.last_shadow_decision.recommended_provider, "value", None),
                    self.last_shadow_decision.recommended_model,
                    getattr(self.last_shadow_decision.confidence, "value", None),
                    self.last_shadow_decision.reason,
                )
            except Exception as error:
                # Shadow analysis must never affect the active conversation path.
                _LOGGER.debug("Shadow routing unavailable (%s)", type(error).__name__)
        pending = [*self.messages, Message(role="user", content=prompt)]
        try:
            answer = self.provider.reply(pending)
        except ProviderError:
            raise
        except Exception:
            raise ProviderError("The AI provider request failed. Please try again.") from None
        answer = answer.strip()
        if not answer:
            raise ProviderError("The AI provider returned an empty response. Please try again.")
        self.messages = [*pending, Message(role="assistant", content=answer)]
        return answer
