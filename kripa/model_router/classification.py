"""Task classification interface and a conservative, local keyword classifier."""

from dataclasses import dataclass
from math import isfinite
from typing import Optional, Protocol

from kripa.model_router.contracts import TaskCategory


@dataclass(frozen=True)
class TaskClassification:
    category: Optional[TaskCategory]
    confidence: float
    reason: str

    def __post_init__(self):
        if self.category is not None and not isinstance(self.category, TaskCategory):
            object.__setattr__(self, "category", TaskCategory(self.category))
        if (
            not isinstance(self.confidence, (int, float))
            or isinstance(self.confidence, bool)
            or not isfinite(self.confidence)
            or not 0.0 <= self.confidence <= 1.0
        ):
            raise ValueError("classification confidence must be between 0 and 1.")


class TaskClassifier(Protocol):
    """Interface implemented by local or future model-backed classifiers."""

    def classify(self, text):
        """Return the task category and classification evidence."""
        ...


class KeywordTaskClassifier:
    """Small deterministic classifier for shadow observations, not execution."""

    _RULES = (
        (TaskCategory.INSTRUCTION_FOLLOWING, ("follow these instructions", "exactly", "only output", "format as")),
        (TaskCategory.RESEARCH, ("research", "sources", "cite", "latest", "current information")),
        (TaskCategory.SUMMARIZATION, ("summarize", "summary", "tl;dr", "condense", "key points")),
        (TaskCategory.CODING, ("code", "python", "function", "debug", "implement", "bug")),
        (TaskCategory.REASONING, ("solve", "calculate", "reason", "equation", "logic", "how many")),
        (TaskCategory.WRITING, ("write", "draft", "compose", "email", "story", "rewrite")),
    )

    def classify(self, text):
        normalized = (text or "").lower()
        for category, terms in self._RULES:
            if any(term in normalized for term in terms):
                return TaskClassification(
                    category=category,
                    confidence=0.7,
                    reason="Matched a local keyword rule; classification is heuristic.",
                )
        return TaskClassification(
            category=TaskCategory.GENERAL_CONVERSATION,
            confidence=0.35,
            reason="No task-specific rule matched; using general conversation as a low-confidence label.",
        )
