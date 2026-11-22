"""Typed, provider-neutral data contracts for future intelligent routing."""

from dataclasses import dataclass, field
from enum import Enum
from math import isfinite
from typing import FrozenSet, Optional, Tuple


class TaskCategory(str, Enum):
    GENERAL_CONVERSATION = "general_conversation"
    REASONING = "reasoning"
    CODING = "coding"
    WRITING = "writing"
    RESEARCH = "research"
    SUMMARIZATION = "summarization"
    INSTRUCTION_FOLLOWING = "instruction_following"


class RoutingPriority(str, Enum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"


class ModelProvider(str, Enum):
    GEMINI = "gemini"
    GROQ = "groq"
    OPENROUTER = "openrouter"
    OLLAMA = "ollama"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"

    @property
    def is_local(self):
        return self is ModelProvider.OLLAMA


class Availability(str, Enum):
    UNKNOWN = "unknown"
    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True)
class RoutingRequirements:
    """Optional preferences supplied for a future routing decision."""

    accuracy_priority: Optional[RoutingPriority] = None
    speed_priority: Optional[RoutingPriority] = None
    cost_priority: Optional[RoutingPriority] = None
    local_only: bool = False

    def __post_init__(self):
        for name in ("accuracy_priority", "speed_priority", "cost_priority"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, RoutingPriority):
                object.__setattr__(self, name, RoutingPriority(value))
        if not isinstance(self.local_only, bool):
            raise ValueError("local_only must be a boolean.")


@dataclass(frozen=True)
class RoutingRequest:
    """A task category and optional constraints; contains no prompt or secrets."""

    task_category: TaskCategory
    requirements: RoutingRequirements = field(default_factory=RoutingRequirements)

    def __post_init__(self):
        if not isinstance(self.task_category, TaskCategory):
            object.__setattr__(self, "task_category", TaskCategory(self.task_category))
        if not isinstance(self.requirements, RoutingRequirements):
            raise ValueError("requirements must be a RoutingRequirements value.")


@dataclass(frozen=True)
class ModelCandidate:
    """One configured provider/model pair and its currently known metadata."""

    provider: ModelProvider
    model_id: str
    availability: Availability = Availability.UNKNOWN
    supported_capabilities: Optional[FrozenSet[TaskCategory]] = None

    def __post_init__(self):
        if not isinstance(self.provider, ModelProvider):
            object.__setattr__(self, "provider", ModelProvider(self.provider))
        if not isinstance(self.availability, Availability):
            object.__setattr__(self, "availability", Availability(self.availability))
        if not self.model_id or not self.model_id.strip():
            raise ValueError("A model candidate must have a non-empty model ID.")
        object.__setattr__(self, "model_id", self.model_id.strip())
        if self.supported_capabilities is not None:
            capabilities = frozenset(
                category if isinstance(category, TaskCategory) else TaskCategory(category)
                for category in self.supported_capabilities
            )
            object.__setattr__(self, "supported_capabilities", capabilities)


@dataclass(frozen=True)
class ModelCandidateRegistry:
    """Immutable candidate collection with unique provider/model identities."""

    candidates: Tuple[ModelCandidate, ...] = ()

    def __post_init__(self):
        candidates = tuple(self.candidates)
        keys = [(item.provider, item.model_id) for item in candidates]
        if len(keys) != len(set(keys)):
            raise ValueError("The candidate registry contains duplicate provider/model pairs.")
        object.__setattr__(self, "candidates", candidates)

    def for_provider(self, provider):
        provider = provider if isinstance(provider, ModelProvider) else ModelProvider(provider)
        return tuple(item for item in self.candidates if item.provider is provider)

    def find(self, provider, model_id):
        provider = provider if isinstance(provider, ModelProvider) else ModelProvider(provider)
        return next(
            (item for item in self.candidates
             if item.provider is provider and item.model_id == model_id),
            None,
        )


@dataclass(frozen=True)
class BenchmarkResult:
    """Aggregate benchmark record shaped to accept Model Arena v2 summaries."""

    task_category: TaskCategory
    provider: ModelProvider
    model_id: str
    sample_count: int
    success_count: int
    failure_count: int
    accuracy: Optional[float] = None
    mean_latency_seconds: Optional[float] = None
    median_latency_seconds: Optional[float] = None
    consistency: Optional[float] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    provider_cost_usd: Optional[float] = None
    usage_reports: int = 0

    def __post_init__(self):
        if not isinstance(self.task_category, TaskCategory):
            object.__setattr__(self, "task_category", TaskCategory(self.task_category))
        if not isinstance(self.provider, ModelProvider):
            object.__setattr__(self, "provider", ModelProvider(self.provider))
        if not self.model_id or not self.model_id.strip():
            raise ValueError("A benchmark result must identify its model.")
        object.__setattr__(self, "model_id", self.model_id.strip())
        counts = (self.sample_count, self.success_count, self.failure_count, self.usage_reports)
        if any(not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in counts):
            raise ValueError("Benchmark sample and usage counts cannot be negative.")
        if self.success_count + self.failure_count > self.sample_count:
            raise ValueError("Success and failure counts cannot exceed sample_count.")
        for name in ("accuracy", "consistency"):
            value = getattr(self, name)
            if value is not None and (
                not isinstance(value, (int, float)) or isinstance(value, bool)
                or not isfinite(value) or not 0.0 <= value <= 1.0
            ):
                raise ValueError("{} must be between 0 and 1 when present.".format(name))
        for name in ("mean_latency_seconds", "median_latency_seconds", "provider_cost_usd"):
            value = getattr(self, name)
            if value is not None and (
                not isinstance(value, (int, float)) or isinstance(value, bool)
                or not isfinite(value) or value < 0
            ):
                raise ValueError("{} cannot be negative or non-finite.".format(name))
        for name in ("input_tokens", "output_tokens"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, int) or isinstance(value, bool) or value < 0):
                raise ValueError("{} must be a nonnegative integer when present.".format(name))

    @classmethod
    def from_arena_summary(cls, summary):
        """Adapt an existing Model Arena v2 per-task/provider aggregate."""
        return cls(
            task_category=TaskCategory(summary.task_id),
            provider=ModelProvider(summary.provider),
            model_id=summary.model,
            sample_count=summary.attempts,
            success_count=summary.successes,
            failure_count=summary.failures,
            accuracy=summary.accuracy,
            mean_latency_seconds=summary.average_seconds,
            median_latency_seconds=summary.median_seconds,
            consistency=summary.exact_response_consistency,
            input_tokens=summary.input_tokens,
            output_tokens=summary.output_tokens,
            provider_cost_usd=summary.provider_cost_usd,
            usage_reports=summary.usage_reports,
        )
