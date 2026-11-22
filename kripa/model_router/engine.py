"""Evidence-gated candidate evaluation for shadow routing decisions."""

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple

from kripa.model_router.contracts import (
    Availability,
    BenchmarkResult,
    ModelCandidate,
    ModelCandidateRegistry,
    ModelProvider,
    RoutingPriority,
    RoutingRequest,
    TaskCategory,
)


class DecisionConfidence(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INSUFFICIENT = "insufficient"


@dataclass(frozen=True)
class CandidateAssessment:
    candidate: ModelCandidate
    benchmark: Optional[BenchmarkResult]
    eligible: bool
    reason: str


@dataclass(frozen=True)
class RoutingDecision:
    task_category: Optional[TaskCategory]
    recommended_provider: Optional[ModelProvider]
    recommended_model: Optional[str]
    reason: str
    alternatives: Tuple[CandidateAssessment, ...]
    confidence: DecisionConfidence


class RoutingEngine:
    """Rank only candidates with confirmed availability, capability and evidence."""

    _METRIC_ORDER = {"accuracy": 0, "speed": 1, "cost": 2}
    _PRIORITY_ORDER = {
        RoutingPriority.HIGH: 0,
        RoutingPriority.NORMAL: 1,
        RoutingPriority.LOW: 2,
    }

    def evaluate(self, request, registry, benchmark_results=(), expired_benchmark_results=()):
        if not isinstance(request, RoutingRequest):
            raise TypeError("request must be a RoutingRequest")
        if not isinstance(registry, ModelCandidateRegistry):
            raise TypeError("registry must be a ModelCandidateRegistry")

        results = tuple(benchmark_results or ())
        priorities = self._priorities(request)
        assessments = []
        eligible = []
        for candidate in registry.candidates:
            assessment = self._assess(candidate, request, results, priorities, tuple(expired_benchmark_results or ()))
            assessments.append(assessment)
            if assessment.eligible:
                eligible.append(assessment)

        if not eligible:
            reasons = sorted(set(item.reason for item in assessments))
            if not assessments:
                reason = "Insufficient evidence: no model candidates are registered."
            elif not results:
                reason = "Insufficient benchmark evidence: no benchmark results were supplied. " + "; ".join(reasons)
            elif not any("benchmark" in item.reason.lower() for item in assessments):
                reason = "Insufficient evidence: " + "; ".join(reasons)
            else:
                reason = "Insufficient benchmark evidence: " + "; ".join(reasons)
            return RoutingDecision(
                request.task_category, None, None, reason,
                tuple(assessments), DecisionConfidence.INSUFFICIENT,
            )

        eligible.sort(key=lambda item: self._sort_key(item, priorities))
        winner = eligible[0]
        benchmark = winner.benchmark
        confidence = self._confidence(winner, eligible, priorities)
        primary_name = priorities[0][0]
        reason = self._selection_reason(winner, len(eligible), primary_name)
        alternatives = tuple(
            assessment for assessment in assessments if assessment is not winner
        )
        return RoutingDecision(
            task_category=request.task_category,
            recommended_provider=winner.candidate.provider,
            recommended_model=winner.candidate.model_id,
            reason=reason,
            alternatives=alternatives,
            confidence=confidence,
        )

    def _priorities(self, request):
        requirements = request.requirements
        selected = []
        if requirements.accuracy_priority is not None:
            selected.append(("accuracy", requirements.accuracy_priority))
        if requirements.speed_priority is not None:
            selected.append(("speed", requirements.speed_priority))
        if requirements.cost_priority is not None:
            selected.append(("cost", requirements.cost_priority))
        if not selected:
            # Conservative explicit policy when the caller supplies no preference.
            selected = [("accuracy", RoutingPriority.NORMAL)]
        return tuple(sorted(
            selected,
            key=lambda item: (self._PRIORITY_ORDER[item[1]], self._METRIC_ORDER[item[0]]),
        ))

    def _assess(self, candidate, request, benchmark_results, priorities, expired_results=()):
        if candidate.availability is Availability.UNKNOWN:
            return CandidateAssessment(candidate, None, False, "Availability is unknown.")
        if candidate.availability is Availability.UNAVAILABLE:
            return CandidateAssessment(candidate, None, False, "Provider/model is marked unavailable.")
        if request.requirements.local_only and not candidate.provider.is_local:
            return CandidateAssessment(candidate, None, False, "Excluded by the local-only requirement.")
        if candidate.supported_capabilities is None:
            return CandidateAssessment(candidate, None, False, "Supported capabilities are unknown.")
        if request.task_category not in candidate.supported_capabilities:
            return CandidateAssessment(candidate, None, False, "Task capability is not declared as supported.")

        matching = [
            item for item in benchmark_results
            if item.task_category is request.task_category
            and item.provider is candidate.provider
            and item.model_id == candidate.model_id
        ]
        if not matching:
            expired = [item for item in expired_results
                       if item.task_category is request.task_category
                       and item.provider is candidate.provider and item.model_id == candidate.model_id]
            if expired:
                return CandidateAssessment(candidate, None, False, "Matching benchmark data is expired; rerun Model Arena.")
            return CandidateAssessment(candidate, None, False, "No benchmark result for this task and model.")
        if len(matching) > 1:
            return CandidateAssessment(candidate, None, False, "Multiple benchmark records are ambiguous.")
        benchmark = matching[0]
        if benchmark.success_count + benchmark.failure_count != benchmark.sample_count:
            return CandidateAssessment(candidate, benchmark, False, "Benchmark result is incomplete.")
        if benchmark.sample_count <= 0 or benchmark.success_count <= 0:
            return CandidateAssessment(candidate, benchmark, False, "Benchmark has no successful samples.")

        for metric, _priority in priorities:
            value = self._metric(benchmark, metric)
            if value is None:
                return CandidateAssessment(
                    candidate, benchmark, False,
                    "Benchmark is missing the {} metric required by the routing policy.".format(metric),
                )
        return CandidateAssessment(candidate, benchmark, True, "Eligible with declared capability and benchmark evidence.")

    @staticmethod
    def _metric(benchmark, name):
        if name == "accuracy":
            return benchmark.accuracy
        if name == "speed":
            return benchmark.median_latency_seconds
        if name == "cost":
            return benchmark.provider_cost_usd
        raise ValueError("Unknown routing metric: {}".format(name))

    def _sort_key(self, assessment, priorities):
        key = []
        for metric, _priority in priorities:
            value = self._metric(assessment.benchmark, metric)
            key.append(-value if metric == "accuracy" else value)
        key.extend((assessment.candidate.provider.value, assessment.candidate.model_id))
        return tuple(key)

    def _confidence(self, winner, eligible, priorities):
        if len(eligible) < 2 or winner.benchmark.success_count < 3:
            return DecisionConfidence.LOW
        runner_up = eligible[1]
        metric = priorities[0][0]
        best = self._metric(winner.benchmark, metric)
        next_best = self._metric(runner_up.benchmark, metric)
        if metric == "accuracy":
            separated = best - next_best >= 0.1
        else:
            separated = next_best > best and (next_best - best) / max(next_best, best, 1e-12) >= 0.1
        if separated and winner.benchmark.failure_count == 0:
            return DecisionConfidence.HIGH
        return DecisionConfidence.MEDIUM

    @staticmethod
    def _selection_reason(winner, candidate_count, metric):
        benchmark = winner.benchmark
        if metric == "accuracy":
            metric_text = "accuracy {:.0%}".format(benchmark.accuracy)
        elif metric == "speed":
            metric_text = "median latency {:.3f}s".format(benchmark.median_latency_seconds)
        else:
            metric_text = "provider-reported cost ${:.6f}".format(benchmark.provider_cost_usd)
        return (
            "Ranked first by {} among {} eligible candidates using {} successful sample(s) and {} failure(s)."
        ).format(metric_text, candidate_count, benchmark.success_count, benchmark.failure_count)
