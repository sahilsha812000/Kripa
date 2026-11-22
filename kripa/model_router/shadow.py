"""Shadow-only classification and routing analysis for live chat turns."""

from dataclasses import replace

from kripa.model_router.classification import KeywordTaskClassifier
from kripa.model_router.contracts import RoutingRequest
from kripa.model_router.engine import DecisionConfidence, RoutingDecision, RoutingEngine
from kripa.model_router.registry import build_configured_registry


class ShadowModelRouter:
    """Create observable recommendations without selecting an execution provider."""

    MINIMUM_CLASSIFICATION_CONFIDENCE = 0.5

    def __init__(self, classifier, engine, registry, benchmark_results=(), requirements=None,
                 expired_benchmark_results=(), data_issues=()):
        self.classifier = classifier
        self.engine = engine
        self.registry = registry
        self.benchmark_results = tuple(benchmark_results or ())
        self.requirements = requirements
        self.expired_benchmark_results = tuple(expired_benchmark_results or ())
        self.data_issues = tuple(data_issues or ())

    def evaluate(self, prompt):
        classification = self.classifier.classify(prompt)
        if classification.category is None:
            return RoutingDecision(
                None, None, None,
                "Insufficient evidence: task classifier returned no category. " + classification.reason,
                (), DecisionConfidence.INSUFFICIENT,
            )
        if classification.confidence < self.MINIMUM_CLASSIFICATION_CONFIDENCE:
            return RoutingDecision(
                classification.category, None, None,
                "Insufficient evidence: task classification confidence {:.0%} is below the {:.0%} routing threshold.".format(
                    classification.confidence, self.MINIMUM_CLASSIFICATION_CONFIDENCE,
                ),
                (), DecisionConfidence.INSUFFICIENT,
            )
        request = RoutingRequest(classification.category, self.requirements) if self.requirements else RoutingRequest(
            classification.category
        )
        decision = self.engine.evaluate(request, self.registry, self.benchmark_results,
                                        self.expired_benchmark_results)
        reason = "{} Task classification confidence {:.0%}: {}".format(
            decision.reason, classification.confidence, classification.reason,
        )
        if self.data_issues and decision.confidence.value == "insufficient":
            reason += " Local router data has incomplete records; inspect `python -m kripa router status`."
        return replace(decision, reason=reason)


def build_shadow_router(settings, benchmark_results=None, registry=None, classifier=None, engine=None,
                        store=None):
    """Build shadow analysis from persisted evidence; never probes providers."""
    from kripa.model_router.store import ModelRouterStore

    store = store or ModelRouterStore()
    snapshot = store.load() if benchmark_results is None or registry is None else None
    if snapshot is not None:
        if benchmark_results is None:
            benchmark_results = snapshot.current_benchmarks()
        availability = { (item.provider, item.model_id): item.availability
                         for item in snapshot.fresh_statuses() }
        capabilities = {}
        for item in snapshot.capabilities:
            capabilities.setdefault((item.provider, item.model_id), set()).add(item.capability)
        if registry is None:
            configured = build_configured_registry(settings)
            from kripa.model_router.contracts import ModelCandidate, ModelCandidateRegistry, Availability
            candidates = []
            for candidate in configured.candidates:
                candidates.append(ModelCandidate(
                    candidate.provider, candidate.model_id,
                    availability.get((candidate.provider, candidate.model_id), Availability.UNKNOWN),
                    frozenset(capabilities[(candidate.provider, candidate.model_id)])
                    if (candidate.provider, candidate.model_id) in capabilities else None,
                ))
            registry = ModelCandidateRegistry(tuple(candidates))
    return ShadowModelRouter(
        classifier=classifier or KeywordTaskClassifier(),
        engine=engine or RoutingEngine(),
        registry=registry or build_configured_registry(settings),
        benchmark_results=benchmark_results,
        expired_benchmark_results=snapshot.expired_benchmarks() if snapshot is not None else (),
        data_issues=snapshot.issues if snapshot is not None else (),
    )
