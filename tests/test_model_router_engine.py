"""Offline tests for Phase 2 shadow routing and classification."""

import unittest
from types import SimpleNamespace

from kripa.engine.conversation import Conversation
from kripa.model_router.classification import KeywordTaskClassifier, TaskClassification
from kripa.model_router.contracts import (
    Availability,
    BenchmarkResult,
    ModelCandidate,
    ModelCandidateRegistry,
    ModelProvider,
    RoutingPriority,
    RoutingRequirements,
    RoutingRequest,
    TaskCategory,
)
from kripa.model_router.engine import DecisionConfidence, RoutingEngine
from kripa.model_router.shadow import ShadowModelRouter, build_shadow_router


class RoutingEngineTests(unittest.TestCase):
    def setUp(self):
        self.category = TaskCategory.CODING
        self.candidates = (
            self._candidate(ModelProvider.GEMINI, "gemini-coder"),
            self._candidate(ModelProvider.GROQ, "groq-coder"),
            self._candidate(ModelProvider.OLLAMA, "local-coder"),
        )
        self.registry = ModelCandidateRegistry(self.candidates)
        self.benchmarks = (
            self._benchmark(ModelProvider.GEMINI, "gemini-coder", 0.95, 1.8, 0.04),
            self._benchmark(ModelProvider.GROQ, "groq-coder", 0.80, 0.3, 0.01),
            self._benchmark(ModelProvider.OLLAMA, "local-coder", 0.75, 2.0, 0.0),
        )
        self.engine = RoutingEngine()

    def _candidate(self, provider, model, availability=Availability.AVAILABLE):
        return ModelCandidate(
            provider=provider,
            model_id=model,
            availability=availability,
            supported_capabilities=frozenset({self.category}),
        )

    def _benchmark(self, provider, model, accuracy, latency, cost, successes=3, failures=0):
        return BenchmarkResult(
            task_category=self.category,
            provider=provider,
            model_id=model,
            sample_count=successes + failures,
            success_count=successes,
            failure_count=failures,
            accuracy=accuracy,
            mean_latency_seconds=latency,
            median_latency_seconds=latency,
            consistency=1.0,
            provider_cost_usd=cost,
        )

    def test_default_policy_uses_accuracy_and_returns_structured_alternatives(self):
        decision = self.engine.evaluate(
            RoutingRequest(self.category), self.registry, self.benchmarks,
        )
        self.assertEqual(decision.recommended_provider, ModelProvider.GEMINI)
        self.assertEqual(decision.recommended_model, "gemini-coder")
        self.assertIn("accuracy", decision.reason)
        self.assertEqual(len(decision.alternatives), 2)
        self.assertIs(decision.confidence, DecisionConfidence.HIGH)

    def test_accuracy_priority_selects_highest_accuracy(self):
        decision = self.engine.evaluate(
            RoutingRequest(self.category, RoutingRequirements(accuracy_priority=RoutingPriority.HIGH)),
            self.registry,
            self.benchmarks,
        )
        self.assertEqual(decision.recommended_provider, ModelProvider.GEMINI)

    def test_speed_priority_selects_lowest_median_latency(self):
        decision = self.engine.evaluate(
            RoutingRequest(self.category, RoutingRequirements(speed_priority=RoutingPriority.HIGH)),
            self.registry,
            self.benchmarks,
        )
        self.assertEqual(decision.recommended_provider, ModelProvider.GROQ)
        self.assertIn("median latency", decision.reason)

    def test_cost_priority_selects_lowest_provider_reported_cost(self):
        decision = self.engine.evaluate(
            RoutingRequest(self.category, RoutingRequirements(cost_priority=RoutingPriority.HIGH)),
            self.registry,
            self.benchmarks,
        )
        self.assertEqual(decision.recommended_provider, ModelProvider.OLLAMA)
        self.assertIn("provider-reported cost", decision.reason)

    def test_priority_levels_order_multiple_preferences(self):
        requirements = RoutingRequirements(
            accuracy_priority=RoutingPriority.LOW,
            speed_priority=RoutingPriority.HIGH,
        )
        decision = self.engine.evaluate(
            RoutingRequest(self.category, requirements), self.registry, self.benchmarks,
        )
        self.assertEqual(decision.recommended_provider, ModelProvider.GROQ)

    def test_local_only_excludes_cloud_candidates(self):
        decision = self.engine.evaluate(
            RoutingRequest(self.category, RoutingRequirements(local_only=True)),
            self.registry,
            self.benchmarks,
        )
        self.assertEqual(decision.recommended_provider, ModelProvider.OLLAMA)
        self.assertTrue(all(
            item.candidate.provider.is_local
            for item in (assessment for assessment in decision.alternatives if assessment.eligible)
        ))
        self.assertTrue(any("local-only" in item.reason for item in decision.alternatives))

    def test_unavailable_and_unknown_candidates_are_not_selected(self):
        candidates = (
            self._candidate(ModelProvider.GEMINI, "gemini-coder", Availability.UNAVAILABLE),
            self._candidate(ModelProvider.GROQ, "groq-coder", Availability.UNKNOWN),
        )
        registry = ModelCandidateRegistry(candidates)
        decision = self.engine.evaluate(
            RoutingRequest(self.category), registry,
            self.benchmarks[:2],
        )
        self.assertIsNone(decision.recommended_provider)
        self.assertIs(decision.confidence, DecisionConfidence.INSUFFICIENT)
        self.assertTrue(all(not item.eligible for item in decision.alternatives))
        self.assertIn("unavailable", decision.reason.lower())

    def test_incomplete_benchmark_is_not_routing_evidence(self):
        incomplete = BenchmarkResult(
            task_category=self.category, provider=ModelProvider.GEMINI,
            model_id="gemini-coder", sample_count=3, success_count=2,
            failure_count=0, accuracy=1.0, median_latency_seconds=1.0,
        )
        decision = self.engine.evaluate(
            RoutingRequest(self.category),
            ModelCandidateRegistry((self.candidates[0],)),
            (incomplete,),
        )
        self.assertIsNone(decision.recommended_provider)
        self.assertIn("incomplete", decision.alternatives[0].reason)

    def test_unknown_capabilities_are_never_assumed(self):
        registry = ModelCandidateRegistry((
            ModelCandidate(ModelProvider.GEMINI, "gemini-coder", Availability.AVAILABLE),
        ))
        decision = self.engine.evaluate(
            RoutingRequest(self.category), registry, self.benchmarks,
        )
        self.assertIsNone(decision.recommended_model)
        self.assertIn("capabilities are unknown", decision.reason)

    def test_missing_benchmark_and_missing_required_metric_report_insufficient_evidence(self):
        decision = self.engine.evaluate(RoutingRequest(self.category), self.registry)
        self.assertIsNone(decision.recommended_provider)
        self.assertIn("no benchmark results", decision.reason)

        no_cost = tuple(
            self._benchmark(c.provider, c.model_id, 0.8, 1.0, None)
            for c in self.candidates
        )
        cost_decision = self.engine.evaluate(
            RoutingRequest(self.category, RoutingRequirements(cost_priority=RoutingPriority.HIGH)),
            self.registry,
            no_cost,
        )
        self.assertIsNone(cost_decision.recommended_model)
        self.assertIn("missing the cost metric", cost_decision.reason)

    def test_candidates_without_matching_task_capability_are_filtered(self):
        registry = ModelCandidateRegistry((
            ModelCandidate(
                ModelProvider.GEMINI, "gemini-writing", Availability.AVAILABLE,
                frozenset({TaskCategory.WRITING}),
            ),
        ))
        decision = self.engine.evaluate(RoutingRequest(self.category), registry, self.benchmarks)
        self.assertIsNone(decision.recommended_model)
        self.assertIn("not declared as supported", decision.reason)


class ShadowIntegrationTests(unittest.TestCase):
    def test_keyword_classifier_covers_task_types_and_general_fallback(self):
        classifier = KeywordTaskClassifier()
        cases = (
            ("Please calculate 4 plus 5", TaskCategory.REASONING),
            ("Write a short email", TaskCategory.WRITING),
            ("Debug this Python function", TaskCategory.CODING),
            ("Research current sources", TaskCategory.RESEARCH),
            ("Summarize these key points", TaskCategory.SUMMARIZATION),
            ("Follow these instructions exactly", TaskCategory.INSTRUCTION_FOLLOWING),
            ("Hello there", TaskCategory.GENERAL_CONVERSATION),
        )
        for prompt, expected in cases:
            with self.subTest(prompt=prompt):
                self.assertEqual(classifier.classify(prompt).category, expected)

    def test_shadow_router_reports_no_benchmark_evidence_without_changing_execution(self):
        provider = _RecordingProvider()
        settings = SimpleNamespace(
            gemini_model="gemini-configured",
            groq_model="groq-configured",
            openrouter_model="router-configured",
            ollama_model="ollama-configured",
            openai_model="openai-configured",
            anthropic_model="anthropic-configured",
        )
        shadow = build_shadow_router(settings)
        conversation = Conversation(provider, shadow_router=shadow)

        answer = conversation.send("Please write a short note")

        self.assertEqual(answer, "active provider answer")
        self.assertEqual(len(provider.calls), 1)
        self.assertEqual(provider.calls[0][-1].content, "Please write a short note")
        self.assertEqual(conversation.last_shadow_decision.task_category, TaskCategory.WRITING)
        self.assertIsNone(conversation.last_shadow_decision.recommended_provider)
        # With no refreshed availability, the router reports that blocker before
        # benchmark coverage. The shadow result must remain advisory only.
        self.assertIn("insufficient evidence", conversation.last_shadow_decision.reason.lower())
        self.assertIn("availability is unknown", conversation.last_shadow_decision.reason.lower())

    def test_shadow_failure_never_blocks_current_provider(self):
        provider = _RecordingProvider()

        class BrokenShadow:
            def evaluate(self, prompt):
                raise RuntimeError("shadow classifier failed")

        conversation = Conversation(provider, shadow_router=BrokenShadow())
        self.assertEqual(conversation.send("hello"), "active provider answer")
        self.assertEqual(len(provider.calls), 1)
        self.assertIsNone(conversation.last_shadow_decision)

    def test_low_confidence_task_label_cannot_drive_a_route(self):
        class LowConfidenceClassifier:
            def classify(self, text):
                return TaskClassification(
                    TaskCategory.GENERAL_CONVERSATION, 0.2, "Ambiguous input",
                )

        shadow = ShadowModelRouter(
            classifier=LowConfidenceClassifier(),
            engine=RoutingEngine(),
            registry=ModelCandidateRegistry(()),
        )
        decision = shadow.evaluate("unclear input")
        self.assertIsNone(decision.recommended_provider)
        self.assertIn("below the 50% routing threshold", decision.reason)


class _RecordingProvider:
    def __init__(self):
        self.calls = []

    def reply(self, messages):
        self.calls.append(messages)
        return "active provider answer"


if __name__ == "__main__":
    unittest.main()
