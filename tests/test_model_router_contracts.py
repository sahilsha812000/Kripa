"""Tests for Phase 1 task-aware routing data contracts."""

import unittest
from types import SimpleNamespace

from kripa.model_arena.runner import ArenaSummary
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
from kripa.model_router.registry import build_configured_registry


class RoutingContractTests(unittest.TestCase):
    def test_task_categories_cover_the_requested_taxonomy(self):
        self.assertEqual(
            {category.value for category in TaskCategory},
            {
                "general_conversation", "reasoning", "coding", "writing",
                "research", "summarization", "instruction_following",
            },
        )

    def test_routing_requirements_are_optional_and_typed(self):
        defaults = RoutingRequirements()
        self.assertIsNone(defaults.accuracy_priority)
        self.assertIsNone(defaults.speed_priority)
        self.assertIsNone(defaults.cost_priority)
        self.assertFalse(defaults.local_only)

        request = RoutingRequest(
            task_category="coding",
            requirements=RoutingRequirements(
                accuracy_priority=RoutingPriority.HIGH,
                speed_priority="normal",
                cost_priority=RoutingPriority.LOW,
                local_only=True,
            ),
        )
        self.assertIs(request.task_category, TaskCategory.CODING)
        self.assertIs(request.requirements.speed_priority, RoutingPriority.NORMAL)
        self.assertTrue(request.requirements.local_only)

    def test_candidate_registry_contains_configured_models_without_claiming_unknown_data(self):
        settings = SimpleNamespace(
            gemini_model="gemini-model",
            groq_model="groq-model",
            openrouter_model="router/model",
            ollama_model="local-model",
            openai_model="openai-model",
            anthropic_model="anthropic-model",
            gemini_api_key="must-not-enter-registry",
        )
        registry = build_configured_registry(settings)

        self.assertEqual(len(registry.candidates), 6)
        gemini = registry.find(ModelProvider.GEMINI, "gemini-model")
        self.assertIsNotNone(gemini)
        self.assertIs(gemini.availability, Availability.UNKNOWN)
        self.assertIsNone(gemini.supported_capabilities)
        self.assertTrue(registry.find("ollama", "local-model").provider.is_local)
        self.assertNotIn("must-not-enter-registry", repr(registry))

    def test_registry_accepts_explicit_availability_and_capabilities(self):
        settings = SimpleNamespace(gemini_model="gemini-model")
        registry = build_configured_registry(
            settings,
            availability={"gemini": "available"},
            capabilities={ModelProvider.GEMINI: {TaskCategory.CODING, TaskCategory.REASONING}},
        )
        candidate = registry.candidates[0]
        self.assertIs(candidate.availability, Availability.AVAILABLE)
        self.assertEqual(
            candidate.supported_capabilities,
            frozenset({TaskCategory.CODING, TaskCategory.REASONING}),
        )

    def test_registry_rejects_duplicate_candidate_identity(self):
        candidate = ModelCandidate(ModelProvider.GROQ, "same-model")
        with self.assertRaises(ValueError):
            ModelCandidateRegistry((candidate, candidate))

    def test_arena_v2_summary_converts_to_benchmark_contract(self):
        summary = ArenaSummary(
            task_id="reasoning",
            provider="groq",
            model="groq-model",
            attempts=3,
            successes=2,
            failures=1,
            accuracy=1.0,
            average_seconds=0.8,
            median_seconds=0.7,
            exact_response_consistency=0.5,
            input_tokens=120,
            output_tokens=45,
            provider_cost_usd=0.001,
            usage_reports=2,
        )
        result = BenchmarkResult.from_arena_summary(summary)

        self.assertEqual(result.task_category, TaskCategory.REASONING)
        self.assertEqual(result.provider, ModelProvider.GROQ)
        self.assertEqual(result.model_id, "groq-model")
        self.assertEqual((result.sample_count, result.success_count, result.failure_count), (3, 2, 1))
        self.assertEqual(result.mean_latency_seconds, 0.8)
        self.assertEqual(result.median_latency_seconds, 0.7)
        self.assertEqual(result.consistency, 0.5)
        self.assertEqual(result.provider_cost_usd, 0.001)

    def test_manual_review_benchmark_can_leave_accuracy_unknown(self):
        result = BenchmarkResult(
            task_category=TaskCategory.WRITING,
            provider=ModelProvider.OLLAMA,
            model_id="local-model",
            sample_count=3,
            success_count=3,
            failure_count=0,
        )
        self.assertIsNone(result.accuracy)
        self.assertIsNone(result.provider_cost_usd)

    def test_benchmark_contract_rejects_invalid_metrics(self):
        with self.assertRaises(ValueError):
            BenchmarkResult(
                task_category=TaskCategory.CODING,
                provider=ModelProvider.GROQ,
                model_id="model",
                sample_count=2,
                success_count=2,
                failure_count=1,
            )
        with self.assertRaises(ValueError):
            BenchmarkResult(
                task_category=TaskCategory.CODING,
                provider=ModelProvider.GROQ,
                model_id="model",
                sample_count=1,
                success_count=1,
                failure_count=0,
                accuracy=1.5,
            )


if __name__ == "__main__":
    unittest.main()
