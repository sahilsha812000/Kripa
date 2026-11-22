"""Offline tests for Model Arena v2 orchestration and evaluation."""

import unittest
from unittest.mock import patch

from kripa.engine.errors import ProviderError
from kripa.model_arena.quality import check_response
from kripa.model_arena.report import format_report
from kripa.model_arena.runner import ModelArena
from kripa.model_arena.tasks import TASKS


GOOD_RESPONSES = {
    "reasoning": "24 - 8 + 5 = 21",
    "general_conversation": "Set a phone reminder for the same time each morning.",
    "research": (
        "The pilot gave packets to 34 households; its voluntary survey did not measure "
        "harvest size, so it is not a yield estimate [Northfield Library, June]."
    ),
    "summarization": (
        "- Route 8 detours Monday through Thursday while Elm Bridge is closed.\n"
        "- The eastbound Pine Street stop moves to Oak Avenue; westbound service is "
        "unchanged, and normal service is expected Friday."
    ),
    "coding": (
        "def is_palindrome(text):\n"
        "    normalized = text.lower().replace(' ', '')\n"
        "    return normalized == normalized[::-1]"
    ),
    "writing": "Student, your robotics practice matters. Keep learning, and your progress will grow. I believe in you.",
    "instruction_following": (
        "KRIPA\nI can help with reasoning, coding, writing, and instructions.\n"
        "If unsure, I will say so.\nI keep API keys private."
    ),
}


class ModelArenaTests(unittest.TestCase):
    def test_three_identical_runs_per_task_and_provider(self):
        received = {"gemini": [], "groq": []}

        class FakeProvider:
            model = "test-model"

            def __init__(self, name):
                self.name = name
                self.last_usage = None
                self.last_provider_cost_usd = None

            def reply(self, messages):
                prompt = messages[-1].content
                received[self.name].append(prompt)
                return GOOD_RESPONSES[next(task.task_id for task in TASKS if task.prompt == prompt)]

        report = ModelArena(
            provider_names=("gemini", "groq"),
            provider_factory=lambda settings, name: FakeProvider(name),
        ).run(settings=object())

        self.assertEqual(len(report.results), len(TASKS) * 2 * 3)
        for provider in received:
            self.assertEqual(received[provider], [task.prompt for task in TASKS for _ in range(3)])
        self.assertTrue(all(result.run_number in (1, 2, 3) for result in report.results))
        self.assertEqual(report.runs_per_task, 3)

    def test_each_repeated_gemini_run_starts_without_prior_interaction_context(self):
        class StatefulProvider:
            model = "gemini-test"

            def __init__(self):
                self.previous_interaction_id = "stale-value"
                self.calls = 0

            def reply(self, messages):
                self.assert_prompt = messages[-1].content
                if self.previous_interaction_id is not None:
                    raise AssertionError("Arena run inherited prior interaction context")
                self.calls += 1
                self.previous_interaction_id = "interaction-{}".format(self.calls)
                return GOOD_RESPONSES["reasoning"]

        provider = StatefulProvider()
        report = ModelArena(
            provider_names=("gemini",),
            provider_factory=lambda settings, name: provider,
            tasks=(TASKS[0],),
        ).run(settings=object())
        self.assertEqual(provider.calls, 3)
        self.assertEqual(len(report.results), 3)

    def test_reasoning_and_instruction_following_use_expected_answers(self):
        by_id = {task.task_id: task for task in TASKS}
        self.assertTrue(check_response("reasoning", "24 - 8 + 5 = 21", 21))
        self.assertFalse(check_response("reasoning", "24 - 8 + 5 = 20", 21))
        self.assertTrue(check_response(
            "instruction_following", GOOD_RESPONSES["instruction_following"],
            by_id["instruction_following"].expected,
        ))
        self.assertFalse(check_response(
            "instruction_following", GOOD_RESPONSES["instruction_following"] + "\nExtra",
            by_id["instruction_following"].expected,
        ))

    def test_new_general_conversation_research_and_summary_checks(self):
        tasks = {task.task_id: task for task in TASKS}
        self.assertTrue(check_response(
            "general_conversation", GOOD_RESPONSES["general_conversation"],
            tasks["general_conversation"].expected,
        ))
        self.assertFalse(check_response(
            "general_conversation", "That sounds difficult.",
            tasks["general_conversation"].expected,
        ))
        self.assertTrue(check_response(
            "research", GOOD_RESPONSES["research"], tasks["research"].expected,
        ))
        self.assertFalse(check_response(
            "research", "34 households took part, but the report omitted the limitation.",
            tasks["research"].expected,
        ))
        self.assertTrue(check_response(
            "summarization", GOOD_RESPONSES["summarization"],
            tasks["summarization"].expected,
        ))
        self.assertFalse(check_response(
            "summarization", "- Route 8 is affected.\n- Westbound service is unchanged.",
            tasks["summarization"].expected,
        ))

    def test_coding_is_checked_against_cases_without_executing_source(self):
        task = next(item for item in TASKS if item.task_id == "coding")
        self.assertTrue(check_response("coding", GOOD_RESPONSES["coding"], task.expected))
        self.assertTrue(check_response(
            "coding", GOOD_RESPONSES["coding"] + "\nprint(is_palindrome('Race car'))", task.expected,
        ))
        incorrect = GOOD_RESPONSES["coding"].replace("normalized == normalized[::-1]", "True")
        self.assertFalse(check_response("coding", incorrect, task.expected))
        malicious = (
            "def is_palindrome(text):\n"
            "    return __import__('os').system('touch /tmp/kripa-arena-should-not-exist')"
        )
        self.assertFalse(check_response("coding", malicious, task.expected))

    def test_writing_has_no_automated_quality_score(self):
        self.assertIsNone(check_response("writing", GOOD_RESPONSES["writing"], None))

    def test_failures_are_preserved_and_do_not_stop_other_providers(self):
        class FakeProvider:
            model = "working-model"

            def reply(self, messages):
                return GOOD_RESPONSES[next(task.task_id for task in TASKS if task.prompt == messages[-1].content)]

        def provider_factory(settings, name):
            if name == "gemini":
                raise ProviderError("GEMINI_API_KEY is missing.")
            return FakeProvider()

        class Settings:
            gemini_model = "configured-gemini-model"

        report = ModelArena(
            provider_names=("gemini", "groq"), provider_factory=provider_factory,
        ).run(settings=Settings())
        failed = [result for result in report.results if result.provider == "gemini"]
        succeeded = [result for result in report.results if result.provider == "groq"]
        self.assertEqual(len(failed), len(TASKS) * 3)
        self.assertTrue(all(result.error for result in failed))
        self.assertTrue(all(result.model == "configured-gemini-model" for result in failed))
        self.assertTrue(all(not result.error for result in succeeded))
        self.assertEqual(report.leaderboard["reasoning"][0].provider, "groq")

    def test_request_failure_counts_and_empty_leaderboard(self):
        class BrokenProvider:
            model = "broken-model"

            def reply(self, messages):
                raise ProviderError("Could not connect to provider.")

        report = ModelArena(
            provider_names=("ollama",),
            provider_factory=lambda settings, name: BrokenProvider(),
        ).run(settings=object())
        summary = next(item for item in report.summaries if item.task_id == "reasoning")
        self.assertEqual((summary.attempts, summary.successes, summary.failures), (3, 0, 3))
        self.assertEqual(report.leaderboard["reasoning"], [])
        self.assertIn("FAILED: Could not connect", format_report(report))

    def test_average_median_and_response_consistency(self):
        class SequenceProvider:
            model = "timed-model"

            def __init__(self):
                self.index = 0

            def reply(self, messages):
                answers = ("24 - 8 + 5 = 21", "24 - 8 + 5 = 21", "24 - 8 + 5 = 21.")
                answer = answers[self.index]
                self.index += 1
                return answer

        ticks = iter((0, 1, 2, 6, 7, 10))
        with patch("kripa.model_arena.runner.perf_counter", side_effect=lambda: next(ticks)):
            report = ModelArena(
                provider_names=("gemini",), provider_factory=lambda settings, name: SequenceProvider(),
                tasks=(TASKS[0],),
            ).run(settings=object())
        summary = report.summaries[0]
        self.assertAlmostEqual(summary.average_seconds, 8.0 / 3.0)
        self.assertEqual(summary.median_seconds, 3.0)
        self.assertEqual(summary.accuracy, 1.0)
        self.assertAlmostEqual(summary.exact_response_consistency, 2.0 / 3.0)

    def test_usage_and_cost_only_appear_when_provider_reports_them(self):
        class MeteredProvider:
            model = "metered-model"

            def __init__(self):
                self.last_usage = None
                self.last_provider_cost_usd = None

            def reply(self, messages):
                self.last_usage = {"input_tokens": 12, "output_tokens": 7}
                self.last_provider_cost_usd = 0.0004
                return GOOD_RESPONSES[next(task.task_id for task in TASKS if task.prompt == messages[-1].content)]

        report = ModelArena(
            provider_names=("openrouter",), provider_factory=lambda settings, name: MeteredProvider(),
        ).run(settings=object())
        summary = next(item for item in report.summaries if item.task_id == "reasoning")
        self.assertEqual(summary.input_tokens, 36)
        self.assertEqual(summary.output_tokens, 21)
        self.assertAlmostEqual(summary.provider_cost_usd, 0.0012)
        self.assertEqual(summary.usage_reports, 3)

        class UnmeteredProvider:
            model = "unmetered"

            def reply(self, messages):
                return GOOD_RESPONSES[next(task.task_id for task in TASKS if task.prompt == messages[-1].content)]

        unmetered = ModelArena(
            provider_names=("ollama",), provider_factory=lambda settings, name: UnmeteredProvider(),
        ).run(settings=object())
        unmetered_summary = next(item for item in unmetered.summaries if item.task_id == "reasoning")
        self.assertIsNone(unmetered_summary.input_tokens)
        self.assertIsNone(unmetered_summary.provider_cost_usd)

    def test_report_separates_metrics_and_keeps_writing_for_review(self):
        class FakeProvider:
            model = "review-model"

            def reply(self, messages):
                return GOOD_RESPONSES[next(task.task_id for task in TASKS if task.prompt == messages[-1].content)]

        report = ModelArena(
            provider_names=("groq",), provider_factory=lambda settings, name: FakeProvider(),
        ).run(settings=object())
        formatted = format_report(report)
        self.assertIn("Accuracy leaderboard", formatted)
        self.assertIn("Average", formatted)
        self.assertIn("Exact-response consistency", formatted)
        self.assertIn("Writing: manual review", formatted)
        self.assertIn("No prices are inferred", formatted)

    def test_unknown_or_duplicate_providers_and_invalid_run_count_are_rejected(self):
        with self.assertRaises(ValueError):
            ModelArena(provider_names=("gemini", "unknown"))
        with self.assertRaises(ValueError):
            ModelArena(provider_names=("gemini", "gemini"))
        with self.assertRaises(ValueError):
            ModelArena(runs_per_task=0)


if __name__ == "__main__":
    unittest.main()
