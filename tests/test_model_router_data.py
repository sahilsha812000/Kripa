"""Offline tests for persisted shadow-router evidence and status refresh."""

import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from datetime import datetime, timedelta, timezone
from io import StringIO
from unittest.mock import patch
from types import SimpleNamespace

from kripa.model_router.availability import AvailabilityRefresher, RefreshResult
from kripa.model_router.cli import format_status, run_status
from kripa.model_router.contracts import Availability, ModelProvider, TaskCategory
from kripa.model_router.registry import build_configured_registry
from kripa.model_router.store import ModelRouterStore, ProviderStatus


class RouterDataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.store = ModelRouterStore(os.path.join(self.temp.name, "router.json"))

    def tearDown(self):
        self.temp.cleanup()

    def test_arena_aggregates_persist_and_expire(self):
        from kripa.model_arena.runner import ArenaSummary, ArenaReport
        now = datetime(2026, 1, 1, tzinfo=timezone.utc)
        summary = ArenaSummary("reasoning", "gemini", "model-a", 3, 2, 1, 0.5,
                               1.2, 1.0, 0.5, None, None, None, 0)
        report = ArenaReport([], [summary], {}, 3)
        self.assertEqual(self.store.save_arena_report(report, now), 1)
        snapshot = self.store.load()
        self.assertEqual(len(snapshot.current_benchmarks(now + timedelta(days=1))), 1)
        self.assertEqual(len(snapshot.expired_benchmarks(now + timedelta(days=31))), 1)
        with open(self.store.path, encoding="utf-8") as handle:
            payload = handle.read()
        self.assertIn('"schema_version": 1', payload)
        self.assertNotIn("api_key", payload.lower())

    def test_capabilities_require_explicit_verification(self):
        with self.assertRaises(ValueError):
            self.store.register_verified_capability("gemini", "model", "coding", "")
        record = self.store.register_verified_capability("gemini", "model", "coding", "manual test")
        self.assertEqual(record.capability, TaskCategory.CODING)
        self.assertEqual(len(self.store.load().capabilities), 1)

    def test_malformed_records_are_reported_and_not_overwritten(self):
        with open(self.store.path, "w", encoding="utf-8") as handle:
            json.dump({"schema_version": 1, "benchmarks": [{"broken": True}]}, handle)
        snapshot = self.store.load()
        self.assertTrue(snapshot.issues)
        with self.assertRaises(ValueError):
            self.store.save_availability(())

    def test_availability_cache_skips_fresh_checks_and_refreshes_stale(self):
        settings = SimpleNamespace(gemini_model="g-model")
        registry = build_configured_registry(settings)
        now = datetime(2026, 1, 1, tzinfo=timezone.utc)
        calls = []
        def checker(settings, candidate, checked_at):
            calls.append(candidate.model_id)
            return ProviderStatus(candidate.provider, candidate.model_id, Availability.AVAILABLE,
                                  checked_at.isoformat(), "checked")
        refresher = AvailabilityRefresher({ModelProvider.GEMINI: checker}, ttl_seconds=300)
        fresh = ProviderStatus(ModelProvider.GEMINI, "g-model", Availability.AVAILABLE,
                               now.isoformat(), "cached")
        result = refresher.refresh(settings, registry, (fresh,), now=now + timedelta(seconds=10))
        self.assertEqual(result.checked, 0)
        result = refresher.refresh(settings, registry, (fresh,), now=now + timedelta(seconds=301))
        self.assertEqual(result.checked, 1)
        result = refresher.refresh(settings, registry, (fresh,), force=True, now=now + timedelta(seconds=10))
        self.assertEqual(result.checked, 1)
        self.assertEqual(len(calls), 2)

    def test_report_discloses_unknown_capabilities_and_not_ready(self):
        settings = SimpleNamespace(gemini_model="g-model")
        registry = build_configured_registry(settings)
        report = format_status(registry, self.store.load(), ())
        self.assertIn("none", report)
        self.assertIn("Not ready", report)
        self.assertIn("general_conversation", report)

    def test_status_separates_benchmark_coverage_and_readiness_for_all_tasks(self):
        from kripa.model_arena.runner import ArenaReport, ArenaSummary

        settings = SimpleNamespace(
            gemini_model="g-model", groq_model="q-model",
            openrouter_model="r-model", ollama_model="o-model",
            openai_model="a-model", anthropic_model="c-model",
        )
        summaries = [ArenaSummary(
            task, provider, model, 3, 3, 0,
            None if task == "writing" else 1.0,
            0.2, 0.2, 1.0, None, None, None, 0,
        ) for task, provider, model in (
            ("reasoning", "gemini", "g-model"),
            ("coding", "groq", "q-model"),
            ("writing", "openrouter", "r-model"),
            ("instruction_following", "ollama", "o-model"),
        )]
        now = datetime.now(timezone.utc)
        self.store.save_arena_report(ArenaReport([], summaries, {}, 3), now=now)
        registry = build_configured_registry(settings)
        statuses = tuple(ProviderStatus(
            candidate.provider, candidate.model_id, Availability.AVAILABLE,
            now.isoformat(), "mocked",
        ) for candidate in registry.candidates)

        report = format_status(registry, self.store.load(), statuses, now=now)
        readiness = report.split("## Routing readiness", 1)[1]
        coverage = report.split("## Benchmark coverage", 1)[1].split("## Registered model", 1)[0]
        coverage_expectations = {
            "general_conversation": "No Arena benchmark exists.",
            "reasoning": "1 current record(s).",
            "coding": "1 current record(s).",
            "writing": "1 current record(s).",
            "research": "No Arena benchmark exists.",
            "summarization": "No Arena benchmark exists.",
            "instruction_following": "1 current record(s).",
        }
        expected = {
            "general_conversation": "No Arena benchmark exists.",
            "reasoning": "Benchmark exists, capability not verified.",
            "coding": "Benchmark exists, capability not verified.",
            "writing": "Benchmark exists, capability not verified. Writing requires manual accuracy review.",
            "research": "No Arena benchmark exists.",
            "summarization": "No Arena benchmark exists.",
            "instruction_following": "Benchmark exists, capability not verified.",
        }
        self.assertIn("## Benchmark coverage", report)
        for task, blocker in expected.items():
            with self.subTest(task=task):
                line = next(line for line in readiness.splitlines() if line.startswith("- {} —".format(task)))
                self.assertIn(blocker, line)
                self.assertIn("- {}: {}".format(task, coverage_expectations[task]), coverage)
        self.assertEqual(len(self.store.load().capabilities), 0)
        self.assertIn("Shadow mode remains enabled", readiness)

    def test_status_command_persists_mocked_check_without_live_calls(self):
        settings = SimpleNamespace(
            gemini_model="g-model", groq_model=None, openrouter_model=None,
            ollama_model=None, openai_model=None, anthropic_model=None,
        )
        class FakeRefresher:
            def refresh(self, _settings, registry, cached, force=False):
                candidate = registry.candidates[0]
                return RefreshResult((ProviderStatus(
                    candidate.provider, candidate.model_id, Availability.AVAILABLE,
                    datetime.now(timezone.utc).isoformat(), "mocked",
                ),), 1)
        output = StringIO()
        with redirect_stdout(output):
            result = run_status([], settings=settings, store=self.store, refresher=FakeRefresher())
        self.assertEqual(result, 0)
        self.assertIn("gemini", output.getvalue())
        self.assertIn("g-model", output.getvalue())
        self.assertEqual(self.store.load().availability[0].availability, Availability.AVAILABLE)

    def test_app_dispatches_router_status_command_and_refresh_flag(self):
        from kripa.app import main

        for command, handler_arguments in (
            (["router", "status"], []),
            (["router", "status", "--refresh"], ["--refresh"]),
        ):
            with self.subTest(command=command):
                with patch("kripa.model_router.cli.run_status", return_value=0) as handler:
                    self.assertEqual(main(command), 0)
                handler.assert_called_once_with(handler_arguments)

    def test_arena_cli_results_are_visible_to_router_status(self):
        from kripa.model_arena.cli import run_arena
        from kripa.model_arena.runner import ArenaReport, ArenaSummary

        task_ids = ("reasoning", "coding", "writing", "instruction_following")
        summaries = [ArenaSummary(
            task, "gemini", "g-model", 3, 3, 0,
            None if task == "writing" else 1.0, 0.2, 0.2, 1.0,
            None, None, None, 0,
        ) for task in task_ids]
        report = ArenaReport([], summaries, {}, 3)

        class FakeArena:
            def run(self, _settings):
                return report

        settings = SimpleNamespace(
            gemini_model="g-model", groq_model=None, openrouter_model=None,
            ollama_model=None, openai_model=None, anthropic_model=None,
        )
        output = StringIO()
        with redirect_stdout(output):
            self.assertEqual(run_arena(["gemini"], settings=settings, arena=FakeArena(), store=self.store), 0)
            self.assertEqual(run_status([], settings=settings, store=self.store,
                                        refresher=type("NoNetwork", (), {
                                            "refresh": lambda _self, _settings, registry, _cached, force=False:
                                            RefreshResult((ProviderStatus(
                                                registry.candidates[0].provider,
                                                registry.candidates[0].model_id,
                                                Availability.AVAILABLE,
                                                datetime.now(timezone.utc).isoformat(), "mocked",
                                            ),), 0)
                                        })()), 0)
        displayed = output.getvalue()
        self.assertIn("Saved 4 task/model benchmark summaries", displayed)
        self.assertIn("reasoning", displayed)
        self.assertIn("instruction_following", displayed)
        snapshot = self.store.load()
        self.assertEqual({item.result.task_category.value for item in snapshot.benchmarks}, set(task_ids))

    def test_arena_reports_persistence_failure_with_nonzero_exit(self):
        from kripa.model_arena.cli import run_arena
        from kripa.model_arena.runner import ArenaReport

        class FakeArena:
            def run(self, _settings):
                return ArenaReport([], [], {}, 3)

        class FailingStore:
            def save_arena_report(self, _report):
                raise ValueError("invalid local data")

        output = StringIO()
        with redirect_stdout(output):
            result = run_arena(["gemini"], settings=object(), arena=FakeArena(), store=FailingStore())
        self.assertEqual(result, 1)
        self.assertIn("Error: benchmark results could not be saved (ValueError)", output.getvalue())


if __name__ == "__main__":
    unittest.main()
