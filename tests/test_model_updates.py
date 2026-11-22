"""Offline tests for read-only model discovery and update reports."""

import hashlib
import json
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from kripa.app import main
from kripa.model_updates.discoverers import (
    _ollama_registry_digest,
    discover_gemini,
    discover_ollama,
    discover_openrouter,
)
from kripa.model_updates.manager import ModelUpdateManager
from kripa.model_updates.models import CatalogSnapshot, ModelRecord
from kripa.model_updates.report import format_report


def make_settings():
    return SimpleNamespace(
        gemini_api_key="placeholder-gemini",
        groq_api_key="placeholder-groq",
        openrouter_api_key="placeholder-openrouter",
        gemini_model="gemini-3.5-flash-lite",
        groq_model="llama-old",
        openrouter_model="openai/gpt-4.2",
        ollama_base_url="http://localhost:11434",
        ollama_model="llama3.2:3b",
    )


class ModelUpdateManagerTests(unittest.TestCase):
    def test_reports_current_outdated_retired_and_ollama_update_states(self):
        snapshots = {
            "Gemini": CatalogSnapshot("Gemini", [
                ModelRecord("gemini-3.5-flash-lite", created_at=1, compatibility="text generation"),
                ModelRecord("gemini-4-flash-lite", created_at=2, compatibility="text generation"),
            ]),
            "Groq": CatalogSnapshot("Groq", [
                ModelRecord("llama-old", created_at=1, active=False),
                ModelRecord("llama-new", created_at=2, active=True),
            ]),
            "OpenRouter": CatalogSnapshot("OpenRouter", [
                ModelRecord("openai/gpt-4.2", created_at=1, expires_at="2099-01-01T00:00:00Z"),
                ModelRecord("openai/gpt-5", created_at=2),
            ]),
            "Ollama": CatalogSnapshot(
                "Ollama",
                [ModelRecord("llama3.2:3b", active=True, compatibility="llama; Q4_K_M")],
                installed_models=[{
                    "model": "llama3.2:3b",
                    "installed": True,
                    "local_digest": "a" * 64,
                    "latest_digest": "b" * 64,
                    "update_available": True,
                    "status": "update available",
                }],
            ),
        }
        manager = ModelUpdateManager(
            discoverers={name: (lambda settings, snapshot=snapshot: snapshot)
                         for name, snapshot in snapshots.items()}
        )
        report = manager.generate_report(make_settings())
        rows = {row.provider: row for row in report.rows}
        self.assertEqual(rows["Gemini"].deprecation_status, "outdated")
        self.assertEqual(rows["Gemini"].recommended_replacement, "gemini-4-flash-lite")
        self.assertEqual(rows["Groq"].deprecation_status, "retired")
        self.assertEqual(rows["Groq"].recommended_replacement, "llama-new")
        self.assertEqual(rows["OpenRouter"].deprecation_status, "deprecated")
        self.assertEqual(rows["OpenRouter"].recommended_replacement, "openai/gpt-5")
        self.assertEqual(rows["Ollama"].deprecation_status, "update available")
        self.assertEqual(rows["Ollama"].recommended_replacement, "llama3.2:3b")
        self.assertEqual(len(report.ollama_installed_models), 1)

    def test_unavailable_catalog_reports_unknown_without_revealing_details(self):
        manager = ModelUpdateManager(discoverers={
            provider: (lambda settings: CatalogSnapshot(
                provider, error="Catalog check failed (AuthenticationError)"
            ))
            for provider in ("Gemini", "Groq", "OpenRouter", "Ollama")
        })
        report = manager.generate_report(make_settings())
        self.assertTrue(all(row.deprecation_status == "not checked" for row in report.rows))
        formatted = format_report(report)
        self.assertNotIn("placeholder-", formatted)
        self.assertIn("AuthenticationError", formatted)

    def test_openrouter_catalog_uses_public_model_list(self):
        data = {"data": [
            {
                "id": "openai/new-model",
                "created": 200,
                "architecture": {
                    "input_modalities": ["text", "image"],
                    "output_modalities": ["text"],
                },
                "context_length": 64000,
                "pricing": {"prompt": "0.1"},
            },
            {
                "id": "audio/speech-model",
                "created": 300,
                "architecture": {"output_modalities": ["audio"]},
            },
        ]}
        with patch("kripa.model_updates.discoverers._get_json", return_value=data) as get_json:
            snapshot = discover_openrouter(make_settings())
        get_json.assert_called_once()
        self.assertIn("/models?sort=newest", get_json.call_args.args[0])
        self.assertEqual([record.model_id for record in snapshot.records], ["openai/new-model"])
        self.assertIn("context window 64000", snapshot.records[0].compatibility)

    def test_gemini_catalog_uses_header_auth_and_filters_non_generation_models(self):
        pages = iter([
            {"models": [
                {
                    "name": "models/gemini-4-flash",
                    "supportedActions": ["generateContent"],
                    "createTime": "2026-01-01T00:00:00Z",
                    "inputTokenLimit": 100000,
                },
                {
                    "name": "models/text-embedding-new",
                    "supportedActions": ["embedContent"],
                },
                {
                    "name": "models/gemini-4-native-audio-preview",
                    "supportedActions": ["generateContent"],
                },
            ], "nextPageToken": "next-page"},
            {"models": [{
                "name": "models/gemini-4-pro",
                "supportedActions": ["generateContent"],
                "createTime": "2026-02-01T00:00:00Z",
                "deprecated": True,
            }]},
        ])

        def get_json(request, timeout=10):
            self.assertNotIn("key=", request.full_url)
            self.assertEqual(request.get_header("X-goog-api-key"), "placeholder-gemini")
            return next(pages)

        with patch("kripa.model_updates.discoverers._get_json", side_effect=get_json):
            snapshot = discover_gemini(make_settings())
        self.assertEqual(
            [record.model_id for record in snapshot.records],
            ["gemini-4-flash", "gemini-4-pro"],
        )
        self.assertTrue(snapshot.records[1].deprecated)
        self.assertIn("input tokens 100000", snapshot.records[0].compatibility)

    def test_ollama_checks_local_tags_and_registry_digest_without_pull(self):
        tags = {"models": [{
            "name": "llama3.2:3b",
            "digest": "a" * 64,
            "details": {"family": "llama", "parameter_size": "3.2B", "quantization_level": "Q4_K_M"},
        }]}
        response = MagicMock()
        response.read.return_value = json.dumps(tags).encode("utf-8")
        with patch("kripa.model_updates.discoverers.urlopen", return_value=response) as open_url, \
                patch("kripa.model_updates.discoverers._ollama_registry_digest", return_value=("b" * 64, "")):
            snapshot = discover_ollama(make_settings())
        open_url.assert_called_once()
        self.assertTrue(snapshot.installed_models[0]["update_available"])
        self.assertEqual(snapshot.installed_models[0]["status"], "update available")
        self.assertIn("Q4_K_M", snapshot.records[0].compatibility)

    def test_ollama_registry_manifest_digest_uses_metadata_body_only(self):
        manifest = b'{"schemaVersion":2,"layers":[]}'
        response = MagicMock()
        response.headers = {}
        response.read.return_value = manifest
        with patch("kripa.model_updates.discoverers._open_url", return_value=response) as open_url:
            digest, note = _ollama_registry_digest("llama3.2:3b")
        self.assertEqual(digest, hashlib.sha256(manifest).hexdigest())
        self.assertFalse(note)
        self.assertIn("/v2/library/llama3.2/manifests/3b", open_url.call_args.args[0])

    def test_cli_routes_model_report_without_changing_chat_default(self):
        with patch("kripa.model_updates.cli.run_report", return_value=0) as run_report:
            self.assertEqual(main(["models", "report"]), 0)
        run_report.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
