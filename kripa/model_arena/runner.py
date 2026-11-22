"""Provider orchestration and measured Model Arena results."""

from dataclasses import dataclass
from math import isfinite
from statistics import mean, median
from time import perf_counter
from typing import Dict, List, Optional

from kripa.engine.conversation import Message
from kripa.engine.errors import ProviderError
from kripa.engine.providers.gemini_provider import GeminiProvider
from kripa.engine.providers.groq_provider import GroqProvider
from kripa.engine.providers.ollama_provider import OllamaProvider
from kripa.engine.providers.openrouter_provider import OpenRouterProvider
from kripa.model_arena.quality import check_response
from kripa.model_arena.tasks import TASKS


PROVIDER_NAMES = ("gemini", "groq", "openrouter", "ollama")
MODEL_SETTING_NAMES = {
    "gemini": "gemini_model",
    "groq": "groq_model",
    "openrouter": "openrouter_model",
    "ollama": "ollama_model",
}


@dataclass(frozen=True)
class ArenaResult:
    task_id: str
    provider: str
    model: str
    run_number: int
    response: str = ""
    elapsed_seconds: Optional[float] = None
    correct: Optional[bool] = None
    error: str = ""
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    provider_cost_usd: Optional[float] = None


@dataclass(frozen=True)
class ArenaSummary:
    task_id: str
    provider: str
    model: str
    attempts: int
    successes: int
    failures: int
    accuracy: Optional[float]
    average_seconds: Optional[float]
    median_seconds: Optional[float]
    exact_response_consistency: Optional[float]
    input_tokens: Optional[int]
    output_tokens: Optional[int]
    provider_cost_usd: Optional[float]
    usage_reports: int


@dataclass(frozen=True)
class ArenaReport:
    results: List[ArenaResult]
    summaries: List[ArenaSummary]
    leaderboard: Dict[str, List[ArenaSummary]]
    runs_per_task: int


def build_provider(settings, provider_name):
    """Build a single adapter, deliberately bypassing adaptive fallback."""
    if provider_name == "gemini":
        return GeminiProvider(settings.gemini_api_key, settings.gemini_model)
    if provider_name == "groq":
        return GroqProvider(settings.groq_api_key, settings.groq_model)
    if provider_name == "openrouter":
        return OpenRouterProvider(settings.openrouter_api_key, settings.openrouter_model)
    if provider_name == "ollama":
        return OllamaProvider(settings.ollama_base_url, settings.ollama_model)
    raise ValueError("Unsupported arena provider: {}".format(provider_name))


class ModelArena:
    """Send identical fixed tasks repeatedly to each selected provider."""

    def __init__(self, provider_names=None, provider_factory=None, tasks=None, runs_per_task=3):
        self.provider_names = tuple(provider_names or PROVIDER_NAMES)
        unknown = set(self.provider_names).difference(PROVIDER_NAMES)
        if unknown:
            raise ValueError("Unsupported arena provider(s): {}".format(", ".join(sorted(unknown))))
        if len(set(self.provider_names)) != len(self.provider_names):
            raise ValueError("Choose each arena provider at most once.")
        if not isinstance(runs_per_task, int) or isinstance(runs_per_task, bool) or runs_per_task < 1:
            raise ValueError("runs_per_task must be a positive integer.")
        self.provider_factory = provider_factory or build_provider
        self.tasks = tuple(tasks or TASKS)
        self.runs_per_task = runs_per_task

    def run(self, settings):
        results = []
        for name in self.provider_names:
            try:
                provider = self.provider_factory(settings, name)
                model = getattr(provider, "model", "unknown")
            except Exception as error:
                safe_message = str(error) if isinstance(error, ProviderError) else "Provider could not be initialized."
                model = getattr(settings, MODEL_SETTING_NAMES[name], "unknown")
                for task in self.tasks:
                    for run_number in range(1, self.runs_per_task + 1):
                        results.append(ArenaResult(
                            task.task_id, name, model, run_number, error=safe_message,
                        ))
                continue

            for task in self.tasks:
                for run_number in range(1, self.runs_per_task + 1):
                    started = perf_counter()
                    try:
                        # A fresh one-turn message list ensures prompt parity on every run.
                        if hasattr(provider, "previous_interaction_id"):
                            provider.previous_interaction_id = None
                        answer = provider.reply([Message(role="user", content=task.prompt)])
                        elapsed = perf_counter() - started
                        if not isinstance(answer, str) or not answer.strip():
                            raise ProviderError("Provider returned an empty response.")
                        usage = getattr(provider, "last_usage", None)
                        cost = getattr(provider, "last_provider_cost_usd", None)
                        results.append(ArenaResult(
                            task_id=task.task_id,
                            provider=name,
                            model=model,
                            run_number=run_number,
                            response=answer,
                            elapsed_seconds=elapsed,
                            correct=check_response(task.task_id, answer, task.expected),
                            input_tokens=_usage_count(usage, "input_tokens"),
                            output_tokens=_usage_count(usage, "output_tokens"),
                            provider_cost_usd=_valid_cost(cost),
                        ))
                    except ProviderError as error:
                        results.append(ArenaResult(
                            task.task_id, name, model, run_number,
                            elapsed_seconds=perf_counter() - started,
                            error=str(error),
                        ))
                    except Exception:
                        results.append(ArenaResult(
                            task.task_id, name, model, run_number,
                            elapsed_seconds=perf_counter() - started,
                            error="Provider request failed.",
                        ))

        summaries = _summarize(results, self.tasks, self.provider_names, self.runs_per_task)
        return ArenaReport(
            results=results,
            summaries=summaries,
            leaderboard=_make_leaderboard(summaries, self.tasks),
            runs_per_task=self.runs_per_task,
        )


def _usage_count(usage, name):
    if usage is None:
        return None
    value = usage.get(name) if isinstance(usage, dict) else getattr(usage, name, None)
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    return None


def _valid_cost(value):
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        number = float(value)
        if number >= 0 and isfinite(number):
            return number
    return None


def _summarize(results, tasks, provider_names, runs_per_task):
    summaries = []
    for provider_name in provider_names:
        model_names = sorted(set(
            result.model for result in results if result.provider == provider_name
        )) or ["unknown"]
        for task in tasks:
            for model_name in model_names:
                group = [
                    result for result in results
                    if result.task_id == task.task_id
                    and result.provider == provider_name
                    and result.model == model_name
                ]
                successes = [result for result in group if not result.error]
                failures = [result for result in group if result.error]
                times = [result.elapsed_seconds for result in successes if result.elapsed_seconds is not None]
                evaluated = [result.correct for result in successes if result.correct is not None]
                exact_matches = {}
                for result in successes:
                    exact_matches[result.response] = exact_matches.get(result.response, 0) + 1
                consistency = (
                    float(max(exact_matches.values())) / len(successes) if successes else None
                )
                usage_items = [
                    result for result in successes
                    if result.input_tokens is not None or result.output_tokens is not None
                    or result.provider_cost_usd is not None
                ]
                summaries.append(ArenaSummary(
                    task_id=task.task_id,
                    provider=provider_name,
                    model=model_name,
                    attempts=len(group) if group else runs_per_task,
                    successes=len(successes),
                    failures=len(failures),
                    accuracy=(float(sum(evaluated)) / len(evaluated) if evaluated else None),
                    average_seconds=(mean(times) if times else None),
                    median_seconds=(median(times) if times else None),
                    exact_response_consistency=consistency,
                    input_tokens=(sum(item.input_tokens or 0 for item in usage_items)
                                  if any(item.input_tokens is not None for item in usage_items) else None),
                    output_tokens=(sum(item.output_tokens or 0 for item in usage_items)
                                   if any(item.output_tokens is not None for item in usage_items) else None),
                    provider_cost_usd=(sum(item.provider_cost_usd or 0.0 for item in usage_items)
                                       if any(item.provider_cost_usd is not None for item in usage_items) else None),
                    usage_reports=len(usage_items),
                ))
    return summaries


def _make_leaderboard(summaries, tasks):
    board = {}
    for task in tasks:
        ranked = [
            item for item in summaries
            if item.task_id == task.task_id and item.accuracy is not None
        ]
        board[task.task_id] = sorted(
            ranked,
            key=lambda item: (
                -item.accuracy,
                item.median_seconds if item.median_seconds is not None else float("inf"),
            ),
        )
    return board
