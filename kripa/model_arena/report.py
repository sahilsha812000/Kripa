"""Human-readable accuracy, speed, consistency, cost, and run details."""


TASK_LABELS = {
    "general_conversation": "General conversation",
    "reasoning": "Reasoning",
    "coding": "Coding",
    "writing": "Writing",
    "research": "Research",
    "summarization": "Summarization",
    "instruction_following": "Instruction following",
}


def _percent(value):
    return "N/A" if value is None else "{:.0f}%".format(value * 100)


def _seconds(value):
    return "N/A" if value is None else "{:.3f}".format(value)


def _tokens(value):
    return "N/A" if value is None else str(value)


def _cost(value):
    return "N/A" if value is None else "${:.6f}".format(value)


def format_report(report):
    lines = [
        "# Kripa Model Arena v2",
        "",
        "{} runs per task and provider. Every run used the same user prompt; each provider was called directly without adaptive fallback.".format(report.runs_per_task),
        "Accuracy, speed, consistency, and provider-reported usage/cost are shown separately. Failures remain in the results below.",
        "No active provider or model settings were changed.",
        "",
        "## Per-task comparison",
        "",
        "| Task | Provider | Model | Accuracy | Success / failures | Average s | Median s | Exact-response consistency | Usage runs | Input tokens | Output tokens | Provider-reported cost (USD) |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in report.summaries:
        accuracy = "Manual review" if item.accuracy is None and item.task_id == "writing" else _percent(item.accuracy)
        lines.append("| {} | {} | {} | {} | {} / {} | {} | {} | {} | {} | {} | {} |".format(
            TASK_LABELS.get(item.task_id, item.task_id), item.provider, item.model,
            accuracy, item.successes, item.failures, _seconds(item.average_seconds),
            _seconds(item.median_seconds), _percent(item.exact_response_consistency),
            "{}/{}".format(item.usage_reports, item.attempts),
            _tokens(item.input_tokens), _tokens(item.output_tokens), _cost(item.provider_cost_usd),
        ))

    lines.extend(["", "## Accuracy leaderboard", ""])
    for task_id, ranked in report.leaderboard.items():
        if task_id == "writing":
            lines.append("- Writing: manual review; no automated quality score or ranking.")
        elif not ranked:
            lines.append("- {}: no successful evaluated responses.".format(TASK_LABELS.get(task_id, task_id)))
        else:
            winners = [item for item in ranked if item.accuracy == ranked[0].accuracy]
            lines.append("- {}: {} ({} accuracy{}).".format(
                TASK_LABELS.get(task_id, task_id),
                ", ".join("{} / {}".format(item.provider, item.model) for item in winners),
                _percent(ranked[0].accuracy),
                " · median {:.3f}s".format(ranked[0].median_seconds)
                if ranked[0].median_seconds is not None else "",
            ))

    lines.extend(["", "## Individual runs", ""])
    for result in report.results:
        lines.append("### {} — {} ({}, run {})".format(
            TASK_LABELS.get(result.task_id, result.task_id), result.provider,
            result.model, result.run_number,
        ))
        if result.error:
            lines.append("FAILED: {}".format(result.error))
            if result.elapsed_seconds is not None:
                lines.append("Elapsed: {:.3f} s".format(result.elapsed_seconds))
        else:
            accuracy = (
                "Manual review" if result.correct is None
                else ("PASS" if result.correct else "FAIL")
            )
            lines.append("Check: {} · Time: {:.3f} s · Input/output tokens: {}/{} · Provider-reported cost: {}".format(
                accuracy,
                result.elapsed_seconds,
                _tokens(result.input_tokens),
                _tokens(result.output_tokens),
                _cost(result.provider_cost_usd),
            ))
            lines.append("")
            lines.append(result.response)
        lines.append("")

    lines.extend([
        "## Metric notes",
        "",
        "Accuracy is the share of successful responses passing the task's fixed answer check or safe coding test cases; request failures are counted separately. Writing responses are provided for manual review and are not scored. Speed uses successful requests only and reports arithmetic mean and median. Exact-response consistency is the most frequent identical response count divided by successful runs; it measures repeatability, not quality. Token totals and cost appear only when the provider SDK returns explicit usage data. No prices are inferred and no cost is estimated from token counts.",
    ])
    return "\n".join(lines)
