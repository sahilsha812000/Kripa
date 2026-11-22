"""Human-readable formatting for model update reports."""


def format_report(report):
    """Format a report as Markdown without printing credentials or SDK errors."""
    lines = [
        "# Kripa Model Update Report",
        "",
        "Read-only discovery. No models were installed and no settings were changed.",
        "",
        "| Provider | Current model | Latest available | Status | Recommended replacement | Compatibility |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in report.rows:
        values = (
            row.provider,
            row.current_model,
            row.latest_available_model,
            row.deprecation_status,
            row.recommended_replacement,
            row.compatibility,
        )
        lines.append("| {} |".format(" | ".join(_cell(value) for value in values)))
        if row.note:
            lines.append("  - {}: {}".format(_cell(row.provider), _cell(row.note)))
    if report.ollama_installed_models:
        lines.extend(("", "## Installed Ollama models", ""))
        lines.append("| Model | Installed | Update status | Local digest | Latest digest |")
        lines.append("| --- | --- | --- | --- | --- |")
        for model in report.ollama_installed_models:
            values = (
                model.get("model", "Unknown"),
                "yes" if model.get("installed") else "no",
                model.get("status", "unknown"),
                _short_digest(model.get("local_digest")),
                _short_digest(model.get("latest_digest")),
            )
            lines.append("| {} |".format(" | ".join(_cell(value) for value in values)))
            if model.get("note"):
                lines.append("  - {}: {}".format(_cell(model.get("model")), _cell(model["note"])))
    lines.extend((
        "",
        "Any model change or installation requires Nitin's approval. This report does not apply recommendations.",
    ))
    return "\n".join(lines)


def _cell(value):
    return str(value or "—").replace("|", "\\|").replace("\n", " ")


def _short_digest(value):
    return value[:12] if value else "—"
