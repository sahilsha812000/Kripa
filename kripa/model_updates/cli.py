"""Command entry point for read-only model discovery."""

from kripa.config import get_settings
from kripa.model_updates.manager import ModelUpdateManager
from kripa.model_updates.report import format_report


def run_report(settings=None, manager=None):
    """Generate and print a provider catalog report; performs no update action."""
    settings = settings or get_settings()
    report = (manager or ModelUpdateManager()).generate_report(settings)
    print(format_report(report))
    return 0
