"""Read-only model catalog discovery and update reporting."""

from kripa.model_updates.manager import ModelUpdateManager
from kripa.model_updates.report import format_report

__all__ = ["ModelUpdateManager", "format_report"]
