"""Data types shared by model catalog discoverers and report generation."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class ModelRecord:
    """A provider's catalog description for one model identifier."""

    model_id: str
    created_at: Any = None
    active: Optional[bool] = None
    deprecated: Optional[bool] = None
    expires_at: Optional[str] = None
    compatibility: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CatalogSnapshot:
    """Available models or a safe explanation for why discovery did not run."""

    provider: str
    records: List[ModelRecord] = field(default_factory=list)
    note: str = ""
    error: str = ""
    installed_models: List[Dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class ModelUpdateRow:
    """Read-only comparison between one configured model and its catalog."""

    provider: str
    current_model: str
    latest_available_model: str
    deprecation_status: str
    recommended_replacement: str
    compatibility: str
    note: str = ""


@dataclass(frozen=True)
class ModelUpdateReport:
    """A point-in-time report; it contains no update actions."""

    rows: List[ModelUpdateRow]
    ollama_installed_models: List[Dict[str, Any]] = field(default_factory=list)
