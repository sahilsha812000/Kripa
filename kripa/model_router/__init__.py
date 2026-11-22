"""Typed contracts for a future task-aware model router.

This package defines routing data only; it does not select or invoke providers.
"""

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
from kripa.model_router.classification import (
    KeywordTaskClassifier,
    TaskClassification,
    TaskClassifier,
)
from kripa.model_router.engine import (
    CandidateAssessment,
    DecisionConfidence,
    RoutingDecision,
    RoutingEngine,
)
from kripa.model_router.registry import build_configured_registry
from kripa.model_router.shadow import ShadowModelRouter, build_shadow_router

__all__ = [
    "Availability",
    "BenchmarkResult",
    "CandidateAssessment",
    "DecisionConfidence",
    "KeywordTaskClassifier",
    "ModelCandidate",
    "ModelCandidateRegistry",
    "ModelProvider",
    "RoutingPriority",
    "RoutingRequirements",
    "RoutingRequest",
    "RoutingDecision",
    "RoutingEngine",
    "ShadowModelRouter",
    "TaskClassification",
    "TaskClassifier",
    "TaskCategory",
    "build_configured_registry",
    "build_shadow_router",
]
