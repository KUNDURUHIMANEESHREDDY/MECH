"""Comparative Science and Cross-Model Alignment Package."""

from .cross_model_alignment import (
    CrossModelUniversalityEngine,
    CrossModelUniversalityReport,
    ModelArchitectureSpec,
    ModelAlignmentResult,
    NormalizedCircuitNode,
    FunctionalRole,
    STANDARD_MODEL_ZOO,
)
from .functional_signatures import (
    FunctionalCrossModelEngine,
    FunctionalComponentMatch,
    FunctionalEmbedding,
    ActivationSignature,
    AttentionSignature,
    CausalSignature,
)
from .heldout_cross_model_validator import (
    IndependentCrossModelValidator,
    IndependentUniversalityVerification,
    HeldoutProbeResult,
    CrossModelCausalInterchangeResult,
    HeldoutTaskType,
)

__all__ = [
    "CrossModelUniversalityEngine",
    "CrossModelUniversalityReport",
    "ModelArchitectureSpec",
    "ModelAlignmentResult",
    "NormalizedCircuitNode",
    "FunctionalRole",
    "STANDARD_MODEL_ZOO",
    "FunctionalCrossModelEngine",
    "FunctionalComponentMatch",
    "FunctionalEmbedding",
    "ActivationSignature",
    "AttentionSignature",
    "CausalSignature",
    "IndependentCrossModelValidator",
    "IndependentUniversalityVerification",
    "HeldoutProbeResult",
    "CrossModelCausalInterchangeResult",
    "HeldoutTaskType",
]


