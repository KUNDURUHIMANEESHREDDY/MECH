"""Backend API Contracts — Shared Pydantic models for request/response validation.

These models define the canonical wire format between backend and frontend.
They are the single source of truth for API shapes.

Version: 2.0.0 — Evidence-first contracts with provenance tracking.
"""

from __future__ import annotations

# Re-export all contract models
from .models import (  # noqa: F401
    # Core provenance
    ProvenanceLabel,
    EvidenceLevel,
    ProvenanceInfo,
    FieldProvenance,
    # Model & Engine
    ModelInfo,
    ModelLoadResponse,
    ModelStatusResponse,
    # Inference
    TokenInfo,
    AttentionMap,
    NeuronActivation,
    InferenceRequest,
    InferenceResponse,
    # GPT-2 specific
    GPT2ArchitectureResponse,
    GPT2LayerResponse,
    GPT2NeuronsResponse,
    GPT2NeuronDetailResponse,
    GPT2AttentionHeadResponse,
    GPT2HeadResponse,
    GPT2ActivationsResponse,
    GPT2FreshPromptResponse,
    GPT2PatchNeuronResponse,
    GPT2RunPromptResponse,
    GPT2PatchHeadResponse,
    GPT2IOIResponse,
    GPT2SteerResponse,
    GPT2LayerActivationsResponse,
    GPT2LogitLensAllResponse,
    SAEStatusResponse,
    SAEInspectResponse,
    SAETrainStartResponse,
    SAERunStatusResponse,
    # Experiments
    Experiment,
    ExperimentCreateRequest,
    ExperimentListResponse,
    Session,
    SessionCreateRequest,
    SessionListResponse,
    # Benchmarks
    BenchmarkListResponse,
    BenchmarkRunRequest,
    BenchmarkRunResponse,
    # Discovery
    DiscoveryListResponse,
    DiscoveryRunRequest,
    DiscoveryRunResponse,
    ValidationResult,
    # Runtime
    RuntimeEngineProbe,
    RuntimeStatusResponse,
    # Health
    HealthSubsystem,
    HealthSnapshotResponse,
    # Generic error/unavailable
    ErrorResponse,
    UnavailableResponse,
)

__all__ = [
    "ProvenanceLabel",
    "EvidenceLevel",
    "ProvenanceInfo",
    "FieldProvenance",
    "ModelInfo",
    "ModelLoadResponse",
    "ModelStatusResponse",
    "TokenInfo",
    "AttentionMap",
    "NeuronActivation",
    "InferenceRequest",
    "InferenceResponse",
    "GPT2ArchitectureResponse",
    "GPT2LayerResponse",
    "GPT2NeuronsResponse",
    "GPT2NeuronDetailResponse",
    "GPT2AttentionHeadResponse",
    "GPT2HeadResponse",
    "GPT2ActivationsResponse",
    "GPT2FreshPromptResponse",
    "GPT2PatchNeuronResponse",
    "GPT2RunPromptResponse",
    "GPT2PatchHeadResponse",
    "GPT2IOIResponse",
    "GPT2SteerResponse",
    "GPT2LayerActivationsResponse",
    "GPT2LogitLensAllResponse",
    "SAEStatusResponse",
    "SAEInspectResponse",
    "SAETrainStartResponse",
    "SAERunStatusResponse",
    "Experiment",
    "ExperimentCreateRequest",
    "ExperimentListResponse",
    "Session",
    "SessionCreateRequest",
    "SessionListResponse",
    "BenchmarkListResponse",
    "BenchmarkRunRequest",
    "BenchmarkRunResponse",
    "DiscoveryListResponse",
    "DiscoveryRunRequest",
    "DiscoveryRunResponse",
    "ValidationResult",
    "RuntimeEngineProbe",
    "RuntimeStatusResponse",
    "HealthSubsystem",
    "HealthSnapshotResponse",
    "ErrorResponse",
    "UnavailableResponse",
]