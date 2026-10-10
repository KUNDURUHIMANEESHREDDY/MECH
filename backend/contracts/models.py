"""Canonical wire contracts for MECH backend <-> frontend.

Every response that can carry a measurement carries:
  status, provenance, field_provenance, attested
Every unavailable response carries:
  status in {"unavailable","error"}, provenance="unavailable", error|reason

Models are intentionally permissive (extra="allow") so adding a new
measured field never breaks validation — but renaming/removing a
contracted field or returning "ok" without provenance does.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

ProvenanceLabel = Literal["live", "seeded", "reference", "unavailable"]
EvidenceLevel = Literal[
    "OBSERVATIONAL",
    "ATTRIBUTIONAL",
    "INTERVENTIONAL",
    "CAUSALLY_VALIDATED",
    "REPLICATED",
]

StatusOk = Literal["ok", "loaded", "completed", "connected", "active", "created", "started"]
StatusNotExecuted = Literal["unavailable", "error", "not_found", "busy"]


class _AllowExtra(BaseModel):
    model_config = ConfigDict(extra="allow")

    @model_validator(mode="after")
    def _fail_closed_ok_requires_evidence(self):
        status = str(getattr(self, "status", "") or "").lower()
        prov = str(getattr(self, "provenance", "") or "").lower()
        if status in {"ok", "loaded", "completed", "connected"} and prov == "unavailable":
            has_cause = bool(
                getattr(self, "error", None)
                or getattr(self, "reason", None)
                or getattr(self, "provenance_note", None)
            )
            if not has_cause:
                raise ValueError(
                    f"status={status!r} with provenance='unavailable' and no "
                    "error/reason/provenance_note is indistinguishable from a "
                    "measurement — fail closed instead"
                )
        return self


class ProvenanceInfo(_AllowExtra):
    provenance: ProvenanceLabel = "unavailable"
    attested: bool = False
    reason: Optional[str] = None
    error: Optional[str] = None
    provenance_note: Optional[str] = None
    model_loaded: Optional[str] = None
    model_requested: Optional[str] = None
    model_mismatch: Optional[bool] = None
    measured: Optional[bool] = None
    evidence_level: Optional[EvidenceLevel] = None
    validation_eligible: Optional[bool] = None
    publication_eligible: Optional[bool] = None


FieldProvenance = Dict[str, ProvenanceLabel]


# ── Model & engine ──


class ModelInfo(_AllowExtra):
    model_name: str
    layers: Optional[int] = None
    hidden_size: Optional[int] = None
    vocab_size: Optional[int] = None
    num_heads: Optional[int] = None
    provenance: ProvenanceLabel = "reference"
    field_provenance: FieldProvenance = Field(default_factory=dict)


class ModelLoadResponse(_AllowExtra):
    status: str
    model_name: Optional[str] = None
    n_layers: Optional[int] = None
    n_heads: Optional[int] = None
    d_model: Optional[int] = None
    d_mlp: Optional[int] = None
    device: Optional[str] = None
    provenance: ProvenanceLabel = "unavailable"
    field_provenance: FieldProvenance = Field(default_factory=dict)
    reason: Optional[str] = None
    error: Optional[str] = None
    attested: bool = False


class ModelStatusResponse(_AllowExtra):
    status: str
    platform: Optional[str] = None
    version: Optional[str] = None
    modules: Optional[List[str]] = None


# ── Inference ──


class TokenInfo(_AllowExtra):
    text: str
    id: int


class AttentionMap(_AllowExtra):
    layer: int
    head: int
    tokens: List[str] = Field(default_factory=list)
    matrix: List[List[float]] = Field(default_factory=list)


class NeuronActivation(_AllowExtra):
    layer: int
    index: int
    activation: float
    token_activations: Optional[List[float]] = None


class InferenceRequest(_AllowExtra):
    prompt: Optional[str] = None
    model_name: Optional[str] = "gpt2-small"


class InferenceResponse(_AllowExtra):
    status: Optional[str] = None
    model_name: Optional[str] = None
    tokens: Optional[List[TokenInfo]] = None
    generated_text: Optional[str] = None
    attention_maps: Optional[List[AttentionMap]] = None
    neuron_activations: Optional[List[NeuronActivation]] = None
    gpu_util: Optional[float] = None
    memory_util: Optional[float] = None
    provenance: ProvenanceLabel = "unavailable"
    field_provenance: FieldProvenance = Field(default_factory=dict)
    provenance_note: Optional[str] = None
    error: Optional[str] = None
    reason: Optional[str] = None
    attested: bool = False


# ── GPT-2 specific ──


class GPT2ArchitectureResponse(_AllowExtra):
    status: str
    model_name: Optional[str] = None
    n_layers: Optional[int] = None
    n_heads: Optional[int] = None
    d_model: Optional[int] = None
    d_mlp: Optional[int] = None
    device: Optional[str] = None
    provenance: ProvenanceLabel = "unavailable"
    field_provenance: FieldProvenance = Field(default_factory=dict)
    error: Optional[str] = None
    reason: Optional[str] = None
    attested: bool = False


class GPT2LayerResponse(_AllowExtra):
    status: str
    layer: Optional[int] = None
    provenance: ProvenanceLabel = "unavailable"
    field_provenance: FieldProvenance = Field(default_factory=dict)
    error: Optional[str] = None
    attested: bool = False


class GPT2NeuronsResponse(_AllowExtra):
    status: str
    layer: Optional[int] = None
    neurons: Optional[List[Dict[str, Any]]] = None
    provenance: ProvenanceLabel = "unavailable"
    field_provenance: FieldProvenance = Field(default_factory=dict)
    error: Optional[str] = None
    attested: bool = False


class GPT2NeuronDetailResponse(_AllowExtra):
    status: str
    layer: Optional[int] = None
    neuron_index: Optional[int] = None
    provenance: ProvenanceLabel = "unavailable"
    field_provenance: FieldProvenance = Field(default_factory=dict)
    error: Optional[str] = None
    attested: bool = False


class GPT2AttentionHeadResponse(_AllowExtra):
    status: str
    layer: Optional[int] = None
    head: Optional[int] = None
    matrix: Optional[List[List[float]]] = None
    str_tokens: Optional[List[str]] = None
    provenance: ProvenanceLabel = "unavailable"
    field_provenance: FieldProvenance = Field(default_factory=dict)
    error: Optional[str] = None
    attested: bool = False


class GPT2RunPromptResponse(_AllowExtra):
    status: str
    prompt: Optional[str] = None
    str_tokens: Optional[List[str]] = None
    provenance: ProvenanceLabel = "unavailable"
    field_provenance: FieldProvenance = Field(default_factory=dict)
    error: Optional[str] = None
    attested: bool = False


class GPT2PatchHeadResponse(_AllowExtra):
    status: str
    layer: Optional[int] = None
    head: Optional[int] = None
    clean_ld: Optional[float] = None
    patched_ld: Optional[float] = None
    delta: Optional[float] = None
    direction: Optional[str] = None
    evidence_level: Optional[EvidenceLevel] = None
    provenance: ProvenanceLabel = "unavailable"
    field_provenance: FieldProvenance = Field(default_factory=dict)
    error: Optional[str] = None
    attested: bool = False


class GPT2IOIResponse(_AllowExtra):
    status: str
    io_name: Optional[str] = None
    subj_name: Optional[str] = None
    io_name_defaulted: Optional[bool] = None
    clean_prompt: Optional[str] = None
    corrupted_prompt: Optional[str] = None
    clean_top1: Optional[str] = None
    corrupted_top1: Optional[str] = None
    clean_ld: Optional[float] = None
    corrupted_ld: Optional[float] = None
    provenance: ProvenanceLabel = "unavailable"
    field_provenance: FieldProvenance = Field(default_factory=dict)
    error: Optional[str] = None
    attested: bool = False


class GPT2SteerResponse(_AllowExtra):
    status: str
    provenance: ProvenanceLabel = "unavailable"
    field_provenance: FieldProvenance = Field(default_factory=dict)
    error: Optional[str] = None
    attested: bool = False


class GPT2LayerActivationsResponse(_AllowExtra):
    status: str
    layer: Optional[int] = None
    prompt: Optional[str] = None
    tokens: Optional[List[str]] = None
    provenance: ProvenanceLabel = "unavailable"
    field_provenance: FieldProvenance = Field(default_factory=dict)
    error: Optional[str] = None
    attested: bool = False


class GPT2ActivationsResponse(_AllowExtra):
    status: str
    layer: Optional[int] = None
    resid_shape: Optional[List[int]] = None
    attn_shape: Optional[List[int]] = None
    mlp_shape: Optional[List[int]] = None
    provenance: ProvenanceLabel = "unavailable"
    field_provenance: FieldProvenance = Field(default_factory=dict)
    error: Optional[str] = None
    attested: bool = False


class GPT2FreshPromptResponse(_AllowExtra):
    status: str
    prompt: Optional[str] = None
    primer: Optional[str] = None
    method: Optional[str] = None
    tokens_added: Optional[int] = None
    provenance: ProvenanceLabel = "unavailable"
    field_provenance: FieldProvenance = Field(default_factory=dict)
    error: Optional[str] = None
    attested: bool = False


class GPT2HeadResponse(_AllowExtra):
    status: str
    layer: Optional[int] = None
    head: Optional[int] = None
    provenance: ProvenanceLabel = "unavailable"
    field_provenance: FieldProvenance = Field(default_factory=dict)
    error: Optional[str] = None
    attested: bool = False


class GPT2PatchNeuronResponse(_AllowExtra):
    status: str
    layer: Optional[int] = None
    neuron_index: Optional[int] = None
    evidence_level: Optional[EvidenceLevel] = None
    provenance: ProvenanceLabel = "unavailable"
    field_provenance: FieldProvenance = Field(default_factory=dict)
    error: Optional[str] = None
    attested: bool = False


class GPT2LogitLensAllResponse(_AllowExtra):
    status: str
    method: Optional[str] = None
    prompt: Optional[str] = None
    layers: Optional[List[Dict[str, Any]]] = None
    provenance: ProvenanceLabel = "unavailable"
    field_provenance: FieldProvenance = Field(default_factory=dict)
    error: Optional[str] = None
    attested: bool = False


# ── SAE ──


class SAEStatusResponse(_AllowExtra):
    status: str
    torch_available: Optional[bool] = None
    gpt2_available: Optional[bool] = None
    pipeline_importable: Optional[bool] = None
    checkpoint_configured: Optional[bool] = None
    checkpoint_source: Optional[str] = None
    training_running: Optional[bool] = None
    provenance: ProvenanceLabel = "unavailable"
    field_provenance: FieldProvenance = Field(default_factory=dict)
    reason: Optional[str] = None
    error: Optional[str] = None
    attested: bool = False


class SAEInspectResponse(_AllowExtra):
    status: str
    feature_indices: Optional[List[int]] = None
    activations: Optional[List[float]] = None
    prompt: Optional[str] = None
    layer: Optional[int] = None
    tensor_source: Optional[str] = None
    d_in: Optional[int] = None
    d_sae: Optional[int] = None
    weights_sha256: Optional[str] = None
    reconstruction_measured: Optional[bool] = None
    provenance: ProvenanceLabel = "unavailable"
    field_provenance: FieldProvenance = Field(default_factory=dict)
    reason: Optional[str] = None
    error: Optional[str] = None
    attested: bool = False


class SAETrainStartResponse(_AllowExtra):
    status: str
    run_id: Optional[str] = None
    poll: Optional[str] = None
    provenance: ProvenanceLabel = "unavailable"
    field_provenance: FieldProvenance = Field(default_factory=dict)
    reason: Optional[str] = None
    error: Optional[str] = None


class SAERunStatusResponse(_AllowExtra):
    status: str
    run_id: Optional[str] = None
    phase: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    provenance: ProvenanceLabel = "unavailable"
    field_provenance: FieldProvenance = Field(default_factory=dict)
    reason: Optional[str] = None
    error: Optional[str] = None


# ── Experiments / sessions ──


class Experiment(_AllowExtra):
    id: Optional[str] = None
    name: Optional[str] = None
    experiment_type: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None
    results: Optional[Dict[str, Any]] = None
    provenance: Optional[ProvenanceLabel] = None
    measured: Optional[bool] = None
    timestamp: Optional[str] = None


class ExperimentCreateRequest(_AllowExtra):
    name: Optional[str] = None
    experiment_type: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None


class ExperimentListResponse(_AllowExtra):
    experiments: List[Dict[str, Any]] = Field(default_factory=list)


class Session(_AllowExtra):
    id: Optional[str] = None


class SessionCreateRequest(_AllowExtra):
    name: Optional[str] = None


class SessionListResponse(_AllowExtra):
    sessions: List[Dict[str, Any]] = Field(default_factory=list)


# ── Benchmarks ──


class BenchmarkListResponse(_AllowExtra):
    benchmarks: List[str] = Field(default_factory=list)
    provenance: ProvenanceLabel = "reference"
    field_provenance: FieldProvenance = Field(default_factory=dict)


class BenchmarkRunRequest(_AllowExtra):
    benchmark_name: str = "IOI"


class BenchmarkRunResponse(_AllowExtra):
    status: str
    benchmark_name: Optional[str] = None
    score: Optional[float] = None
    pass_rate: Optional[float] = None
    eval_samples: Optional[Any] = None
    provenance: ProvenanceLabel = "unavailable"
    field_provenance: FieldProvenance = Field(default_factory=dict)
    reason: Optional[str] = None
    error: Optional[str] = None


# ── Discovery / validation ──


class DiscoveryListResponse(_AllowExtra):
    discoveries: List[str] = Field(default_factory=list)
    provenance: ProvenanceLabel = "reference"
    field_provenance: FieldProvenance = Field(default_factory=dict)


class DiscoveryRunRequest(_AllowExtra):
    hypothesis_statement: Optional[str] = ""
    n_prompts: Optional[int] = 4
    max_heads: Optional[int] = 10


class DiscoveryRunResponse(_AllowExtra):
    status: str
    discovery_id: Optional[str] = None
    provenance: ProvenanceLabel = "unavailable"
    field_provenance: FieldProvenance = Field(default_factory=dict)
    measured: Optional[bool] = None
    validation_eligible: Optional[bool] = None
    publication_eligible: Optional[bool] = None
    evidence_level: Optional[EvidenceLevel] = None
    reason: Optional[str] = None
    error: Optional[str] = None


class ValidationResult(_AllowExtra):
    status: str
    discovery_id: Optional[str] = None
    validated: Optional[bool] = None
    provenance: ProvenanceLabel = "unavailable"
    field_provenance: FieldProvenance = Field(default_factory=dict)
    validation_eligible: Optional[bool] = None
    publication_eligible: Optional[bool] = None
    reason: Optional[str] = None
    error: Optional[str] = None


# ── Runtime / health ──


class RuntimeEngineProbe(_AllowExtra):
    reachable: bool = False
    detail: Optional[str] = None


class RuntimeStatusResponse(_AllowExtra):
    status: str
    engines: Optional[List[str]] = None
    engine_detail: Optional[Dict[str, RuntimeEngineProbe]] = None
    reason: Optional[str] = None
    provenance: ProvenanceLabel = "live"
    validation_eligible: bool = False
    publication_eligible: bool = False


class HealthSubsystem(_AllowExtra):
    name: Optional[str] = None
    status: Optional[str] = None
    detail: Optional[str] = None


class HealthSnapshotResponse(_AllowExtra):
    status: str
    subsystems: Optional[Any] = None


# ── Generic ──


class ErrorResponse(_AllowExtra):
    status: str = "error"
    error: Optional[str] = None
    provenance: ProvenanceLabel = "unavailable"
    field_provenance: FieldProvenance = Field(default_factory=dict)


class UnavailableResponse(_AllowExtra):
    status: str = "unavailable"
    provenance: ProvenanceLabel = "unavailable"
    field_provenance: FieldProvenance = Field(default_factory=dict)
    reason: Optional[str] = None
    error: Optional[str] = None
    attested: bool = False
