"""Epic 8: Runtime API — typed data transfer objects for all endpoints."""

from pydantic import BaseModel
from typing import Any


# ── Model endpoints ──────────────────────────────────────────────

class ModelLoadRequest(BaseModel):
    model_name: str = "gpt2"


class ModelLoadResponse(BaseModel):
    model_name: str
    status: str
    num_layers: int
    num_heads: int
    hidden_dim: int


class ModelInfoResponse(BaseModel):
    model_name: str
    hf_id: str
    family: str
    loaded: bool
    n_layer: int | None = None
    n_head: int | None = None
    n_embd: int | None = None


# ── Inference endpoints ──────────────────────────────────────────

class InferenceRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 10
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 0.9
    session_id: str | None = None  # optional; creates new if absent


class AttentionMap(BaseModel):
    layer: int
    head: int
    tokens: list[str]
    matrix: list[list[float]]


class NeuronActivation(BaseModel):
    layer: int
    index: int
    activation: float


class TokenInfo(BaseModel):
    text: str
    id: int


class InferenceResponse(BaseModel):
    session_id: str
    model_name: str
    tokens: list[TokenInfo]
    generated_text: str
    attention_maps: list[AttentionMap]
    neuron_activations: list[NeuronActivation]
    profiling: dict[str, Any] = {}
    gpu_util: float = 0.0
    memory_util: float = 0.0
    cache_ids: list[str] = []


# ── Session endpoints ────────────────────────────────────────────

class SessionResponse(BaseModel):
    session_id: str
    model_name: str
    prompt: str
    generated_text: str
    created_at: float
    closed_at: float | None = None
    is_open: bool
    num_tokens: int = 0
    num_layers: int = 0


class SessionCreateResponse(BaseModel):
    session_id: str


# ── Hook endpoints ───────────────────────────────────────────────

class HookRegisterRequest(BaseModel):
    hook_type: str  # "attention", "mlp", "residual", "embedding", "logit", "custom"
    layer_idx: int
    name: str | None = None


class HookHandleResponse(BaseModel):
    id: str
    name: str
    enabled: bool
    layer_idx: int
    hook_type: str


# ── Performance endpoints ────────────────────────────────────────

class PerformanceReportResponse(BaseModel):
    samples: list[dict[str, Any]]
    total_duration_ms: float
    breakdown: dict[str, float]


# ── Cache endpoints ──────────────────────────────────────────────

class CacheQueryRequest(BaseModel):
    session_id: str | None = None
    layer: int | None = None
    component: str | None = None
    head: int | None = None
    prompt_id: str | None = None


class CacheEntryResponse(BaseModel):
    activation_id: str
    session_id: str
    prompt_id: str
    layer: int
    head: int | None = None
    component: str
    shape: list[int]
    timestamp: float
    metadata: dict[str, Any] = {}


# ── Error responses ──────────────────────────────────────────────

class ErrorResponse(BaseModel):
    error: str
    detail: str
    type: str
