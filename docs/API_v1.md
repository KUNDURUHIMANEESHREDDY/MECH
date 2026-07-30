# Runtime API v1 — Formal Contract

> **Status:** FROZEN — no breaking changes without Architect approval.
> **Version:** 1.0.0
> **Base URL:** `http://localhost:8000/api/v1`
>
> Legacy root aliases (e.g. `/models`, `/inference`) remain for backward
> compatibility but new consumers should use the versioned path.

---

# Table of Contents

1. [Endpoints](#1-endpoints)
2. [DTO Schemas](#2-dto-schemas)
3. [Events](#3-events)
4. [Hooks](#4-hooks)
5. [Capabilities](#5-capabilities)
6. [Error Codes](#6-error-codes)
7. [Session Schema](#7-session-schema)
8. [Versioning Policy](#8-versioning-policy)

---

# 1. Endpoints

## 1.1 Health

```
GET /api/v1/health
```

Returns system health and resource metrics.

**Response (200):**
```json
{
  "status": "ok",
  "timestamp": 1715000000.0,
  "system": {
    "platform": "Windows-10-10.0.19041",
    "python": "3.11.5",
    "pid": 12345
  },
  "cpu": {
    "cores": 16,
    "physical_cores": 8,
    "percent": 12.5,
    "process_percent": 2.1
  },
  "ram": {
    "total_mb": 32768.0,
    "available_mb": 12288.0,
    "used_mb": 20480.0,
    "percent": 62.5,
    "process_mb": 1234.5
  },
  "pytorch": {
    "version": "2.5.1",
    "cuda_available": true,
    "cuda_version": "12.1"
  },
  "gpu": {
    "available": true,
    "name": "NVIDIA RTX 4090",
    "total_mb": 24576.0,
    "allocated_mb": 4096.0,
    "cached_mb": 8192.0,
    "utilization_percent": 15.0
  },
  "runtime": {
    "model_loaded": true,
    "model_name": "gpt2",
    "cache_entries": 42,
    "cache_memory_estimate_mb": 21.0,
    "active_sessions": 3,
    "loaded_hooks": 6,
    "events_logged": 1024,
    "events_per_sec": 5.2
  }
}
```

---

## 1.2 Capabilities

```
GET /api/v1/capabilities
```

Describes what the runtime supports. Frontend uses this to adapt UI.

**Response (200):**
```json
{
  "version": "2.0.0",
  "models": ["gpt2", "gpt2-medium", "distilgpt2", "gemma-2b", "llama-7b", "pythia-1b", "mistral-7b", "qwen-1.5b"],
  "hooks": ["embedding", "attention", "mlp", "residual", "logit", "custom"],
  "supports_streaming": true,
  "supports_sessions": true,
  "supports_attention": true,
  "supports_residuals": true,
  "supports_mlp": true,
  "supports_embeddings": true,
  "supports_logits": true,
  "supports_patching": false,
  "supports_sae": false,
  "supports_circuits": false,
  "supports_gradients": false,
  "supports_client_auth": true,
  "supports_persistence": true,
  "supports_event_replay": true,
  "supports_hook_dependencies": true
}
```

---

## 1.3 Models

### List models

```
GET /api/v1/models
```

**Response (200):**
```json
{
  "models": ["gpt2", "gpt2-medium", "distilgpt2", "gemma-2b", "llama-7b", "pythia-1b", "mistral-7b", "qwen-1.5b"]
}
```

### Model info

```
GET /api/v1/models/{name}
```

**Response (200):**
```json
{
  "model_name": "gpt2",
  "hf_id": "gpt2",
  "family": "gpt2",
  "architecture": "decoder",
  "hidden_size": 768,
  "num_layers": 12,
  "num_heads": 12,
  "vocab_size": 50257,
  "max_position": 1024,
  "activation_function": "gelu_new",
  "description": "GPT-2 small (124M parameters)",
  "supports_attention": true,
  "supports_residuals": true,
  "supports_mlp": true,
  "loaded": true
}
```

**Response (404):**
```json
{"detail": "Model 'unknown' not found"}
```

### Load model

```
POST /api/v1/models/load
```

**Request:**
```json
{
  "model_name": "gpt2"
}
```

**Response (200):**
```json
{
  "model_name": "gpt2",
  "status": "loaded",
  "num_layers": 12,
  "num_heads": 12,
  "hidden_dim": 768
}
```

**Response (400):**
```json
{
  "error": "ModelLoadError",
  "detail": "Failed to load model 'gpt2': Connection error"
}
```

### Unload model

```
POST /api/v1/models/unload
```

**Response (200):**
```json
{
  "status": "unloaded"
}
```

---

## 1.4 Sessions

### Create session

```
POST /api/v1/sessions?model_name=gpt2&prompt=Hello
```

**Headers:** `Client-Id: my-client` (optional)

**Response (200):**
```json
{
  "session_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890"
}
```

### List sessions

```
GET /api/v1/sessions
```

**Response (200):**
```json
[
  {
    "session_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
    "model_name": "gpt2",
    "prompt": "Hello world",
    "generated_text": "Hello world, it's a beautiful day!",
    "created_at": 1715000000.0,
    "closed_at": null,
    "is_open": true,
    "num_tokens": 2,
    "num_layers": 12
  }
]
```

### Get session

```
GET /api/v1/sessions/{session_id}
```

**Response (200):** Same shape as list item.

**Response (404):**
```json
{"detail": "Session not found"}
```

### Close session

```
POST /api/v1/sessions/{session_id}/close
```

**Response (200):**
```json
{
  "status": "closed",
  "session_id": "a1b2c3d4-..."
}
```

---

## 1.5 Inference

```
POST /api/v1/inference
```

**Headers:** `Client-Id: model-explorer-ui` (optional)

**Request:**
```json
{
  "prompt": "Hello world",
  "max_new_tokens": 10,
  "session_id": "a1b2c3d4-..." (optional; omitted = auto-create)
}
```

**Response (200):**
```json
{
  "session_id": "a1b2c3d4-...",
  "model_name": "gpt2",
  "tokens": [
    {"text": "Hello", "id": 15496},
    {"text": " world", "id": 995}
  ],
  "generated_text": "Hello world, it's a beautiful day!",
  "attention_maps": [
    {
      "layer": 0,
      "head": 0,
      "tokens": ["Hello", " world"],
      "matrix": [[0.8, 0.2], [0.3, 0.7]]
    }
  ],
  "neuron_activations": [
    {"layer": 0, "index": 0, "activation": 0.95}
  ],
  "profiling": {
    "total_duration_ms": 245.3,
    "phases": {"load": 0, "inference": 245.3},
    "timeline": [
      {
        "phase": "inference",
        "name": "inference.run",
        "duration_ms": 245.3,
        "cpu_percent": 15.0,
        "memory_mb": 2048.0,
        "gpu_memory_mb": 4096.0,
        "metadata": {"prompt": "Hello world", "session": "..."}
      }
    ]
  },
  "gpu_util": 1.0,
  "memory_util": 0.45,
  "cache_ids": ["act-001", "act-002"]
}
```

---

## 1.6 Layers & Heads

### List layers

```
GET /api/v1/layers
```

**Response (200):**
```json
{
  "layers": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
}
```

### List heads for a layer

```
GET /api/v1/layers/{layer_idx}/heads
```

**Response (200):**
```json
{
  "layer": 0,
  "heads": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
}
```

**Response (404):**
```json
{"detail": "Layer 99 out of range (0-11)"}
```

---

## 1.7 Hooks

### List hooks

```
GET /api/v1/hooks
```

**Response (200):**
```json
[
  {
    "id": "hook-001",
    "name": "attention",
    "enabled": true,
    "layer_idx": 0,
    "hook_type": "attention"
  }
]
```

### Register hook

```
POST /api/v1/hooks
```

**Request:**
```json
{
  "hook_type": "attention",
  "layer_idx": 0,
  "name": "my-attention-hook"
}
```

**Response (200):**
```json
{
  "id": "hook-001",
  "name": "attention",
  "enabled": true,
  "layer_idx": 0,
  "hook_type": "attention"
}
```

### Remove hook

```
DELETE /api/v1/hooks/{hook_id}
```

**Response (200):**
```json
{
  "status": "removed",
  "hook_id": "hook-001"
}
```

**Response (404):**
```json
{"detail": "Hook hook-001 not found"}
```

---

## 1.8 Activation Cache

### Search cache

```
GET /api/v1/cache?session_id=...&layer=0&component=attention&head=7
```

Query parameters (all optional):
- `session_id` — filter by session
- `layer` — filter by layer index
- `component` — `"attention"` | `"mlp"` | `"residual"` | `"embedding"` | `"logit"`
- `head` — filter by head index (attention only)

**Response (200):**
```json
[
  {
    "activation_id": "act-001",
    "session_id": "a1b2c3d4-...",
    "prompt_id": "a1b2c3d4",
    "layer": 0,
    "head": 7,
    "component": "attention",
    "shape": [1, 12, 5, 5],
    "timestamp": 1715000000.0,
    "metadata": {"model": "gpt2"}
  }
]
```

### Get cached activation

```
GET /api/v1/cache/{activation_id}
```

**Response (200):** Same shape as search item.

**Response (404):**
```json
{"detail": "Activation act-001 not found"}
```

---

## 1.9 Events

```
GET /api/v1/events?event_type=inference.started&limit=100
```

**Response (200):**
```json
{
  "total": 1024,
  "events_per_sec": 5.2,
  "events": [
    {
      "type": "inference.started",
      "timestamp": 1715000000.0,
      "payload": {"prompt": "Hello", "model": "gpt2"}
    }
  ]
}
```

---

## 1.10 Performance

```
GET /api/v1/performance
```

**Response (200):**
```json
{
  "samples": [...],
  "total_duration_ms": 245.3,
  "phases": {
    "load": 0.0,
    "inference": 245.3,
    "hooks": 0.0,
    "cache": 0.0,
    "serialization": 0.0,
    "api": 0.0
  },
  "timeline": [...]
}
```

---

## 1.11 Streaming Inference

```
GET /api/v1/inference/stream?prompt=Hello&max_new_tokens=10
```

**Media type:** `text/event-stream`

**Events:**
```
event: inference.started
data: {"prompt": "Hello", "num_tokens": 1}

event: layer.processed
data: {"layer": 0, "component": "attention", "shape": [1, 12, 1, 1], "timestamp": ...}

event: layer.processed
data: {"layer": 0, "component": "mlp", "shape": [1, 768], "timestamp": ...}

event: layer.processed
data: {"layer": 1, "component": "attention", ...}
...

event: inference.finished
data: {"generated_text": "Hello world!", "num_layers": 12, "num_attention_maps": 144}
```

---

# 2. DTO Schemas

## 2.1 AttentionMap

```python
class AttentionMap(BaseModel):
    layer: int
    head: int
    tokens: list[str]
    matrix: list[list[float]]   # [seq_len × seq_len], softmax-normalized
```

## 2.2 NeuronActivation

```python
class NeuronActivation(BaseModel):
    layer: int
    index: int                  # neuron index within layer
    activation: float           # tanh(GELU_activation), 0..1 range
```

## 2.3 TokenInfo

```python
class TokenInfo(BaseModel):
    text: str                   # decoded token text
    id: int                     # token ID (vocabulary index)
```

## 2.4 InferenceRequest

```python
class InferenceRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 10
    session_id: str | None = None
```

## 2.5 InferenceResponse

```python
class InferenceResponse(BaseModel):
    session_id: str
    model_name: str
    tokens: list[TokenInfo]
    generated_text: str
    attention_maps: list[AttentionMap]
    neuron_activations: list[NeuronActivation]
    profiling: dict
    gpu_util: float
    memory_util: float
    cache_ids: list[str]
```

## 2.6 ModelLoadRequest / ModelLoadResponse

```python
class ModelLoadRequest(BaseModel):
    model_name: str = "gpt2"

class ModelLoadResponse(BaseModel):
    model_name: str
    status: str
    num_layers: int
    num_heads: int
    hidden_dim: int
```

## 2.7 HookRegisterRequest / HookHandleResponse

```python
class HookRegisterRequest(BaseModel):
    hook_type: str       # "embedding" | "attention" | "mlp" | "residual" | "logit" | "custom"
    layer_idx: int
    name: str | None = None

class HookHandleResponse(BaseModel):
    id: str
    name: str
    enabled: bool
    layer_idx: int
    hook_type: str
```

## 2.8 SessionResponse / SessionCreateResponse

```python
class SessionResponse(BaseModel):
    session_id: str
    model_name: str
    prompt: str
    generated_text: str
    created_at: float
    closed_at: float | None
    is_open: bool
    num_tokens: int
    num_layers: int

class SessionCreateResponse(BaseModel):
    session_id: str
```

## 2.9 CacheEntryResponse

```python
class CacheEntryResponse(BaseModel):
    activation_id: str
    session_id: str
    prompt_id: str
    layer: int
    head: int | None
    component: str
    shape: list[int]
    timestamp: float
    metadata: dict
```

## 2.10 ErrorResponse

```python
class ErrorResponse(BaseModel):
    error: str        # exception class name
    detail: str       # human-readable message
    type: str         # error type identifier
```

---

# 3. Events

All events are emitted by the `EventBus` and recorded in the `EventLog`.

| Event Type | Payload | Emitted When |
|---|---|---|
| `model.loading` | `model_name`, `hf_id` | Model download starts |
| `model.loaded` | `model_name`, `info` | Model ready |
| `model.unloaded` | `model_name` | Model unloaded |
| `model.switched` | `model_name` | Switch completes |
| `session.created` | `session_id`, `model` | New session |
| `session.closed` | `session_id` | Session closed |
| `inference.started` | `prompt`, `model` | Inference begins |
| `inference.layer_processed` | `layer`, `component` | Single layer done |
| `inference.finished` | `tokens`, `generated_length` | Inference complete |
| `inference.error` | `error`, `detail` | Inference fails |
| `activation.captured` | — | Raw activation captured |
| `activation.cached` | `activation_id`, `layer`, `head` | Stored in cache |
| `hook.registered` | `hook_name`, `layer` | Hook attached |
| `hook.removed` | `hook_name`, `layer` | Hook detached |
| `hook.error` | `hook_name`, `error` | Hook execution error |
| `cache.updated` | — | Cache entry added |
| `cache.hit` | `activation_id` | Cache hit |
| `cache.miss` | — | Cache miss |

---

# 4. Hooks

## 4.1 Built-in Hook Types

| Type | Module | Captures |
|---|---|---|
| `embedding` | `model.transformer.wte` | Token embedding vectors `[batch, seq, hidden]` |
| `attention` | `model.transformer.h[i].attn` | Attention weights (true Q×K^T, softmaxed) `[batch, heads, seq, seq]` |
| `mlp` | `model.transformer.h[i].mlp` | MLP GELU activation `[batch, seq, intermediate_dim]` |
| `residual` | `model.transformer.h[i]` | Residual stream after block `[batch, seq, hidden]` |
| `logit` | `model.lm_head` | Final logits `[batch, seq, vocab]` |
| `custom` | (user-specified) | Arbitrary |

## 4.2 Hook Dependencies

Hooks can declare dependencies to ensure correct execution order:

```python
handle = hook_manager.register_hook(
    layer_idx=5,
    hook_type="residual",
    depends_on=["mlp"],       # residual requires MLP to have run
    runs_after=["attention"], # residual runs after attention
)
```

---

# 5. Capabilities

Returned by `GET /api/v1/capabilities`. Describes runtime feature support.

| Field | Type | Description |
|---|---|---|
| `version` | string | Runtime version |
| `models` | string[] | Available model names |
| `hooks` | string[] | Available hook type names |
| `supports_streaming` | bool | SSE layer-by-layer streaming |
| `supports_sessions` | bool | Session management |
| `supports_attention` | bool | Attention hook available |
| `supports_residuals` | bool | Residual hook available |
| `supports_mlp` | bool | MLP hook available |
| `supports_embeddings` | bool | Embedding hook available |
| `supports_logits` | bool | Logit hook available |
| `supports_patching` | bool | Activation patching (future) |
| `supports_sae` | bool | Sparse autoencoders (future) |
| `supports_circuits` | bool | Circuit discovery (future) |
| `supports_gradients` | bool | Gradient-based methods (future) |
| `supports_client_auth` | bool | Client-ID header auth |
| `supports_persistence` | bool | SQLite session persistence |
| `supports_event_replay` | bool | Event log with query/replay |
| `supports_hook_dependencies` | bool | depends_on / runs_after |

---

# 6. Error Codes

All errors return HTTP 400 with a JSON body.

| Exception | `error` field | Meaning |
|---|---|---|
| `ModelLoadError` | `ModelLoadError` | Model cannot be loaded |
| `ModelNotFoundError` | `ModelNotFoundError` | Unknown model name |
| `InferenceError` | `InferenceError` | Inference failure |
| `HookError` | `HookError` | Hook registration/execution failure |
| `SessionNotFoundError` | `SessionNotFoundError` | Unknown session ID |
| `CacheError` | `CacheError` | Cache operation failure |
| `MemoryError` | `MemoryError` | Runtime OOM |
| `APIError` | `APIError` | API contract violation |
| `HealthCheckError` | `HealthCheckError` | Health check component failure |

---

# 7. Session Schema

Sessions are stored in SQLite (`runtime.db` by default, configurable via `MODEL_EXPLORER_DB` env var).

## SQLite Table

```sql
CREATE TABLE sessions (
    session_id   TEXT PRIMARY KEY,
    model_name   TEXT,
    prompt       TEXT,
    generated_text TEXT,
    created_at   REAL,
    closed_at    REAL,
    token_ids    TEXT,    -- JSON array of ints
    token_texts  TEXT,    -- JSON array of strings
    num_layers   INTEGER,
    num_heads    INTEGER,
    metadata     TEXT     -- JSON object (includes client_id, etc.)
);
```

## Python Dataclass

```python
@dataclass
class Session:
    session_id: str
    model_name: str
    prompt: str
    generated_text: str
    created_at: float
    closed_at: float | None
    token_ids: list[int]
    token_texts: list[str]
    num_layers: int
    num_heads: int
    activations: dict[str, Tensor]   # runtime only, not persisted
    metadata: dict
```

---

# 8. Versioning Policy

## Rules

1. **`/api/v1/` is frozen.** No breaking changes without Architect approval.
2. **New endpoints** are added to v1 freely (backward-compatible additions only).
3. **Deprecation path:** Mark endpoint as deprecated in v1, keep it running for 2 releases, then remove.
4. **`/api/v2/`** (future) may break contracts when necessary.
5. **Root aliases** (`/models`, `/inference`) mirror v1 but new consumers should use versioned paths.

## What constitutes a breaking change

- Removing an endpoint
- Renaming a DTO field
- Changing a DTO field type
- Adding a required field to a request body
- Changing event type names
- Removing a hook type
- Changing session schema columns

## What does NOT require approval

- Adding new endpoints
- Adding new hook types
- Adding new model descriptors
- Performance improvements
- Internal refactoring
- Bug fixes that don't change contracts
- Adding optional fields to response DTOs
