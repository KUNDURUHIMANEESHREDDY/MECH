# Runtime Engine & Forward Debugger API Documentation

This document describes the **Runtime Engineer** architecture, activation patching engine, forward stepping debugger, model comparison DTOs, intermediate logits, and batch experiment runner.

---

## 1. Architecture Overview

```text
               backend/runtime/
  ┌───────────────────┼───────────────────┐
  ▼                   ▼                   ▼
execution_context.py  patching.py         debugger.py
(Shared State)        (PatchSet)          (Stepping Sessions)

  ▼                   ▼                   ▼
comparison.py         logits.py           batch_runner.py
(Model Compare)       (Intermed Logits)   (Batch Experiments)
```

---

## 2. Debug Session Lifecycle (`runtime/debugger/*`)

### Session Lifecycle Flow
```text
start_debug_session -> set_breakpoint -> step / continue -> BreakpointHit -> stop / finish
```

### Endpoints & Payloads

#### `runtime/debugger/start`
- **Payload**: `{ "session_id": "dbg_1", "prompt": "When Mary and John...", "breakpoint": 8 }`
- **Returns**: Debugger state dict containing `current_layer`, `status`, and `breakpoints`.

#### `runtime/debugger/step`
- **Payload**: `{ "session_id": "dbg_1" }`
- **Returns**: Advances 1 layer, returns updated layer index and status (`running`, `paused`, `finished`).

#### `runtime/debugger/continue`
- **Payload**: `{ "session_id": "dbg_1" }`
- **Returns**: Executes until next breakpoint or completion.

---

## 3. Activation Patching Engine (`runtime/patch`)

### `ActivationPatch` Schema
```json
{
  "session_id": "sess_ioi",
  "layer": 8,
  "component": "mlp",
  "neuron_index": 402,
  "operation": "replace",
  "value": 2.5
}
```

### Supported Intervention Operations
- `replace`: Set activation to exact `value`.
- `add`: Add `value` to original tensor activation.
- `multiply`: Scale activation by `value`.
- `zero`: Zero out activation tensor.
- `mask`: Pass-through if `value > 0`, else zero out.
- `custom`: User-defined intervention.

---

## 4. Model Comparison DTOs (`runtime/compare`)

Normalized comparison payload comparing two architectures (e.g. GPT-2 vs Pythia):

```json
{
  "prompt": "The capital of France is",
  "model_a": "GPT-2 Small",
  "model_b": "Pythia 160M",
  "activations": { "cosine_similarity": 0.875, "kl_divergence": 0.142 },
  "predictions": { "top_token_match": true, "model_a_top": { "token": " Paris", "prob": 0.82 } },
  "attention": { "head_alignment_score": 0.91 },
  "residuals": { "layer_norm_ratio": 1.04 }
}
```

---

## 5. Debugger Event Flow

The debugger emits real-time events during execution:
- `DebuggerStarted`: Emitted when debug session initiates.
- `BreakpointHit`: Emitted when forward execution pauses at a breakpoint.
- `StepCompleted`: Emitted after stepping 1 layer forward.
- `DebuggerFinished`: Emitted when forward pass reaches final layer.
- `DebuggerStopped`: Emitted on explicit cancellation.
