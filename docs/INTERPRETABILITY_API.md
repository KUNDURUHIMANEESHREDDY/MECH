# Mechanistic Interpretability API & Architecture Documentation

This document describes the architecture, inspector responsibilities, activation repository query engine, search APIs, data flow, and example requests for the **Neural Debugger** backend.

---

## 1. Architecture Overview

```text
Electron Desktop Shell
          │
     (JSON Stdio)
          ▼
   backend/main.py
          │
          ▼
  backend/api/dispatcher.py
   ┌──────┴─────────────┬─────────────────┬──────────────────┐
   ▼                    ▼                 ▼                  ▼
runtime/         interpretability/    repository/        search/
(Engine)         (Inspectors)         (ActivationRepo)   (Search Services)
```

---

## 2. Inspectors Package (`backend/interpretability/inspectors/`)

| Inspector Module | Class Name | Responsibility |
|---|---|---|
| `neuron_inspector.py` | `NeuronInspector` | Analyzes neuron activations, firing frequency, and feature labels. |
| `attention_inspector.py` | `AttentionInspector` | Computes multi-head attention weights, induction patterns, and head classifications. |
| `residual_inspector.py` | `ResidualInspector` | Decomposes residual stream norms and layer contribution percentages. |
| `layer_inspector.py` | `LayerInspector` | Profiles layer-wide MLP vs. Attention activation norms. |
| `token_inspector.py` | `TokenInspector` | Inspects per-token residual stream state and peak firing neurons. |
| `prediction_inspector.py` | `PredictionInspector` | Inspects output logits, top-k probabilities, entropy, Logit Lens, and Tuned Lens projections. |

---

## 3. Repository Query Engine (`backend/repository/activation_repository.py`)

The `ActivationRepository` provides structured, multi-field filtering over cached activation records.

### Query Filter Fields
- `layer` *(int)*: Filter by transformer layer index.
- `component` *(str)*: Filter by component (`mlp`, `attention`, `residual`).
- `token` *(str)*: Filter by token text string.
- `threshold` / `min_activation` *(float)*: Minimum activation magnitude cutoff.
- `session_id` *(str)*: Filter by research session ID.
- `head` *(int)*: Filter by attention head index.
- `prompt_id` *(str)*: Filter by input prompt ID.
- `activation_id` *(str)*: Filter by specific activation record ID.
- `top_k` *(int)*: Limit number of returned results.
- `sort_by` *(str)*: Sort by `activation` (descending) or `layer`.

### Cache Telemetry Metrics
Exposes real-time cache performance stats:
- `hits`: Total successful queries.
- `misses`: Queries yielding zero records.
- `hit_ratio`: Ratio of hits / (hits + misses).
- `cached_items`: Count of records in cache.
- `memory_bytes`: Approximate memory consumed by cache in bytes.
- `oldest_entry` & `newest_entry`: ISO timestamps of oldest/newest records.
- `average_lookup_ms`: Average query latency in milliseconds.

---

## 4. Search API

Generic `search` endpoint supporting query types:
- `neuron`: Search neuron IDs (e.g. `L8_N402`) or feature labels.
- `token`: Search token text matches.
- `session`: Search active research sessions.
- `experiment`: Search experiment matrix runs.
- `layer`: Filter activations by layer index.
- `attention_head`: Query attention head records.

---

## 5. Example Protocol Requests & Responses

### Request: `inspectors:neuron`
```json
{
  "id": 1,
  "method": "inspectors:neuron",
  "payload": { "layer": 8, "neuron_index": 402 }
}
```
### Response:
```json
{
  "id": 1,
  "result": {
    "layer": 8,
    "neuron_index": 402,
    "neuron_id": "L8_N402",
    "mean_activation": 0.782,
    "max_activation": 2.41,
    "firing_frequency": 0.4,
    "feature_label": "Layer 8 Feature #402"
  }
}
```

### Request: `repository:query`
```json
{
  "id": 2,
  "method": "repository:query",
  "payload": { "layer": 8, "min_activation": 2.0, "sort_by": "activation" }
}
```

### Request: `repository:metrics`
```json
{
  "id": 3,
  "method": "repository:metrics",
  "payload": {}
}
```
