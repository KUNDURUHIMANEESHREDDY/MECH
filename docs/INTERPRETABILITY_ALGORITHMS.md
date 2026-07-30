# Mechanistic Interpretability Algorithms Documentation (Sprint 2)

This document describes the Sparse Autoencoder (SAE) integration, Projection Lens models, Attention Head Ranking, and Search API architecture for **AI 3**.

---

## 1. Architecture Overview

```text
               backend/interpretability/
  ┌───────────────────┬───────────────────┬──────────────────┐
  ▼                   ▼                   ▼                  ▼
sae/                inspectors/         algorithms/        repository/
(Loader & Cache)    (DTO Inspectors)    (Lenses & Ranker)  (Feature Repo)
```

---

## 2. Sparse Autoencoder (SAE) Architecture (`interpretability/sae/*`)

### SAE Package Structure
- `loader.py`: Loads SAE checkpoint files and verifies feature dimension (e.g. `d_sae = 16384`).
- `feature_dictionary.py`: Manages feature IDs, firing frequencies, and descriptions.
- `inspector.py`: Maps feature IDs to connected neurons and dataset examples.
- `cache.py`: Memory manager for loaded SAE model weights.

---

## 3. Projection Models (`interpretability/projections/*`)

Implements the common `ProjectionModel` interface:

### Logit Lens (`interpretability/projections/logit_lens`)
Directly projects intermediate residual stream vectors through the model unembedding matrix to inspect intermediate layer prediction progression.

### Tuned Lens (`interpretability/projections/tuned_lens`)
Applies learned layer-specific affine transformations prior to unembedding projection for higher projection fidelity.

---

## 4. Attention Head Ranking (`interpretability/ranking/heads`)

Ranks multi-head attention components across transformer layers using configurable ranking metrics:
- `entropy`: Attention distribution entropy.
- `importance`: Gradient/activation impact score.
- `sparsity`: L0 sparsity of attention matrix.
- `variance`: Attention weight variance.
- `distance`: Average attention distance across token positions.
- `diversity`: Head representation diversity.

---

## 5. Search Engine (`interpretability/search/*`)

### Activation Search (`ActivationQuery`)
Query model activations by `layer`, `component`, `threshold`, `token`, `top_k`, and `sort_by`.

### Feature Search (`FeatureSearchEngine`)
Multi-modal search over SAE features matching query strings, feature descriptions, and dataset examples.
