# MECH — Mechanistic Interpretability IDE

A research platform for reading the internal behaviour of neural networks and language models.

Instead of asking *"what did the model output?"*, MECH asks *"what is the model actually computing?"* — inspecting neurons, attention heads, residual streams, and Sparse Autoencoder features, then tracing how they causally compose into the final prediction.

Built by a 2nd-year CSE (AI & ML) student. Currently validated against **GPT-2 Small**.

---

## Why this exists

Interpretability research is usually a notebook you write and never reuse. The analysis lives in one place, the model lives in another, and reproducing last month's result means rewriting the plumbing.

MECH is a platform, not a notebook. Models load once, analyses are registered as reusable algorithms, and every run produces a comparable report against established baselines.

---

## Core capabilities

### Inspectors
| Inspector | What it answers |
|---|---|
| `neuron_inspector` | Which tokens does this neuron fire on, and how strongly? |
| `attention_inspector` | Where is this head looking, and what pattern is it detecting? |
| `residual_inspector` | How does information accumulate across layers? |
| `logit_inspector` | What does the model predict at each step, before the softmax? |
| `token_inspector` | How does a specific token's representation change through depth? |
| `feature_inspector` | What has this SAE feature learned to represent? |

### Algorithms
- **Logit Lens** and **Tuned Lens** — read intermediate layer representations directly as vocabulary predictions
- **Attention Head Ranker** — score and rank heads by importance
- **Activation / Feature Search** — find neurons and features matching a query
- **Causal Tracing** and **Attribution Patching** — intervene on a component and measure the effect on the output

### Sparse Autoencoders
Feature dictionaries with caching and loader support, for decomposing dense activations into interpretable sparse features.

### Discovery
The largest subsystem — 60+ modules covering:
- **Circuit discovery and evolution** — find the circuits that implement a behaviour, then track how they change across models
- **Autonomous research loops** — hypothesis generation, automated paper replication, regression suites
- **Cross-model circuits** — does the same mechanism implement a behaviour in GPT-2 and in Pythia?
- **Polysemanticity detection** — find neurons doing more than one job
- **Concept & representation evolution** — track how internal representations drift
- **Superposition analysis** — how many features are packed into how many dimensions
- **Confidence calibration and evidence ranking** — which discoveries are actually trustworthy

---

## Reproducibility pipeline

Every supported task runs as a standardised `Pipeline`, executed by `BenchmarkRunner`, producing a `ReproducibilityReport` that compares observed metrics against known baselines (e.g. TransformerLens) and assigns a fidelity tier: **Gold**, **Silver**, **Bronze**, or **Needs Investigation**.

Built-in benchmarks:

| Benchmark | Validates |
|---|---|
| **IOI** (Indirect Object Identification) | Circuit extraction logic |
| **Induction Heads** | In-context learning mechanics |
| **Greater Than** | Mathematical circuitry |
| **Copy Task** | Zero- and one-layer copying behaviour |
| **Arithmetic** | Modulo and base-10 addition |
| **Factual Recall** | ROME / MEMIT tracing |

---

## Architecture

Modular, multi-agent design built around five abstractions:

- **`ResearchSociety`** — orchestrator managing specialised sub-agents (Planner, GoalOptimizer, Critic) through autonomous research loops
- **`UnifiedRegistry`** — single source of truth for Papers, Circuits, Mechanisms, Datasets, Experiments
- **`ModelManager`** — HuggingFace weights, lazy loading, device placement, VRAM pressure handling
- **`ExecutionBackend`** — abstraction letting workloads run on local CUDA, Kubernetes, Ray, or Slurm
- **`ArtifactStore`** — persists discoveries, figures, and models to local storage, S3, or GCS

Supporting layers: `provenance_viewer` and `evidence_graph` for traceability, `workflow_dsl` for reproducible pipelines, `vector_store` for semantic search across artifacts.

---

## Stack

**Backend:** Python 3.11+, FastAPI, Pydantic v2, SQLAlchemy 2
**ML:** PyTorch, HuggingFace Transformers, TransformerLens, NumPy, SciPy, pandas, Matplotlib, seaborn
**Frontend:** TypeScript, Vite, React, Electron, Tailwind
**Testing:** pytest, Vitest, Playwright

---

## Running it

```bash
git clone https://github.com/KUNDURUHIMANEESHREDDY/MECH.git
cd MECH

python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Backend
uvicorn main:app --reload          # API at http://localhost:8000

# Frontend
cd frontend && npm install && npm run dev
```

Requires a CUDA GPU for real model inference. Runs on CPU with reduced benchmark scope.

### Running benchmarks

```bash
python run_benchmarks.py
```

---

## Project layout

```
backend/
  interpretability/     the core: inspectors, algorithms, SAE, causal tracing, discovery
  core/                 registry, provenance, evidence graph, artifact store, workflow DSL
  agents/               multi-agent orchestration
  research/             research platform and campaign management
  benchmarking/         benchmark runners and registries
  api/                  FastAPI routes
frontend/               Electron + React IDE interface
docs/                   architecture, algorithms, API reference, benchmarks, roadmap
tests/                  pytest suite
```

---

## Status

Active development. Benchmark coverage is expanding; some discovery modules are experimental. Roadmap in [`docs/roadmap.md`](docs/roadmap.md).

---

## Notes on scope

This is a student research platform, not a production product. It's honest about that, and the benchmark system is specifically designed to make claims falsifiable — including MECH's own. If a discovery doesn't replicate against baseline, the report says so.

## License

MIT — see [`LICENSE`](LICENSE).