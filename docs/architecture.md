# Architecture Overview

The platform is designed around a modular, multi-agent architecture that orchestrates mechanistic interpretability research.

## Core Abstractions
1. **ResearchSociety**: A unified orchestrator managing all specialized AI sub-agents (Planner, GoalOptimizer, Critic, etc.) to conduct autonomous research loops.
2. **UnifiedRegistry**: The central truth-source for tracking Papers, Circuits, Mechanisms, Datasets, and Experiments.
3. **ModelManager**: Handles Hugging Face weights, lazy loading, device placement, and VRAM pressure.
4. **ExecutionBackend**: An abstraction layer permitting workloads to run on Local CUDA, Kubernetes, Ray, or Slurm seamlessly.
5. **ArtifactStore**: Persists discoveries, figures, and models to Local Storage, S3, or GCS.

## The Reproducibility Pipeline
Every supported task (IOI, Induction Heads, SAE) is implemented as a standardized Pipeline. The `BenchmarkRunner` executes these pipelines, yielding a `ReproducibilityReport` that objectively compares observed metrics against established baselines (e.g. TransformerLens).


## Rust-native application boundary

MECH is migrating application ownership to Rust. The active desktop host is Tauri 2; Vue 3 remains the renderer. Rust owns renderer IPC validation, bearer-token injection, workspace SQLite access, and compatibility-worker lifecycle. The FastAPI control plane and PyTorch/TransformerLens scientific engine remain temporary compatibility services until their Rust replacements pass behavior and numerical parity checks. See [RUST_MIGRATION.md](RUST_MIGRATION.md).
