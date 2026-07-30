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
