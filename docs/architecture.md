# MECH Architecture Overview

MECH is an **AI-Native Mechanistic Interpretability Research Studio and Autonomous Scientist Platform**. 

It is designed around a three-tier architecture:
1. **The MI Core Engine:** High-performance, hook-based tensor analysis (TransformerLens, PyTorch, Sparse Autoencoders, Causal Mediation).
2. **The AI Research Assistant Layer:** A cognitive layer providing hypothesis formulation, literature RAG, verified tool execution, multi-agent debate, and closed-loop self-reflection.
3. **The Interactive Research Canvas:** A dual-target desktop (Electron) and web (React 18 + Vite) IDE providing dockable panels, live heatmaps, and ReactFlow circuit diagrams.

---

## 1. Core Architectural Abstractions

```text
┌────────────────────────────────────────────────────────────────────────┐
│                        INTERACTIVE RESEARCH CANVAS                     │
│   • ReactFlow Circuit Viewer    • Token & Prediction Panels            │
│   • Multi-Pane Dock Manager     • Interactive Research Notebooks       │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
┌───────────────────────────────────▼────────────────────────────────────┐
│                    AI RESEARCH ASSISTANT LAYER                         │
│   • Literature RAG & Knowledge Graph                                   │
│   • Hypothesis Formulation & Experiment Designer                       │
│   • Multi-Agent Research Society (Planner, Critic, Governance)         │
│   • Closed-Loop Self-Reflection & Anomaly Detection                    │
│   • Traceable Evidence Graph & Statistical Validation                  │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
┌───────────────────────────────────▼────────────────────────────────────┐
│                      VERIFIED CAPABILITY REGISTRY                      │
│   • Logit Lens Probing          • Causal Attribution & Patching        │
│   • SAE Feature Dictionary      • Automated Circuit Discovery (ACDC)   │
│   • Attention Pattern Profiling • Monosemanticity / Polysemanticity    │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
┌───────────────────────────────────▼────────────────────────────────────┐
│                         MODEL RUNTIME ENGINE                           │
│   • PyTorch / TransformerLens Multi-Model Engine (GPT-2, Gemma, LLaMA) │
│   • Pluggable Drivers (Local CUDA, Distributed, Ray, Slurm, K8s)       │
│   • Multi-Threaded SQLite Storage Engine (WAL Mode, Deterministic IDs) │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Layer Specifications

### Layer 1: The Model Runtime & Execution Engine
* **TransformerLens & PyTorch Adapters:** In-memory hook management for residual stream capture, MLP activations, and attention matrices across GPT-2, Gemma, LLaMA, Qwen, Mistral, and DeepSeek.
* **Storage & Persistence:** Multi-threaded SQLite database utilizing Write-Ahead Logging (WAL) and collision-free deterministic hashing.
* **Distributed Runtime:** Adaptive scheduler capable of offloading heavy SAE sweeps to local GPU pools, Ray, or Slurm clusters.

### Layer 2: Verified Capability Registry
* **Strongly-Typed Tool Primitives:** Central registry (`backend/core/capability_registry.py`) providing deterministic mathematical tools that return real tensor metrics rather than simulated text.
* **Causal Patching & Attribution:** Corrupt-and-restore mediation analysis for calculating direct and indirect causal effects.
* **Sparse Autoencoders (SAE):** Dictionary learning for decomposing dense representations into monosemantic concept directions.

### Layer 3: AI Research Assistant Layer
* **Literature Grounding:** RAG engine indexing mechanistic interpretability literature to align experimental designs with established science.
* **Autonomous Experiment Loop:** Generates prompt pairs, executes multi-layer interventions, evaluates $p$-values and Cohen's $d$, and updates the Evidence Graph.
* **Empirical Self-Reflection:** Diagnoses null results and hypothesizes alternative mechanisms (backup heads, redundant pathways).

### Layer 4: Human-in-the-Loop Canvas
* **Bidirectional Sync:** When the AI assistant runs an experiment, the resulting circuit graph is rendered immediately in ReactFlow, allowing the human researcher to interactively steer, edit, or extend the investigation.

---

## 3. Related Documentation
* [AI Research Assistant Layer Specification](file:///c:/Users/himaneeshreddyk/Downloads/MECH/docs/AI_RESEARCH_ASSISTANT_LAYER.md)
* [Interpretability Algorithms & Causal Methods](file:///c:/Users/himaneeshreddyk/Downloads/MECH/docs/INTERPRETABILITY_ALGORITHMS.md)
* [IDE Design System & Layout](file:///c:/Users/himaneeshreddyk/Downloads/MECH/frontend/DESIGN.md)
