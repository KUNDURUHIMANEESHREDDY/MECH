# Autonomous AI Scientist Platform Documentation

## Overview

The **Autonomous AI Scientist Platform** in MECH is an AI-native orchestration engine that formulates hypotheses, designs controlled experiments, executes interventions via the **Verified Capability Registry**, and analyzes empirical results to discover and validate neural mechanisms.

---

## Key Modules & Components

| Component | Code Location | Purpose |
| :--- | :--- | :--- |
| **Master Orchestrator** | `backend/research_platform/autonomous/ai_scientist_engine.py` | Coordinates the end-to-end scientific research lifecycle from prompt formulation to final publication draft. |
| **Multi-Agent Research Society** | `backend/research_platform/autonomous/multi_agent_society.py` | Orchestrates specialized agents (Planner, Experiment Designer, Critic, Governance) in structured scientific debate. |
| **Literature Learning Pipeline** | `backend/research_platform/meta/literature_learning_pipeline.py` | Vector-indexes mechanistic interpretability literature to ground hypotheses in published findings. |
| **Traceable Evidence Graph** | `backend/core/evidence_graph.py` | Assembles a directed graph of hypotheses, interventions, observations, and calculated effect sizes ($p$-values, Cohen's $d$). |
| **Self-Reflection Engine** | `backend/research_platform/meta/self_reflection_engine.py` | Detects null or contradictory experimental results and formulates refined follow-up hypotheses. |
| **Uncertainty & Governance Manager** | `backend/research_platform/autonomous/uncertainty_manager.py` | Quantifies statistical uncertainty and enforces experimental rigor before findings are accepted. |
| **Research Roadmap Generator** | `backend/research_platform/autonomous/research_roadmap_generator.py` | Dynamically produces multi-stage research roadmaps to systematically explore model capabilities. |

---

## Closed-Loop Execution Lifecycle

```text
1. Research Goal Ingestion ("Investigate why GPT-2 fails on greater-than task")
       ↓
2. Literature Ingestion & Semantic Retrieval (RAG)
       ↓
3. Multi-Agent Hypothesis Formulation & Debate
       ↓
4. Experiment Recommendation & Design (Counterfactual Prompt Pairs)
       ↓
5. Verified Capability Registry Execution (Logit Lens, Patching, SAE Features)
       ↓
6. Evidence Graph Ingestion & Statistical Significance Validation
       ↓
7. Uncertainty Evaluation & Self-Reflection (Refine or Accept)
       ↓
8. Circuit Synthesis on Canvas & LaTeX Report Generation
```

For full architectural details, see the [AI Research Assistant Layer Specification](file:///c:/Users/himaneeshreddyk/Downloads/MECH/docs/AI_RESEARCH_ASSISTANT_LAYER.md).
