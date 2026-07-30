# Research Platform Documentation (Sprint 3)

This document describes the overarching Research Platform architecture, unifying Workflows, Pipelines, Plugins, and Datasets.

---

## Architecture Overview

```text
                  backend/platform/
   ┌──────────────────────┼──────────────────────┐
   ▼                      ▼                      ▼
workflow/              pipelines/             datasets/
(State Machine DTO)    (Graph Execution)      (Streaming Engine)
```

---

## Core Platform Components
1. **Workflow Engine**: State machine lifecycle (`Draft -> Question -> Hypothesis -> Experiment -> Running -> Observation -> Analysis -> Conclusion -> Published`).
2. **Pipeline Engine**: Multi-stage graph pipelines (`Prompt -> Activation Search -> SAE -> Circuit Discovery -> Patch -> Compare -> Report`).
3. **Plugin SDK**: Extensible interfaces (`AlgorithmPlugin`, `PanelPlugin`, `ReportPlugin`, `ExporterPlugin`, `WorkflowPlugin`, `PipelineNodePlugin`, `DatasetPlugin`, `CommandPlugin`).
4. **Dataset Manager**: Multi-corpus streaming manager (`OpenWebText`, `The Pile`, `WikiText`, `Custom`).
