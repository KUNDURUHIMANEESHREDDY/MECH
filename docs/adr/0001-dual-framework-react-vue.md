# ADR-0001: Dual React+Vue Frontend Framework

**Date**: 2026-08-10
**Status**: accepted
**Deciders**: Architecture Team

## Context

The MECH platform frontend was initially built with Vue 3 (Options API) for the core shell and dashboard. As the platform grew, React was introduced for complex interactive components (visualizations, neuron explorers, circuit editors) due to:
- Better ecosystem for scientific visualization libraries (D3, Three.js, React Flow)
- Richer component library ecosystem (Radix UI, shadcn/ui)
- Team familiarity with React patterns

## Decision

We will maintain a **dual-framework approach** during transition:
- **Vue 3** for: App shell, routing, layout, settings, simple CRUD pages
- **React 18** for: Complex interactive panels (Circuit Explorer, Neuron Inspector, Transformer Visualizer, Knowledge Graph, Debugger)

Migration path: New features in React; migrate Vue pages incrementally.

## Alternatives Considered

### Alternative 1: Full Migration to React
- **Pros**: Single framework, simpler build, unified state management
- **Cons**: High risk, 3-6 month effort, disrupts ongoing feature work
- **Why not**: Current Vue code is stable; incremental migration is safer

### Alternative 2: Full Migration to Vue
- **Pros**: Single framework, team has Vue expertise
- **Cons**: Weaker visualization ecosystem, would need to rewrite React panels
- **Why not**: React panels are complex and working well

## Consequences

### Positive
- Can leverage best framework for each use case
- Incremental migration reduces risk
- Teams can work in preferred framework

### Negative
- Dual build configuration (Vite + both frameworks)
- Bundle size overhead (~150KB extra)
- Context switching for developers
- State sharing requires bridge (custom event bus)

### Risks
- Version conflicts between Vue/React ecosystems
- Long-term maintenance burden
- **Mitigation**: Set 12-month target for full React migration

## Related ADRs
- ADR-0004: Zustand for State Management (chosen to work across both frameworks)