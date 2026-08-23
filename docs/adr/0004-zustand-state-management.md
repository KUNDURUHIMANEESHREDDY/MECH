# ADR-0004: Zustand for Frontend State Management

**Date**: 2026-08-10
**Status**: accepted
**Deciders**: Frontend Team

## Context

The MECH frontend needs a state management solution that works across:
- Vue 3 (Options/Composition API) components
- React 18 components
- Shared state between frameworks (user session, workspace, model selection)

Requirements:
- Framework-agnostic (works in both Vue and React)
- TypeScript-first with good inference
- DevTools support
- Minimal boilerplate
- Async action support

## Decision

Use **Zustand** as the primary state management library with:
- **Vanilla store** (`createStore`) for framework-agnostic access
- **React hooks** (`useStore`) for React components
- **Vue composable** (`useStore`) via custom wrapper for Vue components
- **Middleware**: `persist` for localStorage, `devtools` for debugging

## Alternatives Considered

### Alternative 1: Redux Toolkit
- **Pros**: Opinionated, great DevTools, TypeScript support
- **Cons**: Boilerplate, React-centric, not Vue-friendly
- **Why not**: Overkill for current needs, dual-framework friction

### Alternative 2: Vuex / Pinia (Vue only)
- **Pros**: Native Vue integration, DevTools
- **Cons**: Not usable in React components
- **Why not**: Doesn't solve cross-framework state

### Alternative 3: React Context + useReducer
- **Pros**: Built-in, no dependencies
- **Cons**: Performance issues at scale, React-only
- **Why not**: Doesn't solve cross-framework state

### Alternative 4: Jotai / Recoil
- **Pros**: Atomic, React-friendly
- **Cons**: React-only, experimental
- **Why not**: Vue compatibility required

## Consequences

### Positive
- Single source of truth for both frameworks
- Minimal API surface (~500 lines for entire store)
- Excellent TypeScript inference
- Works outside React/Vue (service workers, Web Workers)
- Easy testing (plain objects/functions)

### Negative
- Less opinionated than Redux (team must define patterns)
- No built-in normalization (manual for relational data)
- DevTools not as rich as Redux DevTools

### Risks
- Store mutation outside components can cause stale renders
- **Mitigation**: Use `shallow` equality, subscribe to specific keys
- **Mitigation**: Document Zustand patterns in CLAUDE.md