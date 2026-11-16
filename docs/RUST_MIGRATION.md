# Rust-native application migration

## Objective

Move MECH's application infrastructure from Electron and Python/FastAPI into a Rust-native desktop and service architecture while retaining the Vue 3 + TypeScript renderer. Python remains a compatibility worker only for scientific operations that have not yet been reimplemented and validated in Rust.

## Target architecture

- **Desktop host:** Tauri 2, with Rust owning application lifecycle, native commands, process supervision, and security boundaries.
- **Renderer:** Vue 3 + TypeScript + Vite. Renderer requests cross typed Tauri commands; it never receives the backend bearer token.
- **Core contracts:** `crates/mech-core` owns shared domain types, execution states, provenance, and validated request contracts.
- **Runtime:** `crates/mech-runtime` owns lifecycle of compatibility workers during migration.
- **Persistence:** Rust-owned SQLite access uses the existing `app.db` schema and path so current projects, sessions, experiments, and settings remain readable.
- **Scientific compute:** migrate data models and deterministic CPU algorithms first; preserve PyTorch/TransformerLens execution behind a supervised Python worker until Rust replacements pass parity tests against real model weights and controlled experiments.

## Migration phases

1. **Native foundation (current):** Rust workspace, Tauri shell, safe API proxy, supervised Python compatibility worker, Rust SQLite adapter, shared scientific result contracts, and fail-closed tests.
2. **Application services:** port settings, workspace/project services, logs, job lifecycle, event delivery, and API routing to Rust. Remove Electron IPC and process-management modules only after parity tests pass.
3. **Research control plane:** port FastAPI routing, authentication/session policy, plugin registry boundaries, MCP transport, and orchestration. Keep an explicit compatibility adapter for routes not yet migrated.
4. **Scientific domain:** port provenance, validation, experiment models, registries, schedulers, and deterministic CPU algorithms. Add numerical parity tests before replacing each implementation.
5. **Tensor/runtime migration:** evaluate Rust tensor backends and model loading for each supported architecture. GPU kernels, activation hooks, patching, and SAE training are migrated only when output and tolerance criteria are documented and verified.
6. **Release cutover:** package the Rust app and required runtime resources, migrate existing user data safely, run end-to-end research workflows, then remove Electron and Python control-plane dependencies. Python may remain an optional external compute adapter if no safe equivalent exists.

## Non-negotiable scientific guarantees

- `NOT_EXECUTED` is distinct from `FAILED` and neither state may contain result data.
- `COMPLETED` requires data and provenance.
- Empty values are not converted into measurements or zeros.
- Exceptions never become successful results.
- Every measurement must retain model/revision, prompt/config hashes, and seed where applicable.
- No Python scientific operation is removed until a real-weight parity or correctness test passes.

## Current compatibility boundary

The Tauri host supervises `backend/main.py` on loopback during this migration. Rust owns the renderer-to-backend proxy and injects the bearer token; the token never crosses into renderer JavaScript. A missing backend or token is reported as unavailable, not replaced with mock output.

## Verification policy

Each phase requires Rust unit tests, frontend type/build checks, and targeted existing regression tests. Full scientific tests are reserved for stable-tree checkpoints because the existing suite loads real model weights and checks source stability. Packaging is not considered complete until a clean-machine install can launch without relying on an undeclared system Python.
