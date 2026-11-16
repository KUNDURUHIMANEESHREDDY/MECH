# ADR 0001: Rust-native desktop and application runtime

- **Status:** Accepted
- **Date:** 2026-11-16

## Context

MECH currently uses Electron for desktop lifecycle and IPC, a FastAPI/Python control plane, SQLite persistence, and a Vue 3 renderer. The repository contains a substantial scientific Python implementation based on PyTorch and TransformerLens. Replacing all of this in one step would risk breaking validated research workflows and silently changing numerical behavior.

## Decision

Adopt a Rust-native application architecture using Tauri 2. Rust will progressively own desktop lifecycle, process supervision, API/control-plane services, persistence, orchestration, and scientific domain contracts. Keep Vue 3 + TypeScript as the renderer. Retain Python as a supervised compatibility worker only for scientific operations until equivalent Rust implementations pass documented parity tests.

The migration is staged. Electron and Python components remain until their replacements cover the same behavior and regression tests prove parity. No cutover is based solely on a successful build or an HTTP 200 response.

## Consequences

- A Cargo workspace is the source of shared Rust domain contracts.
- The renderer communicates with native code through Tauri commands.
- Rust owns API request validation and bearer-token injection; secrets are not exposed to the renderer.
- Rust opens the existing SQLite storage location/schema to preserve user data.
- A compatibility worker remains a temporary runtime dependency and release-packaging constraint.
- The project cannot claim a complete Rust migration until the FastAPI control plane and required application services are replaced and real research workflows are verified.

## Rejected alternatives

- **Tauri as a thin wrapper around the unchanged Electron-era architecture:** would change the shell without moving application ownership into Rust.
- **Immediate deletion of Python:** would discard the validated PyTorch/TransformerLens engine before numerical parity is established.
- **Rewrite the Vue renderer:** adds unrelated UI risk and does not advance the Rust migration objective.
