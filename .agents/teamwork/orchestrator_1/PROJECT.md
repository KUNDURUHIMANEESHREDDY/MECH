# Project: MECH Research Platform Orchestration

## Architecture
MECH is a full-stack mechanistic interpretability and AI research desktop platform comprising:
- **Backend**: FastAPI web service running on `127.0.0.1:8000` (`backend/main.py`), exposing modular routers (`/api`, `/api/v1`), SQLite database storage (`backend/storage/database.py`), transformer interpretability engines, autonomous research agents (Society), and reproducibility pipelines.
- **Frontend**: Vue 3 + Pinia desktop shell (`frontend/src/`) hosting 32 analytical tools, bundled with Vite and optionally executed inside Electron (`frontend/electron/`), proxying `/api/*` requests to port 8000.
- **Persistence**: SQLite WAL-mode database (`mech.db`) with evidence graph persistence, experiment logging, and knowledge graph writeback.
- **Test Infrastructure**: Backend `pytest` suite under `tests/pytest/`, Frontend Vitest suite under `frontend/tests/vitest/`, and Playwright E2E specs.

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| 1 | Server Orchestration & Lifecycle Control | Entry point unification, lifespan handler, WAL checkpoint, process cleanup, /health verification | None | **DONE** |
| 2 | Backend Test Suite 100% Pass Rate | Pathing configuration, GPT2Adapter mock tokens, AI scientist error guard, test fixture alignment | M1 | **DONE** |
| 3 | Loophole Remediation & Error Hardening | gpt2_engine.infer(), dispatcher input validation, StorageError handler, frontend CSP/error handling | M2 | **DONE** |
| 4 | Dedicated Regression Suite & Upgrades | test_remediation_regression.py (9 tests), full stack 244/244 test suite | M3 | **DONE** |
| 5 | Live Full-Stack Verification & Forensic Audit | E2E service validation, forensic audit, prompt-cache bug discovered & fixed | M4 | **DONE** |

## Feature Inventory
| # | Feature | Description | Milestone |
|---|---------|-------------|-----------|
| 1 | Canonical Entry Point & Health Endpoint | backend/main.py canonical, root main.py aligned, GET /health → `{"status":"healthy"}` on port 8000 | M1 |
| 2 | FastAPI Lifespan & Clean Resource Lifecycle | `@asynccontextmanager` lifespan, SQLite WAL truncation on shutdown, Windows tree-kill in scripts | M1 |
| 3 | Frontend Build & Vite Dev Proxy Verification | Clean `npm run build:renderer` (1858 modules), Vite dev proxying verified | M1 |
| 4 | Pytest Collection & Pathing Fix | `pythonpath = . backend` in pytest.ini, REPO_ROOT injected in conftest.py | M2 |
| 5 | Mock Token Coverage in GPT2Adapter | Regex-based IOI name-pair mock token resolution so IOI pipeline passes deterministically | M2 |
| 6 | AI Scientist Defensive Key Handling | `val_res.get("confidence") or {}` guard against KeyError when validation returns status:unavailable | M2 |
| 7 | Discovery Lifecycle & Evidence Policy Test Alignment | Full orchestration through Publication; test fixtures updated with live provenance flags | M2 |
| 8 | GPT2 Engine `infer()` Implementation | Live forward-pass `infer(prompt, model_name)` returning tokens, attention maps, neuron activations | M3 |
| 9 | Dispatcher Payload Validation | HTTP 400 on non-string prompt; type-validated layer/head/seq_len across gpt2 endpoints | M3 |
| 10 | StorageError & Exception Handlers | Custom FastAPI handlers: StorageError→400, ValueError→400, TypeError→400 | M3 |
| 11 | Frontend CSP & Error Hardening | CSP meta tag in index.html; structured `detail`/`error` extraction in api.ts | M3 |
| 12 | Dedicated Regression Suite | `tests/pytest/test_remediation_regression.py` — 9 tests covering all remediation fixes | M4 |
| 13 | Full Stack Test Verification | 244/244 backend pytest passed; 119/119 frontend Vitest passed | M4 |
| 14 | **Prompt-Cache Bug Fix** | `/api/gpt2/attention_head` and `/api/gpt2/activations` were returning stale-cache data regardless of the submitted prompt. Fixed by calling `run_prompt(prompt)` before reading cache. Two regression tests added. | M5 |
| 15 | Live Full-Stack E2E Verification | 500-request /health burst (500/500 healthy @ ~500 req/s); live infer, run_prompt, attention_head, activations, models all verified with `provenance: live` | M5 |

## Interface Contracts
### Client ↔ Backend API
- `GET /health` → `{"status": "healthy"}` (HTTP 200)
- `POST /api/infer` → live tokens, attention maps, neuron activations; `400` on non-string prompt
- `POST /api/gpt2/run_prompt` → top-5/16 predictions, next token, provenance:live; `400` on null/non-string prompt
- `POST /api/gpt2/attention_head` → prompt-specific attention matrix (cache refreshed per request); `400` on invalid layer/head types
- `POST /api/gpt2/activations` → prompt-specific residual/MLP shapes (cache refreshed per request); `400` on invalid layer/seq_len types
- `POST /api/gpt2/patch_head` → logit-diff patching result; `400` on invalid layer/head types
- `POST /api/experiments` → `{"status": "saved", "id": str}` or `400 {"detail": str, "error_type": "StorageError"}`
- All error responses: `{"detail": "<message>", "status_code": <int>}`

### Server Lifecycle Contract
- `@asynccontextmanager` lifespan on Uvicorn startup/shutdown.
- Shutdown calls `PRAGMA wal_checkpoint(TRUNCATE)` to release SQLite WAL.
- Electron/dev scripts use `taskkill /T /F /PID <pid>` on Windows to eliminate orphaned sub-processes.

## Code Layout
- `backend/main.py` — canonical server entry point, CORS, lifespan handler, router mounting, exception handlers
- `main.py` — root entry point, re-exports `backend.main:app`, binds `127.0.0.1:8000`
- `backend/api/dispatcher.py` — API routes, payload validation, prompt-cache priming
- `backend/services/gpt2_engine.py` — GPT-2 Small transformer inference, `infer()`, `run_prompt()`, `attention_head()`, `activations()`
- `backend/storage/database.py` — SQLite persistence, `checkpoint_wal()`
- `frontend/src/` — Vue 3 application source
- `frontend/electron/` — Electron wrapper and main process
- `tests/pytest/` — Backend pytest suite (244 tests)
- `tests/pytest/test_remediation_regression.py` — Dedicated regression suite (9 tests)
- `tests/pytest/test_challenger_m1_adversarial.py` — Adversarial/chaos suite (22 tests)
- `frontend/tests/vitest/` — Frontend Vitest suite (119 tests)

## Audit Log
| Milestone | Event | Result |
|-----------|-------|--------|
| M1 | 500-req /health burst | 606.9 req/s, 500/500 HTTP 200 — ✅ CLEAN |
| M2 | Full pytest suite | 219/219 passed — ✅ CLEAN |
| M2 | Frontend Vitest suite | 119/119 passed — ✅ CLEAN |
| M2 | Frontend build | 1858 modules, 0 errors — ✅ CLEAN |
| M3 | Regression suite (7 tests) | 7/7 passed — ✅ CLEAN |
| M4 | Full pytest suite | 242/242 passed — ✅ CLEAN |
| M5 | Live API probe | infer, run_prompt, attention_head, activations, models — all `provenance: live` ✅ |
| M5 | **Forensic finding: prompt-cache stale-read** | `attention_head` & `activations` returned prior-run cache regardless of submitted prompt. Fixed in `dispatcher.py`. 2 regression tests added. |
| M5 | Full pytest suite post-fix | 244/244 passed — ✅ CLEAN |
| M5 | Live prompt-cache fix verification | Different prompts → different tokens ✅; seq length scales with prompt length ✅ |
