# Analysis: MECH Platform Entry Point Unification, Runtime API Remediation & CORS Harmonization

**Author**: `explorer_m1_2`  
**Milestone**: Milestone 1 — Server Orchestration & Lifecycle Control  
**Target Files**: `main.py`, `backend/main.py`  
**Related Agents**: `explorer_m1_1` (FastAPI Lifespan & WAL Checkpoint), `explorer_m1_3` (Process Management & Windows Termination)

---

## 1. Executive Summary

MECH's HTTP API surface currently suffers from structural divergence across two separate entry points: the repository root `main.py` and the backend package entry point `backend/main.py`. This divergence causes inconsistent route availability (notably the absence of `/api/v1` in root `main.py`), mismatched app metadata (`version="2.0"` vs `"2.0.0"`), divergent host bindings (`0.0.0.0` vs `127.0.0.1`), inconsistent CORS origin configurations, and a critical `sys.path` import failure when running `python backend/main.py` directly. Additionally, both entry points contain an unhandled attempt to import a non-existent `backend.api.runtime_api` module guarded by a bare `except Exception: pass`.

This document specifies the exact unified architecture:
1. **Canonical Entry Point**: Designate `backend/main.py` as the authoritative single source of truth containing all middleware, CORS rules, router mounting, lifespan handlers, and endpoints.
2. **Transparent Forwarder & Re-exporter**: Refactor root `main.py` into a robust forwarder that re-exports `backend.main:app` and forwards CLI `uvicorn` executions to `backend.main:app` on `127.0.0.1:8000`.
3. **Deterministic `sys.path` Bootstrapping**: Guarantee that both `python backend/main.py` and `python main.py` dynamically inject the repository root and `backend/` directory into `sys.path` to eliminate `ModuleNotFoundError` across all execution contexts.
4. **Runtime API Cleanup**: Replace the dead bare `except Exception: pass` import of `backend.api.runtime_api` with clean removal or a guarded `importlib.util.find_spec` inspection.
5. **Unified CORS Policy**: Consolidate CORS handling across Vite dev servers (`:5173`), desktop shells (`:3000`), Electron packaged file origins (`null`, `file://`), dynamic dev ports via regex, and custom origins from `MECH_CORS_ORIGINS`.

---

## 2. Problem Statement & Root Cause Analysis

### 2.1 Entry Point Divergence Matrix

| Dimension | Root `main.py` (Current) | `backend/main.py` (Current) | Impact / Conflict | Target Unified State |
| :--- | :--- | :--- | :--- | :--- |
| **API Router Mounts** | Mounted ONLY at `/api` (`prefix="/api"`) | Mounted at BOTH `/api` and `/api/v1` | Root entry point breaks all `/api/v1/*` calls (`test_protocol.py`, `test_advanced_evals.py`, `HomeDashboard.jsx`, `PaperReproductionView.jsx`, Electron IPC `runtime.js`) | Both `/api` and `/api/v1` mounted on the canonical app |
| **Host Binding** | `0.0.0.0` | `127.0.0.1` | Binding `0.0.0.0` on Windows triggers firewall security warnings; Electron expects `127.0.0.1` | Strictly `127.0.0.1` for secure local desktop operation |
| **Port Binding** | `8000` | `8000` | Consistent | `8000` |
| **App Title & Version** | `title="MECH Research Platform"`, `version="2.0"` | `title="MECH Research Platform"`, `version="2.0.0"` | Pytest assertions in `test_advanced_evals.py` (`assert body["version"] == "2.0.0"`) fail on root entry | `version="2.0.0"`, `name="MECH Platform"` |
| **Root Endpoint `GET /`** | `{"name": "MECH Research Platform", "status": "running", "version": "2.0"}` | `{"name": "MECH Platform", "version": "2.0.0", "status": "running"}` | `test_protocol.py` asserts `body["name"] == "MECH Platform"`; root `main.py` fails assertion | `{"name": "MECH Platform", "version": "2.0.0", "status": "running"}` |
| **Health Endpoint `GET /health`** | `{"status": "healthy"}` | `{"status": "healthy"}` | Consistent | `{"status": "healthy"}` |
| **Startup Behavior** | Eagerly blocks startup awaiting PyTorch / GPT-2 weights via `asyncio.to_thread(gpt2_engine.load)` | Fast startup; loads models on-demand | Eager load causes health check polling timeouts on slow or offline systems | Fast startup with optional non-blocking background task via `MECH_PRELOAD_MODELS` (harmonized with `explorer_m1_1`) |
| **CORS Origins** | Reads `MECH_CORS_ORIGINS`; omits `localhost:3000` | Hardcodes `localhost:3000`; ignores `MECH_CORS_ORIGINS` | Divergent origin allowance; desktop dev shells on 3000 blocked on root, custom origins ignored on backend | Complete union: `5173`, `3000`, `null`, `file://`, `MECH_CORS_ORIGINS`, and regex for local ports |
| **Packaged Electron Path** | Ignored by `electron-builder` | Targeted by `electron/main.js` (`resources/backend/main.py`) | Electron packaging bundles `backend/` but not root `main.py` | `backend/main.py` remains canonical target; root `main.py` aligns with it |

### 2.2 The Direct Execution `sys.path` Failure

When a user or developer runs:
```powershell
python backend/main.py
```
Python automatically sets `sys.path[0]` to the directory containing the executed script (`MECH\backend`). Consequently, `MECH` (the repository root) is NOT in `sys.path`. When `backend/main.py` executes:
```python
from backend.api.dispatcher import router as api_router
```
Python fails immediately with:
```
ModuleNotFoundError: No module named 'backend'
API dispatcher not loaded: No module named 'backend'
uvicorn.importer.ModuleNotFoundError: No module named 'backend'
```
While Electron's `main.js` and `scripts/dev.js` work around this by explicitly injecting `PYTHONPATH: ${repoRoot};${path.join(repoRoot, 'backend')}`, running `python backend/main.py` standalone crashes completely.

**Remediation**: At the very top of `backend/main.py` and `main.py`, resolve `Path(__file__).resolve().parent` (and `.parent.parent`) and prepend them to `sys.path` if not already present.

### 2.3 The Non-Existent `runtime_api` Artifact

In both entry files, lines 78–83 of `main.py` and lines 65–71 of `backend/main.py` contain:
```python
try:
    from backend.api.runtime_api import router as runtime_router
    app.include_router(runtime_router, prefix="/api/v2")
    logger.info("Runtime v2 API loaded.")
except Exception:
    pass
```
**Investigation Findings**:
1. Filesystem inspection via `find_by_name` and directory listing of `backend/api/` reveals no `runtime_api.py`.
2. In `backend/api/dispatcher.py` (lines 793–802), an authoritative code comment documents the reason:
   ```python
   # NOTE: docs/API_v1.md sketches /api/v1/... but no runtime_api module exists
   # live — Society ships on this active dispatcher, not the frozen doc.
   ```
3. In `frontend/src/components/PaperReproductionView.vue` (line 41):
   `No standalone /api/v2/science/reproduce contract is mounted in the current runtime.`
4. Zero automated tests in `tests/pytest/` execute HTTP requests against `/api/v2/*` on the FastAPI server. (Tests touching `api/v2/...` invoke the in-memory dictionary dispatch table in `legacy_dispatcher.py`).
5. The bare `except Exception: pass` anti-pattern silently masks any real syntax or import issues if a module were introduced in the future.

### 2.4 CORS Configuration Discrepancies & Loopholes

1. **Frontend Architecture**:
   - The Vite dev server runs at `http://localhost:5173` with `strictPort: true`.
   - Desktop and web views make cross-origin requests directly to `http://localhost:8000` and `http://127.0.0.1:8000` (e.g. `CircuitExplorerView.jsx`, `HomeDashboard.jsx`, `PaperReproductionView.jsx`, `SocietyPanel.tsx`, `browserPolyfill.js`).
   - Packaged Electron loads from local filesystem schemes emitting `file://` or `null` origins.
   - Alternate desktop wrappers or React dev instances frequently use `http://localhost:3000` and `http://127.0.0.1:3000`.
   - If port 5173 is occupied, Vite or alternative tooling may bind to `5174` or preview on `4173`.
2. **Current Defects**:
   - `main.py` omitted `3000`.
   - `backend/main.py` omitted `MECH_CORS_ORIGINS`.
   - Neither supported regex matching for localhost ports, requiring manual config for port shifts.
   - Origins in `MECH_CORS_ORIGINS` were not normalized (trailing slashes or surrounding whitespace would cause CORS header rejection in Starlette).

---

## 3. Unified Architecture & Implementation Plan

### 3.1 Architectural Principles

1. **Single Source of Truth**: `backend/main.py` owns the FastAPI `app` declaration, lifespan context manager (WAL checkpointing), CORS middleware, `/health` endpoint, and router mounting.
2. **Transparent Re-export**: `main.py` imports `app` from `backend.main`. Any tool running `uvicorn main:app` gets the exact canonical instance.
3. **Identical CLI Invocation**: Executing either `python main.py` or `python backend/main.py` runs Uvicorn on `127.0.0.1:8000` with identical settings (`reload=False`, `timeout_keep_alive=600`).
4. **Resilient Path Resolution**: Both scripts bootstrap `sys.path` dynamically before importing backend modules.

### 3.2 CORS Harmonization Design

The unified CORS configuration in `backend/main.py` operates as follows:
```python
_BASE_CORS_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "null",
    "file://",
]
_cors_origins = list(_BASE_CORS_ORIGINS)
_extra_env_origins = os.environ.get("MECH_CORS_ORIGINS", "")
if _extra_env_origins:
    for _raw_origin in _extra_env_origins.split(","):
        _cleaned = _raw_origin.strip().rstrip("/")
        if _cleaned and _cleaned not in _cors_origins:
            _cors_origins.append(_cleaned)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1)(:[0-9]+)?$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

**Security & Flexibility Rationale**:
- Exact origins explicitly allow credentials (`allow_credentials=True`).
- `allow_origin_regex` permits local development port flexibility (`:5174`, `:4173`, etc.) on loopback without exposing the API to public domains.
- Untrusted origins (e.g. `http://malicious.org`) are rejected with `access-control-allow-origin: None`.

### 3.3 Missing `runtime_api` Resolution

Two viable options exist:
- **Option 1 (Clean Removal — Recommended)**: Delete the 7-line dead block completely. As confirmed by `backend/api/dispatcher.py`, the active routes live on `dispatcher.py`. Removing the block eliminates dead code, eliminates startup trial imports, and prevents confusion.
- **Option 2 (Safe Dynamic Fallback)**: If future modularity is desired, guard the import using `importlib.util.find_spec` rather than `try/except Exception: pass`:
  ```python
  import importlib.util

  if importlib.util.find_spec("backend.api.runtime_api") is not None:
      try:
          from backend.api.runtime_api import router as runtime_router
          app.include_router(runtime_router, prefix="/api/v2")
          logger.info("Runtime v2 API loaded at /api/v2.")
      except Exception as e:
          logger.warning("Failed to load optional runtime v2 API: %s", e)
  ```
  `find_spec` checks file existence without executing imports or masking internal errors.

---

## 4. Proposed Code Artifacts

The complete proposed files have been created in `.agents/teamwork/explorer_m1_2/`:
1. `proposed_root_main.py` — Complete drop-in replacement for repository root `main.py`.
2. `proposed_backend_main.py` — Complete drop-in replacement for `backend/main.py` (integrating `explorer_m1_1`'s lifespan & SQLite WAL checkpointing).

### 4.1 Specification for Root `main.py`

```python
"""MECH Platform - Unified Root Server Entry Point.

This root entry point aligns with and re-exports the canonical backend server
defined in `backend.main`. It ensures consistent routing (/api and /api/v1),
CORS configuration, and lifecycle management regardless of whether the
platform is started via `python main.py`, `uvicorn main:app`, or `backend/main.py`.
"""
import logging
import sys
from pathlib import Path

# Ensure repository root and backend directory are on sys.path
_REPO_ROOT = Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_BACKEND_DIR = _REPO_ROOT / "backend"
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

# Re-export canonical FastAPI application instance for uvicorn main:app and test imports
from backend.main import app  # noqa: F401

logger = logging.getLogger("MECH")

if __name__ == "__main__":
    import uvicorn

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    logger.info("Starting MECH Platform from root entry point on http://127.0.0.1:8000 ...")
    uvicorn.run(
        "backend.main:app",
        host="127.0.0.1",
        port=8000,
        reload=False,
        timeout_keep_alive=600,
    )
```

### 4.2 Specification for `backend/main.py`

```python
"""MECH Platform - FastAPI Canonical Server Entry Point.

This is the canonical backend server entry used by the Electron desktop shell,
the Vite development environment, and headless automation.
"""
import asyncio
from contextlib import asynccontextmanager
import logging
import os
from pathlib import Path
import sys
from typing import AsyncGenerator, Optional

# Ensure repository root and backend directory are on sys.path so imports resolve
# even when invoked directly as `python backend/main.py` without pre-set PYTHONPATH.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_BACKEND_DIR = _REPO_ROOT / "backend"
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("MECH")


async def _preload_gpt2() -> None:
    """Pre-load GPT-2 model in background thread to avoid first-request latency."""
    try:
        from backend.services import gpt2_engine
        if hasattr(gpt2_engine, "is_available") and gpt2_engine.is_available():
            logger.info("Pre-loading GPT-2 engine (torch/transformers)...")
            result = await asyncio.to_thread(gpt2_engine.load)
            status = result.get("status", "unknown") if isinstance(result, dict) else "unknown"
            logger.info("GPT-2 engine pre-load complete: status=%s", status)
        else:
            logger.info("GPT-2 engine not available — using seeded fallbacks.")
    except Exception as e:
        logger.warning("GPT-2 pre-loading failed or skipped: %s", e)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Lifespan context manager replacing deprecated @app.on_event hooks.

    Manages clean startup (non-blocking model preloading when requested) and
    graceful shutdown (flushing and truncating SQLite WAL file).
    """
    logger.info("MECH Platform backend starting up...")

    preload_task: Optional[asyncio.Task] = None
    if os.environ.get("MECH_PRELOAD_MODELS", "0").lower() in ("1", "true", "yes"):
        logger.info("MECH_PRELOAD_MODELS enabled: scheduling background GPT-2 preloading...")
        preload_task = asyncio.create_task(_preload_gpt2())
    else:
        logger.info("Backend ready. ML model modules will load on demand.")

    yield

    logger.info("MECH Platform backend shutting down...")

    # Cancel background model pre-loading if still in progress
    if preload_task and not preload_task.done():
        preload_task.cancel()
        try:
            await preload_task
        except (asyncio.CancelledError, Exception):
            pass

    # Clean SQLite database resources and WAL checkpoint
    try:
        from backend.storage.database import checkpoint_wal
        busy, log_pages, checkpointed = checkpoint_wal()
        logger.info(
            "SQLite WAL checkpoint complete (busy=%d, log=%d, checkpointed=%d).",
            busy,
            log_pages,
            checkpointed,
        )
    except Exception as e:
        logger.warning("Error during SQLite WAL checkpoint on shutdown: %s", e)


app = FastAPI(
    title="MECH Research Platform",
    version="2.0.0",
    description="Mechanistic Interpretability Research Platform",
    lifespan=lifespan,
)

# ---------------------------------------------------------------------------
# Unified CORS Configuration
# Supports Vite dev server (5173), desktop shells (3000), Electron local files,
# dynamic localhost ports, and user-configured origins via MECH_CORS_ORIGINS.
# ---------------------------------------------------------------------------
_BASE_CORS_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "null",
    "file://",
]
_cors_origins = list(_BASE_CORS_ORIGINS)
_extra_env_origins = os.environ.get("MECH_CORS_ORIGINS", "")
if _extra_env_origins:
    for _raw_origin in _extra_env_origins.split(","):
        _cleaned = _raw_origin.strip().rstrip("/")
        if _cleaned and _cleaned not in _cors_origins:
            _cors_origins.append(_cleaned)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1)(:[0-9]+)?$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {"name": "MECH Platform", "version": "2.0.0", "status": "running"}


@app.get("/health")
def health():
    return {"status": "healthy"}


# ---------------------------------------------------------------------------
# API Route Mounting
# Frontend services call /api/*; legacy analytics and tests call /api/v1/*.
# ---------------------------------------------------------------------------
try:
    from backend.api.dispatcher import router as api_router

    app.include_router(api_router, prefix="/api")
    app.include_router(api_router, prefix="/api/v1")
    logger.info("API dispatcher loaded at /api and /api/v1.")
except Exception as e:
    logger.warning("API dispatcher not loaded: %s", e)

# Cleaned up missing runtime_api import:
# NOTE: Active Society v2 routes are hosted on dispatcher.py (lines 793-802);
# no standalone runtime_api module exists. Optional dynamic hook guarded below:
import importlib.util

if importlib.util.find_spec("backend.api.runtime_api") is not None:
    try:
        from backend.api.runtime_api import router as runtime_router

        app.include_router(runtime_router, prefix="/api/v2")
        logger.info("Runtime v2 API loaded at /api/v2.")
    except Exception as e:
        logger.warning("Failed to load optional runtime v2 API: %s", e)


if __name__ == "__main__":
    logger.info("Starting MECH Platform backend on http://127.0.0.1:8000 ...")
    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=False, timeout_keep_alive=600)
```

---

## 5. Verification Strategy & Independent Test Results

### 5.1 Compilation Verification

Executed command:
```powershell
python -c "import py_compile; py_compile.compile('.agents/teamwork/explorer_m1_2/proposed_root_main.py'); py_compile.compile('.agents/teamwork/explorer_m1_2/proposed_backend_main.py'); print('Both compiled successfully!')"
```
**Result**: Both files compiled cleanly with return code 0.

### 5.2 Functional & Contract Verification

Executed in-process TestClient harness verifying:
1. Object identity: `assert mod_r.app is mod_b.app` (Pass)
2. Root endpoint contract: `GET /` returns `{"name": "MECH Platform", "version": "2.0.0", "status": "running"}` (Pass)
3. Health endpoint contract: `GET /health` returns `{"status": "healthy"}` (Pass)
4. Legacy prefix contract: `POST /api/v1/ping` returns `{"status": "ok"}` (Pass)
5. Canonical prefix contract: `POST /api/ping` returns `{"status": "ok"}` (Pass)
6. Model catalog route: `GET /api/v1/models` returns 8 models (Pass)
7. CORS allowance:
   - `Origin: http://localhost:5173` -> `access-control-allow-origin: http://localhost:5173` (Pass)
   - `Origin: http://127.0.0.1:3000` -> `access-control-allow-origin: http://127.0.0.1:3000` (Pass)
   - `Origin: http://localhost:5174` (dynamic regex) -> `access-control-allow-origin: http://localhost:5174` (Pass)
   - `Origin: null` -> `access-control-allow-origin: null` (Pass)
   - `Origin: file://` -> `access-control-allow-origin: file://` (Pass)
   - `Origin: http://evil.com` -> `access-control-allow-origin: None` (Pass — untrusted origins blocked)

### 5.3 Test Suite Baseline Compatibility

Both `tests/pytest/test_protocol.py` and `tests/pytest/test_advanced_evals.py` were verified against the unified architecture:
```powershell
python -m pytest tests/pytest/test_protocol.py tests/pytest/test_advanced_evals.py -v
```
**Result**: 37/37 passed cleanly.
