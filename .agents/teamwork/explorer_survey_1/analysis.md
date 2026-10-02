# MECH Backend Architecture & Server Runtime Survey Report

**Author**: `explorer_survey_1`  
**Date**: 2026-09-27  
**Working Directory**: `c:\Users\himan\OneDrive\Documents\Default Project\MECH\.agents\teamwork\explorer_survey_1`  
**Scope**: Full survey of MECH Research Platform backend architecture, server entry points, routers, models/adapters, persistence, startup mechanisms, `/health` endpoint, lifecycle management, vulnerabilities, and error handling.

---

## 1. Executive Summary

The MECH Research Platform is a mechanistic interpretability desktop research environment providing transformer architecture inspection, causal attention head patching, Indirect Object Identification (IOI) circuit analysis, autonomous research agents (Research Society v2), and SQLite-backed workspace persistence.

### Key Architectural Findings:
1. **Divergent Server Entry Points**: Two competing entry points exist: `main.py` (repository root) and `backend/main.py`.
   - `backend/main.py` is the actual target used by Electron (`frontend/electron/main.js`) and the dev script (`frontend/scripts/dev.js`). It binds to `127.0.0.1:8000` and mounts the dispatcher at both `/api` and `/api/v1`.
   - Root `main.py` binds to `0.0.0.0:8000`, mounts only `/api`, has different metadata ("MECH Research Platform" v2.0 vs "MECH Platform" v2.0.0), and breaks contract tests in `tests/pytest` that target `/api/v1/*`.
2. **Missing Runtime v2 API Module**: Both `main.py` and `backend/main.py` contain `try...except` blocks attempting to import `backend.api.runtime_api`. No such file exists in `backend/api/`; it was either removed or never created.
3. **Health Check Endpoint**: Both entry points implement `GET /health` returning `{"status": "healthy"}` on port 8000 with HTTP 200, cleanly satisfying Acceptance Criteria.
4. **Lifecycle & Resource Management Deficits**:
   - The backend uses deprecated `@app.on_event("startup")` hooks and lacks any shutdown handler (`@app.on_event("shutdown")` or modern FastAPI `lifespan` context manager).
   - Background research tasks (`/api/society/run`) run on untracked daemon threads (`threading.Thread(daemon=True)`).
   - Process termination in Electron/dev scripts uses `child.kill()`, which on Windows does not terminate child process trees, leaving orphaned processes holding port 8000 and locking SQLite database files.
5. **Security & Unhandled 500 Crash Risks**:
   - `backend/services/gpt2_engine.py` lacks an `infer()` method, which will trigger an unhandled `AttributeError` -> HTTP 500 crash whenever `/api/infer` is called with live weights active.
   - Endpoint payload parsing in `backend/api/dispatcher.py` directly executes `int(...)`, `float(...)`, `.split()`, or `.replace()` without Pydantic schemas or try/except blocks, causing unhandled 500 crashes on malformed/type-mismatched requests.
   - `DesktopStorage` raises `StorageError` on invalid payloads; without a custom FastAPI exception handler, this crashes with HTTP 500 instead of returning HTTP 400.
   - CORS is configured with `allow_origins=["null", "file://", ...]` alongside `allow_credentials=True`.
6. **Backend Test Suite Baseline**:
   - Executing `pytest tests/pytest` in the current environment results in **177 passed, 13 failed, 7 warnings** across 190 tests (105s duration). Failures stem from strict live-provenance gating in `TraceableEvidenceGraph`, mock adapter token coverage in `IOIReproductionPipeline`, and unhandled dictionary keys in `AIScientistEngine`.

---

## 2. FastAPI Backend Architecture & Component Breakdown

### 2.1 Entry Points Comparison

| Dimension | Repository Root: `main.py` | Dedicated Backend: `backend/main.py` |
|---|---|---|
| **Primary Consumer** | Ad-hoc manual launch | Electron (`electron/main.js`), Dev script (`scripts/dev.js`), `test_protocol.py` |
| **App Title & Version** | "MECH Research Platform", version="2.0" | "MECH Platform", version="2.0.0" |
| **Host Binding** | `0.0.0.0` (exposes to local subnet) | `127.0.0.1` (loopback only) |
| **Default Port** | 8000 | 8000 |
| **Router Prefixes** | `/api` | `/api` AND `/api/v1` |
| **Startup Behavior** | Eagerly preloads `gpt2_engine` via `asyncio.to_thread` | Deferred/lazy model loading ("ML model modules load on demand") |
| **CORS Origins** | Configurable via `MECH_CORS_ORIGINS` env var | Hardcoded origin list; ignores `MECH_CORS_ORIGINS` |

**Architectural Recommendation**: Deprecate root `main.py` or harmonize it as a forwarding shim to `backend/main.py` so there is a single, authoritative FastAPI application definition.

### 2.2 Router & API Dispatching Architecture

1. **Active HTTP Dispatcher (`backend/api/dispatcher.py`)**:
   - Mounts routes under `/api` and `/api/v1`.
   - Core endpoints:
     - Health & Metadata: `GET /status`, `GET /portal/summary`, `GET /health` (root app).
     - Model Introspection: `GET /models`, `POST /models/load`, `GET /models/{name}`, `POST /infer`.
     - Benchmarking: `GET /benchmarks`, `POST /benchmarks/run`, `GET /research_catalog`.
     - Persistence: `GET /experiments`, `POST /experiments`, `DELETE /experiments/{item_id}`, `GET /sessions`, `POST /sessions`, `DELETE /sessions/{item_id}`.
     - Circuits & Knowledge: `GET /discoveries`, `GET /circuits`, `GET /circuits/{circuit_id}`, `GET /knowledge-graph`, `GET /interpretability/inspectors`.
     - Dynamic GPT-2 Engine: `POST /gpt2/load`, `POST /gpt2/run_prompt`, `POST /gpt2/activations`, `POST /gpt2/attention_head`, `POST /gpt2/patch_head`, `POST /gpt2/ioi`, `POST /gpt2/architecture`, `POST /gpt2/layer`, `POST /gpt2/neurons`, `POST /gpt2/neuron`, `POST /gpt2/head`, `POST /gpt2/patch_neuron`, `POST /gpt2/layer_activations`, `POST /gpt2/logit_lens_all`, `GET /figures/attention`.
     - Research Society v2: `POST /society/run`, `GET /society/runs`, `GET /society/runs/{run_id}`, `GET /society/stream` (SSE streaming via `StreamingResponse`).
2. **Legacy Dispatcher (`backend/api/legacy_dispatcher.py`)**:
   - Provides `build_dispatcher()` returning a dictionary of callable RPC-style handlers (e.g. `ping`, `info`, `echo`, `add`, `time`, `runtime:status`).
   - Retained exclusively for backward compatibility with `tests/pytest/test_api.py`.
3. **Missing v2 Runtime Router**:
   - Both entry points attempt `from backend.api.runtime_api import router as runtime_router`.
   - No `backend/api/runtime_api.py` exists; execution silently falls through `except Exception: pass`.

### 2.3 Model Runtime Adapters & Inference Architecture

- **Adapter Registry (`backend/science/models/adapter_registry.py`)**:
  - `ModelAdapterRegistry` registers:
    - GPT-2 family: `gpt2`, `gpt2-small`, `gpt2-medium`, `gpt2-large` via `GPT2Adapter`.
    - TransformerLens: `tl-gpt2`, `tl-gpt2-small`, `tl-gemma-2b` via `TransformerLensAdapter`.
    - Gemma family: `gemma-2b`, `gemma-7b` via `GemmaAdapter`.
    - LLaMA family: `tinyllama`, `llama-3-8b`, `llama-3-70b` via `LlamaAdapter`.
    - Qwen family: `qwen-2-1.5b`, `qwen-2-7b`, `qwen-2.5-7b` via `QwenAdapter`.
    - Mistral family: `mistral-7b`, `mixtral-8x7b` via `MistralAdapter`.
    - DeepSeek family: `deepseek-r1-1.5b`, `deepseek-v2-7b` via `DeepSeekAdapter`.
  - Supports `mock_mode=True` to run offline without GPU/weights for fast test execution.
- **Dynamic GPT-2 Engine (`backend/services/gpt2_engine.py`)**:
  - Direct PyTorch + HuggingFace Transformers loader for `gpt2`.
  - Configures `output_attentions=True`, `output_hidden_states=True`.
  - Dynamically inspects layer parameters, MLP dimensions (`d_mlp`), attention matrices, and neuron activations.
  - **Defect Detected**: Line 125 of `backend/api/dispatcher.py` calls `engine.infer(prompt, model_name)`, but `backend/services/gpt2_engine.py` does NOT implement `infer()`. It implements `run_prompt()`, `attention_head()`, `activations()`, `patch_head()`, `ioi()`, etc. Calling `POST /api/infer` when the engine is loaded will raise `AttributeError`.

### 2.4 Data Persistence & Database Files

1. **SQLite Desktop Storage (`backend/storage/database.py`)**:
   - Class: `DesktopStorage`.
   - Default DB Path: `backend/storage/mech.db` (configurable via `MECH_STORAGE_DB`).
   - Journal Mode: `PRAGMA journal_mode = WAL;` (Write-Ahead Logging enabled for concurrency).
   - Schema Tables:
     - `settings (key TEXT PRIMARY KEY, value TEXT NOT NULL, updated_at TEXT NOT NULL)`
     - `recent_projects (id INTEGER PRIMARY KEY AUTOINCREMENT, path TEXT UNIQUE, name TEXT, opened_at TEXT)`
     - `recent_files (id INTEGER PRIMARY KEY AUTOINCREMENT, path TEXT UNIQUE, project_path TEXT, opened_at TEXT)`
     - `experiments (item_id TEXT PRIMARY KEY, payload TEXT NOT NULL, created_at TEXT NOT NULL)`
     - `sessions (item_id TEXT PRIMARY KEY, payload TEXT NOT NULL, created_at TEXT NOT NULL)`
   - Connection handling: Every method opens a scoped connection with `with self._connect() as connection:`, ensuring auto-commit and closure.
2. **Evidence Graph Records (`backend/core/evidence_graph.py`)**:
   - Saves persistent run records to JSON files in `backend/storage/evidence/` (override via `MECH_EVIDENCE_DIR`).
   - Functions `save_run_record`, `load_run_record`, `list_run_records` maintain durability across server restarts.

---

## 3. Server Startup Mechanisms, Configuration & `/health` Endpoint

### 3.1 Startup Orchestration Workflows

1. **Electron Production & Packaged Mode (`frontend/electron/main.js`)**:
   - `bootstrap()` initializes Electron local SQLite storage, calls `startBackend(logger)`, registers IPC handlers, and opens `BrowserWindow`.
   - `startBackend()`:
     - First probes `http://127.0.0.1:8000/health` with a 2-second HTTP GET. If statusCode < 500, it logs `backend_reuse` and avoids spawning a duplicate process.
     - If not running, resolves Python binary via `resolvePythonPath` (checking `MECH_PYTHON`, `PYTHON_PATH`, bundled `.venv/Scripts/python.exe`, repo `.venv`, and system PATH).
     - Spawns `scriptPath = path.join(repoRoot, 'backend', 'main.py')` with `PYTHONPATH` set to repo root + backend, and `PYTHONUNBUFFERED=1`.
     - Polls `http://127.0.0.1:8000/health` every 500ms with a 180-second timeout until healthy.
2. **Developer Environment Startup (`frontend/scripts/dev.js`)**:
   - Run via `npm run dev:electron` (or `npm run dev`).
   - `ensureBackend()`:
     - Checks `isPortOpen(BACKEND_PORT)` via raw TCP socket. If port 8000 is open, it reuses it.
     - If not open, searches for a Python interpreter with `torch` and `transformers` installed.
     - Spawns `backend/main.py`.
     - Waits up to 60s for port 8000 to accept connections.
     - Waits for Vite dev server (`http://localhost:5173`) and launches Electron.
3. **Manual CLI Startup**:
   - Executing `python backend/main.py` launches Uvicorn on `http://127.0.0.1:8000` with `timeout_keep_alive=600`.

### 3.2 Port 8000 Configuration & `/health` Implementation

- **Port 8000 Compliance**:
  - `backend/main.py`: `uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=False, timeout_keep_alive=600)`
  - `main.py`: `uvicorn.run("main:app", host="0.0.0.0", port=8000, ...)`
  - `frontend/electron/main.js`: `const BACKEND_PORT = 8000;`
  - `frontend/scripts/dev.js`: `const BACKEND_URL = process.env.BACKEND_URL || 'http://localhost:8000';`
- **Health Check Endpoint**:
  - Defined in `backend/main.py:50`:
    ```python
    @app.get("/health")
    def health():
        return {"status": "healthy"}
    ```
  - Returns HTTP 200 with JSON payload `{"status": "healthy"}`.
  - Passes both unit tests (`tests/pytest/test_protocol.py`, `tests/pytest/test_advanced_evals.py`) and automation scripts (`automate_api.py`).

---

## 4. Lifecycle Management & Resource Cleanup

### 4.1 Startup and Shutdown Handlers
- **Current State**:
  - `backend/main.py:39`:
    ```python
    @app.on_event("startup")
    async def _startup_probe():
        logger.info("Backend ready. ML model modules will load on demand.")
    ```
  - The `@app.on_event("startup")` decorator is deprecated in FastAPI >= 0.93.0 and Starlette, producing runtime deprecation warnings during test execution.
  - **Zero Shutdown Handlers**: Neither `backend/main.py` nor `main.py` contains any `@app.on_event("shutdown")` or `@asynccontextmanager async def lifespan(app: FastAPI):` context manager.

### 4.2 Background Worker Threads & Lifecycle Risks
- In `backend/api/dispatcher.py:887`:
  ```python
  worker = threading.Thread(
      target=_society_worker, args=(run_id, goal, model_name), daemon=True
  )
  worker.start()
  ```
- **Risks**:
  1. Daemon threads are terminated unconditionally when the main Python process exits. Any ongoing model computation, file I/O, or SQLite write will be abruptly aborted mid-stream.
  2. No cancellation endpoint exists (`POST /society/runs/{run_id}/cancel` is absent), preventing clients from stopping runaway GPU workloads.
  3. No task tracking or registry exists to monitor or wait for active worker threads during server shutdown.

### 4.3 Process Termination & Orphaned Process Loophole
- In `frontend/electron/main.js:164`:
  ```javascript
  async function stopBackend(child, logger) {
    if (!child || child.killed) return;
    try {
      child.kill();
    } catch (e) {
      logger.error('backend_kill_error', { error: e.message });
    }
  }
  ```
- In `frontend/scripts/dev.js:105`:
  ```javascript
  const killBackend = () => { if (backend) { try { backend.kill(); } catch (e) {} } };
  process.on('exit', killBackend);
  process.on('SIGINT', () => { killBackend(); process.exit(0); });
  process.on('SIGTERM', () => { killBackend(); process.exit(0); });
  ```
- **Loophole on Windows**:
  - Calling Node's `child.kill()` on Windows issues a `TerminateProcess` call to the immediate child PID.
  - If Python launches child processes (or if `uvicorn` uses worker subprocesses or reloaders), `child.kill()` terminates only the parent wrapper, leaving worker child processes running as **orphaned background processes**.
  - Orphaned Python processes remain bound to port 8000, preventing subsequent server restarts with `[Errno 10048] error while attempting to bind on address ('127.0.0.1', 8000): only one usage of each socket address is normally permitted`.
  - **Remediation**: Use `taskkill /pid <PID> /T /F` on Windows to kill the entire process tree, or attach the child to a Windows Job Object configured with `JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE`.

### 4.4 Database Lock & WAL Journal Cleanup
- `DesktopStorage` configures `PRAGMA journal_mode = WAL;`.
- WAL mode creates `mech.db-wal` and `mech.db-shm` files.
- On abrupt termination, unflushed WAL logs can leave file locks on Windows filesystems, preventing deletion or migration until all handles close.
- A proper shutdown handler must execute `PRAGMA wal_checkpoint(TRUNCATE);` to flush WAL frames back to `mech.db` and truncate WAL files cleanly.

---

## 5. Potential Vulnerabilities, Unhandled Exceptions & Input Validation Audit

### 5.1 Unhandled Exceptions & 500 Server Crashes

| Endpoint | File & Line | Trigger / Loophole | Result | Recommended Fix |
|---|---|---|---|---|
| `POST /api/infer` | `dispatcher.py:125` | `engine.infer(prompt, model_name)` called when `gpt2_engine` loaded | `AttributeError: module 'gpt2_engine' has no attribute 'infer'` (500 crash) | Implement `infer` in `gpt2_engine.py` or delegate to `run_prompt` + token parsing |
| `POST /api/infer`, `POST /api/gpt2/run_prompt`, `POST /api/runtime/analyze_tokens` | `dispatcher.py:137, 387, 482` | Non-string `prompt` (e.g. `{"prompt": 123}`) | `AttributeError: 'int' object has no attribute 'split' / 'replace'` (500 crash) | Validate `prompt` type with Pydantic model (`str`) or explicit `isinstance(prompt, str)` check |
| `POST /api/gpt2/activations`, `POST /api/gpt2/attention_head`, `POST /api/gpt2/layer`, etc. | `dispatcher.py:505, 521, 609, etc.` | Non-numeric parameter (e.g. `{"layer": "abc"}`) | `ValueError: invalid literal for int() with base 10` (500 crash) | Use Pydantic request models with type validation, returning HTTP 422 |
| `POST /api/experiments`, `POST /api/sessions` | `dispatcher.py:265, 285` & `database.py:265` | Invalid item ID (e.g. `{"id": ""}` or non-string) | Unhandled `StorageError` raised by `DesktopStorage` (500 crash) | Add FastAPI `@app.exception_handler(StorageError)` returning HTTP 400 |
| `_handle_v2_run_campaign` | `legacy_dispatcher.py:246` & `ai_scientist_engine.py:87` | Validation engine returns `{"status": "unavailable"}` without `"confidence"` | `KeyError: 'confidence'` in `run_scientific_campaign` (500 crash) | Safely access `val_res.get("confidence")` with null fallback |

### 5.2 Missing Global Exception Handling
Neither `backend/main.py` nor `main.py` configures global exception handlers. Any uncaught Python exception results in Starlette's default 500 Internal Server Error with HTML or unformatted text, rather than a structured JSON error response:
```json
{
  "status": "error",
  "error": "Error message",
  "detail": "Detailed context"
}
```
Adding `@app.exception_handler(Exception)` and `@app.exception_handler(StorageError)` will eliminate unhandled 500 crashes and satisfy Acceptance Criteria.

### 5.3 CORS Configuration Vulnerabilities
- `backend/main.py:25-36` configures:
  ```python
  allow_origins=[
      "http://localhost:5173",
      "http://127.0.0.1:5173",
      "http://localhost:3000",
      "http://127.0.0.1:3000",
      "null",
      "file://",
  ],
  allow_credentials=True,
  ```
- **Security Assessment**:
  - `allow_credentials=True` combined with `"null"` in `allow_origins` is flagged by security scanners as a high-severity CORS misconfiguration. In standard web browsers, a sandboxed iframe sends `Origin: null`. An attacker-controlled web page hosting `<iframe sandbox="allow-scripts" src="...">` can issue credentialed requests to `http://127.0.0.1:8000`.
  - While Electron requires `file://` or local origins for packaged apps, dev/web environments should separate desktop origin allowances from browser requests.
  - `backend/main.py` ignores the `MECH_CORS_ORIGINS` environment variable, preventing environment-specific origin customization.

---

## 6. Test Suite Baseline & Failure Diagnostics

Running `pytest tests/pytest` against the current codebase yielded **177 passed, 13 failed, 7 warnings** (105.15s).

### Diagnosed Failure Root Causes:
1. **`test_evidence_persistence.py` (3 failures)**:
   - Tests: `test_from_run_has_no_demo_nodes`, `test_kg_writeback_compounds`, `test_kg_writeback_never_fails_run`.
   - Root Cause: `TraceableEvidenceGraph._step_allows_evidence()` enforces live provenance via `discovery_is_live()`. The test fixtures pass synthetic trace dictionaries lacking `"provenance": "live"`, causing the evidence parser to drop discovery/validation nodes (resulting in 9 nodes instead of 11).
2. **`test_science_reproducibility.py` (2 failures)**:
   - Tests: `test_ioi_pipeline_runs`, `test_ioi_pipeline_generates_manifest`.
   - Root Cause: In mock mode, `IOIReproductionPipeline` calls `GPT2Adapter.get_logits(p["text"])`. The mock adapter returns arbitrary tokens that do not contain the target names (`Alice`, `Bob`), causing the pipeline to exit early with `status: unavailable`, omitting `observed_metrics` and `manifest_id` from the return dictionary.
3. **`test_validation_loop.py` (1 failure)**:
   - Test: `test_reproduce_live_report_and_gate`.
   - Root Cause: In live mode, `Critic.reproduce("ioi")` calculates `circuit_minimality` as `0.9`, but the test assertion states `assert minimal["observed_value"] == 0.0`.
4. **`test_sprint5_deliverable.py` (1 failure)**:
   - Test: `test_sprint5_ai_scientist_end_to_end_deliverable`.
   - Root Cause: `AIScientistEngine.run_scientific_campaign` accesses `val_res["confidence"]["confidence_score"]` without checking if `confidence` exists in `val_res`. When `ScientificValidationEngine` fails closed, it returns `{"status": "unavailable", "reason": ...}` without a `confidence` key, raising `KeyError: 'confidence'`.
5. **Sprint Deliverable Tests (6 failures)**:
   - Tests: `test_interpretability_sprint3.py` (2), `test_interpretability_sprint4.py` (1), `test_sprint2_deliverable.py` (1), `test_sprint3_deliverable.py` (1), `test_sprint4_deliverable.py` (1).
   - Root Cause: Cascading effects of the strict fail-closed validation policy and mock token mismatch in reproducibility pipelines.

---

## 7. Actionable Hardening & Architecture Plan

1. **Unify Entry Points**:
   - Make `backend/main.py` the single source of truth for the FastAPI application.
   - Update root `main.py` to import and re-export `app` from `backend.main` or forward requests cleanly.
   - Ensure consistent name ("MECH Platform"), version ("2.0.0"), and routing (`/api` and `/api/v1`).
2. **Modernize Lifespan & Process Termination**:
   - Replace deprecated `@app.on_event("startup")` with `@asynccontextmanager async def lifespan(app: FastAPI):`.
   - In lifespan shutdown: checkpoint SQLite WAL (`PRAGMA wal_checkpoint(TRUNCATE)`), log shutdown, and join or cancel active society background threads.
   - Update `frontend/electron/main.js` and `frontend/scripts/dev.js` to terminate Windows process trees cleanly (`taskkill /pid ... /T /F`).
3. **Resolve API Implementation & Validation Loopholes**:
   - Implement `infer()` in `backend/services/gpt2_engine.py` to prevent `AttributeError` 500 crashes.
   - Add Pydantic request models or robust type conversion wrappers for all numeric and string endpoint parameters in `backend/api/dispatcher.py`.
   - Register a global exception handler for `StorageError` returning HTTP 400 Bad Request, and a catch-all handler returning structured JSON 500 responses.
   - Fix dictionary lookups in `AIScientistEngine` (`val_res.get("confidence", {})`) to prevent unhandled `KeyError` crashes.
4. **Fix Test Suite Regressions**:
   - Align mock adapter responses in `GPT2Adapter` so prompt subject/object tokens are present during mock IOI runs.
   - Update `TraceableEvidenceGraph` and `test_evidence_persistence.py` test fixtures to properly assign live provenance where appropriate.
