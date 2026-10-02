# Handoff Report — MECH Backend Architecture Survey

**Agent**: `explorer_survey_1`  
**Working Directory**: `c:\Users\himan\OneDrive\Documents\Default Project\MECH\.agents\teamwork\explorer_survey_1`  
**Handoff Type**: Hard (Investigation complete)  

---

## 1. Observation

### 1.1 Backend Entry Points Discrepancy
- In `backend/main.py` (lines 23–36, 56–71, 74–77):
  - Binds to `127.0.0.1:8000`: `uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=False, timeout_keep_alive=600)`
  - Mounts API router at both `/api` and `/api/v1`:
    ```python
    app.include_router(api_router, prefix="/api")
    app.include_router(api_router, prefix="/api/v1")
    ```
  - App metadata: `title="MECH Research Platform", version="2.0.0"`, root endpoint returns `{"name": "MECH Platform", "version": "2.0.0", "status": "running"}`.
- In root `main.py` (lines 24–32, 70–76, 85–94):
  - Binds to `0.0.0.0:8000`: `uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False, timeout_keep_alive=600)`
  - Mounts API router ONLY at `/api`:
    ```python
    app.include_router(api_router, prefix="/api")
    ```
  - Root endpoint returns `{"name": "MECH Research Platform", "status": "running", "version": "2.0"}`.
  - Startup hook eagerly pre-loads model: `@app.on_event("startup") async def _preload_gpt2_engine()`.
- In `frontend/electron/main.js` (lines 14–16, 114–120):
  - `const BACKEND_PORT = 8000; const BACKEND_HOST = '127.0.0.1';`
  - Spawns `scriptPath = path.join(repoRoot, 'backend', 'main.py');`
- In `frontend/scripts/dev.js` (lines 80–85):
  - Spawns `child = spawn(py.cmd, [...py.args, 'backend/main.py'], ...)`

### 1.2 Non-Existent Runtime API Module
- In both `main.py` (lines 78–83) and `backend/main.py` (lines 65–71):
  ```python
  try:
      from backend.api.runtime_api import router as runtime_router
      app.include_router(runtime_router, prefix="/api/v2")
      logger.info("Runtime v2 API loaded.")
  except Exception:
      pass
  ```
- File search via `find_by_name` for `*runtime_api*` returned zero results. Directory `backend/api/` contains only `dispatcher.py`, `legacy_dispatcher.py`, `project_export.py`, `project_validator.py`.

### 1.3 Health Endpoint
- In `backend/main.py` (lines 50–52) and `main.py` (lines 60–64):
  ```python
  @app.get("/health")
  def health():
      return {"status": "healthy"}
  ```
- Directly verified via in-process `TestClient` in `tests/pytest/test_protocol.py` (lines 37–40) and `tests/pytest/test_advanced_evals.py` (lines 41–44): both assert `r.status_code == 200` and `r.json() == {"status": "healthy"}`.

### 1.4 Lifecycle and Resource Cleanup
- In `backend/main.py` (line 39):
  - Contains `@app.on_event("startup")`.
  - Zero shutdown handlers (`@app.on_event("shutdown")` is absent).
  - Modern FastAPI `lifespan` context manager is absent. Deprecation warnings are emitted on every startup:
    `DeprecationWarning: on_event is deprecated, use lifespan event handlers instead.`
- In `backend/api/dispatcher.py` (lines 887–889):
  - Spawns unmanaged daemon thread: `worker = threading.Thread(target=_society_worker, args=(run_id, goal, model_name), daemon=True); worker.start()`
  - No cancellation endpoint or tracking registry exists.
- In `frontend/electron/main.js` (lines 164–171) and `frontend/scripts/dev.js` (lines 105–108):
  - Terminate child process using Node's `child.kill()`. On Windows, child process trees (e.g. Python worker processes) are not killed, remaining orphaned in the background.
- In `backend/storage/database.py` (line 42):
  - Sets `PRAGMA journal_mode = WAL;`. Abrupt process termination leaves SQLite `-wal` and `-shm` files uncheckpointed.

### 1.5 Crash and Vulnerability Loopholes
- **Missing `infer` function**:
  - In `backend/api/dispatcher.py` (lines 124–125): `res = engine.infer(prompt, model_name)`
  - In `backend/services/gpt2_engine.py`: Grep search for `def infer(` returned 0 matches. Calling `/api/infer` when live weights are loaded raises `AttributeError: module 'backend.services.gpt2_engine' has no attribute 'infer'`, crashing with HTTP 500.
- **Missing type validations**:
  - In `backend/api/dispatcher.py` (lines 137, 387, 482): `prompt.split()` and `prompt.replace()` called on unvalidated payload values; non-string types crash with `AttributeError`.
  - In `backend/api/dispatcher.py` (lines 505, 521, 548, 609, 622, 640): Direct calls to `int(payload.get(...))` crash with `ValueError` on non-numeric strings.
  - In `backend/storage/database.py` (lines 261–266): Raises `StorageError` on invalid payloads; with no exception handler in `backend/main.py`, FastAPI returns unhandled 500 crashes instead of 400 Bad Request.
- **CORS Misconfiguration**:
  - In `backend/main.py` (lines 24–36): `allow_origins` includes `"null"` and `"file://"` alongside `allow_credentials=True`. Ignores `MECH_CORS_ORIGINS`.

### 1.6 Backend Test Suite Baseline
- Command: `python -m pytest tests/pytest -q`
- Output: `13 failed, 177 passed, 7 warnings in 105.15s`
- Failures:
  - `tests/pytest/test_evidence_persistence.py` (3 tests): `AssertionError: assert 9 == 11` due to `_step_allows_evidence()` filtering out synthetic non-live evidence nodes.
  - `tests/pytest/test_sprint5_deliverable.py` (1 test): `KeyError: 'confidence'` in `AIScientistEngine.run_scientific_campaign` line 87 when validation returns `status: unavailable`.
  - `tests/pytest/test_science_reproducibility.py` (2 tests): `KeyError: 'observed_metrics'` and `KeyError: 'manifest_id'` in `IOIReproductionPipeline` mock mode.
  - `tests/pytest/test_validation_loop.py` (1 test): `AssertionError: assert 0.9 == 0.0` on minimality metric.
  - Sprint deliverable tests (6 tests): Cascading validation and mock token mismatch failures.

---

## 2. Logic Chain

1. **Entry Point Inconsistency**:
   - `frontend/electron/main.js:117` and `tests/pytest/test_protocol.py:20` specifically target `backend/main.py`.
   - `tests/pytest/test_protocol.py:44` hits `/api/v1/ping` and `tests/pytest/test_advanced_evals.py:51` hits `/api/v1/models`.
   - `backend/main.py:60` includes `/api/v1`, whereas root `main.py` only includes `/api`.
   - Therefore, `backend/main.py` is the true architectural entry point; root `main.py` is an inconsistent legacy clone that violates routing contracts.

2. **Server Lifecycle & Process Orphan Risks**:
   - Neither `main.py` nor `backend/main.py` registers any shutdown handler.
   - On Windows, `child.kill()` in `electron/main.js` only sends a terminate signal to the parent Node child handle; sub-processes or spawned Uvicorn workers survive as orphans.
   - Because SQLite uses `PRAGMA journal_mode = WAL;`, abrupt termination leaves `mech.db-wal` uncheckpointed and file locks held by surviving processes.
   - Therefore, clean server lifecycle requires an async FastAPI `lifespan` handler (with `wal_checkpoint(TRUNCATE)`) and process tree termination (`taskkill /T /F`) in Node wrappers.

3. **500 Server Crash Loophole**:
   - In `backend/api/dispatcher.py:125`, `engine.infer()` is invoked.
   - In `backend/services/gpt2_engine.py`, no `infer` function exists.
   - In `backend/main.py`, no exception handler is registered for `AttributeError` or `StorageError`.
   - Therefore, whenever live models are loaded and `/api/infer` is called, or when invalid items are saved to `/api/experiments`, the FastAPI server will crash with an unhandled HTTP 500 error instead of a structured client error.

---

## 3. Caveats

- **Torch / ML Environment**: The current python environment is Python 3.11.9 with PyTorch and Transformers installed, allowing tests to run live GPT-2 operations where applicable.
- **Frontend Survey**: Detailed UI store/component state was not deeply audited in this report, as that is the primary objective of peer subagent `explorer_survey_2`.
- **Read-Only Constraint**: No source code changes were made; this report and `analysis.md` were written strictly inside the dedicated agent folder.

---

## 4. Conclusion

1. **Authoritative Entry Point**: `backend/main.py` must be established as the single canonical entry point. Root `main.py` should be aligned or refactored as a forwarder.
2. **Health Check Verified**: `/health` on port 8000 functions as required by Acceptance Criteria (`{"status": "healthy"}`).
3. **Lifecycle Hardening Required**: Migrate from `@app.on_event("startup")` to `@asynccontextmanager async def lifespan(app: FastAPI):` with SQLite WAL checkpointing (`PRAGMA wal_checkpoint(TRUNCATE)`) and thread cleanup. Implement tree killing (`taskkill /T /F`) in Electron/dev scripts.
4. **Vulnerability Remediation Needed**:
   - Implement `infer()` in `backend/services/gpt2_engine.py`.
   - Add Pydantic schema validation or type guarding in `backend/api/dispatcher.py` to prevent `int()` / `.split()` crashes.
   - Register exception handlers for `StorageError` (HTTP 400) and generic `Exception` (structured JSON HTTP 500).
   - Safeguard `val_res.get("confidence", {})` in `backend/research_platform/autonomous/ai_scientist_engine.py`.
5. **Backend Test Suite Baseline**: 177 passing, 13 failing out of 190 tests. The 13 failures are isolated and understood (evidence policy provenance tagging, mock adapter vocabulary coverage, and dictionary key guards).

---

## 5. Verification Method

### 5.1 Backend Test Execution
To independently verify the backend test suite and reproduce the exact test baseline:
```bash
python -m pytest tests/pytest -q
```
Expected current output: `13 failed, 177 passed, 7 warnings in ~105s`.

### 5.2 Specific Test Invalidation Conditions
- Test health endpoint contract:
  ```bash
  python -m pytest tests/pytest/test_protocol.py -k test_health_roundtrip -v
  ```
  Expected output: `1 passed in ~0.5s`.
- Test evidence persistence failure:
  ```bash
  python -m pytest tests/pytest/test_evidence_persistence.py -k test_from_run_has_no_demo_nodes -v
  ```
  Expected output: `FAILED (assert 9 == 11)`.
- Test AI scientist `KeyError: 'confidence'` failure:
  ```bash
  python -m pytest tests/pytest/test_sprint5_deliverable.py -v
  ```
  Expected output: `FAILED (KeyError: 'confidence')`.

### 5.3 Files to Inspect
- `backend/main.py`: lines 23–36 (CORS), 39–42 (startup probe), 50–52 (`/health`), 56–63 (routers).
- `backend/api/dispatcher.py`: lines 123–136 (`/infer` calling non-existent `engine.infer`), 261–273 (`/experiments` unhandled `StorageError`).
- `backend/services/gpt2_engine.py`: lines 101–120 (`info()`), observe absence of `infer()`.
- `backend/storage/database.py`: lines 39–45 (`PRAGMA journal_mode = WAL;`), 261–266 (`StorageError`).
- `frontend/electron/main.js`: lines 14–16 (port 8000), 98–162 (`startBackend`), 164–171 (`stopBackend`).
