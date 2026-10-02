# Handoff Report — MECH Platform Entry Point Unification & CORS Harmonization

**Agent**: `explorer_m1_2`  
**Working Directory**: `c:\Users\himan\OneDrive\Documents\Default Project\MECH\.agents\teamwork\explorer_m1_2`  
**Handoff Type**: Hard (Investigation complete)  

---

## 1. Observation

### 1.1 Entry Point Divergence & Missing Router Prefix
- In root `main.py` (lines 20–32, 51–64, 70–76, 85–94):
  - Line 24: `_CORS_ORIGINS = ["http://localhost:5173", "http://127.0.0.1:5173", "null", "file://"]` (omits port 3000).
  - Line 25: Appends `_os.environ.get("MECH_CORS_ORIGINS", "")`.
  - Line 56: Root endpoint returns `{"name": "MECH Research Platform", "status": "running", "version": "2.0"}`.
  - Line 72: Router mounted ONLY at `/api`: `app.include_router(api_router, prefix="/api")`. Missing `/api/v1`.
  - Line 89: `uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False, timeout_keep_alive=600)`.
- In `backend/main.py` (lines 16–36, 45–53, 56–63, 74–77):
  - Lines 25–32: Hardcodes origins including `http://localhost:3000` and `http://127.0.0.1:3000`, but ignores `MECH_CORS_ORIGINS`.
  - Line 47: Root endpoint returns `{"name": "MECH Platform", "version": "2.0.0", "status": "running"}`.
  - Lines 59–60: Router mounted at both `/api` and `/api/v1`:
    ```python
    app.include_router(api_router, prefix="/api")
    app.include_router(api_router, prefix="/api/v1")
    ```
  - Line 76: `uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=False, timeout_keep_alive=600)`.
- In `frontend/src/components/HomeDashboard.jsx` (line 12), `PaperReproductionView.jsx` (line 10), and `frontend/electron/ipc/runtime.js` (line 6):
  - Requests target `http://127.0.0.1:8000/api/v1/...` and `http://localhost:8000/api/v1`.
- In `tests/pytest/test_protocol.py` (lines 33, 44) and `tests/pytest/test_advanced_evals.py` (lines 38, 51):
  - Asserts `body["name"] == "MECH Platform"` and `body["version"] == "2.0.0"`.
  - Requests `POST /api/v1/ping` and `GET /api/v1/models`.

### 1.2 Standalone Direct Execution Failure (`sys.path`)
- Command executed:
  ```powershell
  python backend/main.py --help
  ```
- Verbatim terminal output:
  ```
  API dispatcher not loaded: No module named 'backend'
  Traceback (most recent call last):
    File "...\backend\main.py", line 76, in <module>
      uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=False, timeout_keep_alive=600)
    ...
  ModuleNotFoundError: No module named 'backend'
  ```
- Cause: Running `python backend/main.py` directly sets `sys.path[0]` to `MECH\backend`. Neither the repo root nor `backend` package root is recognized without manual `PYTHONPATH` configuration.

### 1.3 Missing `backend.api.runtime_api` Artifact
- In `main.py` (lines 78–83) and `backend/main.py` (lines 65–71):
  ```python
  try:
      from backend.api.runtime_api import router as runtime_router
      app.include_router(runtime_router, prefix="/api/v2")
      logger.info("Runtime v2 API loaded at /api/v2.")
  except Exception:
      pass
  ```
- Command executed:
  ```powershell
  python -c "import importlib.util; print(importlib.util.find_spec('backend.api.runtime_api'))"
  ```
- Output: `None`.
- File search for `*runtime_api*` across the repository returned zero files.
- In `backend/api/dispatcher.py` (lines 799–800):
  `# NOTE: docs/API_v1.md sketches /api/v1/... but no runtime_api module exists live — Society ships on this active dispatcher, not the frozen doc.`
- In `frontend/src/components/PaperReproductionView.vue` (line 41):
  `No standalone /api/v2/science/reproduce contract is mounted in the current runtime.`
- Zero tests in `tests/pytest/` execute HTTP requests against `/api/v2/*` on the FastAPI server.

### 1.4 CORS Configuration Test Output
- In-process TestClient run with `CORSMiddleware(allow_origins=[...], allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1)(:[0-9]+)?$", allow_credentials=True)`:
  - `Origin: http://localhost:5173` -> `access-control-allow-origin: http://localhost:5173`
  - `Origin: http://127.0.0.1:3000` -> `access-control-allow-origin: http://127.0.0.1:3000`
  - `Origin: http://localhost:5174` -> `access-control-allow-origin: http://localhost:5174`
  - `Origin: null` -> `access-control-allow-origin: null`
  - `Origin: file://` -> `access-control-allow-origin: file://`
  - `Origin: http://evil.com` -> `access-control-allow-origin: None` (rejected)

---

## 2. Logic Chain

1. **Inconsistent Routing & Metadata in Root `main.py`**:
   - `test_protocol.py:44` hits `/api/v1/ping`, `test_advanced_evals.py:51` hits `/api/v1/models`, and frontend components hit `/api/v1/research_catalog`.
   - Root `main.py:72` only mounts `/api`, causing all `/api/v1/*` requests to 404.
   - `test_protocol.py:33` asserts `body["name"] == "MECH Platform"`, while root `main.py:54` returns `"MECH Research Platform"`.
   - Therefore, root `main.py` was an out-of-sync fork of `backend/main.py`. Refactoring root `main.py` to re-export `app` from `backend.main` and forward `uvicorn` invocation guarantees 100% contract compliance and eliminates code divergence.

2. **`sys.path` Resolution**:
   - When Python runs `python backend/main.py`, `sys.path[0]` is the `backend` folder.
   - Statements like `from backend.api.dispatcher import router` fail because `backend` cannot import itself as a package from within `backend`.
   - Adding `sys.path.insert(0, str(_REPO_ROOT))` and `sys.path.insert(0, str(_BACKEND_DIR))` at the top of both files guarantees that direct execution works without requiring external environment variables.

3. **Dead Code Elimination for `runtime_api`**:
   - `backend.api.runtime_api` does not exist on disk (`find_spec` returns `None`).
   - The active research society routes reside directly on `backend/api/dispatcher.py` (as documented at line 799).
   - The bare `except Exception: pass` suppresses potential syntax and import errors.
   - Therefore, the dead block should either be cleanly removed or guarded with `importlib.util.find_spec` without trial-importing or swallowing general exceptions.

4. **Complete CORS Coverage**:
   - Vite runs on `:5173`, Electron dev and desktop shells on `:3000`, Electron packaged builds on `file://` and `null`.
   - If port 5173 is occupied, Vite binds dynamically to `:5174`.
   - Combining explicit origins with `allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1)(:[0-9]+)?$"` safely accommodates all local ports while blocking untrusted web origins.

---

## 3. Caveats

- **Coordinate with `explorer_m1_1`**: `explorer_m1_1` drafted `proposed_backend_main.py` focusing on lifespan and SQLite WAL checkpointing. Our proposed `proposed_backend_main.py` incorporates their lifespan/WAL checkpointing and completes it with dynamic `sys.path` bootstrapping, unified CORS origin parsing/regex, and `runtime_api` cleanup.
- **Port 8000 Binding**: Both files bind to `127.0.0.1:8000` rather than `0.0.0.0:8000`. This prevents Windows Firewall prompts while matching Electron's `BACKEND_HOST = '127.0.0.1'` configuration. If remote access is needed, developers can specify `--host 0.0.0.0` or set environment variables.
- **Read-Only Scope**: In compliance with the explorer role, no repository source code files were edited directly. Full drop-in proposed code is provided in this agent's folder.

---

## 4. Conclusion

1. **Establish `backend/main.py` as Authoritative**: Retain `backend/main.py` as the canonical server containing router mounts, CORS middleware, `/health`, and the lifespan context manager.
2. **Refactor Root `main.py` as Re-exporter & Forwarder**: Root `main.py` imports `app` from `backend.main`, exposes it at module scope for ASGI servers, and runs `uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=False, timeout_keep_alive=600)`.
3. **Add `sys.path` Bootstrap**: Inject repository root and `backend` directory at module start in both files.
4. **Remediate `runtime_api`**: Remove the dead bare-except block, or guard via `importlib.util.find_spec`.
5. **Harmonize CORS**: Support `5173`, `3000`, `null`, `file://`, `MECH_CORS_ORIGINS` (stripped and deduplicated), and localhost port regex.
6. **Artifacts Delivered**:
   - `proposed_root_main.py`: Complete replacement for `main.py`.
   - `proposed_backend_main.py`: Complete replacement for `backend/main.py`.
   - `analysis.md`: Detailed architecture and rationale document.

---

## 5. Verification Method

### 5.1 Syntax & Import Verification
```powershell
python -c "import py_compile; py_compile.compile('.agents/teamwork/explorer_m1_2/proposed_root_main.py'); py_compile.compile('.agents/teamwork/explorer_m1_2/proposed_backend_main.py'); print('Compiled!')"
```
Expected: Exits with code 0 and prints `Compiled!`.

### 5.2 Functional End-to-End Contract Verification
Run the automated verification script:
```powershell
python -c "import sys, importlib.util; spec_b = importlib.util.spec_from_file_location('proposed_backend_main', '.agents/teamwork/explorer_m1_2/proposed_backend_main.py'); mod_b = importlib.util.module_from_spec(spec_b); sys.modules['backend.main'] = mod_b; spec_b.loader.exec_module(mod_b); spec_r = importlib.util.spec_from_file_location('proposed_root_main', '.agents/teamwork/explorer_m1_2/proposed_root_main.py'); mod_r = importlib.util.module_from_spec(spec_r); spec_r.loader.exec_module(mod_r); assert mod_r.app is mod_b.app; from fastapi.testclient import TestClient; c = TestClient(mod_r.app); assert c.get('/').json()['name'] == 'MECH Platform'; assert c.get('/health').json() == {'status': 'healthy'}; assert c.post('/api/v1/ping').json() == {'status': 'ok'}; assert c.post('/api/ping').json() == {'status': 'ok'}; assert len(c.get('/api/v1/models').json()['models']) == 8; assert c.get('/health', headers={'Origin': 'http://localhost:5173'}).headers.get('access-control-allow-origin') == 'http://localhost:5173'; assert c.get('/health', headers={'Origin': 'http://127.0.0.1:3000'}).headers.get('access-control-allow-origin') == 'http://127.0.0.1:3000'; assert c.get('/health', headers={'Origin': 'http://localhost:5174'}).headers.get('access-control-allow-origin') == 'http://localhost:5174'; assert c.get('/health', headers={'Origin': 'http://evil.com'}).headers.get('access-control-allow-origin') is None; print('ALL PROPOSED CODE VERIFICATIONS PASSED 100%!')"
```
Expected: Prints `ALL PROPOSED CODE VERIFICATIONS PASSED 100%!`.

### 5.3 Full Test Suite Regression Check
```powershell
python -m pytest tests/pytest/test_protocol.py tests/pytest/test_advanced_evals.py -v
```
Expected: 37 passed, 0 failed.

### 5.4 Files to Inspect
- `c:\Users\himan\OneDrive\Documents\Default Project\MECH\.agents\teamwork\explorer_m1_2\proposed_root_main.py`
- `c:\Users\himan\OneDrive\Documents\Default Project\MECH\.agents\teamwork\explorer_m1_2\proposed_backend_main.py`
- `c:\Users\himan\OneDrive\Documents\Default Project\MECH\.agents\teamwork\explorer_m1_2\analysis.md`

### 5.5 Invalidation Conditions
- If running `python backend/main.py` fails with `ModuleNotFoundError: No module named 'backend'`.
- If `POST /api/v1/ping` or `GET /api/v1/models` returns HTTP 404 when started via root `main.py`.
- If requests from `http://localhost:5173`, `http://127.0.0.1:5173`, `http://localhost:3000`, `http://127.0.0.1:3000`, or `MECH_CORS_ORIGINS` fail CORS preflight.
