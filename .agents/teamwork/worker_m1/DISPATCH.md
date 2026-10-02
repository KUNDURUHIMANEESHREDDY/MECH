# Dispatch for worker_m1

## 2026-09-27T01:40:28Z

## Milestone 1 Implementation: Server Orchestration & Lifecycle Control
Read:
- `c:\Users\himan\OneDrive\Documents\Default Project\MECH\.agents\teamwork\ORIGINAL_REQUEST.md`
- `c:\Users\himan\OneDrive\Documents\Default Project\MECH\.agents\teamwork\orchestrator_1\PROJECT.md`
- Explorer 1 handoff & proposed files: `c:\Users\himan\OneDrive\Documents\Default Project\MECH\.agents\teamwork\explorer_m1_1\handoff.md`, `proposed_backend_main.py`, `proposed_database.py`
- Explorer 2 handoff & proposed files: `c:\Users\himan\OneDrive\Documents\Default Project\MECH\.agents\teamwork\explorer_m1_2\handoff.md`, `proposed_root_main.py`, `proposed_backend_main.py`
- Explorer 3 handoff: `c:\Users\himan\OneDrive\Documents\Default Project\MECH\.agents\teamwork\explorer_m1_3\handoff.md`

## Write Ownership
You exclusively own:
- `backend/main.py`
- `main.py`
- `backend/storage/database.py`
- `frontend/electron/main.js`
- `frontend/scripts/dev.js`

## Implementation Tasks
1. `backend/main.py`: Implement the modernized lifespan context manager (`@asynccontextmanager async def lifespan(app: FastAPI):`) replacing deprecated `@app.on_event`. On shutdown, call `checkpoint_wal()`. Keep `/health` returning `{"status": "healthy"}` on port 8000. Include both `/api` and `/api/v1` router mounts. Add `sys.path` bootstrap.
2. `main.py`: Align root entry point to re-export `backend.main:app` and run uvicorn targeting `backend.main:app` on `127.0.0.1:8000`.
3. `backend/storage/database.py`: Implement `checkpoint_wal()` method executing `PRAGMA wal_checkpoint(TRUNCATE)`.
4. `frontend/electron/main.js`: Implement synchronous Windows process tree termination (`taskkill /T /F /PID <pid>`) in `stopBackend` and lifecycle shutdown hooks (`before-quit`, `will-quit`, `window-all-closed`, process exit).
5. `frontend/scripts/dev.js`: Implement Windows process tree termination for `backend` and `electronChild` in `killProcessTree` and cleanup handlers.

## Mandatory Verification
- Verify backend health check and protocol tests: `pytest tests/pytest/test_protocol.py -v`
- Verify frontend renderer build: `cd frontend && npm run build:renderer`
- Verify database WAL checkpoint execution.

Deliver your report to `c:\Users\himan\OneDrive\Documents\Default Project\MECH\.agents\teamwork\worker_m1\handoff.md`.
