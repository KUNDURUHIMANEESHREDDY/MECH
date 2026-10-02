# Dispatch for explorer_m1_2

## Scope: Milestone 1 - Server Orchestration & Lifecycle Control
Read:
- `c:\Users\himan\OneDrive\Documents\Default Project\MECH\.agents\teamwork\ORIGINAL_REQUEST.md`
- `c:\Users\himan\OneDrive\Documents\Default Project\MECH\.agents\teamwork\orchestrator_1\PROJECT.md`
- Survey report at `c:\Users\himan\OneDrive\Documents\Default Project\MECH\.agents\teamwork\explorer_survey_1\handoff.md`

Your specific task:
Design the implementation strategy for entry point unification:
1. Reconcile root `main.py` and `backend/main.py`. Ensure root `main.py` cleanly imports or forwards to `backend.main:app` (or runs uvicorn targeting `backend.main:app` on port 8000), mounting both `/api` and `/api/v1` so no contracts or tests break.
2. Clean removal or safe fallback for the missing `backend.api.runtime_api` import in both files.
3. CORS configuration unification: ensure `http://localhost:5173`, `http://127.0.0.1:5173`, `http://localhost:3000`, `http://127.0.0.1:3000`, and custom `MECH_CORS_ORIGINS` are respected.

Write your recommendations to `c:\Users\himan\OneDrive\Documents\Default Project\MECH\.agents\teamwork\explorer_m1_2\analysis.md` and deliver `handoff.md`.

## 2026-09-27T01:31:00Z
Investigate and design the exact implementation plan for entry point unification:
- Align root `main.py` and `backend/main.py` to eliminate divergence. Root `main.py` should cleanly run or re-export `backend.main:app` (or match routes `/api` and `/api/v1` and port 8000).
- Clean up or guard the missing `backend.api.runtime_api` import.
- Unify CORS headers to allow Vite (5173), desktop shells, and environment configured origins.
Write your analysis to `c:\Users\himan\OneDrive\Documents\Default Project\MECH\.agents\teamwork\explorer_m1_2\analysis.md` and deliver `handoff.md`.
Notify your parent via send_message when complete.
