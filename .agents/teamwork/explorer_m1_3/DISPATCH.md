# Dispatch for explorer_m1_3 (spec_miner)

## Scope: Milestone 1 - Server Orchestration & Lifecycle Control
Read:
- `c:\Users\himan\OneDrive\Documents\Default Project\MECH\.agents\teamwork\ORIGINAL_REQUEST.md`
- `c:\Users\himan\OneDrive\Documents\Default Project\MECH\.agents\teamwork\orchestrator_1\PROJECT.md`
- Survey report at `c:\Users\himan\OneDrive\Documents\Default Project\MECH\.agents\teamwork\explorer_survey_1\handoff.md`
- Survey report at `c:\Users\himan\OneDrive\Documents\Default Project\MECH\.agents\teamwork\explorer_survey_2\handoff.md`

Your specific task:
Specify the exact process management and termination mechanisms for Windows:
1. In `frontend/electron/main.js` and `frontend/scripts/dev.js`: analyze how Python backend child processes are started and stopped. Specify replacement of naive `child.kill()` with Windows process tree termination (`taskkill /T /F /PID <pid>`) when on Windows (`process.platform === 'win32'`).
2. Specify graceful shutdown hooks so that closing Electron or interrupting `npm run dev` cleanly frees port 8000 and releases SQLite database locks.
3. Define the exact acceptance verification procedure for R1 Service Health & Execution.

Write your specifications to `c:\Users\himan\OneDrive\Documents\Default Project\MECH\.agents\teamwork\explorer_m1_3\analysis.md` and deliver `handoff.md`.

## 2026-09-27T01:30:49Z
<USER_REQUEST>
You are explorer_m1_3, a specification mining subagent.
Your dedicated working directory is:
c:\Users\himan\OneDrive\Documents\Default Project\MECH\.agents\teamwork\explorer_m1_3

MANDATORY REQUIREMENT:
Read ORIGINAL_REQUEST.md before starting work at:
c:\Users\himan\OneDrive\Documents\Default Project\MECH\.agents\teamwork\ORIGINAL_REQUEST.md
Also read PROJECT.md at:
c:\Users\himan\OneDrive\Documents\Default Project\MECH\.agents\teamwork\orchestrator_1\PROJECT.md
Also read your dispatch task at:
c:\Users\himan\OneDrive\Documents\Default Project\MECH\.agents\teamwork\explorer_m1_3\DISPATCH.md

Mission:
Extract and specify the exact process management and clean termination logic on Windows:
- In `frontend/electron/main.js` and `frontend/scripts/dev.js`: specify replacing Node's naive `child.kill()` with Windows process tree termination (`taskkill /T /F /PID <pid>`) when running on Windows (`process.platform === 'win32'`), preventing orphaned processes holding port 8000 and locking SQLite files.
- Document exact verification steps and commands for R1 acceptance criteria.
Write your specifications to `c:\Users\himan\OneDrive\Documents\Default Project\MECH\.agents\teamwork\explorer_m1_3\analysis.md` and deliver `handoff.md`.
Notify your parent via send_message when complete.
</USER_REQUEST>
