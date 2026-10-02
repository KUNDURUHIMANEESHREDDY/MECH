# Dispatch for challenger_m2_2_repl (Replacement for hung challenger_m2_2)

## Task
Adversarially challenge test suite isolation and flake resistance:
1. Run individual deliverable test files in cold isolated processes:
   - `pytest tests/pytest/test_sprint2_deliverable.py -q`
   - `pytest tests/pytest/test_evidence_persistence.py -q`
   - `pytest tests/pytest/test_sprint3_deliverable.py -q`
   - `pytest tests/pytest/test_sprint4_deliverable.py -q`
   - `pytest tests/pytest/test_sprint5_deliverable.py -q`
   - `pytest tests/pytest/test_interpretability_sprint3.py -q`
   - `pytest tests/pytest/test_interpretability_sprint4.py -q`
   - `pytest tests/pytest/test_science_reproducibility.py -q`
2. Run randomized or reverse-order file execution.
3. Test consecutive multi-run stability to ensure no database locking or shared state leaks.

Read:
- `c:\Users\himan\OneDrive\Documents\Default Project\MECH\.agents\teamwork\ORIGINAL_REQUEST.md`
- `c:\Users\himan\OneDrive\Documents\Default Project\MECH\.agents\teamwork\orchestrator_1\PROJECT.md`
- Worker handoff: `c:\Users\himan\OneDrive\Documents\Default Project\MECH\.agents\teamwork\worker_m2\handoff.md`

Deliver empirical test results and verdict (APPROVE or REQUEST_CHANGES) in `c:\Users\himan\OneDrive\Documents\Default Project\MECH\.agents\teamwork\challenger_m2_2_repl\handoff.md`.
