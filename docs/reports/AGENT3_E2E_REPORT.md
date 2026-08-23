# AGENT 3 — Researcher Experience E2E Verification Report

**Date:** 2026-08-20
**Environment:** Windows, backend (FastAPI) on `http://localhost:8000` (UP), Vite frontend on `http://localhost:5173` (UP)
**Scope:** End-to-end verification of the Researcher → UI → Agent 2 → Agent 1 → Real experiment chain, all UI failure states, and the frontend test suites.
**Boundary:** No changes to PyTorch experiment implementation, model hooks, causal metric math, or hypothesis-truth determination.

---

## 1. Executive Summary

The researcher's acceptance path — **CONCLUSION → EVIDENCE → EXPERIMENT → INTERVENTION → COMPONENT → MODEL → REAL EXECUTION** — is fully verified end-to-end against the live backend:

- All **16 Playwright E2E tests pass** (10 research E2E + 6 shell smoke).
- All **83 Vitest unit/component tests pass** (31 files).
- **28 relevant backend pytest tests pass** (API, research router, fail-closed epistemic integrity, immutable reproducibility).
- A real GPT-2 causal intervention was executed through the full chain from the Experiment Builder UI and via the API, returning live metadata (`delta_logit=0.401`, `is_reproducible=true`, `used_mock_data=false`).
- All previously-failing frontend crash bugs exposed by opening panels (drawer hook-order, model panel `availableModels`, SAE feature envelope, notebook panel envelope) are fixed and regression-tested.

**Verdict: ACCEPTED** — the researcher experience satisfies the acceptance criterion: an experiment result can be traced from the UI back to a real executed intervention with provenance and integrity metadata, and all failure modes fail closed without masking or fabrication.

---

## 2. Evidence: Test Suites (final green runs)

### 2.1 Playwright E2E — `npm run test:e2e` → **16 passed**

| # | Test | Result | Key assertion |
|---|------|--------|---------------|
| 1 | research pages render via hash navigation + breadcrumb | PASS | header crumb matches each hash |
| 2 | experiment builder runs real IOI experiment through full chain | PASS | "Experiment Completed" appears exactly once after clicking **Run Experiment** |
| 3 | experiment monitor renders runs with real execution metadata | PASS | COMPLETED tab count > 0; live run metadata shown |
| 4 | experiment monitor shows mock-data warning only when mock runs exist | PASS | MOCK DATA badge absent for live runs |
| 5 | evidence explorer renders evidence + provenance chains | PASS | Evidence content rendered |
| 6 | research graph renders investigation pipeline | PASS | investigation/hypothesis/mechanism graph rendered |
| 7 | research notebook renders with epistemic types | PASS | notebook panel renders, no crash |
| 8 | acceptance test verifier confirms agent 3 boundary | PASS | boundary checks listed |
| 9 | ui adversarial tester exposes all failure-state checks | PASS | failure-state checks rendered |
| 10 | **agent 1 epistemic gate failure is surfaced, not masked as COMPLETED** | PASS | `UNEXECUTED_EXPERIMENT` envelope → gate error shown, no crash, no false completion |
| 11–16 | shell smoke: sidebar, drawer toggle, **all 29 nav pages**, hash-only pages, activity bar, python status | PASS | every nav page updates crumb AND hash |

### 2.2 Vitest — `npm run test:js` → **83 passed (31 files)**

All existing component/unit suites pass with the refactored drawer/shell/model services in place (Sprint2–6 deliverables, explorer panels, knowledge graph, scientific health, settings, evidence fusion, reasoning trace, etc.).

### 2.3 Backend pytest (relevant subset) — `pytest tests/pytest/...` → **28 passed**

`test_api.py`, `test_research_router.py`, `test_fail_closed_epistemic_integrity.py`, `test_immutable_experiment_reproducibility.py`.

---

## 3. Real Execution Evidence (API → UI trace)

The chain was executed with a live model through the actual endpoints the UI calls.

**Run request** → `POST /api/v1/research/experiments/run` (IOI, `inv_ioi_default`, `hyp_l9h9_namemover`, gpt2):

```json
{
  "clean_prompt": "When Mary and John went to the store, John gave a drink to",
  "target_token": " Mary",
  "source_component": "L8_H4",
  "intervention_type": "ABLATION_ZERO",
  "model_id": "gpt2",
  "dataset_id": "ioi",
  "repeats": 1, "seed": 42
}
```

**Response (abridged):**

```json
{
  "status": "success",
  "run": {
    "id": "run_ef3fdc8da7",
    "execution_status": "COMPLETED",
    "used_mock_data": false,
    "delta_logit": 0.4013824462890625,
    "baseline_target_prob": 0.4460166096687317,
    "intervened_target_prob": 0.39687395095825195,
    "execution_time_ms": 326.63,
    "model_id": "gpt2",
    "is_reproducible": true
  }
}
```

**Downstream endpoints confirm the run is visible with real metadata and provenance:**

- `GET /api/v1/research/runs` → `count=39`, newest run `run_ef3fdc8da7` `COMPLETED`, `mock=False`, `delta=0.401`.
- `GET /api/v1/research/evidence?investigation_id=inv_ioi_default` → `count=20`, first evidence record `evi_007960b076` typed `INFERENCE`, level `CONTRADICTED`, linked to `run=run_ef3fdc8da7`.
- The Experiment Builder UI (`/#experiment_builder`) renders this chain and shows "Experiment Completed".

**Conclusion of the trace:** CONCLUSION/verdict → EVIDENCE (`evi_…` with knowledge type + evidence level) → EXPERIMENT (`run_ef3fdc8da7`) → INTERVENTION (`ABLATION_ZERO` on `L8_H4`) → COMPONENT (`L8_H4`) → MODEL (`gpt2`) → REAL EXECUTION (326 ms live forward pass, reproducible). **Acceptance criterion met.**

---

## 4. Failure-State Verification

All failure modes were tested against the real backend; the UI fails closed and never fabricates or masks results.

### 4.1 Backend killed during/after experiment (verified in-session)

| Check | Observed | Correct? |
|-------|----------|----------|
| Monitor after reload with backend down | "No Experiments Found" — no stale `Δ =` delta results shown | YES |
| Python status indicator | "Python Offline" shown | YES |
| Running an experiment with backend down | run FAILS with error; does NOT show "Experiment Completed" | YES |
| Note | "COMPLETED" text is always present in monitor filter-tab buttons (count 0) — tests must NOT rely on bare `includes('COMPLETED')` | documented in test suite |

### 4.2 Epistemic gate / UNEXECUTED envelope (regression test added)

`POST /api/v1/research/experiments/run` with invalid payload (missing `investigation_id` / bad `intervention_type` enum) returns fail-closed:

```json
{
  "status": "UNEXECUTED_EXPERIMENT",
  "manifest_id": null,
  "data": null,
  "integrity_status": "UNVERIFIED",
  "error": "Epistemic Gate Violation: ..."
}
```

New E2E test **"agent 1 epistemic gate failure is surfaced, not masked as COMPLETED"** routes the run endpoint to this envelope and asserts:
- the gate error text is visible in the UI,
- no page crash (`Cannot read properties of undefined`),
- no false "Experiment Completed".

This was the **root cause of a real crash**: `research.ts` previously accessed `res.run.id` on the UNEXECUTED envelope → crash. Now guarded (`res.status === 'UNEXECUTED_EXPERIMENT' || !res.run`).

### 4.3 Client-side validation (verified in-session)

Empty `clean_prompt` → "Clean prompt is required." shown, no API call made.

---

## 5. Pre-Existing Bugs Found & Fixed During This Verification

All were exposed by opening panels via hash navigation / the refactored drawer. All fixes are in the frontend only (within Agent 3 scope).

| # | File | Bug | Fix |
|---|------|-----|-----|
| 1 | `src/shell/features/FeaturesDrawer.tsx` | Hook-order violation: `if (!open) return null` before two `useCallback` hooks → "Rendered more hooks than during the previous render" on every drawer open | Moved all hooks above the early return |
| 2 | `src/services/modelService.ts` | `/api/models` returns a raw ARRAY of model records; code returned `data.models` (undefined) → `availableModels=undefined` → every model panel crashed with `Cannot read properties of undefined (reading '0')` | Normalize `Array.isArray(data)` → map `model_id`, else `data.models` |
| 3 | `src/science/api/scienceApi.ts` | `fetchLayerFeatures` returned the whole fail-closed envelope; panels called `.find()` on it → `saeFeatures.find is not a function` | Unwrap `env.result.result` (array) or return `[]` |
| 4 | `src/app/Shell.tsx` | `PAGE_LABEL_MAP` missing `real_time_dag`, `gpt2explorer`, `transformerExplorer` → hash-only pages showed stale crumb and did not open panels | Added the three entries; verified crumbs "GPT-2 Neuron Explorer", "Transformer Explorer", "Experiment Builder" with no page errors |
| 5 | `src/science/api/scienceApi.ts` | `saveExperiment` returned the whole envelope instead of `envelope.result`; with no model loaded the envelope's `result` is null → `ExperimentNotebookPanel` seeded a malformed run → infinite render-loop crash on `r.specification.clean_prompt` (hung the page, broke nav sweep) | Unwrap `envelope.result`; throw with the envelope's `statistical_caveat` when `result` is null (panel now shows empty state instead of crashing) |
| 6 | `src/shell/features/FeaturesDrawer.tsx` | Drawer `zIndex: 20` same as `BottomWorkspace` → the bottom status bar painted over lower nav items and swallowed clicks after panels opened | Raised drawer to `zIndex: 30` |
| 7 | `tests/playwright/shell-smoke.spec.js` | NAV_PAGES stale vs. refactored drawer; duplicate `nav-*` testids (dataset_viewer, sae_feature, intervention_lab, hypothesis_lab, evidence_graph, mechanism_builder appear twice) caused strict-mode failures | Rewrote NAV_PAGES to current 29 navKeys, added `.first()` for duplicates, re-scroll + retry loop |

### API behavior note (backend, verified, not modified)

`POST /api/v1/science/experiments/save` with no loaded model returns a fail-closed UNEXECUTED envelope: `"Cannot derive experiment metrics: no model loaded. ... Do not fabricate scientific results."` — the notebook panel's seed path now handles this without crashing (fix #5).

---

## 6. How to Reproduce

```powershell
# 1. Backend (already running on :8000)
$env:MECH_DISABLE_AUTH="1"; .venv\Scripts\python.exe main.py

# 2. Frontend (already running on :5173)
npm run dev   # in frontend/

# 3. Full E2E (requires backend UP)
cd frontend; npm run test:e2e        # 16 passed

# 4. Vitest
cd frontend; npm run test:js         # 83 passed

# 5. Backend subset relevant to this verification
.venv\Scripts\python.exe -m pytest tests/pytest/test_api.py tests/pytest/test_research_router.py tests/pytest/test_fail_closed_epistemic_integrity.py tests/pytest/test_immutable_experiment_reproducibility.py -q
```

Playwright config: `testDir: ./tests/playwright`, `webServer` reuses the existing Vite dev server (`reuseExistingServer: !process.env.CI`).

---

## 7. Files Touched by Agent 3

- `frontend/src/shell/features/FeaturesDrawer.tsx` — hook-order fix + z-index 30.
- `frontend/src/services/modelService.ts` — `fetchAvailableModels` shape normalization.
- `frontend/src/science/api/scienceApi.ts` — envelope unwraps (`fetchLayerFeatures`, `saveExperiment`).
- `frontend/src/app/Shell.tsx` — `PAGE_LABEL_MAP` additions.
- `frontend/src/shared/stores/research.ts` — UNEXECUTED_EXPERIMENT guard.
- `frontend/tests/playwright/shell-smoke.spec.js` — NAV_PAGES updated to current drawer.
- `frontend/tests/playwright/research-e2e.spec.js` — epistemic-gate failure-state test added (10th test).
- `AGENT3_E2E_REPORT.md` — this report.

**Boundary respected:** no changes to PyTorch experiment code, model hooks, causal metric calculations, or hypothesis-truth determination.