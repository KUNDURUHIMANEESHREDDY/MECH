# MECH Research Platform: Full-Stack Testing Infrastructure Survey & Analysis

**Author:** explorer_survey_3  
**Date:** 2026-09-27  
**Scope:** Backend test suite (`tests/pytest`), Frontend test suite (`frontend/tests/vitest`), E2E suites (`tests/playwright`, `frontend/tests/playwright`), Core Workflow Coverage, and Test Failure Remediation Audit.

---

## 1. Executive Summary & Inventory

The MECH Research Platform maintains testing suites across both backend (FastAPI / scientific compute sidecar) and frontend (React / Vite / Electron Desktop OS renderer) layers.

| Tier | Test Runner & Version | Test Location | Test Count | Passing | Failing | Skipped | Run Duration |
|------|-----------------------|---------------|------------|---------|---------|---------|--------------|
| **Backend** | Pytest 9.1.1 (Python 3.11.9) | `tests/pytest/` | 190 tests (32 files) | 177 | 13 | 0 | 97.71s |
| **Frontend Unit/Component** | Vitest 2.1.9 (jsdom) | `frontend/tests/vitest/` | 119 tests (27 files) | 119 | 0 | 0 | 35.72s |
| **Frontend E2E** | Playwright Test 1.47.0 | `frontend/tests/playwright/`, `tests/playwright/` | 6 spec files | Spec-based | - | - | On demand |

### Key High-Level Findings
1. **Frontend Vitest Suite is 100% Passing**: All 119 tests across 27 component/view test suites pass cleanly. A minor warning exists regarding un-wrapped React state updates in `VisualizationSprint2.test.jsx`.
2. **Backend Pytest Suite Has a Critical Pathing Loophole**: Pytest cannot run directly via `pytest tests/pytest` without `PYTHONPATH=".;backend"` set. Both `pytest.ini` and `tests/pytest/conftest.py` fail to register repository root in `sys.path`, resulting in immediate `ModuleNotFoundError: No module named 'backend'` during test collection.
3. **Backend Has Exactly 13 Test Failures**: Out of 190 collected tests, 177 pass and 13 fail. All 13 failures stem from 5 distinct architectural discrepancies between recent fail-closed evidence policy hardening and older test assertions / mock fallbacks.

---

## 2. Backend Test Suite Architecture (`tests/pytest`)

### 2.1 Pytest Configuration & Runner Setup
- **Config File:** `pytest.ini` in project root:
  ```ini
  [pytest]
  testpaths = tests
  python_files = test_*.py
  addopts = --basetemp=.pytest-tmp
  ```
- **Conftest:** `tests/pytest/conftest.py`:
  ```python
  HERE = os.path.dirname(os.path.abspath(__file__))
  BACKEND_DIR = os.path.abspath(os.path.join(HERE, "..", "..", "backend"))
  if BACKEND_DIR not in sys.path:
      sys.path.insert(0, BACKEND_DIR)
  ```
- **Identified Pathing Loophole**:
  In `frontend/package.json`, script `"test:py": "cd .. && pytest tests/pytest -q"` invokes pytest from root. Because `backend/` was inserted into `sys.path` but the workspace root was not, statements like `from backend.storage import ...` inside backend submodules fail with `ModuleNotFoundError: No module named 'backend'` during collection of `test_acdc_search.py` and `test_adaptive_runtime.py`.
  - **Remediation**: Add `pythonpath = . backend` to `pytest.ini` and append root `os.path.abspath(os.path.join(HERE, "..", ".."))` to `sys.path` in `conftest.py`.

### 2.2 Test Suite File Inventory (32 Test Files)
1. `test_acdc_search.py` (5 tests): Causal circuit discovery via ACDC pruning.
2. `test_adaptive_runtime.py` (11 tests): Dynamic batching, execution caching, layer streaming.
3. `test_advanced_evals.py` (34 tests): Comprehensive FastAPI endpoint verification against `/api/v1/` routes.
4. `test_api.py` (11 tests): Core sidecar API dispatcher methods (`ping`, `info`, `echo`, `add`, `time`, `runtime:status`).
5. `test_autonomous_platform.py` (7 tests): Autonomous Research Agent run, typed research graph, memory store.
6. `test_batch_comparison.py` (1 test): Multi-model comparison workflow.
7. `test_checkpoint_restore.py` (1 test): Runtime debugger session state save/restore.
8. `test_compression_roundtrip.py` (1 test): Activation tensor compression and decompression.
9. `test_end_to_end_inspection.py` (1 test): End-to-end model inspection workflow.
10. `test_evidence_persistence.py` (7 tests): SQLite run record persistence, EvidenceGraph DAG, KG write-back.
11. `test_feature_workflow.py` (1 test): SAE feature search, feature inspection, and dataset examples.
12. `test_inspectors.py` (6 tests): Neuron, Attention, Residual, Layer, Token, and Prediction inspectors.
13. `test_interpretability_sprint2.py` (6 tests): SAE dictionary loading, logit lens, attention head intervention.
14. `test_interpretability_sprint3.py` (7 tests): Circuit discovery, causal tracing, attribution patching, automated labeling.
15. `test_interpretability_sprint4.py` (8 tests): Discovery engine orchestration, hypothesis testing, cross-model alignment.
16. `test_lens_comparison.py` (1 test): Logit lens vs tuned lens comparison.
17. `test_model_adapters.py` (13 tests): ModelAdapterRegistry across 6 model families (`gpt2`, `gemma`, `llama`, `qwen`, `mistral`, `deepseek`).
18. `test_neural_explorer.py` (19 tests): Model tree structure, induction head flags, neuron inspector pagination, circuit explorer.
19. `test_patch_and_continue.py` (1 test): Activation patch injection during execution stepping.
20. `test_protocol.py` (4 tests): HTTP server root `/`, `/health`, `/ping`, and 404 handler.
21. `test_repository.py` (4 tests): Activation repository query filtering, search API, aggregations, cache metrics.
22. `test_runtime_epics.py` (5 tests): Activation patching, forward debugger stepping, model comparison.
23. `test_runtime_sprint3.py` (4 tests): Distributed target routing, multi-GPU planner, on-demand layer streaming.
24. `test_runtime_sprint4.py` (7 tests): Industrial runtime backends (K8s, Ray, Slurm), priority queue, cost tracking.
25. `test_science_reproducibility.py` (17 tests): Paper registry, dataset versioning manifests, gold-tier reporting, IOI, induction heads, greater-than, logit lens, SAE pipelines.
26. `test_sprint2_deliverable.py` (1 test): Sprint 2 debugger, inspector, patch, comparison, and report deliverable.
27. `test_sprint3_deliverable.py` (1 test): Sprint 3 workflow state machine, streaming, circuit discovery deliverable.
28. `test_sprint3_platform.py` (4 tests): Research workflow state machine, graph pipeline engine, dataset manager, plugin SDK.
29. `test_sprint4_deliverable.py` (1 test): Sprint 4 autonomous research deliverable.
30. `test_sprint5_deliverable.py` (1 test): Sprint 5 AI Scientist autonomous campaign deliverable.
31. `test_sprint6_deliverable.py` (9 tests): Sprint 6 meta-research engine, self-reflection, policy repository, campaign vector search, experience replay.
32. `test_validation_loop.py` (2 tests): Critic reproduction loop, metric gating against baselines, unimplemented paper handling.

---

## 3. Frontend Test Suite Architecture (`frontend/tests`)

### 3.1 Vitest Unit / Component Framework
- **Config File:** `frontend/vitest.config.js`:
  - Framework: Vitest 2.1.9 with `@vitejs/plugin-react`.
  - Environment: `jsdom`.
  - Globals: Enabled (`describe`, `it`, `test`, `expect`).
  - Setup File: `./tests/vitest/setup.js`.
  - Path alias: `@` -> `frontend/src`.
  - Included patterns: `tests/vitest/**/*.test.{js,jsx}`.

### 3.2 Mocking Infrastructure (`tests/vitest/setup.js`)
- Injects a complete browser mock for `window.appApi` (the Electron preload IPC bridge).
- Stubs:
  - Settings: `getSettings()`, `setSettings()`, `resetSettings()`.
  - Projects: `listProjects()`, `addProject()`, `removeProject()`.
  - Files: `listRecentFiles()`, `addRecentFile()`, `clearRecentFiles()`.
  - Python Sidecar: `pythonPing()` (returns `{ ok: true, echo: 'pong' }`), `pythonCall(method, payload)`.
  - Desktop Build / System: `startBuild()`, `getBuildLogs()`, `getAppLogs()`, `openExternal()`, `showInFolder()`.

### 3.3 Vitest Test Execution Results (100% Pass)
Command: `npm run test:js` (inside `frontend/`)
- **Result:** 27 test files passed (27 / 27), 119 tests passed (119 / 119), 0 failed. Duration: 35.72s.
- **Component Coverage Areas:**
  - Shell & Navigation: `App.test.jsx`, `desktopWindowState.test.js`, `desktopRouteRegistry.test.js`, `Settings.test.jsx`.
  - Model Inspection & Visualization: `TransformerVisualizer.test.jsx`, `Visualization.test.jsx`, `VisualizationSprint2.test.jsx` (SAEFeaturePanel, LogitLensViewer, ModelComparisonPanel), `VisualizationSprint3.test.jsx`, `VisualizationSprint4.test.jsx`, `VisualizationSprint5.test.jsx`.
  - Multi-Agent & Autonomous Research: `CampaignWorkspaceView.test.jsx`, `ReasoningTraceView.test.jsx`, `EvidenceFusionView.test.jsx`, `KnowledgeGraphView.test.jsx`, `ScientificHealthView.test.jsx`, `ResearchAnalyticsView.test.jsx`, `DiscoveryMemoryModal.test.jsx`, `ProvenanceBanner.test.jsx`.
  - Sprint Deliverable Views: `Sprint2Deliverable.test.jsx`, `Sprint3Deliverable.test.jsx`, `Sprint3Platform.test.jsx`, `Sprint4Deliverable.test.jsx`, `Sprint5Deliverable.test.jsx`, `Sprint5ExplorerPanels.test.jsx`, `Sprint6Deliverable.test.jsx`.
  - Client Communication: `desktopApiClient.test.js`.

---

## 4. Platform Workflow Test Coverage Matrix

| Core Platform Workflow | Backend Tests | Frontend Tests | Assessment & Coverage Gaps |
|---|---|---|---|
| **1. API Dispatching** | `test_api.py`, `test_protocol.py`, `test_advanced_evals.py`, `test_sprint6_deliverable.py` | `desktopApiClient.test.js`, `desktopRouteRegistry.test.js` | **High Coverage**. Dispatches over 60 RPC commands and 20 REST endpoints. **Gap:** Missing tests for malformed JSON error handling and HTTP status 422 validation. |
| **2. Model Runtime Adapters** | `test_model_adapters.py`, `test_adaptive_runtime.py`, `test_runtime_epics.py`, `test_runtime_sprint3.py`, `test_runtime_sprint4.py` | `DebuggerWorkflow.test.jsx`, `VisualizationSprint2.test.jsx` | **Comprehensive Coverage**. All 6 model families tested with mock mode. Causal patching, attention matrices, logits, and KV cache streaming verified. |
| **3. Data Persistence** | `test_repository.py`, `test_evidence_persistence.py`, `core/database.py` | `desktopWindowState.test.js` | **Moderate Coverage**. Activation repository and JSON graph store covered. **Gap:** SQLite database schema migrations and transactions lack dedicated integration tests. |
| **4. Interactive Visualization** | `test_neural_explorer.py`, `test_inspectors.py` | 15 Vitest visualization suites, 5 Playwright spec files | **Excellent Coverage**. All inspection panels, attention head heatmaps, circuit graphs, and DAG flows have dedicated component tests. |

---

## 5. Root-Cause Analysis of 13 Pytest Failures

When running `pytest tests/pytest` with `PYTHONPATH=".;backend"`, exactly 13 tests fail. Below is the exact root-cause analysis for each failure:

### Category A: IOI Reproduction Pipeline Mock Mode Contract (4 Failures)
- **Failing Tests:**
  1. `tests/pytest/test_science_reproducibility.py::test_ioi_pipeline_runs` (`KeyError: 'observed_metrics'`)
  2. `tests/pytest/test_science_reproducibility.py::test_ioi_pipeline_generates_manifest` (`KeyError: 'manifest_id'`)
  3. `tests/pytest/test_interpretability_sprint3.py::test_circuit_discovery` (`assert 0.5 < 0.0`)
  4. `tests/pytest/test_sprint3_deliverable.py::test_sprint3_deliverable_full_workflow` (`assert 0.5 < 0.0`)
- **Root Cause:**
  In `backend/science/reproducibility/ioi_pipeline.py` (lines 331–361), `run(mock_mode=True)` calls `self.adapter.get_logits(prompt)`. However, `GPT2Adapter._KNOWN_TOP_TOKENS` in `backend/science/models/gpt2_adapter.py` only contains 3 hardcoded prompt prefixes. The generated IOI prompts (`"When Alice and Bob went to the store..."`) do not match these prefixes, returning default fallback `" the"`. Because the indirect object (`Bob`) and subject (`Alice`) are not in `" the"`, `io_token` is `None`, and `ioi_pipeline.run()` returns `{"status": "unavailable", "reason": "Required IOI comparison tokens were not returned..."}` without `observed_metrics` or `manifest_id`.
  When `_handle_circuits_discover` in `backend/api/legacy_dispatcher.py` calls `_ioi_pipeline.run()`, `metrics` is empty, defaulting `circuit_score` to `0.0`.
- **Fix:** In `GPT2Adapter.get_logits` or `_KNOWN_TOP_TOKENS`, support IOI prompts by returning the subject and indirect object tokens in `top_tokens` when prompt contains `"When "` and `" went to the store"`, or support synthetic mock logit distribution for IOI comparisons.

### Category B: Evidence Graph & KG Write-Back Policy Mismatches (3 Failures)
- **Failing Tests:**
  5. `tests/pytest/test_evidence_persistence.py::test_from_run_has_no_demo_nodes` (`assert len(graph.nodes) == 1 + 7 + 3` -> actual 9)
  6. `tests/pytest/test_evidence_persistence.py::test_kg_writeback_compounds` (`assert len(first.get("stored", [])) >= 5` -> actual 0)
  7. `tests/pytest/test_evidence_persistence.py::test_kg_writeback_never_fails_run` (`assert 'error' in out` -> actual has `'reason'`)
- **Root Cause:**
  1. In `test_evidence_persistence.py`, the fixture `TRACE` lacks the fail-closed provenance keys required by `backend/agents/evidence_policy.py` (`"provenance": "live"`, `"status": "completed"`, `"validation_eligible": True`, `"publication_eligible": True`). Because of this, `_step_allows_evidence(step)` in `backend/core/evidence_graph.py` excludes synthetic numbers, creating 1 evidence node instead of 3.
  2. In `Scribe.write_back` (`backend/agents/scribe.py`), write-back is blocked because `publication_block_reason(trace, ...)` detects missing live provenance in `TRACE`.
  3. In `test_kg_writeback_never_fails_run`, `Scribe.write_back` returns `{"stored": [], "status": "blocked", "reason": "..."}`. The test asserted `'error' in out`, whereas the API contract uses `'reason'`.
- **Fix:** Update the test fixture `TRACE` to include explicit live provenance metadata conforming to `evidence_policy.py`, and align the test assertion in `test_kg_writeback_never_fails_run` to check `'reason' in out or 'error' in out`.

### Category C: Discovery Lifecycle State Boundary (2 Failures)
- **Failing Tests:**
  8. `tests/pytest/test_interpretability_sprint4.py::test_discovery_engine_run` (`assert 'Validation' == 'Publication'`)
  9. `tests/pytest/test_sprint4_deliverable.py::test_sprint4_end_to_end_deliverable` (`assert disc_res["lifecycle"]["state"] == "Publication"`)
- **Root Cause:**
  In `backend/interpretability/discovery/discovery_engine.py` (lines 98–106), `discover_and_orchestrate` transitions the discovery lifecycle up to `"Validation"` (`lifecycle.transition_to("Validation", ...)`). It deliberately leaves publication to the downstream `Scribe` agent in the Research Society workflow. However, the Sprint 4 unit and deliverable tests assert `assert res["lifecycle"]["state"] == "Publication"`.
- **Fix:** Either allow `discover_and_orchestrate` to complete the lifecycle transition to `"Publication"` upon successful live discovery, or provide a parameter/transition step allowing final publication state.

### Category D: AI Scientist Campaign Unhandled KeyError (1 Failure)
- **Failing Test:**
  10. `tests/pytest/test_sprint5_deliverable.py::test_sprint5_ai_scientist_end_to_end_deliverable` (`KeyError: 'confidence'`)
- **Root Cause:**
  In `backend/research_platform/autonomous/ai_scientist_engine.py` (line 87), `run_scientific_campaign` evaluates uncertainty:
  ```python
  uncertainty_decision = self.uncertainty_manager.evaluate_uncertainty(
      confidence_score=val_res["confidence"]["confidence_score"],
      uncertainty_interval=val_res["confidence"]["uncertainty_interval"],
      ...
  )
  ```
  However, `validation_engine.validate_discovery` returns `{"status": "unavailable", "validated": False, "reason": "..."}` when no live discovery was recalled. It does not contain a `"confidence"` key. Accessing `val_res["confidence"]` raises an unhandled `KeyError`.
- **Fix:** Safely handle missing confidence in `ai_scientist_engine.py` using `.get("confidence", {})` with fallback default confidence values (or handle unavailable validation status).

### Category E: Stale Test Expectations for Refactored Services (3 Failures)
- **Failing Tests:**
  11. `tests/pytest/test_sprint2_deliverable.py::test_sprint2_deliverable_workflow` (`assert "Layer 8 Pause" in report["markdown"]`)
  12. `tests/pytest/test_interpretability_sprint3.py::test_mechanistic_report` (`assert len(report["circuit_components"]) >= 3` -> actual 0)
  13. `tests/pytest/test_validation_loop.py::test_reproduce_live_report_and_gate` (`assert minimal["observed_value"] == 0.0` -> actual 0.9)
- **Root Causes:**
  1. `ReportService.generate_report` (`backend/services/report_service.py`) was refactored into a fail-closed transport summary ("No layer, intervention, probability, or mechanistic claim is synthesized here"). The test still expects the legacy synthetic phrase `"Layer 8 Pause"`.
  2. `_handle_mechanistic_report` in `legacy_dispatcher.py` delegates to `_ioi_pipeline.run()`, which returns 0 components in mock mode due to the same mock token issue as Category A.
  3. In `test_validation_loop.py`, the test asserted `# Minimality is honestly unmeasured, never fabricated: assert minimal["observed_value"] == 0.0`. However, `IOIReproductionPipeline` was upgraded to actively measure circuit minimality on live weights using leave-one-out head ablation (`minimality = 0.9`). The test assertion is stale.
- **Fix:** Align `test_sprint2_deliverable.py` and `test_validation_loop.py` to match the current live implementation behavior.

---

## 6. Deprecation Warnings & Modernization Opportunities

During the pytest run, 7 warnings were captured:
1. **SQLAlchemy 2.0 `declarative_base()` Warning**:
   - Location: `backend/core/database.py:16`
   - Issue: `declarative_base()` should be imported from `sqlalchemy.orm.declarative_base()`.
2. **FastAPI Lifespan Warning**:
   - Location: `backend/main.py:39`
   - Issue: `@app.on_event("startup")` is deprecated in modern FastAPI; use `lifespan` context manager.
3. **Starlette TestClient Warning**:
   - Location: `fastapi/testclient.py`
   - Issue: Using `httpx` with Starlette testclient is deprecated in favor of updated client configs.
4. **Pytest RemovedIn10Warning**:
   - Location: `tests/pytest/test_advanced_evals.py` lines 84 and 353.
   - Issue: Class-scoped fixtures defined as instance methods (`def infer_result(self)`) should be `@classmethod` or function-scoped.
