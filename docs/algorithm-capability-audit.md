# Algorithm Capability Audit (2026-10-23)

This matrix ties registered algorithm names to code paths and regression tests. A registration, UI label, or successful mock run is not evidence that a published scientific method was executed. “Implemented” means an executable path exists; release readiness still requires the relevant live-weight test and applicable UI/API path test to pass on the current CI head.

| Capability | Code inspected | Existing verification | Readiness status |
|---|---|---|---|
| Algorithm/API registry | `backend/interpretability/discovery/algorithms/registry.py` | `tests/pytest/test_acdc_search.py` | Metadata registry exists. This test exercises ACDC registration and CircuitDiscoveryEngine integration, not every registry entry. |
| ACDC search and target-logit fidelity | `backend/interpretability/discovery/algorithms/acdc.py` | `tests/pytest/test_acdc_search.py`; `tests/pytest/test_acdc_fidelity_measured.py`; `tests/pytest/test_acdc_head_intervention_semantics.py` | Search is implemented. Mock mode intentionally reports unmeasured; live metric requires real weights. |
| Path patching | `backend/interpretability/discovery/algorithms/path_patching.py`; `backend/interpretability/discovery/live_measure.py` | `tests/pytest/test_live_discovery_path_patching.py` | Implemented for supported IOI inputs; refuses when `io_id`/`subject_id` are absent. |
| Attribution patching | `backend/interpretability/discovery/algorithms/attribution_patching.py` | `tests/pytest/test_attribution_patching_direction.py` | Implementation and direction-specific regression tests exist; current live CI evidence remains required. |
| Causal scrubbing | `backend/interpretability/discovery/algorithms/causal_scrubbing.py` | `tests/pytest/test_causal_scrubbing_semantics.py` | Implementation exists; semantic equivalence-class membership is only structurally checked by prompt length. Do not imply semantic classes are fully validated. |
| Sparse feature clustering | `backend/interpretability/discovery/algorithms/sparse_feature_clustering.py` | No dedicated end-to-end test file found in the current `tests/pytest` filename inventory. | Clustering math exists, but the complete user-facing path lacks a clearly dedicated regression test in the current inventory. |
| Transcoder dictionary learning | `backend/interpretability/discovery/algorithms/transcoders.py` | No dedicated transcoder-specific test file found in the current `tests/pytest` filename inventory. | Not implemented; refuses instead of simulating trained weights. Needs a trained dictionary and MLP input/output capture. |
| Cross-model feature universality | `backend/interpretability/discovery/algorithms/feature_universality.py` | No dedicated feature-universality-specific test file found in the current `tests/pytest` filename inventory. | Not implemented; refuses instead of returning fabricated alignment scores. |
| Tuned lens | `backend/interpretability/algorithms/tuned_lens.py` | `tests/pytest/test_tuned_lens_full_affine.py`; `tests/pytest/test_logit_and_tuned_lens.py` | Translator fit/apply/save/load implemented; live training tests are weight-gated and artifacts must be available at runtime. |
| Sparse autoencoders (SAE) | `backend/interpretability/sae/loader.py` | `tests/pytest/test_sae_checkpoint_live.py`; `tests/pytest/test_sae_real_training.py`; `tests/pytest/test_sae_routes.py`; `tests/pytest/test_sae_tensor_binding.py` | Loader/training routes and tensor-binding tests exist; runtime requires actual dictionaries or trained weights. Do not generalize one checkpoint test to universal availability. |

## CI acceptance

- The aggregate CI gate must include focused backend regressions and fail on failure, cancellation, skip, or any other non-success result.
- The full backend suite must complete from a fresh run. The prior run printed a failure at 2% and was cancelled at 21% before emitting the failing test name or traceback, so the first failure remains unidentified until the new fail-fast run starts.
- Weight-dependent live tests must remain visibly skipped/unavailable when model weights cannot be fetched; they must not be represented as live successes.
- The roadmap’s older “third of the advertised algorithm surface is still a stub” statement conflicts with later sections that describe path patching/ACDC as measured while also listing earlier open items. Reconcile those statements using test and live-run evidence rather than assuming all registered methods are production-ready.

## Confirmed gaps that must remain explicit

1. Transcoder dictionary learning is registered but unavailable without a trained dictionary and MLP input/output capture.
2. Cross-model feature universality is registered but unavailable; it must not report fixed alignment scores.
3. Sparse feature clustering has clustering math, but a dedicated end-to-end test for the user-facing path was not found in the test filename inventory.
4. Causal scrubbing’s semantic equivalence classes are not fully validated; the implementation records a structural prompt-length check, which is necessary but not sufficient.
5. The roadmap still lists Electron packaging as incomplete in Phase 3; verify packaged release artifacts before release.

This audit is a source/test inventory, not a claim that tests passed in the current CI run.