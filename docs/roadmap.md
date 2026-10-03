# Roadmap

This roadmap is derived from the code as it stands, not from aspiration. Every item below
corresponds to a verified gap. Evidence and measured results live in the
[root README](../README.md) and [`docs/results/capture.json`](results/capture.json).

## Where MECH is now

- Real GPT-2 small (124 M) loaded and measured via `transformers`.
- Working causal tooling: logit lens, head/neuron activation patching, layer ablation,
  contrast-vector steering, IOI clean-vs-corrupted, 144-head screening.
- 7-step Society workflow completing end to end with live provenance on every step.
- A 31-tool Vue 3 UI that renders measured data and refuses to invent what it lacks.
- **But:** a fresh install is broken by unpinned dependencies, CI is red, and roughly a
  third of the advertised algorithm surface is still a stub.

---

## Phase 1 — Make it installable and trustworthy

### 1.1 Pin dependencies ✅ done

`requirements.txt` had no upper bounds, so a clean install pulled `transformers` 5.x and
`transformer-lens` 4.x. TransformerLens 4.0 **removed `HookedTransformer`**, breaking
`gpt2_steps.py` and `backend/interpretability/gpt2_model.py`.

Fixed: the ML stack is pinned to a verified combination
(`torch 2.14.0+cu126`, `transformers 4.57.6`, `transformer-lens 2.18.0`). Note that
`transformer-lens` 2.x declares `transformers>=4.57` with no ceiling, so **both** libraries
need capping — capping transformer-lens alone is not enough.

Also fixed: all three call sites now resolve the class through
`backend/interpretability/tl_compat.py`, which distinguishes a version mismatch from a broken
environment and reports the supported range plus the exact install command.

- [x] Pin `transformers` and `transformer-lens` with upper bounds.
- [x] Add a compat shim with actionable errors.
- [x] Add `tests/pytest/test_dependency_contract.py`, including a negative control that
      proves the check rejects a simulated 4.x install.
- [ ] Add a CI job that runs `python -c "import backend.main"` as an import smoke test.
- [ ] Migrate to the TransformerLens 4.x `TransformerBridge` API and drop the `<3.0` pin.
      Substantial: the code depends on hook-name conventions (`run_with_cache`, `hook_z`)
      whose v4 equivalents differ, so it is a rewrite rather than a rename.

### 1.4 Fix the `datasets` package shadowing ✅ done

Found while writing the dependency tests. MECH shipped `backend/datasets/`, importable as the
top-level name `datasets` whenever `backend/` is on `sys.path` — which is exactly what
`tests/pytest/conftest.py` does. Anything doing `import datasets` (transformer-lens included)
then resolved to MECH's package and died with `No module named 'datasets.arrow_dataset'`.

This actively hid the version check, because transformer-lens became unimportable and the
compat probe skipped instead of failing. Confirmed empirically: with the shadowing present,
7 `test_dependency_contract.py` tests were red; after the rename all 7 pass.

Fixed by renaming `backend/datasets/` → `backend/benchmark_datasets/` and updating all
callers, including six modules that built the old path as a *string* and silently recreated
the directory at runtime (`GraphStore`, `CircuitDiscoveryEngine`, `MechanismClaimRegistry`,
`ResearchCampaignManager`, `ScientificPublicationEngine`, `distributed/scheduler`).

- [x] Rename the package and update importers, including the six runtime path writers that
      were recreating `backend/datasets/` after every publish.
- [x] `test_backend_datasets_shadows_huggingface_datasets` is no longer an `xfail`; it is a
      hard regression guard asserting `backend/datasets/` is never recreated.

### 1.2 Get CI green

- [ ] **Vitest fails to load a file.** `frontend/tests/vitest/desktopWindowState.test.js:3`
      imports `../../src/stores/desktop`, but `frontend/src/stores/` is an empty directory
      left from the removed Desktop OS shell. Delete the test or restore the store.
- [x] ~~**Pytest: 27 failed / 2 collection errors.**~~ **done** — the contract is now
      decided: *fail closed*. Where the code was already correct and the test asserted the
      old synthetic-success contract, the test was rewritten. Where the test found a real
      defect, the code was fixed. Both collection errors were real missing features and
      are now implemented (`lifespan` in `backend/main.py`, `checkpoint_wal` in
      `backend/storage/database.py`).

      Real defects found and fixed along the way:
      - `ai_scientist_engine.py` indexed `val_res["confidence"]["confidence_score"]`
        blind, so `{}`, `None`, or a bare string from validation raised
        `AttributeError`/`TypeError`/`IndexError` and killed the campaign. Now extracted
        defensively and routed to "More experiments" when evidence is absent.
      - The dispatcher coerced raw JSON with bare `int()`/`float()`, so `{"layer": "x"}`
        was a 500. All routes now validate and return 400.
      - `attention_head` / `activations` / the legacy inspectors ignored the request's
        prompt and read whatever the engine's one-prompt cache held — the stale-cache bug.
      - `interpretability/circuits/discover` turned a blocked pipeline into
        `circuit_score: 0.0` with no nodes and no reason; the mechanistic report then
        narrated "reaches faithfulness 0.0 via 0 nominated heads".
      - `PolysemanticityDetectorEngine` returned fixed fixtures with fabricated activation
        evidence and no provenance.
      - `circuit_minimality` was reported from a 6-prompt majority vote (your 2.2 item);
        now gated on `MIN_PROMPTS_FOR_MINIMALITY` and reported unmeasured below it.
- [ ] Make the live-weight test (`test_validation_loop.py`) a required CI job, not optional.

### 1.3 Fix confirmed defects

- [ ] `backend/benchmarking/benchmark_tasks.py:382,466` — `logger.warning(...)` without
      importing `logging` → `NameError` on the fallback path.
- [ ] `backend/reasoning/__init__.py:3` — imports two modules that do not exist;
      `import backend.reasoning` raises `ModuleNotFoundError`.
- [x] ~~`backend.main` had no `lifespan`.~~ **done** — added as an
      `@asynccontextmanager` wired via `FastAPI(lifespan=...)`, replacing the deprecated
      `@app.on_event` hooks. Shutdown checkpoints the SQLite WAL and tolerates a
      checkpoint failure rather than failing shutdown.
- [x] ~~`backend.storage.database` had no `checkpoint_wal`.~~ **done** — added as both a
      module function and a `DesktopStorage` method, using `PRAGMA wal_checkpoint(TRUNCATE)`
      and returning `(busy, log_pages, checkpointed)`. Lock contention is reported as
      `busy >= 1`, not raised.
- [x] ~~CORS allowlist allowed any origin / used `allow_methods=["*"]`.~~ **done** —
      loopback origins are matched with an anchored regex
      (`^https?://(localhost|127\.0\.0\.1|\[::1\])(:\d+)?$`) because Vite binds a random
      port; the anchoring is what rejects `localhost.evil.com` and
      `127.0.0.1.attacker.org`. Methods and headers are now explicit, `MECH_CORS_ORIGINS`
      entries are slash-normalised, and the env var can only *add* origins.
- [x] ~~`backend/main.py` imports a nonexistent `runtime_api` under a bare `except: pass`.~~
      **done** — the module's absence is announced at startup as a WARNING naming the
      consequence (404 = capability absent, not degraded); a module that exists but fails to
      import is logged with a traceback. Dispatcher load failure is now fatal.
- [x] ~~Two divergent FastAPI apps.~~ **done** — root `main.py` built a second `FastAPI()`
      instance with a different version string and a different CORS allowlist, so the active
      security posture depended on which file was launched. `main.py` is now a thin shim
      re-exporting `backend.main.app`; the GPT-2 preload moved into `backend/main.py`.
      The root shim also bound `0.0.0.0`; it now binds `127.0.0.1` like `backend/main.py`.
- [x] ~~`/health` returns `{"status": "healthy"}` and `/runtime/status` returns a fixed
      engine list.~~ **done** — `/health` stays a pure liveness probe (a probe that loads
      torch turns a slow dependency into a restart loop) and a new `GET /health/subsystems`
      probes process, dependency, model, executor, storage, and evidence. The status
      vocabulary has no `healthy` member: nothing there proves health, only that a
      subsystem was exercised and what it reported. `POST /api/runtime/status` now probes
      each engine for real reachability (including `sbatch` on PATH for Slurm) and only
      says `connected` when one verified.
- [x] ~~Make the live-weight test a required CI job.~~ **done** — `.github/workflows/ci.yml`
      has a separate `live-evidence` job that loads GPT-2, runs a real forward pass (so a
      stub cannot satisfy it), then runs `test_validation_loop.py`, the attestation hash
      check, and `test_evidence_boundary.py`. A `required` job `needs` all three, so a
      skipped live check cannot report success. The `backend` job also gained an import
      smoke test and an explicit `datasets`-shadowing assertion.
- [ ] `backend/interpretability/algorithms/registry.py:23` — uses `List` without importing it.

- [ ] ~~`python backend/main.py` failed with `ModuleNotFoundError: No module named 'backend'`~~
      — **done**, fixed by putting the repo root on `sys.path` in `backend/main.py`.

---

## Phase 2 — Make the science defensible

### 2.1 Real IOI circuit discovery

The current screen is a single-prompt, single-direction zero-ablation. It does not
separate S1/S2/S3, and the recovery metric it reports is not the same quantity as the
published circuit's — see below before drawing any conclusion from the two numbers.

- [x] ~~Average over many name pairs instead of one template.~~ **done** — all 100
      prompts shared one frame (`When A and B went to the store, A gave a drink to`),
      so every result was a statement about that frame. The pipeline now rotates
      through eight IOI frames deterministically (`_IOI_FRAMES`), varying the
      lead-in, place, verb, and item while preserving the ABB corruption, and
      reports `per_template` with `n_templates`, `templates_with_usable_prompts`,
      and `cross_template_consistent` (false unless ≥2 distinct frames produce a
      usable measurement that agree within 0.25).
- [ ] Add corrupted-prompt patching so the *IOI-specific* signal separates from generic
      token-identity effects (this is why L0/L2 heads dominate the current screen).
- [ ] Add a path-patching pass. `live_discovery.py` already has injection recovery — build on it.
- [x] ~~Report faithfulness of the recovered circuit, not just head ranking.~~ **done** —
      `_run_live` measures `(rec_diff - corrupted) / (clean - corrupted)` for the discovered
      circuit: the fraction of the corrupted-to-clean logit-diff gap that injecting the
      circuit recovers.

#### The 0.86 baseline is not the same measurement

`paper_registry.py` carries a published `circuit_faithfulness` of 0.86 (Wang et al. 2022).
Measuring **the published circuit itself** through this pipeline's harness gives
**0.60–0.71**, not 0.86, across prompt sets. The two numbers are therefore not
like-for-like, and reporting "MECH does not reproduce IOI" on the strength of
0.70 < 0.86 was an invalid comparison — a claim previously repeated in this file
and in session summaries.

What was true, and is now measured on every run:

| quantity | value |
|---|---|
| discovered circuit, this harness | ~0.72 |
| published circuit, **same harness, same prompts** | ~0.60 |
| ratio | **~1.21** |
| registry's published 0.86 | not comparable |

`observed_metrics` now carries `reference_circuit_faithfulness_same_harness`,
`faithfulness_vs_reference_circuit`, and `published_baseline_is_comparable: false`
with the reason attached. `Critic.reproduce` hoists the ratio to
`calibrated_vs_published_circuit`, and the gate records
`reference_basis_is_like_for_like: false` so `passed: false` cannot be misread as a
like-for-like failure.

The remaining gap is methodological, not a failure to find the circuit: this
harness measures head-set sufficiency, whereas the paper's figure comes from its
own path-patching procedure over a specific edge set. Closing that means
implementing the path-patching pass above, then re-deriving the reference through
the same code rather than citing a number from elsewhere.

#### What discovery actually returns

Worth recording, because the discovery is real (`discovery_provenance: live`, 10
heads, 16 edges, discovered per run) and it is *not* the published set:

| | |
|---|---|
| discovered (typical) | `L0H10 L0H8 L11H10 L1H3 L5H1 L6H9 L7H9 L8H10 L8H3 L8H6` |
| published (`REFERENCE_ONLY_DISCOVERED_HEADS`) | `L0H1 L0H10 L10H0 L10H7 L5H1 L5H5 L7H3 L8H6 L9H6 L9H9` |
| overlap | 3 of 10 — `L0H10 L5H1 L8H6` |

Yet the discovered set scores **higher** than the published set under the same
harness (0.724 vs 0.598, ratio 1.21). Two readings, both consistent with the data:
either the published head list is one sufficient set among several, or this
harness's discovery objective rewards redundancy over parsimony. Distinguishing them
needs the minimality statistic, which is gated at 10 usable prompts and therefore
not yet available at `n_prompts=6`. That gate is the blocker, and it is why the
minimality threshold is not negotiable.


### 2.2 Statistically meaningful benchmarks

- [x] ~~The live endpoint uses 6 samples and reports `score: 1.0`.~~
      **done** — `MIN_PROMPTS_FOR_MINIMALITY = 10` gates the majority-vote minimality
      statistic, and below the threshold it now reports `observed_value: None` with
      `circuit_minimality_measured: false` and a `Not Measured` tier. Reporting `0.0`
      made "not attempted" indistinguishable from "attempted and scored zero", and the
      unmeasured metric was dragging `overall_fidelity_pct` down with it — that average
      now excludes unmeasured metrics and says how many were excluded.
- [ ] Port the full 100-template IOI panel (the pipeline default is now 100 across eight
      frames; the HTTP endpoint still requests 6).
- [x] ~~Bootstrap confidence intervals instead of the closed-form `1.96·√(p(1−p)/n)`.~~
      **done differently** — the closed form was being applied to an *aggregate* metric
      with no trial counts behind it, and it is unreliable exactly where these benchmarks
      sit (at p≈0.9 it overruns the boundary; a 0.0 control yields a zero-width interval
      implying false precision). A **Wilson** interval is now computed from the pipelines'
      real per-prompt boolean outcomes (`correct`/`trials`), `confidence_interval_derived`
      records whether trials were available, and `confidence_interval_target` names what
      the interval bounds — for induction-heads that is the behavioural accuracy, not the
      headline score, which is a mean attention fraction.
- [ ] Add result persistence so the Benchmark dashboard can show history.

### 2.3 Implement or delete the stubs
Each of these currently returns a convincing-looking number without measuring anything.

- [ ] Sparse autoencoders — weights are never fetched (`sae/loader.py:61`).
- [ ] Tuned lens — no trained translators; confidence is `+0.12` on a number.
- [ ] Causal tracing / attribution patching — closed-form formulas, not gradients.
- [ ] Path patching — self-documented as simulated.
- [ ] ACDC fidelity — analytic formula that can never drop below 0.90.
- [ ] Non-GPT-2 adapters (Gemma, Llama, Qwen, Mistral, DeepSeek) — unconditional mocks.
- [ ] Delete or clearly quarantine the fixture files: `benchmark_database.json`,
      `*_certificate.json`, `ioi_benchmark.py`. **They read as results and are not.**

#### The continuous validation suite could not fail

`ValidationBenchmarkScheduler.execute_validation_suite` executed nothing. It computed
`current_fid = published_baseline_fidelity * 0.995`, `current_rt = baseline * 1.01`,
`current_vram = baseline * 1.0`, then scored `PASS` against a `baseline * 0.95`
threshold. Since 0.995 > 0.95, **every benchmark passed on every run.**
`HealthDashboardEngine.run_continuous_validation` turned that into
`pass_rate: 100.0` and wrote it into the knowledge graph as a `Continuous Validation
Run` experiment node.

Now each benchmark with an implemented pipeline is executed and scored on what it
measured; the rest report `NOT_RUN` with the reason. Current state:

| benchmark | status | fidelity |
|---|---|---|
| IOI | MEASURED | 0.724 |
| Induction heads | MEASURED | 0.670 |
| Greater-Than | NOT_RUN | — (task not performed: 0/10 above chance) |
| Arithmetic, SAE | NOT_RUN | — (no measurement implemented) |

The two NOT_RUN reasons are now different in kind, which matters: Greater-Than is
wired to a real pipeline that *measured* whether the model performs the comparison
and found it does not on this template (mean valid-minus-invalid logit difference
−0.81, so there is no circuit to localise and no fidelity number exists), while
Arithmetic and SAE have no measurement at all. Both report `NOT_RUN`; only one of
them has evidence behind the claim.

`MEASURED` is a distinct status from `PASS`: the pipelines ran, but their published
baselines are not the same measurement (see 2.1), so issuing a pass/fail verdict
would be spurious. Consequently `overall_pass_rate` is now `None` with
`pass_rate_measured: false`, rather than 100% or 0% — both of which were lies about
the same suite. `benchmarks_unscored: 5` says what the number is hiding — it counts
both `MEASURED` and `NOT_RUN`, since neither is a pass or a failure.

`RegressionDetector` also assumed `current_fidelity` was always a float and raised
`TypeError` on an unrun benchmark; it now skips those rather than treating a missing
value as zero, which would have manufactured a 100% regression for each.

### 2.4 Provenance attestation ✅ model half done

`ModelFingerprintEngine` used to return the literal `"sha256:8f43c...model_weights_mock"`
and a hardcoded `parameter_count=124_000_000`. Because `verify()` compared two calls to
`capture()`, `is_match` was **always `True`**: drift detection could never fire, while
`ScientificValidator` presented the result as provenance.

- [x] `capture()` hashes real artifacts — every tensor's name/dtype/shape/bytes in key order,
      the canonicalised config, and the tokenizer vocabulary. Parameter count, precision,
      architecture, and quantization are read from the model, not hardcoded.
- [x] Fails closed: a `mock_mode` adapter, a missing model, a missing tokenizer, or a hashing
      error all yield `attested=False` with a reason, and `verify()` refuses to report a match.
- [x] Bound downstream: `ScientificValidator.generate_validation_artifacts` raises rather than
      minting a certificate from unattested weights; `PaperValidator` rejects an unattested
      fingerprint rather than just checking it is non-empty.
- [x] `tests/pytest/test_model_attestation.py` (10 tests) — including a drift case that proves
      `verify()` can now actually fail.
- [ ] Still missing: bind `run_id`, `dataset_sha256`, and `code_revision` into the same
      attestation, so a result is bound to the full chain rather than only to weights.
- [ ] Hashing GPT-2 small reads ~500 MB on a cache miss. Acceptable now, but this belongs
      behind an explicit `attest()` call rather than running inside artifact generation.

**Principle:** a stub returning `0.94` is worse than an honest `unavailable`.

Applied so far:
- [x] `PolysemanticityDetectorEngine` — fixed fixtures, now labelled `seeded` /
      `unavailable` and ineligible; its fabricated activations carry `measured: False`.
- [x] `GPT2Adapter.get_logits` in mock mode — labels every response `seeded` +
      ineligible, and deliberately has **no** name-parsing heuristic that could turn
      "When Xavier and Yolanda ... gave a drink to" into a plausible IOI answer.
- [x] `interpretability/circuits/discover` — a blocked pipeline no longer becomes
      `circuit_score: 0.0`; it returns the block and its reason with no score at all.
- [x] `interpretability/reports/mechanistic` — no longer narrates a mechanism it did
      not measure.
- [x] ~~Still to sweep: the remaining stubs.~~ **done** — every stub now carries
      provenance, a reason, and eligibility flags: `sae/loader.py`,
      `algorithms/tuned_lens.py`, `causal/causal_tracing.py`,
      `causal/attribution_patching.py`, `discovery/algorithms/path_patching.py`,
      `discovery/algorithms/acdc.py`, `science/models/model_adapters.py` (all five
      families), `circuit_evolution.py`, `feature_genealogy.py`,
      `cross_model_circuits.py`, `semantics/auto_circuit_namer.py`,
      `evidence_ranker.py`, `confidence_scorer.py`, `confidence_calibration.py`,
      `polysemanticity_detector.py`.
- [x] ~~Delete or clearly quarantine the fixture files~~ **done** —
      `benchmark_database.json` and all five `completion_certificate.json` files
      now carry `fixture: true`, `measured: false`, `provenance: "reference"`,
      `validation_eligible: false`, `publication_eligible: false`, and a
      `fixture_notice` naming the literals. Each notice also states why it could
      not have been measured (the Gemma adapter runs no model; SAE loads no
      encoder weights; ACDC fidelity is never computed).
- [x] ~~ACDC fidelity — analytic formula that can never drop below 0.90.~~
      **done** — `logit_recovery_fidelity` was `0.90 + 0.09 * (1 - circuit_score)`,
      a rescaled pruning ratio reported under a name asserting recovered logits,
      and it doubled as the algorithm's confidence. It is now `None` with
      `logit_recovery_fidelity_measured: false`; the graph score is the pruning
      ratio under its own name; confidence is 0.0 because nothing here
      calibrates one.
- [x] ~~Non-GPT-2 adapters — unconditional mocks.~~ **done** — the five adapters
      ignored `mock_mode` entirely and fabricated regardless of it. Each
      constructor now calls `_force_simulated`, forcing `spec.mock_mode = True`
      so downstream provenance checks see simulated data; `ModelAdapter` carries
      `simulated` / `simulation_reason`.
- [x] Further stubs found and quarantined in the same pass:
      `semantics/feature_labeler.py` (confidence 0.94 with no model called),
      `sae/inspector.py` (hardcoded neuron weights 0.85/0.62 and activations
      4.2/3.8 for *every* feature id, from which `max_act` looked measured),
      `meta/scientific_skill_library.py` (`execute_skill` returned
      `output_state="Success"` with six extracted circuit nodes and
      `reproducibility_verified=True` while running nothing),
      `runtime/orchestration/learned_runtime_optimizer.py` (a constant `92.5`
      reported as `predicted_gpu_utilization_pct`).
- [ ] Remaining sweep, not yet started: `research_platform/meta/` still has
      hardcoded confidences in `experience_replay.py` (0.94), `self_reflection_engine.py`
      (0.94), `meta_research_engine.py` (0.94), `research_strategy_optimizer.py`
      (0.94), `autonomous/knowledge_base.py` (0.94), and
      `autonomous/scientific_consensus_engine.py` (0.94), plus
      `discovery/benchmark_registry.py`, `discovery/ioi_benchmark.py`,
      `discovery/induction_head_detector.py`, and `discovery/circuit_discovery.py`'s
      mock branch.



### 2.5 Provenance is derived, not supplied ✅ done

`Scribe.report()` accepted a `provenance` argument, passed it into
`ReportService.generate_report()`, and returned it as the report's own provenance —
so `report(experiment_id="fake", provenance="live")` produced a live-labelled
artifact with no evidence chain behind it. Provenance is now computed from the
trace the caller supplies: no trace, or a trace that does not clear
`publication_block_reason`, returns a blocked envelope and no report.

- [x] `report()` takes the trace and derives the label; there is no longer a way to
      state your own provenance.

### 2.6 Single EvidenceBoundary ✅ done

`backend/core/evidence_boundary.py` is the one place a measurement becomes a result.

- `EvidenceResult` is a frozen dataclass, not a dict. An algorithm returns a
  measurement; only the boundary can hand a caller something that looks like a finding.
- `RunAttestation` binds a claim to `run_id`, `executor_id`, `model_id`,
  `weights_sha256`, `dataset_sha256`, and `code_revision`. It rejects the placeholders
  this repo used to ship — anything containing `...` or `mock`, anything that is not a
  64-char hex digest, and the literal `unattested`.
- `EvidenceBoundary.admit()` **only ever downgrades a provenance claim.** A `seeded`,
  `reference`, or `unavailable` claim stays that way; nothing is promoted except by a
  verifying attestation. `live` without an attestation is rejected as "a claim is not a
  proof", and an attested run that produced no measurement is "not a finding".
- `tests/pytest/test_evidence_boundary.py` (11 tests) covers each of these, plus a guard
  that eligibility is computed in exactly one place.

- [ ] Still to do: route the remaining call sites through it. `evidence_graph.from_run`
      and the dispatcher `_mark()` are now wired (see 2.7), but the Society agents
      (`Scribe`, `Critic`, `Discoverer`) still construct their own envelopes.
- [ ] Full attestation binding: a result should carry `dataset_sha256` and
      `code_revision` derived from the run, not merely required to be non-empty.

### 2.7 Evidence graph and dispatcher wired to the boundary ✅ done

- `evidence_graph.from_run` initialised `evidence_provenance = "live"` and only
  overwrote it when a result happened to carry a `provenance` key — so any step
  whose result omitted one produced a **live-labelled evidence node**. Steps
  other than `discover`/`validate` skip the scientific eligibility gate
  entirely, which is how a seeded measurement could reach the graph as live.
  Provenance is now derived by `_step_provenance`, which reads the step's own
  declaration and returns `unavailable` for anything absent or unrecognised.
  A node labelled `live` also carries `attested: False`, because a declaration
  is not an attestation.
- The dispatcher's `_mark()` let a route assert `live` on the engine's behalf
  (`engine.is_available()` says the stack imports, not that this response came
  from a forward pass over known weights). It now preserves any provenance the
  engine set, and marks every response `attested: False`.

### 2.8 Real implementations replacing the three load-bearing stubs ✅ partly

**Path patching — now real.** The procedure needs *two simultaneous*
interventions on the corrupted run: freeze the sender at its clean output, and
swap the clean value into the receiver's input. The old code patched only the
receiver, so the measured delta was the sender's *total* effect, and it attached
a fabricated `0.5 + delta` confidence to every edge. `live_measure.path_patch`
now installs both hooks (a forward hook on the sender's `attn.c_proj`, a
forward-pre hook on the receiver's), and `PathPatchingAlgorithm` delegates to
it. `implements_published_method` is True again, and confidence is the mean
absolute isolated path effect — a measured quantity, not a rescaled score.
Without `io_id`/`subject_id` it reports unavailable rather than an edge it
cannot justify.

**ACDC fidelity — now measured.** `live_measure.circuit_fidelity` injects the
retained heads into the corrupted run and divides the recovered logit
difference by the clean-minus-corrupted gap. `ACDCAlgorithm` calls it and
reports `logit_recovery_fidelity_measured`. When the live engine is absent it
says so — the generic adapter's single-site `patch_activation` cannot restore a
whole circuit at once, so there is no approximation to fall back on.

**SAE — loader fixed, no weights available.** There are no trained SAE encoder
weights in this repository, so a genuinely working encoder cannot be finished
here. The surrounding machinery is fixed:

- Providers raise `SAELoadError` when real weights cannot be fetched, instead of
  returning a config-only `SAE` reported as `status: "loaded"`.
- `checkpoint_sha` was synthesised from the checkpoint's own name
  (`f"sha256_{identifier}_{version}"`), so `verify_integrity` was string
  equality against a value derived from the argument being checked. It now
  hashes the actual tensors and the file on disk.
- `activate()` performs the real encoding — `ReLU((h - b_pre) @ W_enc + b_enc)`,
  top-k — when weights are present, and returns an explicit unavailable
  envelope when they are not.
- `AnthropicProvider` raises instead of returning a config.
- [ ] Train or vendor an SAE for GPT-2 layer 8/10 and point `MECH_SAE_PATH` at
      it. Until then every SAE view must stay unavailable.

### 2.9 `research_platform/meta/` sweep ✅ done

Six modules fed an agent confident-looking conclusions. All now say what they
actually have:

- `experience_replay` — the seed "remembered" that an SAE L8 checkpoint was
  available, that the IOI benchmark scored 0.94, and that a logit boost was
  confirmed, then recommended a policy from it. The seed is `recorded: False`
  with `provenance: "reference"`, and `policy_insight` is `None` for an
  unrecorded trajectory.
- `self_reflection_engine` — invented two successes (including "SAE Feature
  #1402", impact 0.92) and two failures attributed to "Low causal effect" when
  no hypotheses were supplied, and credited "utilized Sparse Autoencoder feature
  genealogy". With no input it now reflects on nothing and says why.
- `research_strategy_optimizer` — reported `historical_effectiveness_score:
  0.94` with reasoning claiming it came from "meta-learning empirical success
  analysis". No analysis exists; the score is now `None` with
  `score_is_assumed: True`.
- `meta_research_engine` — saved a policy with `success_rate=0.94`,
  `average_cost_usd=3.85`, and `average_runtime_min=8.2`, all literals. Now
  `None`, with `policy_metrics_measured: False`.
- `knowledge_base` — seed facts carried `confidence: 0.94` / `0.96`, including a
  "SAE Feature #1402 fires on name tokens" fact for a subsystem that cannot run.
  Now `0.0` with `measured: False` and a reason.
- `scientific_consensus_engine` — returned the fixed sentence "L8_N402 acts as
  primary Indirect Object Identifier across IOI prompts" at confidence 0.94 for
  *any* input, including an empty list. It now reports the verdict distribution
  of the supplied outcomes and synthesises no mechanism at all.



---

## Phase 3 — Frontend and packaging

- [ ] **Fix Tailwind.** It is registered in `vite.config.mts` but no CSS file imports it, so
      no utility class resolves. `NeuralExplorerView.vue` and `TransformerExplorer.vue` are
      styled entirely with utilities and have no `<style>` block — they render unstyled today.
- [ ] Delete dead code: ~95 % of `frontend/desktop.css`, ~28 of 34 files in `src/services/`,
      the stale `Sidebar.vue` / `Topbar.vue` / `StatusBar.vue` / `ActivityBar.vue`, and the
      unused `react` / `react-dom` / `reactflow` / `lucide-react` dependencies.
- [ ] Repair or remove `npm run lint` — `eslint` is not a dependency and no config exists.
- [ ] Complete the Electron packaging build; `frontend/release/` has never been produced.
- [ ] Replace stale root debug captures (`broken-state.png`, `current-state*.png`,
      `fixed-state.png`) with the current screenshots under `docs/images/ui/`.

---

## Phase 4 — Scale

Only worth starting once Phases 1–2 hold.

- [ ] Additional real models beyond GPT-2 (currently the registry is reference-only).
- [ ] Persistent benchmark result store and run comparison.
- [ ] Multi-node execution for large sweeps (the 144-head screen takes ~25 s on CPU).
- [ ] Paper-level export: LaTeX figure/table generation wired to the measured pipeline.

---

## Invariants to protect

These are working today. Regressions against them are bugs, even when tests pass.

1. **Every number carries provenance.** `live` / `seeded` / `reference` / `unavailable`.
2. **The evidence graph rejects synthetic results** (`backend/core/evidence_graph.py`).
3. **Publication is gated on live provenance** (`backend/agents/evidence_policy.py`).
   Provenance is *derived from the trace*, never accepted as an argument.
4. **Views state their own limits** rather than rendering a placeholder metric.
5. **A missing executor returns `unavailable` with a reason** — not a plausible number.
   This includes a blocked pipeline: it must not be flattened into a `0.0`.
6. **Absent evidence is not low-confidence evidence.** A validation call that returns
   nothing usable routes to "More experiments", never to a publishable default score.
7. **A provenance claim is not a proof.** `live` is granted only by
   `EvidenceBoundary.admit()` against a verifying `RunAttestation`; nothing is promoted
   from a lower provenance by assertion.
8. **Health is probed, not asserted.** `/health` is liveness only; `/health/subsystems`
   exercises each subsystem. No endpoint reports `healthy` or `connected` without a probe.

