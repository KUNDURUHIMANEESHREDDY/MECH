<div align="center">

# MECH

**A mechanistic interpretability workbench that measures GPT-2 instead of describing it.**

[![provenance](https://img.shields.io/badge/results-measured%20from%20live%20weights-2563eb?style=flat-square)](#results)
[![backend](https://img.shields.io/badge/backend-FastAPI%20%2B%20torch-13795f?style=flat-square)](#architecture)
[![frontend](https://img.shields.io/badge/frontend-Vue%203%20%2B%20Vite-42b883?style=flat-square)](#architecture)
[![status](https://img.shields.io/badge/tests-186%20passed%20%C2%B7%2027%20failed%20%C2%B7%202%20errors-ca9500?style=flat-square)](#testing-status)

</div>

---

MECH loads **real GPT-2 small weights** (124 M, `transformers` + `torch`), runs genuine
causal interventions on them — activation patching, layer ablation, activation steering,
IOI circuit discovery — and shows you the numbers in a desktop workbench.

Every value MECH returns carries an explicit **provenance** tag: `live`, `seeded`,
`reference`, or `unavailable`. Views refuse to draw a conclusion the backend did not
actually measure. That constraint is the point of the project, and it is documented
honestly below, including the parts that are **not** implemented yet.

---

## Contents

- [What it does](#what-it-does)
- [Screenshots](#screenshots)
- [Results](#results)
- [Provenance: the honesty contract](#provenance-the-honesty-contract)
- [Real vs. not implemented yet](#real-vs-not-implemented-yet)
- [Quickstart](#quickstart)
- [Architecture](#architecture)
- [What's next](#whats-next)
- [Testing status](#testing-status)
- [Contributing](#contributing)

---

## What it does

MECH is an interpretability workbench built around one rule: **a result is either measured
from real weights or it is labelled as not-measured.** There is no third option.

Concretely, it does six things:

1. **Runs real GPT-2.** Loads actual HuggingFace `gpt2` weights and exposes architecture,
   per-token activations, attention readouts, neuron and head inspection, and sampling.
2. **Intervenes causally.** Zero-ablate any of the 144 attention heads, ablate any layer,
   clamp or patch any of the 36 864 MLP neurons, and inject a measured contrast vector into
   the residual stream. Effects are reported as logit differences, not scores.
3. **Discovers circuits.** Runs the standard IOI benchmark (clean vs. corrupted prompts),
   then screens every head to rank which ones the circuit actually depends on.
4. **Checks the engine against known behaviour.** Induction and counting are reproduced
   from real weights, so you can confirm the engine is reading GPT-2 rather than producing
   plausible noise. (Only `IOI` has a full benchmark executor today.)
5. **Refuses to fake the rest.** Validation and publication are gated behind proof that
   discovery was live. A missing executor returns `"unavailable"` with a reason, not a number.
6. **Shows its work.** A Vue 3 desktop UI with 31 tools, plus an Electron shell that boots
   the backend for you.

It is a research instrument, not a demo: the point is that a negative result stays negative.

---

## Screenshots

All screenshots below were captured from a live run of this repository against real GPT-2
weights — real API responses, zero console errors. Regenerate them with
`scripts/capture_screenshots_interactive.cjs`.

### Model Explorer — live inference and neuron inspection

Prompt run through real weights, with field-level provenance (`status: live · prompt: live ·
top5: live · …`) and the live MLP neuron inspector for `L0.mlp.N908`.

![Model Explorer](docs/images/ui/01-model-explorer.png)

### Network — parameter ledger counted from live weights

`124,439,808` parameters, 12 blocks, 144 attention heads and 36,864 MLP neurons, counted
directly from the loaded tensors rather than read from a config file.

![Network](docs/images/ui/03-network.png)

### Transformer Visualizer — architecture and attention readout

Residual-stream topology with per-head query/key/value norms read off the live model.

![Transformer Visualizer](docs/images/ui/02-transformer-visualizer.png)

### Steering Lab — a real intervention that flips the prediction

A contrast vector measured from `The capital of France is` minus `The capital of Japan is`,
injected at layer 10. The model's next token flips ` the` → ` Paris`.

![Steering Lab](docs/images/ui/04-steering-lab.png)

### Research Society — seven-agent workflow, all steps live

`load → reproduce → inspect → patch → discover → validate → publish`, every step stamped
`Provenance: live`, streamed to the UI over SSE.

![Research Society](docs/images/ui/06b-research-society-progress.png)

### Benchmark dashboard — live benchmark contract

The IOI benchmark executed against live weights: score `1`, pass rate `1`, provenance `live`.

![Benchmark dashboard](docs/images/ui/05-benchmark-dashboard.png)

---

## Results

Measured on GPT-2 small (124 M), CPU, `transformers` + `torch`. Reproduce with:

```bash
python scripts/capture_results.py     # ~35 s -> docs/results/capture.json
```

That script performs a real forward pass for every number below. Nothing is sampled from a
distribution or copied from a paper.

### Sanity checks — the engine reproduces genuine GPT-2 behaviour

| Task | Prompt | Top prediction | Expected |
|------|--------|----------------|----------|
| Induction | `Hello, my name is Julien. Hello, my name is` | `" Jul"` (p=0.437) | repeats the name stem ✅ |
| Counting | `1, 2, 3, 4, 5, 6,` | `" 7"` (p=0.951) | continues the sequence ✅ |

Measured perplexity on held-out prose: **44.9**, in GPT-2 small's normal range. The engine
is reading real weights, not generating plausible noise.

### IOI — the canonical circuit, reproduced

Indirect Object Identification, using the template from
[Wang et al. 2022](https://arxiv.org/abs/2211.00593). Three templates, all measured:

| Template | Clean prompt predicts | Corrupted prompt predicts | Correct / Flipped |
|----------|----------------------|---------------------------|-------------------|
| `…went to the store. John gave a bottle of milk to` | ` Mary` | ` John` | ✅ / ✅ |
| `…went to the park. John gave a bottle of milk to` | ` Mary` | ` John` | ✅ / ✅ |
| `…went to the office. John gave a key to` | ` Mary` | ` John` | ✅ / ✅ |

**Clean accuracy 1.0 (3/3), corrupted flip rate 1.0 (3/3).** Baseline clean logit difference
`2.1681`.

### Head ablation — which heads the IOI behaviour depends on

All 144 heads zero-ablated on the IOI prompt, ranked by how far the logit difference moves.
Baseline `clean_ld = 2.1681` on every row.

| Head | Ablated logit diff | Δ | Relative drop |
|------|--------------------|---|---------------|
| `L0H7` | `3.6692` | `+1.5011` | 69 % |
| `L0H0` | `3.4965` | `+1.3284` | 61 % |
| `L2H3` | `3.3778` | `+1.2097` | 56 % |
| `L2H0` | `1.2290` | `−0.9391` | 43 % |
| `L8H10` | `1.2671` | `−0.9010` | 42 % |
| `L5H9` | `2.9457` | `+0.7776` | 36 % |
| `L8H6` | `1.4272` | `−0.7409` | 34 % |
| `L10H7` | `2.8598` | `+0.6917` | 32 % |

![IOI head sweep](docs/images/ioi-head-sweep.png)

A positive Δ means the logit difference *grew* when the head was removed — the head was
suppressing the IOI answer. Early-layer heads (L0, L2) dominate because they carry the name
tokens themselves.

> **Honest caveat.** This is a single-prompt, single-direction zero-ablation screen, not a
> validated circuit. It does **not** reproduce the published IOI circuit, which needs
> corrupted-prompt patching and path patching to separate S1/S2/S3 components. Treat this as
> a first-pass screen, not a circuit claim. See [What's next](#whats-next).

### Layer ablation

Leave-one-layer-out on the same prompt. Every layer matters except L6.

| Layer | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 |
|-------|---|---|---|---|---|---|---|---|---|----|----|----|
| Δ | −3.58 | −3.90 | **−9.11** | −6.73 | −3.39 | −3.95 | +0.74 | −4.38 | −5.73 | −2.95 | −4.55 | −3.46 |

![Layer ablation](docs/images/ioi-layer-ablation.png)

Layer 2 is the most load-bearing (−9.11), and ablating layer 6 slightly *helps* (+0.74) —
a small negative-correlation component that a healthy measurement can surface.

### Activation steering — a measured flip

Contrast vector = residual stream on the positive prompt minus the negative prompt, added to
one block's output. Rank of `" Paris"` in the steered top-5:

![Steering sweep](docs/images/steering-sweep.png)

At **layer 10, α = 40** the prediction genuinely flips:

| | Clean | Steered |
|---|-------|---------|
| Top-1 | `" the"` | **`" Paris"`** |
| Top-5 | `the, now, a, France, Paris` | `Paris, the, France, a, now` |

Vector norm `79.6192`, provenance `live`.

### Honest negative results

These are real findings, not gaps to hide:

- **Logit lens does not recover `" Paris"` on GPT-2 small.** ` Paris` never becomes the argmax
  at any of the 12 layers for `The capital of France is`, and its probability stays near zero.
  GPT-2 124 M is too small to do this — the same run gives `" the"` as the top prediction.
- **The 6-sample IOI benchmark is trivially easy.** The live benchmark endpoint reports
  `score: 1.0, pass_rate: 1.0, eval_samples: 6`. That number is real but nearly meaningless
  statistically; it needs the full 100-template panel from the paper to be worth reporting.

---

## Provenance: the honesty contract

This is MECH's main design decision, so it is worth explaining. Every API response carries a
`provenance` field, and most carry a per-field `field_provenance` map:

```jsonc
{
  "status": "completed",
  "benchmark_name": "IOI",
  "score": 1.0,
  "pass_rate": 1.0,
  "eval_samples": 6,
  "provenance": "live",
  "field_provenance": {
    "status": "live", "benchmark_name": "live",
    "score": "live", "pass_rate": "live", "eval_samples": "live"
  }
}
```

| Tag | Meaning |
|-----|---------|
| `live` | Measured from loaded weights in this request. |
| `seeded` | Deterministic placeholder because weights are unavailable. Rendered with an amber badge, never as a measurement. |
| `reference` | Literature or catalogue metadata. Not a measurement of this model. |
| `unavailable` | No executor connected. Carries a `reason`. |

Three consequences worth knowing:

- **The evidence graph refuses synthetic numbers.** `backend/core/evidence_graph.py` will not
  promote a discover/validate result into the graph unless its provenance is `live`.
- **Validation and publication are gated.** `backend/agents/evidence_policy.py` requires
  `provenance == "live"` *and* an explicit opt-in before anything can be validated or published.
- **Missing endpoints stay missing.** Several views (`Analytics`, `Health`, `Evidence Fusion`,
  `Reasoning`) have no aggregate endpoint to call, and they say so in the UI rather than
  inventing a score. This is intentional and should be preserved.

If you are reviewing a MECH number, check its provenance tag first.

---

## Real vs. not implemented yet

An honest map, because the gap between these two columns is the most important thing to know
about the project.

### ✅ Real — measured from live weights

| Capability | Where |
|------------|-------|
| GPT-2 inference, sampling, architecture | `backend/services/gpt2_engine.py` |
| Logit lens (all layers) | `gpt2_engine.py:1356` |
| Attention-head zero-ablation / patching | `gpt2_engine.py:965` |
| MLP-neuron inspection and patching | `gpt2_engine.py:1015` |
| Leave-one-layer-out ablation | `gpt2_engine.py:1244` |
| Activation steering (contrast vectors) | `gpt2_engine.py:1288` |
| IOI clean vs. corrupted measurement | `gpt2_engine.py:1094` |
| Live IOI circuit screening | `backend/interpretability/discovery/live_discovery.py` |
| TransformerLens path (desktop sidecar) | `backend/interpretability/gpt2_model.py` |
| Society v2 seven-agent workflow + SSE | `backend/agents/society.py` |
| SQLite persistence, plugins, logging, KG | `backend/storage`, `backend/plugins`, … |

### ⚠️ Not real yet — do not cite these

| Feature | Reality |
|---------|---------|
| SAE / feature dictionaries | Config-only. Weights are never fetched; feature indices and activations are hardcoded (`interpretability/sae/loader.py:61`). |
| Tuned lens | No trained translators; the "affine translation" is `+0.12` on a number. |
| Causal tracing / attribution patching | Closed-form linear formulas, not measured gradients (`interpretability/causal/causal_tracing.py:19`). |
| Path patching | Self-documented as simulated (`discovery/algorithms/path_patching.py:76`). |
| ACDC fidelity | Analytic formula that can never fall below 0.90 (`discovery/algorithms/acdc.py:163`). |
| Non-GPT-2 models (Gemma, Llama, Qwen, Mistral, DeepSeek) | Unconditional mocks; `mock_mode` is accepted but never read (`science/models/model_adapters.py`). |
| `benchmark_database.json`, `*_certificate.json`, `ioi_benchmark.py` | Hand-written fixtures. Each JSON record now carries `fixture: true`, `measured: false`, `provenance: "reference"`, eligibility `false`, and a `fixture_notice`. **Never quote these.** |
| `backend/discovery/*` (13 modules) | Random grids via `np.random`. |
| `/api/v2/*` | Silently absent — the import is swallowed by a bare `except`. |

`GET /api/models` lists eight models; only GPT-2 is ever loaded. `/api/models/load` returns
`provenance: "unavailable"` with that reason. The registry is reference-only, by design.

---

## Quickstart

Requires Python 3.11+ and Node 18+.

```bash
git clone https://github.com/KUNDURUHIMANEESHREDDY/MECH.git
cd MECH

# Backend
pip install -r requirements.txt
python backend/main.py              # http://127.0.0.1:8000

# Frontend (second terminal)
cd frontend
npm install
npm run dev:renderer                # http://localhost:5173
```

Then open <http://localhost:5173/#explorer>, press **Load model**, and **Run prompt**.

Model weights download once (~500 MB) and cache under `~/.cache/huggingface`. Everything runs
on CPU; a GPU is optional.

### Desktop app

```bash
cd frontend
npm run dev          # Vite + Electron together; Electron spawns the backend
```

### Reproduce the results in this README

```bash
python scripts/capture_results.py                      # measurements + figures
python scripts/capture_screenshots_interactive.cjs     # screenshots (needs both servers up)
```

### Pinned ML stack

The ML dependencies carry **load-bearing upper bounds**. Verified working set:

| Package | Constraint | Verified on |
|---------|-----------|--------------|
| `torch` | `>=2.6` | 2.14.0+cu126 |
| `transformers` | `>=4.57,<5.0` | 4.57.6 |
| `transformer-lens` | `>=2.18.0,<3.0` | 2.18.0 |

`transformer-lens` 4.0 **removed `HookedTransformer`**, which three code paths depend on
(`backend/interpretability/gpt2_model.py`, `gpt2_steps.py`,
`backend/science/models/transformer_lens_adapter.py`). Compounding it, `transformer-lens` 2.x
declares `transformers>=4.57` with **no ceiling**, so pip will happily pair it with an
incompatible transformers 5.x — which is why *both* need capping.

All three call sites now resolve the class through `backend/interpretability/tl_compat.py`,
so an unsupported install fails with the supported range and the exact fix rather than a bare
`ImportError`. `tests/pytest/test_dependency_contract.py` guards the pins, including a
negative control that proves the check rejects a simulated 4.x install.

> **Resolved:** `backend/datasets/` was renamed to `backend/benchmark_datasets/`. The package
> no longer shadows HuggingFace `datasets` when `backend/` is on `sys.path`, and
> `tests/pytest/test_dependency_contract.py` now guards this as a hard regression test.

---

## Architecture

```
MECH/
├── backend/                    FastAPI, ~480 Python modules
│   ├── main.py                 entry point; mounts the dispatcher at /api and /api/v1
│   ├── services/gpt2_engine.py the real interpretability engine (1.4k lines)
│   ├── interpretability/       logit lens, patching, live circuit discovery, SAE (stub)
│   ├── agents/society.py       7-step agent workflow with fail-fast gates
│   ├── validation/             benchmark runner + provenance enforcement
│   ├── benchmarking/           9-task catalogue with literature baselines
│   └── science/                reproducibility pipelines, statistics, LaTeX export
├── frontend/                   Vue 3 + Vite + Tailwind, 31 tools
│   ├── src/desktop/routeRegistry.ts   34 routes, 31 in the sidebar
│   ├── src/desktop.css                the live "blue-primary" design system
│   └── electron/                     desktop shell that boots the backend
├── scripts/                    result + screenshot capture (this README's evidence)
└── docs/results/capture.json   the raw measurements quoted above
```

**Backend:** ~61 route decorators, mounted at both `/api` and `/api/v1`.
**Frontend:** 34 hash routes across Explore / Develop / Research / General groups.

---

## What's next

Ordered by what actually blocks trustworthy use, not by what is most exciting.

### 1. ~~Make a fresh install work~~ ✅ fixed

`requirements.txt` had no upper bounds, so a clean install pulled `transformers` 5.x and
`transformer-lens` 4.x — and TransformerLens 4.0 **removed `HookedTransformer`**, which
breaks `gpt2_steps.py` and the desktop sidecar.

**Done.** The ML stack is now pinned to a verified combination
(`torch 2.14` / `transformers 4.57.6` / `transformer-lens 2.18.0`), all three call sites resolve
through `backend/interpretability/tl_compat.py`, and
`tests/pytest/test_dependency_contract.py` guards it with a negative control.

Remaining in this area:

- [ ] Migrate to the TransformerLens 4.x `TransformerBridge` API and drop the `<3.0` pin.
      This is a real rewrite — the code uses hook-name conventions
      (`run_with_cache`, `hook_z`) whose v4 equivalents differ.
- [ ] Fix the `backend/datasets/` → HuggingFace `datasets` shadowing (see [1.4](#14-fix-the-datasets-package-shadowing)).

### 1.4 Fix the `datasets` package shadowing

MECH ships `backend/datasets/`, which is importable as the top-level name `datasets` whenever
`backend/` is on `sys.path` — which is exactly what `tests/pytest/conftest.py` does. Any
dependency doing `import datasets` (including transformer-lens) then resolves to MECH's package
and fails with `No module named 'datasets.arrow_dataset'`.

- [ ] Rename `backend/datasets/` (e.g. `backend/benchmark_datasets/`) and update importers.
- [ ] Or have `conftest.py` append `backend/` rather than prepending it, so installed packages win.
- [ ] The `xfail` control in `test_dependency_contract.py` flips to XPASS when fixed.

### 2. Get CI green

The suite is currently red and CI cannot be trusted.

- **Vitest fails outright**: `frontend/tests/vitest/desktopWindowState.test.js:3` imports
  `../../src/stores/desktop`, but `frontend/src/stores/` is an empty directory left over from
  the removed Desktop OS shell. Delete the test (or restore the store).
- **Pytest is 186 passed / 27 failed / 2 collection errors.** Most failures share one root
  cause: the code was hardened to **fail closed**, but the tests still assert the old
  synthetic-success contract. Typical examples — `test_ioi_pipeline_runs` expects
  `observed_metrics`, `test_sprint4_deliverable` expects lifecycle state `Publication`, both
  now correctly return `blocked` / `Validation` with `provenance: unavailable`.
  **Decide, then align the tests to the fail-closed contract.**

### 3. Fix four confirmed defects

| File | Defect |
|------|--------|
| `backend/benchmarking/benchmark_tasks.py:382,466` | `logger.warning(...)` is called but `logging` is never imported → `NameError` on the fallback path. |
| `backend/reasoning/__init__.py:3` | Imports `journey_tracer` and `neuron_debugger`; the directory contains only `__init__.py`. `import backend.reasoning` raises `ModuleNotFoundError`. |
| `backend/main.py:93` | Imports `backend.api.runtime_api`, which does not exist; a bare `except: pass` hides it so `/api/v2` silently vanishes. |
| `backend/interpretability/algorithms/registry.py:23` | Uses `List` without importing it → `NameError` when listing algorithms. |

### 4. Turn the IOI screen into a real circuit

The single-prompt zero-ablation screen in [Results](#head-ablation--which-heads-the-ioi-behaviour-depends-on)
does not separate S1/S2/S3. Real IOI circuit discovery needs: corrupted-prompt patching,
multi-template averaging, and a path-patching pass. The injection-recovery machinery in
`live_discovery.py` already exists and is the right foundation.

### 5. Make benchmarks statistically meaningful

The live IOI benchmark runs on 6 prompts and reports `score: 1.0`. Port the full
100-template panel and report a confidence interval. Bootstrap rather than the current
closed-form CI.

### 6. Implement or delete the stubs

Sparse autoencoders, tuned lens, attribution patching and path patching are currently
placeholders with convincing-looking output. Either implement them against real weights or
remove them — a stub that returns `0.94` is worse than an honest `unavailable`.

### 7. Frontend cleanup

- Tailwind is registered in `vite.config.mts` but **no CSS file imports it**, so no utility
  class resolves. `NeuralExplorerView.vue` and `TransformerExplorer.vue` are styled *entirely*
  with utilities and have no `<style>` block — they currently render unstyled.
- `frontend/desktop.css` is ~95 % dead (5 surviving class references).
- ~28 of 34 files in `src/services/` are unreferenced; `react`, `react-dom`, `reactflow`,
  `lucide-react`, `@vitejs/plugin-react` and `@vitejs/plugin-vue-jsx` are installed but unused.
- `npm run lint` fails: `eslint` is not a dependency and no config exists.

### 8. Packaging

`frontend/release/` has never been produced — `electron-builder` has not completed. Pin and
verify the build, then ship the installers the roadmap promises.

---

## Testing status

| Suite | Command | Status |
|-------|---------|--------|
| Python | `pytest tests/pytest -q` | **186 passed, 27 failed, 2 collection errors** |
| Frontend unit | `npm run test:js` | **8 passed, 1 file failed to load** |
| Playwright (offline) | `npx playwright test` | 4 specs, run in CI |
| Playwright (live backend) | `trust-online.spec.js` | Not in CI |

Only `tests/pytest/test_validation_loop.py` exercises real weights. Everything else runs
against mocks, so **a green suite would not tell you the interpretability is correct** — one
more reason to get the live path covered.

---

## Contributing

See [`docs/CONTRIBUTING.md`](docs/CONTRIBUTING.md). The short version:

- PEP 8, and type hints on new public functions.
- Add tests — and prefer a test against live weights over another mock.
- **If you touch anything that produces a number, preserve its provenance tag.** A result
  that loses its `live` marker is a regression even if the test passes.
- Document API changes in `docs/api_reference.md`.

---

## License

No license file is present in the repository yet. Treat the code as unlicensed until one is
added.

## References

- Wang et al., *Interpretability in the Wild: a Circuit for Indirect Object Identification in
  GPT-2 Small* — <https://arxiv.org/abs/2211.00593>
- Olsson et al., *In-context Learning and Induction Heads* — <https://arxiv.org/abs/2209.11895>
- Hanna et al., *Does Circuit Analysis Interpretability Scale?* (greater-than) —
  <https://arxiv.org/abs/2305.00586>
- Cunningham et al., *Sparse Autoencoders Find Highly Interpretable Features in Language Models* —
  <https://arxiv.org/abs/2309.08600>
- Elhage et al., *A Mathematical Framework for Transformer Circuits* —
  <https://transformer-circuits.pub>
