# MECH

Mechanistic interpretability research platform. Python + PyTorch +
TransformerLens backend, FastAPI control plane, React frontend. GPT-2 Small is
the initial model.

## Agent skills

### Issue tracker

Issues live in GitHub Issues on `KUNDURUHIMANEESHREDDY/MECH`, operated via the `gh` CLI. See `docs/agents/issue-tracker.md`.

### Triage labels

The five canonical triage roles use their own names verbatim: `needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`. See `docs/agents/triage-labels.md`.

### Domain docs

Single-context: one root `GLOSSARY.md` and one root `docs/adr/`, neither created yet. Read `docs/scientific_methods.md`, `docs/SCIENTIFIC_VALIDATION.md` and `docs/reproducibility.md` as the working vocabulary until they exist. See `docs/agents/domain.md`.

## Non-negotiable: no fabricated results

This is a research IDE. A test must never pass because an endpoint returned
200, a chart rendered, or a payload contained plausible-looking numbers. It must
verify the computation and its relationship to the correct model and experiment.

Concretely:

- An exception is never converted into a successful result.
- A registered-but-unimplemented engine reports `NOT_EXECUTED`, distinct from
  `FAILED`, and carries no data.
- Empty outputs never become zeros. Hard-coded metrics never become
  measurements.
- Evidence and confidence levels are never raised above what the run supports.
- Results carry provenance (model, revision, prompt, config, seed) and stale
  results can never be presented as current.

A negative test passes when MECH correctly rejects the operation or returns a
qualified result — not when it manufactures an answer.

## Testing

The suite is under `tests/pytest/` and takes roughly nine minutes.

```powershell
python -m pytest tests/pytest -q
```

`tests/pytest/conftest.py` installs a source-stability guard: it snapshots
watched sources at session start and end, and a run whose tree changed mid-run
is reported as **not trustworthy** rather than as a failure or a pass. That
report is not a test failure — re-run on a still tree. Override with
`MECH_SKIP_SOURCE_STABILITY_CHECK=1`. See
`docs/audit/source-stability-guard.md`.

Run targeted files while iterating; reserve full runs for checkpoints.

## Layering

| Layer | Location | Notes |
| ----- | -------- | ----- |
| Runtime | `backend/runtime/` | Model lifecycle, hooks, activation cache, patching |
| Engines | `backend/interpretability/` | Activations, attention, logit lens, ACDC, SAE |
| Validation | `backend/validation/` | Evidence grading, reproducibility, provenance |
| Control plane | `backend/main.py`, `backend/api/` | FastAPI, bearer-token gated |
| Storage | `backend/storage/` | |

`backend/interpretability/gpt2_model.py` is the real model adapter;
`backend/interpretability/mock_runtime.py` is not, and a passing test against
it proves nothing about GPT-2 behaviour.