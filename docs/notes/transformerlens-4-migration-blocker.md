# TransformerLens 4.x (`TransformerBridge`) migration — blocked on a dependency conflict

Status: **not started, and not startable as a local refactor.** Recorded here so the
next person does not spend a day discovering it the way this one did.

## The finding

Every TransformerLens release from 3.6.0 onward declares **`transformers>=5.9.0`** (and 3.5.1 declares `transformers>=5.4.0`).
Verified by reading the wheel METADATA, not by inference:

| transformer-lens | requires |
|---|---|
| 3.5.1 | `transformers>=5.4.0` |
| 3.6.0 | `transformers>=5.9.0` |
| 3.7.0 | `transformers>=5.9.0` |
| 3.8.0 | `transformers>=5.9.0` |
| 3.9.0 | `transformers>=5.9.0` |
| 4.0.0 | `transformers>=5.9.0` |
| 4.1.0 | `transformers>=5.9.0` |
| 4.2.0 | `transformers>=5.9.0` |

MECH pins `transformers>=4.57,<5.0` in `requirements.txt`, and that upper bound is
**load-bearing**, documented in the README's "Pinned ML stack" table:

* `transformer-lens` 2.x declares `transformers>=4.57` with no ceiling, so pip will
  pair it with an incompatible 5.x.
* transformers 5.x imports `ResolvedRevision`, which `huggingface-hub` 1.0 removed,
  while `transformer-lens` 2.18 requires `hub<1.0`.

So "migrate to `TransformerBridge` and drop the `<3.0` pin" is not a refactor of
`gpt2_model.py`. It is a **coordinated ML-stack upgrade** (transformers 5.x +
huggingface-hub 1.x + transformer-lens 4.x), which is a project-level decision
because it invalidates the pinned combination the README calls verified and would
need re-validation of every live-weight path.

## Blast radius if the decision is made

33 dependent call sites across four files:

* `backend/interpretability/gpt2_model.py` — `from_pretrained`, `run_with_cache`
  (x2), `blocks.{l}.attn.hook_z` (reads at x4, forward hooks at x2)
* `backend/interpretability/tl_compat.py` — the compat shim itself
* `backend/science/models/transformer_lens_adapter.py` — `run_with_cache` (x3),
  `hook_z` patch site
* `gpt2_steps.py` — `from_pretrained` through the shim
* `backend/ui/circuit_explorer.py` — documentation reference only

## What a correct migration requires

1. Install the 5.x stack and re-verify the live-weight paths (the suite's
   `live-evidence` CI job loads GPT-2 for real; that is the gate).
2. Replace `HookedTransformer.from_pretrained` with the 4.x bridge entry point.
3. Replace `run_with_cache` / `hook_z` name conventions with their 4.x equivalents,
   preserving the `(1, seq, n_heads, d_head)` head-output shape the attention and
   patching code slices, or update every consumer of that shape.
4. Rewrite `tl_compat.py`: its whole purpose is to fail loudly on 4.x, so it either
   goes away or inverts into a "must be >= 4" assertion.
5. Re-run `tests/pytest/test_dependency_contract.py` and update the negative control,
   which currently asserts that a simulated 4.x install is *rejected*.

## Why it was not done in this session

The blocker is a version conflict, not missing code. Forcing
`transformers>=5.9.0` to satisfy the migration would silently invalidate the
verified pin the rest of the platform documents — exactly the class of change that
should be made deliberately, with the live-weight CI job green, rather than
alongside other work.
