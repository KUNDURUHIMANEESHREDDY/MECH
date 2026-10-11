# TL block-11 residual anomaly (open investigation)

Date: 2026-10-07. Status: OPEN, bounded, not load-bearing for any shipped
measurement.

## What was measured

Comparing MECH's `gpt2_engine` cache against TransformerLens
`HookedTransformer(gpt2-small)` on the prompt `"Hello world"`
(`prepend_bos=False`, all TL reparameterizations off):

| quantity | agreement |
|---|---|
| resid_post layers 0–10 | exact (max\|d\| ≤ 7.4e-4) |
| attention patterns layers 0, 11 | exact (max\|d\| = 7.7e-7) |
| MLP c_fc/c_proj weights, biases (all layers) | exact (0.0) |
| LayerNorm weights, eps (all layers) | exact (0.0) |
| MLP pre/post-GELU activations, layer 11 | exact (≤ 6e-6) |
| Q/K/V/O projection weights, layer 11 | exact (0.0) |
| final logits (incl. top-1) | exact (max\|d\| ≤ 1.4e-4) |
| final-layer LogitLens top-1 vs model top-1 | exact match |
| **resid_post layer 11** | **DIVERGES (max\|d\| ≈ 357)** |

## What was ruled out

- Stale cache: snapshot prompt asserted; plain-vs-hooked forwards identical.
- Leaked hooks: zero hooks registered on any block after a forward.
- Hooks perturbing numerics: plain forward (no hooks) == hooked forward exactly.
- Tokenization shift: both sides tokenize to `['Hello', ' world']`.
- TL BOS prepend: pinned `prepend_bos=False` (with BOS: max\|d\| = 16.5 everywhere).
- TL reparameterizations: `center_writing_weights=False`,
  `center_unembed=False`, `fold_ln=False`, `fold_value_biases=False`.
- Weight mismatch: every weight/bias compared at 0.0.
- Per-token affine difference: least-squares fit leaves max\|d\| = 357.

## Why it is bounded (not load-bearing)

1. Both pipelines are internally self-consistent: recomputing block 11
   from either side's intermediates reproduces that side's output.
2. Final logits agree to 1.4e-4 with identical top-1. Every causal
   quantity MECH reports (logit diffs, patch deltas, IOI verdicts,
   LogitLens) is a function of logits, so none of them can see this
   difference.
3. The divergence is confined to the last block's residual stream.
   Layers 0–10 — where all published IOI/induction heads live — agree.

## Hypotheses (unverified)

- A last-block code-path difference between HF transformers'
  `GPT2Block` and TL's `TransformerBlock` that only manifests with
  large outlier activations (GPT-2's known outlier dims 64/373/447/481
  carry values in the hundreds at layer 11).
- An unexamined TL processing flag beyond the four disabled above.

## RESOLVED (2026-10-10): the two sides were not measuring the same quantity

Measured, not inferred. `gpt2_engine._snapshot()["hidden"]` has **13** entries
for a 12-layer model, where a faithful residual record needs **14**:

| index | contents |
| ----- | -------- |
| `hidden[0]` | `wte + wpe` |
| `hidden[1]` … `hidden[11]` | block 0 … block 10 outputs, **pre**-`ln_f` |
| `hidden[12]` | `ln_f(block 11)`, **post**-`ln_f` |

The source is `gpt2_engine.py:456`, which copies HuggingFace's
`out.hidden_states`. That list appends the post-`ln_f` final residual as its
last element, so `hidden[12]` is post-norm — and, because the list is only 13
long, **block 11's own pre-norm output is not recorded at all**. The
post-norm value occupies its slot rather than appearing alongside it.

Evidence:

- MECH's `hidden[12]` equals `ln_f(block 11)` to `max|d| = 0.000e+00`.
- `hidden[i] == tl_resid_post[i-1]` for every `i` in 1…11, at
  `max|d| = 4.9e-04`.
- `tl_resid_post[11]` matches **no** entry in MECH's list. The best match is
  `hidden[12]` at `max|d| = 357` — the number this note recorded — and the
  true pre-norm block-11 residual is simply absent.

So the "divergence" was a comparison of a pre-`ln_f` quantity against a
post-`ln_f` one. The first hypothesis above (a block-level code-path
difference) was never needed, and neither was the outlier-dimension theory:
both were proposed to explain a difference that a single index shift and a
norm convention fully account for.

### Consequence found and fixed

`logit_lens(layer, ...)` reads `hidden[layer + 1]` and applies `ln_f`. That is
correct for layers 0–10, but for the last layer the entry is *already*
normalised, so `ln_f` ran a second time and the final LogitLens reported the
wrong distribution — see `backend/services/gpt2_engine.py` for the fix and
`tests/pytest/test_residual_decomposition.py` for the test.

It went unnoticed because the argmax still matched the model's, so the existing
token-identity check passed. Measured before the fix:

| token | lens reported | model |
| ----- | ------------- | ----- |
| `' the'` | 0.0381 | 0.0846 |
| `','` | 0.0193 | 0.0036 |

A token the model considered unlikely was promoted into the reported top five.

### Not fixed: the missing pre-norm entry

Block 11's pre-norm residual is still not recorded, so a reader cannot obtain
it. No current caller needs it — the logit-lens path is fixed, and
`test_each_block_reconstructs_from_its_own_components` documents the gap by
replaying it independently — so this is recorded rather than changed.
`test_last_block_resid_anomaly_is_tracked` still xfails, which is correct: it
compares `tl_resid_post[11]` against `hidden[12]`, and those remain different
quantities for as long as the recorded entry is post-norm.

Closing it would mean recording the pre-norm final residual as well, which
makes the list 14 entries and changes every `hidden[layer + 1]` read at the
final layer. That is a deliberate migration, not a bug fix.

## Tracking

`test_last_block_resid_anomaly_is_tracked` (xfail, non-strict) in
`tests/pytest/test_tensor_shapes_and_logit_lens.py` pins the current
state. If it ever XPASSes, someone explained it — update this note.
If it starts failing differently (different magnitude), the anomaly
moved and needs re-investigation.
