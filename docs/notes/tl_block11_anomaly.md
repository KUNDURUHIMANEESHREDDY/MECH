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

## Tracking

`test_last_block_resid_anomaly_is_tracked` (xfail, non-strict) in
`tests/pytest/test_tensor_shapes_and_logit_lens.py` pins the current
state. If it ever XPASSes, someone explained it — update this note.
If it starts failing differently (different magnitude), the anomaly
moved and needs re-investigation.
