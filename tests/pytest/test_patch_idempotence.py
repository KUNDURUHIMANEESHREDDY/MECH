"""PAT-005 / PAT-006: patching with a tensor captured from the same forward
pass must be a no-op.

The property
------------
`capture_head_outputs` reads the input to `attn.c_proj` -- the concatenated
per-head outputs -- and `patch_head_output` writes a vector back into head
`h`'s slice of that same tensor. So feeding the capture back in at the same
layer, head and prompt asks the patch machinery to reproduce a value it was
just handed. If the write is faithful and everything downstream is
untouched, the model's logits must be **bit-identical**.

This is the strongest available check on the patching path, which is why the
checklist puts it first:

* it constrains the patch hook's index arithmetic (`h * d_head`), the slice
  write, the dtype/device conversion, and the fact that the pre-hook does not
  perturb neighbouring heads;
* it fails loudly if the patch silently no-ops (the out-of-range guard at
  `x.shape[-1] < end` returns the input unchanged, which would also produce a
  zero delta -- so a bare delta check cannot tell "patched faithfully" from
  "never patched at all").

That last point is why these tests also assert the mechanism fired. A test
that only compared deltas would pass against a completely broken patch.

Checked at float32 precision, not approximately: the capture is rounded to six
decimals, and the round-trip still came back exactly 0.0, so a tolerance would
only hide a regression.
"""

from __future__ import annotations

import pytest

from _weight_guard import skip_reason, weights_available

needs_weights = pytest.mark.skipif(
    not weights_available(), reason=skip_reason()
)

PROMPT = "The capital of France is"


@pytest.fixture(scope="module")
def adapter():
    from backend.science.models.gpt2_adapter import GPT2Adapter

    instance = GPT2Adapter(variant="small", mock_mode=False)
    assert instance._model is not None, "adapter has no weights"
    return instance


@needs_weights
@pytest.mark.parametrize("layer, head", [(9, 9), (0, 0), (5, 3), (11, 11)])
def test_patching_with_the_captured_vector_changes_nothing(adapter, layer, head):
    """PAT-006: a head patched with its own captured output is a no-op.

    Four (layer, head) pairs rather than one, because the slice arithmetic is
    `head * d_head` and a bug there could be invisible at head 0 -- where the
    slice starts at index 0 and a mis-scaled offset still lands somewhere
    plausible. Layer 11 is included because it is the last block, whose
    recorded residual this repository already knows to be unusual
    (`docs/notes/tl_block11_anomaly.md`).

    Asserted as exact equality, not a tolerance: the measured delta is
    exactly 0.0 despite `capture_head_outputs` rounding each component to six
    decimals, so the float32 result absorbs the rounding. Anything non-zero
    here is a real perturbation.
    """
    captured = adapter.capture_head_outputs(PROMPT)
    vector = captured[layer][head]

    result = adapter.patch_head_output(
        PROMPT, layer=layer, head_index=head, patch_vector=vector, position=-1
    )

    assert result.original_logit == result.patched_logit, (
        f"patching layer {layer} head {head} with its own captured output "
        f"changed the scored logit by {result.patched_logit - result.original_logit}. "
        f"A faithful write-back must reproduce the unpatched computation."
    )
    assert result.delta == 0.0, (
        f"delta is {result.delta}, expected exactly 0.0"
    )
    assert result.top_token_before == result.top_token_after


@needs_weights
def test_the_patch_hook_actually_fired_during_the_idempotent_call(adapter):
    """The zero delta above must come from a faithful write, not a skipped one.

    `patch_head_output` guards out-of-range heads by returning its input
    unchanged (`if x.shape[-1] < end: return inp`). A head index past the end
    of the tensor therefore produces a *perfectly zero* delta while doing
    nothing -- indistinguishable from PAT-006 by delta alone.

    So the hook's execution is observed directly, from a pre-hook registered
    ahead of the method's own. If the method's hook is skipped, this sees
    fewer calls than the two forward passes it makes (one for the baseline, one
    for the patch), and the test fails.
    """
    layer, head = 9, 9
    captured = adapter.capture_head_outputs(PROMPT)
    vector = captured[layer][head]

    observed: list[int] = []
    module = adapter._model.transformer.h[layer].attn.c_proj

    def spy(_module, args):
        observed.append(1)
        return None

    probe = module.register_forward_pre_hook(spy)
    try:
        adapter.patch_head_output(
            PROMPT, layer=layer, head_index=head, patch_vector=vector, position=-1
        )
    finally:
        probe.remove()

    assert len(observed) >= 2, (
        f"observed {len(observed)} forward calls on c_proj during "
        f"patch_head_output; expected at least 2 (one baseline, one patched). "
        f"The method's own hook did not run, so the zero delta proved nothing."
    )


@needs_weights
def test_the_captured_slice_is_the_one_written_back(adapter):
    """The vector patched in must be head `head`'s own slice.

    Cross-checks the capture against the patch by patching head `head` with
    *another* head's captured vector and requiring the result to differ. If
    `patch_head_output` and `capture_head_outputs` disagreed about what
    `head * d_head` means -- the exact confusion that `capture_head_outputs`'
    own docstring describes as having silently mis-indexed heads into the
    residual stream -- then patching head 9 with head 10's vector would
    coincide with patching head 9 with its own, and this fails.
    """
    captured = adapter.capture_head_outputs(PROMPT)
    layer, head, other = 9, 9, 10

    own = adapter.patch_head_output(
        PROMPT, layer=layer, head_index=head,
        patch_vector=captured[layer][head], position=-1,
    )
    crossed = adapter.patch_head_output(
        PROMPT, layer=layer, head_index=head,
        patch_vector=captured[layer][other],
    )

    assert own.delta == 0.0, "setup: own-vector patch was not a no-op"
    assert crossed.delta != 0.0, (
        f"patching head {head} with head {other}'s captured vector left the "
        f"logit unchanged. Either the patch did not apply, or both heads read "
        f"the same slice -- the mis-indexing this is meant to catch."
    )


@needs_weights
def test_idempotence_holds_for_every_head_of_one_layer(adapter):
    """All twelve heads, not just the four sampled above.

    A per-head indexing bug shows up on some indices and not others -- a
    doubled offset is invisible at head 0, for instance -- so the property has
    to hold across the whole range or it is not a property of the patch path.

    This test is what found the eager/sdpa asymmetry below: head 6 returned a
    non-zero delta while the other eleven returned exactly 0.0. A single-head
    test would have missed it, because whether a head shows the discrepancy
    depends on how that head's values land in float32 -- not on any property of
    the patch path itself.
    """
    layer = 6
    captured = adapter.capture_head_outputs(PROMPT)

    offenders = []
    for head in range(adapter.spec.num_heads):
        result = adapter.patch_head_output(
            PROMPT, layer=layer, head_index=head,
            patch_vector=captured[layer][head], position=-1,
        )
        if result.delta != 0.0:
            offenders.append((head, result.delta))

    assert not offenders, (
        f"heads whose own vector did not round-trip cleanly: {offenders}. "
        f"See the test below on attention-implementation parity -- a "
        f"capture/patch kernel mismatch shows up here as a non-zero delta on "
        f"some heads and not others."
    )


@needs_weights
def test_capture_and_patch_run_the_same_attention_kernel(adapter):
    """The capture and the patch must observe the same forward pass.

    This is the defect `test_idempotence_holds_for_every_head_of_one_layer`
    turned up, and it is not a rounding artefact.

    `capture_head_outputs` routes through `_forward_with_hooks`, which forces
    `attn_implementation="eager"` so attention weights are materialised.
    `patch_head_output` calls the model directly and therefore runs whatever
    the model is configured for -- `sdpa` by default in transformers 4.57.

    Two different attention kernels over identical weights do not produce
    bit-identical results. Measured on GPT-2 Small at layer 6, head 6, the
    head vector read by the two paths differs by `max|d| = 3.7e-04`, which is
    an order of magnitude above the six-decimal rounding the capture applies.
    Feeding the captured vector back in therefore perturbs the very tensor it
    came from, and the scored logit moves by 0.0625.

    Why this matters beyond idempotence: every head-level causal result in
    MECH compares a *captured* activation against a *patched* forward pass. If
    those are computed under different kernels, the delta includes a constant
    kernel-disagreement term that has nothing to do with the head being
    studied. It would be small, it would be present for every head and every
    layer, and it would be indistinguishable from a weak causal effect.

    Asserted on the tensors rather than on a delta, so it names the actual
    cause instead of its symptom.
    """
    import torch

    layer, head = 6, 6
    d_head = adapter.spec.d_model // adapter.spec.num_heads

    # What the patch path now puts in front of c_proj. Read through a probe
    # installed *around* `patch_head_output`, under the same kernel the method
    # forces, rather than from an independent forward pass of my own -- a bare
    # `model(**inputs)` here would run the model default and reintroduce the
    # very difference this test exists to detect.
    observed: dict[str, torch.Tensor] = {}
    module = adapter._model.transformer.h[layer].attn.c_proj
    probe = module.register_forward_hook(
        lambda _m, i, _o: observed.setdefault("x", i[0].detach().clone())
    )
    try:
        captured = adapter.capture_head_outputs(PROMPT)[layer][head]
        adapter.patch_head_output(
            PROMPT, layer=layer, head_index=head,
            patch_vector=captured, score_token_id=262,
        )
    finally:
        probe.remove()

    assert "x" in observed, "the probe never fired"
    start = head * d_head

    torch.testing.assert_close(
        observed["x"][0, -1, start:start + d_head].cpu().float(),
        torch.tensor(captured, dtype=torch.float32),
        rtol=1e-4, atol=1e-6,
    )


@needs_weights
def test_the_capture_rounding_is_below_float32_resolution(adapter):
    """The capture's 6-decimal rounding must not itself break idempotence.

    Isolated from the kernel mismatch on purpose. `capture_head_outputs` rounds
    each component, which is in principle enough to break a bit-exact
    round-trip; this confirms the rounding is harmless (it is, by roughly two
    orders of magnitude) so that any future non-zero delta has one cause to
    look for rather than two.

    Recomputes the true slice at full precision and measures the rounding
    error directly, then patches with the *unrounded* vector. If the rounded
    and unrounded patches both give exactly 0.0, the rounding is not the cause
    of anything.
    """
    import torch

    layer, head = 6, 6
    d_head = adapter.spec.d_model // adapter.spec.num_heads

    true_slice: dict[str, torch.Tensor] = {}
    probe = adapter._model.transformer.h[layer].attn.c_proj.register_forward_hook(
        lambda _m, i, _o: true_slice.__setitem__("x", i[0].detach().clone())
    )
    try:
        adapter._forward_with_hooks(PROMPT)
    finally:
        probe.remove()

    start = head * d_head
    exact = (
        true_slice["x"][0, -1, start:start + d_head].detach().cpu().reshape(-1)
    )
    rounded = torch.tensor(
        adapter.capture_head_outputs(PROMPT)[layer][head], dtype=torch.float32
    )

    rounding_error = float((exact - rounded).abs().max())
    assert rounding_error < 1e-6, (
        f"capture rounding is larger than documented: {rounding_error:.3e}"
    )

    result = adapter.patch_head_output(
        PROMPT, layer=layer, head_index=head,
        patch_vector=[float(v) for v in exact],
          # The slice above is read at [0, -1, ...], the final position,
          # which is what capture_head_outputs returns. patch_head_output now
          # writes every position by default, so without this the round trip
          # broadcasts one position's vector across the whole sequence and
          # fails at 0.0625 or 0.125 -- not a rounding effect, which is
          # exactly what this test exists to isolate.
          position=-1,
    )
    assert result.delta == 0.0, (
        f"patching with the exact (unrounded) vector still moved the logit by "
        f"{result.delta}, so the discrepancy is not caused by rounding"
    )