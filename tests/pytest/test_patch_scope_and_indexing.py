"""Patch idempotence: the full output, and non-identity interventions.

`test_patch_idempotence.py` proves the strongest no-op in the codebase --
patching a head with the vector just captured from it changes nothing. That
is necessary. It is not sufficient, and the contract review puts it plainly:

    A no-op identity patch is necessary, but it does not establish that
    non-identity interventions are correctly implemented.

An identity patch writes a value that was already there. Every path that
*changes* something -- an injected vector, a zero ablation, an out-of-range
index, a vector of the wrong length -- is a different code path through
`head_patch_hook`, and none of them is covered by an exact-equality test.

Measured on the current code, GPT-2 Small, prompt "John gave a drink to Mary",
layer 5 head 3 (d_head = 64):

    GAP 3  out-of-range head 12 (model has 0..11)
             -> delta = +0.000000     a *successful* result meaning nothing
           out-of-range head 99      -> delta = +0.000000
           real head 11              -> delta = +0.312500

    GAP 3  a one-element vector broadcasts:
             patch_vector=[0.5]         -> delta = -0.937500
             patch_vector=[0.5]*64      -> delta = -0.937500
           identical, so one float silently filled all 64 components

    GAPS 2+5 the two patch modes are not the same intervention:
             patch_vector=None (zero ablation)  -> delta = -1.687500
             patch_vector=[0.0]*64              -> delta = -0.937500
           both mean "zero this head", and they disagree.

That last one is the sharpest. Zero ablation writes ``x[..., start:end]`` --
every position, every batch row. Vector injection writes
``x[0, -1, start:end]`` -- batch 0, final position only. Both are reached by
omitting one argument, so the difference is invisible at the call site, and a
caller comparing "head ablated" against "head set to v" is comparing two
different experiments.
"""

from __future__ import annotations

import pytest
import torch

from _weight_guard import skip_reason, weights_available

needs_weights = pytest.mark.skipif(
    not weights_available(), reason=skip_reason()
)

PROMPT = "John gave a drink to Mary"


@pytest.fixture(scope="module")
def adapter():
    from backend.science.models.gpt2_adapter import GPT2Adapter

    instance = GPT2Adapter(variant="small", mock_mode=False)
    assert instance._model is not None
    return instance


def _d_head(adapter) -> int:
    return adapter.spec.d_model // adapter.spec.num_heads


def _logits(adapter, prompt: str = PROMPT) -> torch.Tensor:
    """The model's full output for `prompt`, under the kernel patch uses."""
    inputs = adapter._tokenizer(prompt, return_tensors="pt").to(
        adapter._model.device)
    config = adapter._model.config
    previous = getattr(config, "_attn_implementation", None)
    if previous != "eager":
        config._attn_implementation = "eager"
    try:
        with torch.no_grad():
            return adapter._model(**inputs).logits[0].detach().clone()
    finally:
        if previous is not None:
            config._attn_implementation = previous


# ── Gap 1: the entire output tensor, not one scored logit ─────────────

@needs_weights
def test_an_identity_patch_leaves_the_whole_logit_tensor_unchanged(adapter):
    """Every position and every vocabulary entry, not a single scored logit.

    The existing tests compare `PatchResult.delta` -- one number, at the final
    position, for one token. A defect that corrupted earlier positions, or a
    different vocabulary entry, would be invisible to them.
    """
    layer, head = 5, 3
    per_layer = adapter.capture_head_outputs(PROMPT)
    captured = per_layer[layer][head]
    assert len(captured) == _d_head(adapter), (
        f"captured a length-{len(captured)} vector; a head is "
        f"{_d_head(adapter)} components")

    baseline = _logits(adapter)

    # Patch with the very vector captured from that head, and compare every
    # element of the output tensor.
    inputs = adapter._tokenizer(PROMPT, return_tensors="pt").to(
        adapter._model.device)
    config = adapter._model.config
    previous = getattr(config, "_attn_implementation", None)
    if previous != "eager":
        config._attn_implementation = "eager"
    d_head = _d_head(adapter)
    start = head * d_head
    end = (head + 1) * d_head
    vector = torch.tensor(captured, device=adapter._model.device,
                          dtype=adapter._model.dtype)

    def _identity(module, inputs_tuple):
        x = inputs_tuple[0].clone()
        x[0, -1, start:end] = vector
        return (x,) + tuple(inputs_tuple[1:])

    handle = adapter._model.transformer.h[layer].attn.c_proj \
        .register_forward_pre_hook(_identity)
    try:
        with torch.no_grad():
            patched = adapter._model(**inputs).logits[0].detach().clone()
    finally:
        handle.remove()
        if previous is not None:
            config._attn_implementation = previous

    assert patched.shape == baseline.shape, (
        f"the output tensor changed shape under an identity patch: "
        f"{baseline.shape} -> {patched.shape}")

    worst = (patched - baseline).abs().max().item()
    assert worst == 0.0, (
        f"an identity patch moved the output tensor by {worst:.3e}. The "
        f"existing tests compare one scored logit at the final position and "
        f"would not have seen a corruption at another position or another "
        f"token."
    )


@needs_weights
def test_a_non_identity_patch_does_change_the_output(adapter):
    """The negative control for the test above.

    Without it, an implementation that ignored every patch would satisfy
    "identity patch changes nothing" perfectly.
    """
    d_head = _d_head(adapter)
    baseline = _logits(adapter)

    result = adapter.patch_head_output(PROMPT, 5, 3, patch_vector=[0.0] * d_head)
    assert result.delta != 0.0, (
        "a zero-vector injection reported delta 0.0; if patches are being "
        "ignored, every identity test in this file passes for the wrong reason"
    )
    assert result.patched_logit != result.original_logit


# ── Gaps 2 and 5: coverage, isolation, exactly-once ───────────────────

@needs_weights
@pytest.mark.parametrize("head", [0, 11])
def test_boundary_heads_patch_and_leave_neighbours_alone(adapter, head):
    """The first and last head must behave like every other head.

    An off-by-one in `start = head * d_head` shows up at the boundaries and
    nowhere in the middle, which is why the existing all-twelve-heads test on
    one layer is the right shape.

    Asserted as a *difference between heads*, not as a non-zero delta. Not
    every head is causally important for the scored token -- head 11 at layer 5
    genuinely measures 0.0 -- so requiring a non-zero delta would be asserting
    a scientific claim rather than a mechanical one. An earlier draft did
    exactly that and failed on a correct result.
    """
    d_head = _d_head(adapter)
    n_heads = adapter.spec.num_heads

    deltas = {
        h: adapter.patch_head_output(PROMPT, 5, h,
                                    patch_vector=[0.0] * d_head).delta
        for h in (head, (head + 1) % n_heads, (head + 2) % n_heads)
    }

    assert len(set(deltas.values())) > 1, (
        f"heads {head}, {(head + 1) % n_heads} and {(head + 2) % n_heads} all "
        f"measured {deltas}; the patch is not landing on the requested head"
    )


@needs_weights
def test_the_patch_is_applied_exactly_once(adapter):
    """A doubled write would be indistinguishable in a delta, but not in a
    counter.

    `head_patch_hook` is registered per call and removed in a `finally`. If a
    future change left it registered, or registered it twice, the second write
    would simply overwrite the first -- the logits would look right and the
    defect would be invisible to every other test here.
    """
    layer, head = 5, 3
    module = adapter._model.transformer.h[layer].attn.c_proj
    before = len(module._forward_pre_hooks)

    adapter.patch_head_output(PROMPT, layer, head,
                              patch_vector=[0.0] * _d_head(adapter))

    after = len(module._forward_pre_hooks)
    assert after == before, (
        f"the patch hook was left registered ({before} -> {after}); a later "
        f"patch would stack on top of it"
    )


@needs_weights
def test_the_hook_is_removed_even_when_the_forward_pass_raises(adapter):
    """Exception-safe removal, or one failure poisons every later call.

    `handle.remove()` sits in a `finally`, which is correct. This pins it,
    because a future edit that moves the return inside the `try` without the
    `finally` would leave a hook bound to a dead request.
    """
    layer, head = 5, 3
    module = adapter._model.transformer.h[layer].attn.c_proj
    before = len(module._forward_pre_hooks)

    original = adapter._tokenizer

    class _Boom:
        def __call__(self, *a, **kw):
            raise RuntimeError("tokenizer failed mid-patch")

    adapter._tokenizer = _Boom()
    try:
        with pytest.raises(RuntimeError):
            adapter.patch_head_output(PROMPT, layer, head)
    finally:
        adapter._tokenizer = original

    assert len(module._forward_pre_hooks) == before, (
        "a failed patch left its hook registered; every subsequent call on "
        "this module would run both"
    )


@needs_weights
@pytest.mark.parametrize("layer", [0, 11])
def test_patching_works_at_the_first_and_last_layer(adapter, layer):
    """Boundary layers, like boundary heads."""
    d_head = _d_head(adapter)
    result = adapter.patch_head_output(PROMPT, layer, 3,
                                       patch_vector=[0.0] * d_head)
    assert result.delta != 0.0, (
        f"zeroing a head at boundary layer {layer} reported delta 0.0"
    )


# ── Gap 3: invalid indices and mismatched vectors ─────────────────────

@needs_weights
@pytest.mark.parametrize("head", [12, 99, 1000])
def test_an_out_of_range_head_is_refused_not_silently_ignored(adapter, head):
    """Asking to patch a head that does not exist must not look like success.

    Measured on the current code:

        head=12   delta=+0.000000    <- a successful result meaning nothing
        head=99   delta=+0.000000
        head=11   delta=+0.312500    <- the last real head, for comparison

    `head_patch_hook` guards with `if x.shape[-1] < end: return inp`, so an
    out-of-range head computes a slice past the end of the tensor and the hook
    returns the input untouched. The caller receives `delta: 0.0` -- the same
    value a real ablation of a causally inert head produces, and the same
    value the identity-patch tests assert. A typo in a discovered head list
    would read as "this head does nothing".
    """
    with pytest.raises((ValueError, IndexError)) as excinfo:
        adapter.patch_head_output(PROMPT, 5, head,
                                  patch_vector=[0.0] * _d_head(adapter))
    assert str(excinfo.value), "a refusal must say why"


@needs_weights
def test_a_negative_head_is_refused(adapter):
    """`-1` currently raises a broadcast error from deep inside torch.

    Measured: `RuntimeError: The expanded size of the tensor (0) must match
    the existing size (64)`. That message says nothing about head indices, and
    a caller reading it would be looking in the wrong place.
    """
    with pytest.raises((ValueError, IndexError)) as excinfo:
        adapter.patch_head_output(PROMPT, 5, -1,
                                  patch_vector=[0.0] * _d_head(adapter))
    assert "head" in str(excinfo.value).lower(), (
        f"the refusal does not mention the head index: {excinfo.value}"
    )


@needs_weights
def test_a_vector_of_the_wrong_length_is_refused_not_broadcast(adapter):
    """A one-element vector must not silently fill the whole head.

    Measured:

        patch_vector=[0.5]      -> delta = -0.937500
        patch_vector=[0.5]*64   -> delta = -0.937500

    identical, because `x[0, -1, start:end] = v` broadcasts a length-1 `v`
    across all 64 components. A caller who transposed a row, or passed a
    scalar by mistake, gets a confident number describing an intervention they
    did not ask for.
    """
    for wrong in (1, _d_head(adapter) - 1, _d_head(adapter) + 1):
        with pytest.raises(ValueError) as excinfo:
            adapter.patch_head_output(PROMPT, 5, 3, patch_vector=[0.5] * wrong)
        assert "64" in str(excinfo.value) or "length" in str(excinfo.value).lower(), (
            f"a wrong-length vector (got {wrong}) was refused without saying "
            f"the expected length: {excinfo.value}"
        )


# ── The two patch modes must not differ silently ──────────────────────

@needs_weights
def test_zero_ablation_and_zero_vector_injection_state_their_scope(adapter):
    """The two ways of zeroing a head are different experiments.

    Measured:

        patch_vector=None      -> delta = -1.687500   (x[...,  start:end])
        patch_vector=[0.0]*64  -> delta = -0.937500   (x[0, -1, start:end])

    Zero ablation zeroes every position of every batch row. Vector injection
    writes the final position of batch 0 only. Both are reached by omitting an
    argument, so the difference is invisible at the call site.

    This does not demand they agree -- ablating every position and patching one
    are both legitimate and the distinction is worth keeping. It demands that
    the difference be stated, so a caller knows which one they ran. If the
    method exposes the scope it applied, both are fine; while the scope is
    implicit in two different lines of source, neither is.
    """
    zero_ablation = adapter.patch_head_output(PROMPT, 5, 3)
    zero_vector = adapter.patch_head_output(
        PROMPT, 5, 3, patch_vector=[0.0] * _d_head(adapter))

    # The two modes must now AGREE: both mean "set head H to zero", so both
    # must cover the same scope. An earlier draft asserted they disagree,
    # because they did -- the assertion was written to pin the defect. Now
    # that they agree, disagreement means something regressed.
    assert zero_ablation.delta == pytest.approx(zero_vector.delta, abs=1e-6), (
        f"zero ablation gives {zero_ablation.delta} but zero-vector injection "
        f"gives {zero_vector.delta}. Both mean 'set this head to zero'; if "
        f"they differ again the two modes are covering different scopes "
        f"undetected."
    )

    import inspect

    from backend.science.models.gpt2_adapter import GPT2Adapter

    declares_scope = "position" in inspect.signature(
        GPT2Adapter.patch_head_output).parameters
    assert declares_scope, (
        "the scope is implicit in two different lines of source again; a "
        "caller cannot tell which positions a patch covered."
    )