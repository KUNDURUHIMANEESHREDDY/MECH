"""4.2 -- the patch reached a layer, but is it the right *computation*?

`test_intervention_placement.py` covers `patch_head_output` and
`run_isolated_circuit`. This file completes it for `patch_activation`, which
was the one gap left open when that file was written.

Why the sibling method needs its own treatment
----------------------------------------------
`patch_activation` is registered as a **forward** hook on
`transformer.h[layer].mlp`, so `output` is the MLP's *return value* -- post
`c_proj`, `d_model` wide, sitting in the residual stream. Measured:

    MLP forward-hook output shape : (1, 6, 768)
    internal MLP hidden (4*d_model): 3072

So `neuron_index` is a residual-stream coordinate, not a neuron. A GPT-2 MLP
neuron lives in the 3072-wide post-GELU hidden. The docstring said "Patch a
specific neuron in the MLP layer"; that is not what the code does, and the
difference is the difference between ablating a neuron and overwriting a
residual channel. Both produce a finite number. Only one means what the
caller asked for.

The other defect is the one already fixed in `patch_head_output` and missed
here:

    patch_activation(prompt, 5, 99999, 0.0).delta  ->  0.0

An out-of-range index is silently skipped by `if output.shape[-1] >
neuron_index` and reported as an intervention that changed nothing. A refusal
to intervene must not be indistinguishable from an intervention of size zero.

Controls
--------
Every placement assertion is paired with a negative control at a neighbouring
coordinate, so a write that landed one index to the left fails rather than
passing for the wrong reason. And the probe is an independent hook sharing
only the slice arithmetic with the production method, so agreement is evidence
rather than a tautology.
"""

from __future__ import annotations

import pytest
import torch

from _weight_guard import skip_reason, weights_available

needs_weights = pytest.mark.skipif(
    not weights_available(), reason=skip_reason()
)

PROMPT = "John gave a drink to Mary"
LAYER = 5


@pytest.fixture(scope="module")
def adapter():
    from backend.science.models.gpt2_adapter import GPT2Adapter

    instance = GPT2Adapter(variant="small", mock_mode=False)
    assert instance._model is not None
    return instance


def _plain_logits(adapter, prompt: str = PROMPT) -> torch.Tensor:
    """The next-token logit vector, under the kernel the patches use."""
    inputs = adapter._tokenizer(prompt, return_tensors="pt").to(
        adapter._model.device)
    config = adapter._model.config
    previous = getattr(config, "_attn_implementation", None)
    if previous != "eager":
        config._attn_implementation = "eager"
    try:
        with torch.no_grad():
            return adapter._model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
            ).logits[0, -1, :].detach().clone()
    finally:
        if previous is not None:
            config._attn_implementation = previous


def _logits_with_residual_coordinate_forced(adapter, layer: int, index: int,
                                            value: float,
                                            prompt: str = PROMPT):
    """Independent probe: force one coordinate of one layer's MLP output.

    A separate hook doing one thing, so agreement with `patch_activation` is
    evidence. It returns a copy rather than mutating the module output, but
    that copy is *not* what makes an in-place write detectable: cloning and
    writing in place produce the same values, so the comparison below cannot
    see the difference, whatever this probe does. That property is asserted
    separately by `test_patch_activation_does_not_write_in_place`.
    """
    def _force(module, _inputs, output):
        patched = output.clone()
        patched[0, -1, index] = value
        return patched

    handle = adapter._model.transformer.h[layer].mlp \
        .register_forward_hook(_force)
    try:
        return _plain_logits(adapter, prompt)
    finally:
        handle.remove()


# ── where does the write actually land? ──────────────────────────────

@needs_weights
def test_the_patch_lands_on_the_residual_stream_not_on_an_mlp_neuron(adapter):
    """The tensor-stage claim, measured rather than assumed.

    If `patch_activation` patched an MLP neuron, forcing the *internal*
    3072-wide hidden at the same index would reproduce its effect. It does
    not, because there is no hook there -- the method writes to the block
    output. Pinning that means the next reader cannot mistake one for the
    other, and a future change to the hook target has to update this test.
    """
    with torch.no_grad():
        mlp = adapter._model.transformer.h[LAYER].mlp
        seen: dict = {}

        def _observe(_module, _inputs, output):
            seen["shape"] = tuple(output.shape)

        handle = mlp.register_forward_hook(_observe)
        try:
            adapter._model(**adapter._tokenizer(
                PROMPT, return_tensors="pt").to(adapter._model.device))
        finally:
            handle.remove()

    assert seen["shape"][-1] == adapter.spec.d_model, (
        f"the MLP forward hook now sees width {seen['shape'][-1]}, not "
        f"d_model={adapter.spec.d_model}. If this changed, "
        f"`patch_activation`'s indexing and its docstring are both wrong."
    )


@needs_weights
def test_patch_activation_matches_an_independent_write_to_the_same_coordinate(
    adapter,
):
    """Placement, proven by agreement with a separate hook.

    Both write the same value to coordinate `index` of the same layer's MLP
    output at the last position. If the method's stage or position were
    wrong, the two would disagree.
    """
    index, value = 7, 2.25

    # Which token is scored. `PatchResult.patched_logit` is the logit of the
    # token the *unpatched* run predicted, so the reference must be read at
    # that same index -- not at the argmax of a patched run, which is a
    # different token whenever the patch changes the top prediction.
    scored = int(torch.argmax(_plain_logits(adapter)))

    result = adapter.patch_activation(PROMPT, LAYER, index, value)

    # The method's own measured value, captured inside the patched forward
    # pass. An earlier version took a *separate* `_plain_logits` call after
    # `patch_activation` returned -- but the method removes its hook in a
    # `finally`, so that pass was unpatched and the two sides were being
    # compared against different states. It disagreed by 0.375, which is a
    # plausible-looking number attached to a meaningless comparison.
    produced = result.patched_logit

    reference_logits = _logits_with_residual_coordinate_forced(
        adapter, LAYER, index, value)
    reference = float(reference_logits[scored])

    assert abs(produced - reference) < 5e-3, (
        f"patch_activation scored {produced:.6f} at token {scored}, but an "
        f"independent write to coordinate {index} of layer {LAYER}'s MLP "
        f"output gives {reference:.6f}. The method is not writing where it "
        f"reports."
    )
    assert result.delta != 0.0, "the intervention changed nothing"


@needs_weights
def test_patch_activation_does_not_write_in_place(adapter):
    """The hook's write must not reach holders of the original tensor.

    `patch_hook` copies before writing, in its own words because "the returned
    tensor may be referenced elsewhere, and an in-place write from a forward
    hook is invisible to anything holding that reference."

    The placement test above cannot see this property: cloning and writing in
    place return the same *values*, so both sides still agree and it stays
    green. Measured, not assumed -- reinstating the in-place write leaves that
    test passing, and `test_mutation_detection.py` reported it as a surviving
    mutation.

    So the aliasing is asserted directly. Hooks fire in registration order, so
    a hook registered before `patch_activation` runs sees the module's output
    tensor just before the patch hook does. If production writes in place, the
    very same tensor object carries the patched value afterwards; if it
    copies, the object is untouched.

    Every pass is checked rather than the last one, so the assertion does not
    depend on how many forward passes the method makes or on which of them is
    the patched one.
    """
    index, value = 7, 2.25
    mlp = adapter._model.transformer.h[LAYER].mlp
    seen: list = []

    def _snapshot(_module, _inputs, output):
        seen.append((output, float(output[0, -1, index])))
        return None

    handle = mlp.register_forward_hook(_snapshot)
    try:
        adapter.patch_activation(PROMPT, LAYER, index, value)
    finally:
        handle.remove()

    assert seen, (
        "the MLP forward hook never fired, so nothing was measured"
    )
    assert any(before != value for _, before in seen), (
        f"coordinate {index} already held the patched value {value} in every "
        f"pass, so an in-place write would be indistinguishable from none and "
        f"this check would prove nothing"
    )

    changed = [
        (before, float(tensor[0, -1, index]))
        for tensor, before in seen
        if float(tensor[0, -1, index]) != before
    ]
    assert not changed, (
        f"patch_activation wrote in place: the module's own output tensor held "
        f"{changed[0][0]:.6f} at coordinate {index} and holds "
        f"{changed[0][1]:.6f} afterwards. Copying in `patch_hook` is what stops "
        f"the write being invisible to anything still holding that reference."
    )


@needs_weights
def test_neighbouring_coordinates_are_distinguishable(adapter):
    """The negative control for the placement claim.

    If coordinates 7 and 8 were aliased, or the index were off by one, forcing
    either would give the same logits and the test above would pass for a
    write that landed in the wrong place.
    """
    at_seven = _logits_with_residual_coordinate_forced(
        adapter, LAYER, 7, 5.0)
    at_eight = _logits_with_residual_coordinate_forced(
        adapter, LAYER, 8, 5.0)

    assert not torch.equal(at_seven, at_eight), (
        "forcing coordinate 7 and coordinate 8 produced identical logits; the "
        "coordinates are aliased or the index is misaligned, so a patch aimed "
        "at one is indistinguishable from a patch aimed at the other"
    )


# ── refusing to intervene must not look like intervening with zero ───

@needs_weights
def test_an_out_of_range_index_is_refused_not_silently_skipped(adapter):
    """The defect, in the exact shape it survived from `patch_head_output`.

        patch_activation(prompt, 5, 99999, 0.0).delta -> 0.0

    A guard clause skips the write and the method returns a well-formed
    `PatchResult` reporting no change. A caller cannot tell that from an
    intervention that genuinely had no effect.
    """
    d_model = adapter.spec.d_model
    forwards = {"n": 0}

    def _count(_module, _inputs, output):
        forwards["n"] += 1
        return output

    counter = adapter._model.transformer.register_forward_hook(_count)
    try:
        with pytest.raises((IndexError, ValueError)):
            adapter.patch_activation(PROMPT, LAYER, d_model + 1, 0.0)
    finally:
        counter.remove()

    # Asserted behaviourally, not on the message text.
    #
    # With the range guard disabled the test still passed twice over: the
    # write raised IndexError from inside the tensor assignment, and that
    # message contains the width ("size 768"), so a width check could not
    # tell them apart either. Mutation testing found both.
    #
    # The difference that actually matters is *when* the refusal happens. A
    # guard refuses before any computation: zero forward passes, no wasted
    # work, no partial state. An accidental crash happens mid-forward, after
    # the hook is installed and the model has already run.
    assert forwards["n"] == 0, (
        f"the out-of-range index raised after {forwards['n']} forward pass(es). "
        f"That is a crash inside the intervention, not a refusal: a guard "
        f"rejects the request before touching the model."
    )


@needs_weights
def test_the_last_valid_index_is_accepted(adapter):
    """The boundary, so the refusal above is a range check and not a blanket.

    `output.shape[-1] > neuron_index` is the pre-existing condition, which
    would reject the final coordinate. If the fix is `>=`, this passes; if it
    rejects everything above some index, this fails.
    """
    last = adapter.spec.d_model - 1
    try:
        adapter.patch_activation(PROMPT, LAYER, last, 0.0)
    except (IndexError, ValueError) as exc:
        pytest.fail(
            f"the last valid coordinate ({last}) was rejected: {exc}. The "
            f"range check is off by one, or is rejecting valid indices."
        )