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
    evidence. It deliberately does *not* mutate the module output in place --
    it returns a copy -- so if the production hook's in-place write is the
    thing being measured, that shows up as a difference rather than being
    mirrored here.
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