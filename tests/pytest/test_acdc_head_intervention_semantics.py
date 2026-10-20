"""Attention-head interventions must touch heads, not MLP neurons.

ACDC swept `(layer, head)` pairs and called

    adapter.patch_activation(prompt, layer=layer, neuron_index=head, patch_value=...)

`patch_activation` is documented and implemented as an *MLP* intervention: it
registers a forward hook on `transformer.h[layer].mlp` and writes

    output[0, -1, neuron_index] = patch_value

so `head=9` wrote MLP neuron 9 out of 3072. The patch succeeded -- 3072 > 12 --
and returned a plausible delta. Nothing downstream could detect the substitution,
so the pruning sweep was ranking MLP neurons while labelling its results
`L{layer}H{head}`, and the component set it selected fed a real fidelity
measurement. A real number computed over a noise-selected circuit is worse than an
obvious mock, because nothing looks wrong.

The sibling misuse: `get_activations(layer=layer, neuron_index=head)` reads
`hidden_states[layer][0, tok, :][head]` -- dimension `head` of the 768-wide
residual stream, not a head.

These tests are behavioural on purpose. A source-level assertion could be
satisfied by a rename while the hook stayed on the wrong module; instead each
test watches the actual tensors during a real forward pass on real weights.
"""

from __future__ import annotations

import re

import pytest

from backend.science.models.adapter_base import LiveUnavailable


from _weight_guard import skip_reason, weights_available


needs_weights = pytest.mark.skipif(
    not weights_available(), reason=skip_reason()
)

PROMPT = "When Mary and John went to the store, John gave a drink to"


@needs_weights
def test_patch_head_output_zeroes_exactly_one_head_slice():
    """The decisive property: head h's slice, and only that slice, is zeroed.

    Note what this does *not* assert. An earlier draft of this test also
    required MLP activations to be bit-identical, on the reasoning that a head
    patch should not touch the MLP. That is wrong physics: zeroing a head changes
    the residual stream at that layer, which is the MLP's input, so the MLP's
    output legitimately differs. Causal influence is exactly what these
    primitives are for. The property that distinguishes the two interventions is
    *which slice of which tensor is written*, not whether downstream activations
    move.
    """
    import torch

    from backend.science.models.gpt2_adapter import GPT2Adapter

    adapter = GPT2Adapter(variant="small", mock_mode=False)
    layer, head = 9, 9
    d_head = adapter.spec.d_model // adapter.spec.num_heads

    inputs = adapter._tokenizer(PROMPT, return_tensors="pt").to(adapter._model.device)

    # Observe with a *forward* hook, not a pre-hook. `patch_head_output`
    # registers a pre-hook, and PyTorch runs hooks in registration order, so an
    # observer registered beforehand sees the unpatched input. A forward hook's
    # `input` argument is delivered after all pre-hooks have run, so it sees the
    # tensor the patch actually wrote.
    baseline = {}
    handle = adapter._model.transformer.h[layer].attn.c_proj.register_forward_hook(
        lambda m, i, o: baseline.__setitem__("heads", i[0].detach().clone())
    )
    # The baseline must be captured under the *same* attention kernel as the
    # patched run below, or the "every other head is untouched" comparison is
    # really comparing two attention implementations.
    #
    # `patch_head_output` forces `attn_implementation="eager"` so that a
    # captured head vector and the tensor patched back into it come from one
    # kernel -- eager and sdpa over identical weights disagree at ~4e-04, which
    # is enough to make an untouched head look changed. A bare
    # `model(**inputs)` runs sdpa, which is what this line used to do, and the
    # test then failed for a reason that had nothing to do with head slicing.
    #
    # Eager is also required for the *capture* here to be the quantity under
    # test; see test_patch_idempotence.py for the measurements.
    previous_impl = getattr(adapter._model.config, "_attn_implementation", None)
    if previous_impl != "eager":
        adapter._model.config._attn_implementation = "eager"
    try:
        with torch.no_grad():
            adapter._model(**inputs)
    finally:
        if previous_impl is not None:
            adapter._model.config._attn_implementation = previous_impl
    handle.remove()

    observed = {}
    handle = adapter._model.transformer.h[layer].attn.c_proj.register_forward_hook(
        lambda m, i, o: observed.__setitem__("heads", i[0].detach().clone())
    )
    try:
        adapter.patch_head_output(PROMPT, layer=layer, head_index=head, patch_vector=None)
    finally:
        handle.remove()

    assert "heads" in observed, "the c_proj hook did not fire"

    # Head 9's slice is zero in the patched run and non-zero in the baseline.
    sl = slice(head * d_head, (head + 1) * d_head)
    assert float(observed["heads"][0, -1, sl].abs().max()) == 0.0, (
        "head 9's slice of the c_proj input was not zeroed"
    )
    assert float(baseline["heads"][0, -1, sl].abs().max()) > 0.0, (
        "head 9 was already all-zero, so this test cannot detect anything"
    )

    # Every other head is untouched.
    for other in range(adapter.spec.num_heads):
        if other == head:
            continue
        s = slice(other * d_head, (other + 1) * d_head)
        assert observed["heads"][0, -1, s].equal(baseline["heads"][0, -1, s]), (
            f"head {other} changed while patching head {head}; the patch is not "
            f"head-specific"
        )


@needs_weights
def test_patch_activation_writes_the_mlp_neuron_indexed_by_its_parameter():
    """The counterpart, so the two primitives cannot be confused again.

    `patch_activation(neuron_index=n)` must write `n` into the MLP output at the
    final position. That is the behaviour ACDC was relying on when it handed this
    function a *head* index.

    Where to observe that write, and why not at the module's output.
    `patch_hook` returns a *clone* carrying the write rather than mutating the
    module output in place; the sibling file pins that on purpose
    (`test_activation_patch_placement.py::test_patch_activation_does_not_write_in_place`).
    A forward hook registered before the patch therefore receives the pre-patch
    tensor, and keeping a reference to it measures the *untouched* MLP output --
    0.688 at this coordinate, measured -- however much of the sentinel landed.
    An earlier version of this test did exactly that and failed against a
    working primitive.

    So observe downstream instead. The block computes `residual + mlp_output`,
    and this pass's residual is measurable because it is `ln_2`'s input. A write
    that lands makes `block_out[n] == residual[n] + sentinel`; no write leaves
    `block_out[n] == residual[n] + mlp_out[n]`. Every comparison below draws all
    of its terms from a single forward pass, so nothing is differenced across
    runs and nothing depends on which pass the method patches.

    The model runs in float16 (measured: `model.dtype == torch.float16`), so the
    residual add rounds: `0.076171875 + 99.0` is 99.076171875 exactly and
    99.0625 as stored. Tolerances here are that rounding, not slack -- a missing
    write is 98 away from the value being asserted.
    """
    import torch

    from backend.science.models.gpt2_adapter import GPT2Adapter

    adapter = GPT2Adapter(variant="small", mock_mode=False)
    layer, neuron, sentinel = 9, 9, 99.0
    block = adapter._model.transformer.h[layer]

    def _hidden(out):
        """GPT2Block returns a tuple; the hidden state is element 0."""
        return out[0] if isinstance(out, tuple) else out

    # One record per forward pass. A pre-hook on the block opens it, so the
    # children below always have somewhere to write.
    passes: list = []
    handles = [
        block.register_forward_pre_hook(lambda m, i: passes.append({})),
        block.ln_2.register_forward_pre_hook(
            lambda m, i: passes[-1].__setitem__("residual", i[0].detach().clone())
        ),
        block.mlp.register_forward_hook(
            lambda m, i, o: passes[-1].update(mlp_out=o.detach().clone(), mlp_ref=o)
        ),
        block.register_forward_hook(
            lambda m, i, o: passes[-1].__setitem__("block", _hidden(o).detach().clone())
        ),
    ]
    try:
        adapter.patch_activation(
            PROMPT, layer=layer, neuron_index=neuron, patch_value=sentinel
        )
    finally:
        for handle in handles:
            handle.remove()

    assert passes, "no forward pass reached the patched block"

    def residual_at(record, index):
        return float(record["residual"][0, -1, index])

    def block_at(record, index):
        return float(record["block"][0, -1, index])

    # float16 spacing at the sentinel's magnitude -- the residual add rounds.
    one_ulp = float(
        torch.nextafter(
            torch.tensor(sentinel, dtype=torch.float16),
            torch.tensor(sentinel + 1.0, dtype=torch.float16),
        )
    ) - sentinel

    # The write is the pass whose block output is `residual + sentinel`.
    witnesses = [
        p for p in passes
        if abs(block_at(p, neuron) - (residual_at(p, neuron) + sentinel)) <= one_ulp
    ]
    assert len(witnesses) == 1, (
        "the sentinel never reached the residual stream at coordinate "
        f"{neuron}: the block's outputs there were "
        f"{[round(block_at(p, neuron), 4) for p in passes]}, while a write of "
        f"{sentinel} into the MLP output would put the patched pass at "
        f"{sentinel} + its residual"
    )
    witness = witnesses[0]

    # And it is the only coordinate that moved. `residual + mlp_out` is exactly
    # what this block computes when the hook returns the module output untouched
    # (same tensors, same fp16 add), so any deviation is the patch itself.
    moved = (witness["block"] - (witness["residual"] + witness["mlp_out"])).abs()
    assert float(moved.max()) > 1.0, (
        "the residual stream is identical to residual + mlp_out, yet the "
        "sentinel was measured downstream"
    )
    moved[0, -1, neuron] = 0.0
    assert float(moved.max()) == 0.0, (
        f"patch_activation altered coordinates other than neuron_index={neuron} "
        f"(largest other change {float(moved.max()):.6f})"
    )

    # Every other pass passed the MLP output through untouched.
    for other in passes:
        if other is witness:
            continue
        diff = (other["block"] - (other["residual"] + other["mlp_out"])).abs()
        assert float(diff.max()) == 0.0, (
            f"an unpatched pass moved the residual stream by {float(diff.max()):.6f}"
        )

    # The *reference* a pre-registered hook holds -- not a copy of it -- keeps
    # the pre-patch values. That is the no-in-place guarantee seen from the
    # caller's side, and it is why the earlier version of this test measured the
    # untouched 0.688. Under an in-place write that reference would carry the
    # sentinel by the time the call returned.
    assert float(witness["mlp_ref"][0, -1, neuron]) != pytest.approx(
        sentinel, abs=1.0
    ), (
        "the patch wrote through the module's output tensor instead of a copy; "
        "the no-in-place guarantee is broken"
    )

    # This is the dimension mismatch that made the original bug survivable: the
    # MLP is 3072 wide, so index 9 is perfectly valid and the patch silently
    # succeeded while the caller believed it had addressed a head.
    assert adapter.spec.d_model * 4 == 3072, "expected a 3072-wide GPT-2 MLP"


@needs_weights
def test_head_and_neuron_patches_are_not_interchangeable():
    """The two primitives must give different answers for the same numbers.

    This is the assertion that would have caught ACDC directly: passing
    `head=9` to both produced two *different* interventions that ACDC treated as
    the same component.
    """
    from backend.science.models.gpt2_adapter import GPT2Adapter

    adapter = GPT2Adapter(variant="small", mock_mode=False)
    layer = 9

    vector = adapter.capture_head_outputs(PROMPT)[layer][9]
    as_head = adapter.patch_head_output(
        PROMPT, layer=layer, head_index=9, patch_vector=vector
    )
    as_neuron = adapter.patch_activation(
        PROMPT, layer=layer, neuron_index=9, patch_value=vector[0]
    )

    assert as_head.delta != as_neuron.delta, (
        "patching head 9 and patching MLP neuron 9 gave the same delta; one of "
        "the primitives is not doing what it claims"
    )


@needs_weights
def test_capture_head_outputs_returns_distinct_real_vectors():
    """Each head must have its own vector, at the correct width.

    Twelve heads sharing one value would make any head-level measurement
    meaningless while still producing numbers.
    """
    from backend.science.models.gpt2_adapter import GPT2Adapter

    adapter = GPT2Adapter(variant="small", mock_mode=False)
    caps = adapter.capture_head_outputs(PROMPT)

    assert len(caps) == adapter.spec.num_layers
    d_head = adapter.spec.d_model // adapter.spec.num_heads

    for layer, heads in caps.items():
        assert len(heads) == adapter.spec.num_heads, f"layer {layer}"
        for head, vector in heads.items():
            assert len(vector) == d_head, (
                f"L{layer}H{head}: vector width {len(vector)}, expected d_head={d_head}"
            )
        values = [tuple(v) for v in heads.values()]
        assert len(set(values)) == adapter.spec.num_heads, (
            f"layer {layer}: heads share output vectors"
        )
        assert any(any(v != 0.0 for v in vec) for vec in values), (
            f"layer {layer}: every head output is zero"
        )


@needs_weights
def test_capture_head_outputs_matches_the_existing_capture_primitive():
    """Cross-check against `live_measure.capture`, which hooks the same module.

    Both read the input to `attn.c_proj`. Agreement to float32 tolerance is
    evidence the slice arithmetic is right.

    The tolerance is relative, not absolute. The two paths differ in attention
    implementation -- `capture_head_outputs` goes through `_forward_with_hooks`,
    which forces eager so the weights materialise, while `live_measure` runs
    whatever the model is configured for (sdpa) -- so they compute the same
    quantity by different kernels and drift. Observed disagreement on this
    prompt is ~4e-3 absolute on a vector of norm ~1.0, i.e. ~0.4%. Asserting
    tighter than that would be asserting the kernels are identical, which they
    are not.
    """
    import torch

    from backend.interpretability.discovery import live_measure
    from backend.science.models.gpt2_adapter import GPT2Adapter

    adapter = GPT2Adapter(variant="small", mock_mode=False)
    d_head = adapter.spec.d_model // adapter.spec.num_heads
    mine = torch.tensor(adapter.capture_head_outputs(PROMPT)[9][9])
    _, caps = live_measure.capture(PROMPT)
    theirs = caps[9][0, -1, 9 * d_head:10 * d_head]

    drift = float((mine - theirs).abs().max())
    scale = max(float(theirs.norm()), 1.0)
    assert drift / scale < 0.01, (
        f"head capture disagrees with live_measure.capture by {drift:.2e} "
        f"({drift / scale:.2%} of the vector norm) -- too large to be kernel drift"
    )


def test_head_primitives_fail_closed_without_weights():
    """No fixture for either primitive; both must refuse."""
    from backend.science.models.gpt2_adapter import GPT2Adapter

    adapter = GPT2Adapter(variant="small", mock_mode=True)

    with pytest.raises(LiveUnavailable):
        adapter.capture_head_outputs(PROMPT)


def test_acdc_no_longer_uses_the_neuron_api():
    """Source guard: ACDC must not sweep heads through a neuron-indexed API.

    Behavioural coverage is in the tests above; this pins the call sites so the
    substitution cannot be reintroduced silently.
    """
    from source_assert import executable_source

    import backend.interpretability.discovery.algorithms.acdc as acdc

    src = executable_source(acdc)

    assert "patch_activation(" not in src, (
        "ACDC is calling patch_activation again; that is an MLP intervention and "
        "its neuron_index parameter is not a head"
    )
    assert "get_activations(" not in src, (
        "ACDC is calling get_activations again; it reads a residual-stream "
        "dimension, not a head"
    )
    assert "capture_head_outputs(" in src
    assert "patch_head_output(" in src


def test_acdc_emits_no_hardcoded_confidences_or_injected_heads():
    """The two substitutions removed from ACDC.

    * When nothing survived pruning it added L9H9 and L10H0 "to ensure the top
      critical heads are retained" -- the published answer replacing the search's.
    * Every retained edge carried `confidence: 0.95` and the edge completing the
      circuit carried `confidence: 1.0`.
    """
    from source_assert import executable_source

    import backend.interpretability.discovery.algorithms.acdc as acdc

    src = executable_source(acdc)

    assert not re.search(r'"confidence"\s*:\s*0\.\d', src), (
        "ACDC is emitting a hardcoded edge confidence again"
    )
    assert not re.search(r'"weight"\s*:\s*1\.0\s*,', src), (
        "ACDC is emitting a hardcoded edge weight again"
    )
    assert "ensure top critical heads" not in src, (
        "ACDC is overriding an empty result with named heads again"
    )
