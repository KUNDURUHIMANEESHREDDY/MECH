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
    with torch.no_grad():
        adapter._model(**inputs)
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

    The observation hook stores a reference rather than a clone, and is read
    after the call: PyTorch runs hooks in registration order, so a hook
    registered before `patch_activation` fires first and would capture the
    *unpatched* value. `patch_activation` mutates the tensor in place, so a
    stored reference reflects the final write.
    """
    import torch

    from backend.science.models.gpt2_adapter import GPT2Adapter

    adapter = GPT2Adapter(variant="small", mock_mode=False)
    layer, neuron, sentinel = 9, 9, 99.0

    seen = {}
    handle = adapter._model.transformer.h[layer].mlp.register_forward_hook(
        lambda m, i, o: seen.__setitem__("mlp", o)
    )
    try:
        adapter.patch_activation(
            PROMPT, layer=layer, neuron_index=neuron, patch_value=sentinel
        )
    finally:
        handle.remove()

    assert "mlp" in seen, "the mlp hook did not fire"
    assert float(seen["mlp"][0, -1, neuron]) == pytest.approx(sentinel), (
        "patch_activation did not write the MLP neuron it claims to patch"
    )
    # A neighbouring neuron is untouched, so the write is index-specific.
    assert float(seen["mlp"][0, -1, neuron + 1]) != pytest.approx(sentinel)

    # This is the dimension mismatch that made the original bug survivable: the
    # MLP is 3072 wide, so index 9 is perfectly valid and the patch silently
    # succeeded while the caller believed it had addressed a head.
    assert adapter.spec.d_model * 4 == 3072, "expected a 3072-wide GPT-2 MLP"
    assert torch.is_tensor(seen["mlp"])


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
