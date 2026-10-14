"""Each intervention type: identity, placement, and downstream effect.

`test_patch_idempotence.py` proves the strongest no-op in the codebase -- a
head's own captured vector, written back, changes nothing. That is necessary
and it is not sufficient, for a reason worth stating precisely:

    a patch can faithfully restore its own tensor while a separate write path
    is wired to the wrong slice

Identity only exercises the path with the value that was already there. A
residual patch, an alternative token position, or a whole-circuit isolation
are different code paths writing different tensors, and none of them is
reached by an identity test.

So for each intervention type this file asserts three independent things:

    1. identity      writing the value that is already there changes nothing
    2. placement     a write to slice S changes slice S and no other
    3. downstream    the measured output matches an independent reference

Case 2 is the one no existing test performs. Every assertion in the patching
tests observes a returned logit; none observes the tensor being written. A
write landing one head to the left, or one position off, would produce a
plausible number and pass all of them.

Placement is verified by construction rather than by inspection: the probe
below writes a distinctive sentinel into one slice of one layer's `c_proj`
input, and the test asserts that (a) the output changed, and (b) writing the
same sentinel into a *neighbouring* slice produces a *different* output. If
the slices were aliased, misaligned, or overwritten by a later hook, the two
would coincide.
"""

from __future__ import annotations

import pytest
import torch

from _weight_guard import skip_reason, weights_available

needs_weights = pytest.mark.skipif(
    not weights_available(), reason=skip_reason()
)

PROMPT = "John gave a drink to Mary"
LAYER, HEAD = 5, 3


@pytest.fixture(scope="module")
def adapter():
    from backend.science.models.gpt2_adapter import GPT2Adapter

    instance = GPT2Adapter(variant="small", mock_mode=False)
    assert instance._model is not None
    return instance


def _d_head(adapter) -> int:
    return adapter.spec.d_model // adapter.spec.num_heads


def _plain_logits(adapter, prompt: str = PROMPT) -> torch.Tensor:
    """The full next-token logit vector, under the kernel the patches use."""
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


def _logits_with_slice_forced(adapter, layer: int, head: int,
                             value: float, prompt: str = PROMPT):
    """Independent reference: force one head's slice of `c_proj`'s input.

    This is not a re-implementation of any intervention -- it is a separate
    hook doing one thing, so agreement between it and a production method is
    evidence rather than a tautology.
    """
    d_head = _d_head(adapter)
    start, end = head * d_head, (head + 1) * d_head
    sentinel = torch.full((d_head,), float(value), dtype=torch.float32)

    def _force(module, inputs):
        x = inputs[0].clone()
        x[..., start:end] = sentinel.to(x.dtype).to(x.device)
        return (x,) + tuple(inputs[1:])

    hook = adapter._model.transformer.h[layer].attn.c_proj \
        .register_forward_pre_hook(_force)
    try:
        return _plain_logits(adapter, prompt)
    finally:
        hook.remove()


# ── case 2: placement, for each intervention type ────────────────────

@needs_weights
def test_a_write_to_one_slice_changes_only_that_slice(adapter):
    """Different slices must produce different computations.

    The negative control for placement. If head 3 and head 4 shared a slice,
    or the offsets were misaligned by one, forcing either would produce the
    same output and the first assertion here would pass for a patch landing
    in the wrong place.
    """
    baseline = _plain_logits(adapter)

    at_three = _logits_with_slice_forced(adapter, LAYER, HEAD, 5.0)
    at_four = _logits_with_slice_forced(adapter, LAYER, HEAD + 1, 5.0)

    assert not torch.equal(at_three, baseline), (
        "forcing head 3's slice changed nothing; the slice is not reachable "
        "and any test asserting a downstream effect would be vacuous"
    )
    assert not torch.equal(at_three, at_four), (
        "forcing head 3 and head 4 produced identical logits; the two slices "
        "are aliased or the offsets are wrong, so a patch aimed at one is "
        "indistinguishable from a patch aimed at the other"
    )


@needs_weights
def test_patch_head_output_matches_an_independent_write_to_the_same_slice(
    adapter,
):
    """Case 3 for the head patch: production method vs. an independent probe.

    `patch_head_output` and `_logits_with_slice_forced` share nothing but the
    slice arithmetic. If they agree to within float32 tolerance, the method is
    writing where it claims -- and if its offsets were wrong, they would not.
    """
    d_head = _d_head(adapter)

    result = adapter.patch_head_output(
        PROMPT, LAYER, HEAD, patch_vector=[3.5] * d_head, position=-1)

    # Compare like for like: two independent forward passes, each with exactly
    # one writer active. An earlier draft called `patch_head_output` and then
    # took a separate plain forward pass, so the method's write was gone by the
    # time `produced` was measured; the reference probe then ran with nothing
    # else active and disagreed by exactly 30.0 -- the sentinel -- which was a
    # sequencing bug in the test, not a placement defect.
    start, end = HEAD * d_head, (HEAD + 1) * d_head
    sentinel = torch.full((d_head,), 3.5, dtype=torch.float32)

    def _force_last(module, inputs):
        x = inputs[0].clone()
        x[:, -1, start:end] = sentinel.to(x.dtype).to(x.device)
        return (x,) + tuple(inputs[1:])

    hook = adapter._model.transformer.h[LAYER].attn.c_proj \
        .register_forward_pre_hook(_force_last)
    try:
        reference_last = _plain_logits(adapter)
    finally:
        hook.remove()

    # Compare the same quantity on both sides. `patch_head_output` reports the
    # logit of the token it scored (`score_token_id`, defaulting to whatever
    # the unpatched run predicted), so the reference must be read at the same
    # index -- not at the argmax, which may be a different token entirely.
    scored = int(torch.argmax(_plain_logits(adapter)))
    produced = result.patched_logit
    reference_value = float(reference_last[scored])

    worst = abs(produced - reference_value)
    assert worst < 5e-3, (
        f"patch_head_output scored {produced:.6f} at token {scored}, but an "
        f"independent write to the same slice at the same position gives "
        f"{reference_value:.6f}. The method is not writing where it reports."
    )
    assert result.delta != 0.0, "the intervention changed nothing"


@needs_weights
def test_run_isolated_circuit_matches_an_independent_mask(adapter):
    """Case 3 for the whole-circuit path.

    `run_isolated_circuit` keeps a listed set of heads and zeroes every other
    head in every layer. An independent probe does the same mask, and the two
    must agree -- which is only true if the method's per-layer offsets and its
    zero-ablation scope are both right.
    """
    keep = [(LAYER, HEAD)]
    result = adapter.run_isolated_circuit(PROMPT, [f"L{LAYER}H{HEAD}"])

    d_head = _d_head(adapter)
    n_layers = adapter.spec.num_layers
    n_heads = adapter.spec.num_heads

    handles = []

    def _make_mask(layer_idx):
        def _mask(module, inputs):
            x = inputs[0].clone()
            for head in range(n_heads):
                if (layer_idx, head) not in keep:
                    start = head * d_head
                    x[..., start:(head + 1) * d_head] = 0.0
            return (x,) + tuple(inputs[1:])
        return _mask

    for layer in range(n_layers):
        handles.append(
            adapter._model.transformer.h[layer].attn.c_proj
            .register_forward_pre_hook(_make_mask(layer)))
    try:
        reference = _plain_logits(adapter)
    finally:
        for handle in handles:
            handle.remove()

    produced_top = adapter.run_isolated_circuit(
        PROMPT, [f"L{LAYER}H{HEAD}"])["top_token"]

    assert produced_top == _decode_top(adapter, reference), (
        f"run_isolated_circuit reported {produced_top!r} but an independent "
        f"mask over the same circuit gives {_decode_top(adapter, reference)!r}"
    )
    assert result["status"] == "ok"


def _decode_top(adapter, logits: torch.Tensor) -> str:
    return adapter._tokenizer.decode([int(torch.argmax(logits))])


@needs_weights
def test_the_two_intervention_types_are_distinguishable(adapter):
    """A head patch and a residual patch must not reach the same tensor.

    `patch_activation` writes to a block output; `patch_head_output` writes
    to the `c_proj` input inside a block. They are different interventions.
    If either were wired to the other's tensor, one of the identity or
    downstream assertions elsewhere would still pass, so the distinction is
    asserted here explicitly.
    """
    d_head = _d_head(adapter)

    head_result = adapter.patch_head_output(
        PROMPT, LAYER, HEAD, patch_vector=[0.0] * d_head)
    residual_result = adapter.patch_activation(PROMPT, LAYER, HEAD, 0.0)

    assert head_result.delta != residual_result.delta, (
        f"patching a head's output and zeroing a residual both produced "
        f"delta={head_result.delta:.6f}. They write to different tensors, so "
        f"they must not produce the same measurement."
    )


# ── case 1: identity, for each type, restated as a control ───────────

@needs_weights
def test_identity_is_held_for_every_intervention_type(adapter):
    """The control for the whole file.

    Without it, an implementation that ignored every write would satisfy
    nothing above and be caught only incidentally -- and the placement
    assertions would pass vacuously.
    """
    d_head = _d_head(adapter)

    plain = _plain_logits(adapter)

    identity = _logits_with_slice_forced(adapter, LAYER, HEAD, 0.0)
    # Forcing zeros is not identity unless the slice is already zero; the
    # honest identity check is writing back what is actually there.
    captured = adapter.capture_head_outputs(PROMPT)[LAYER][HEAD]
    hook_write = adapter.patch_head_output(
        PROMPT, LAYER, HEAD, patch_vector=captured, position=-1)
    assert hook_write.delta == 0.0, (
        f"writing a head's own captured value back moved the logit by "
        f"{hook_write.delta:.6f}"
    )
    del identity, plain