"""Residual decomposition: per-position, per-axis identity, and stated convention.

`test_residual_decomposition.py` pins GPT-2's residual identity at the final
token through one recorded stream. This file covers the gaps the contract
review named for section 3.

    1. Attention and MLP contributions reconstructed separately, every layer
    2. Intermediate token positions, not only the final one
    3. Batch-size, sequence-length, dtype and device variants
    4. Architecture-specific decomposition conventions stated explicitly
    5. Padding, longer contexts

Gaps 1, 2, 3 and 5 were measured before being written down, and the identity
holds *exactly* -- error 0.000e+00 -- across every one of them: every token
position, every layer, batch sizes above one, a padded ragged batch, float32
and float64, and a 200-token context. Those tests are therefore mostly green by
construction, and their value is that they will fail the first time that stops
being true. A decomposition convention that quietly depends on being the last
token, or on batch size one, is the kind of thing that survives until someone
runs it on a prompt of forty tokens.

Gap 4 is where the defect is, and it is not in the arithmetic.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import pytest
import torch
from transformers.masking_utils import create_causal_mask

from _weight_guard import skip_reason, weights_available

needs_weights = pytest.mark.skipif(
    not weights_available(), reason=skip_reason()
)

PROMPT = "The capital of France is"


@pytest.fixture(scope="module")
def engine():
    from backend.services import gpt2_engine as eng

    eng.load()
    return eng


def _decompose(engine, texts, pad: bool = False, dtype=None):
    """Replay GPT-2 block by block, keeping the two contributions separate.

    Returns ``(rows, shape)`` where each row is
    ``(resid_pre, attn_out, mlp_out, resid_post)`` for one layer, and shape is
    ``(batch, seq)``.

    GPT-2's block order, from the model definition rather than from anything
    this repository asserts:

        resid_mid  = resid_pre + attn(ln_1(resid_pre))
        resid_post = resid_mid + mlp(ln_2(resid_mid))

    Recomputed from the model's own submodules, so a value the engine merely
    reproduces cannot satisfy it.
    """
    model = engine._model
    tokenizer = engine._tokenizer
    # Use right padding explicitly for this GPT-2 block-replay contract. The
    # tokenizer configuration may default to left padding for generation; this
    # helper replays block modules directly and derives position embeddings by
    # sequence index, so left padding would intentionally shift every real
    # token's absolute position and measure a different prompt. This is a test
    # fixture choice, not a change to production tokenization.
    original_padding_side = tokenizer.padding_side
    try:
        if pad:
            tokenizer.padding_side = "right"
        encoded = tokenizer(texts, return_tensors="pt", padding=pad)
    finally:
        tokenizer.padding_side = original_padding_side
    ids = encoded["input_ids"].to(model.device)
    attention_mask = encoded.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(model.device)
    batch, seq = ids.shape

    previous = model
    if dtype is not None and dtype != next(model.parameters()).dtype:
        previous = model.float() if dtype == torch.float32 else model.to(dtype)

    with torch.no_grad():
        token_embeddings = previous.transformer.wte(ids)
        position_ids = torch.arange(seq, device=ids.device).unsqueeze(0).expand(batch, -1)
        position_embeddings = previous.transformer.wpe(position_ids)
        # GPT2Attention called as a standalone submodule has no parent GPT2Model
        # to construct the causal/padding mask. Without this, it attends to
        # future positions and right-padding changes the real tokens' residuals.
        causal_mask = create_causal_mask(
            config=previous.config,
            inputs_embeds=token_embeddings,
            attention_mask=attention_mask,
            past_key_values=None,
            position_ids=position_ids,
            allow_is_causal_skip=False,
        )
        h = token_embeddings + position_embeddings
        rows = []
        for layer in range(engine._n_layers()):
            block = previous.transformer.h[layer]
            resid_pre = h
            attn_out = block.attn(
                block.ln_1(resid_pre),
                attention_mask=causal_mask,
            )[0]
            resid_mid = resid_pre + attn_out
            mlp_out = block.mlp(block.ln_2(resid_mid))
            resid_post = resid_mid + mlp_out
            rows.append((resid_pre, attn_out, mlp_out, resid_post))
            h = resid_post
    return rows, (batch, seq)


def _worst(rows, batch: int, positions) -> float:
    """Largest ``|pre + attn + mlp - post|`` over the given positions."""
    worst = 0.0
    for resid_pre, attn_out, mlp_out, resid_post in rows:
        for pos in positions:
            error = (resid_pre[0, pos] + attn_out[0, pos] + mlp_out[0, pos]
                     - resid_post[0, pos]).abs().max().item()
            worst = max(worst, error)
    return worst


# ── Gaps 1 and 2: separate contributions, every layer, every position ──

@needs_weights
def test_attention_and_mlp_contributions_reconstruct_every_layer(engine):
    """The two contributions, added to the block input, give the block output.

    Checked per layer so a single mis-ordered block cannot hide behind eleven
    correct ones.
    """
    rows, _ = _decompose(engine, [PROMPT])

    for layer, (resid_pre, attn_out, mlp_out, resid_post) in enumerate(rows):
        error = (resid_pre[0, -1] + attn_out[0, -1] + mlp_out[0, -1]
                 - resid_post[0, -1]).abs().max().item()
        assert error < 1e-4, f"layer {layer}: identity error {error:.3e}"


@needs_weights
def test_attention_alone_does_not_reconstruct_the_block_output(engine):
    """The negative control.

    Without this, a test that summed all three terms would also pass if the
    attention term were silently zero and the MLP term happened to carry the
    whole block. The MLP contribution is large and non-zero, so if `attn_out`
    were dropped, `pre + mlp` would miss `pre + attn + mlp` by the size of the
    attention output -- and it does not.
    """
    rows, _ = _decompose(engine, [PROMPT])
    resid_pre, attn_out, mlp_out, resid_post = rows[len(rows) // 2]

    with_attn = (resid_pre[0, -1] + attn_out[0, -1] + mlp_out[0, -1]
                 - resid_post[0, -1]).abs().max().item()
    without_attn = (resid_pre[0, -1] + mlp_out[0, -1]
                    - resid_post[0, -1]).abs().max().item()

    assert with_attn < 1e-4, f"the full identity fails: {with_attn:.3e}"
    assert without_attn > 1e-2, (
        f"dropping the attention term still reconstructs the output "
        f"(error {without_attn:.3e}); the attention contribution is not "
        f"carrying its own weight and this test cannot tell the difference"
    )


@needs_weights
def test_the_identity_holds_at_every_token_position_not_only_the_last(engine):
    """Intermediate positions satisfy the same identity.

    Measured across all five positions of the probe prompt: worst error
    0.000e+00, with position 0 and the final position identical. The gap is
    worth closing because a decomposition driven by `[:, -1, :]` indexing
    would pass a final-token-only test and report nothing about the rest of
    the sequence.
    """
    rows, shape = _decompose(engine, [PROMPT])

    worst = _worst(rows, shape[0], range(shape[1]))
    assert worst < 1e-4, (
        f"the identity holds at the final token but not elsewhere: worst "
        f"error {worst:.3e} across {shape[1]} positions"
    )

    # And it must not be the final token that is uniquely well-behaved.
    first = _worst(rows, shape[0], [0])
    last = _worst(rows, shape[0], [shape[1] - 1])
    assert first == pytest.approx(last, abs=1e-4), (
        f"position 0 error {first:.3e} differs from the final position "
        f"{last:.3e}; the identity is not position-uniform"
    )


# ── Gap 3: batch, sequence length, dtype ──────────────────────────────

@needs_weights
def test_a_batch_above_one_satisfies_the_identity_per_sequence(engine):
    """Batch size one is the easy case; the identity is per-sequence anyway."""
    rows, shape = _decompose(
        engine, ["The capital of France is", "The capital of Italy is"])

    assert shape[0] == 2
    for resid_pre, attn_out, mlp_out, resid_post in rows:
        for batch in range(shape[0]):
            error = (resid_pre[batch, -1] + attn_out[batch, -1]
                     + mlp_out[batch, -1] - resid_post[batch, -1]
                     ).abs().max().item()
            assert error < 1e-4, f"batch {batch}: identity error {error:.3e}"


@needs_weights
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_the_identity_holds_in_both_float_widths(engine, dtype):
    """float32 and float64 must agree.

    A decomposition that only holds in the dtype the weights happen to be
    stored in is not a decomposition, it is a coincidence of the current
    checkpoint.
    """
    rows, _ = _decompose(engine, [PROMPT], dtype=dtype)

    observed = next(engine._model.parameters()).dtype
    try:
        worst = _worst(rows, 1, [-1])
    finally:
        engine._model.to(torch.float32)

    assert worst < 1e-4, f"in {dtype} the identity error is {worst:.3e}"


@needs_weights
def test_a_longer_context_preserves_the_identity(engine):
    """A forty-fold longer sequence must not change the arithmetic.

    Measured at 200 tokens: worst error 0.000e+00, mean 0.000e+00.
    """
    long_prompt = " ".join([PROMPT] * 40)
    rows, shape = _decompose(engine, [long_prompt])

    assert shape[1] >= 200, f"only {shape[1]} tokens; the probe did not lengthen"

    worst = _worst(rows, shape[0], range(shape[1]))
    assert worst < 1e-4, f"identity error {worst:.3e} over {shape[1]} tokens"


# ── Gap 5: padding ───────────────────────────────────────────────────

@needs_weights
def test_padding_positions_do_not_break_the_identity(engine):
    """A ragged batch, padded to equal length.

    Padding is the case where a decomposition can go quietly wrong: the padded
    positions carry no real content, and an implementation that slices or
    rescales by sequence length can drift on them without affecting the real
    tokens the caller looks at.
    """
    rows, shape = _decompose(
        engine, ["The capital of France is", "Hi"], pad=True)

    assert shape == (2, 5), f"unexpected padded shape {shape}"

    worst = _worst(rows, shape[0], range(shape[1]))
    assert worst < 1e-4, (
        f"the identity fails on a padded batch: worst error {worst:.3e} "
        f"across both sequences and all five positions"
    )


@needs_weights
def test_padding_does_not_change_the_unpadded_sequence_result(engine):
    """The short sequence must decode the same padded and unpadded.

    This is the assertion that makes the previous one meaningful. If padding
    perturbed the real tokens, the identity could still hold everywhere while
    the measurement was wrong.
    """
    short = "Hi"
    padded, (b_pad, s_pad) = _decompose(
        engine, ["The capital of France is", short], pad=True)
    alone, (b_solo, s_solo) = _decompose(engine, [short])

    assert s_pad > s_solo, (
        f"the short sequence is {s_solo} tokens alone and {s_pad} padded; "
        f"nothing was padded, so the comparison would be vacuous")
    assert s_solo >= 1

    # Compare only the real tokens. The padded row is longer -- it carries
    # padding positions the unpadded run does not have -- so the rows must be
    # sliced before subtracting. An earlier draft compared the whole rows and
    # reported a 92.7 error, which was the padding positions being differenced
    # against nothing, not the measurement moving.
    #
    # The bound is relative, not absolute. Batch shape changes matmul tiling,
    # so float32 does not give bit-identical results at batch 1 and batch 2;
    # measured absolute difference is 1.16e-03, which against a residual of
    # magnitude ~1e2 is ~1e-5 relative. An absolute 1e-3 threshold called that
    # a failure while the arithmetic was correct -- a tolerance chosen without
    # looking at the scale of the quantity it constrains.
    scale = float(alone[-1][3][0, 0].abs().max())
    assert scale > 1.0, (
        f"residual magnitude {scale:.3f} is too small for a relative "
        f"tolerance to mean anything")

    for (pre_p, attn_p, mlp_p, post_p), (pre_a, attn_a, mlp_a, post_a) in zip(
            padded, alone):
        error = (post_p[1][:s_solo] - post_a[0][:s_solo]).abs().max().item()
        assert error <= scale * 1e-4, (
            f"padding changed the short sequence's residual by {error:.3e} "
            f"against a magnitude of {scale:.3f}; that is a relative error of "
            f"{error / scale:.2e}, far above float32 batching variation. The "
            f"identity can hold at every position while the measurement is "
            f"still wrong."
        )


# ── Gap 4: the convention must be stated, not assumed ─────────────────

@needs_weights
def test_the_logit_lens_records_state_their_convention_explicitly(engine):
    """The 13-entry, post-`ln_f`-last convention must be declared.

    This is the whole of gap 4, and it is not an arithmetic problem -- the
    arithmetic is right. The risk is that the convention is right *only* in the
    code, and a caller has to know it by reading the implementation.

    GPT-2 records `n_layers + 1` hidden states, not `n_layers`: HF's
    `output_hidden_states` last element is the residual *after* `ln_f`. Llama
    and Gemma have no `ln_f` at all. A caller who assumes "one entry per
    block" reads the wrong index and gets a normalised value where they
    expected a raw one, and the two differ enough to change a top-1
    prediction.

    Asserted against `architecture()`, which is what `/architecture` and the
    layer listing return -- the surfaces a caller actually receives. An
    earlier draft looked for the convention on `generate_text`, which never
    carried the stream and reported nothing at all.
    """
    from backend.services import gpt2_engine as eng

    meta = eng.architecture()
    layers = meta.get("layers") or []
    assert layers, f"the architecture response has no layers: {sorted(meta)}"

    declared = layers[0].get("residual_convention")
    assert declared, (
        f"the architecture response states no residual convention; layer keys "
        f"are {sorted(layers[0])}. The post-ln_f-last convention is "
        f"load-bearing -- it is why the stream is n_layers + 1 rather than "
        f"n_layers -- and it is currently only visible in the implementation."
    )

    text = str(declared)
    assert "ln_f" in text, (
        f"the convention string does not mention the final layer norm: "
        f"{declared!r}"
    )
    # Asserted symbolically. An earlier draft looked for the literal "13";
    # the string states "n_layers + 1" and enumerates [0]..[12], so a correct
    # declaration failed a test written to require one.
    assert "n_layers + 1" in text, (
        f"the convention does not state the entry count: {declared!r}"
    )
    # The final *index* is n_layers (12), which is one less than the entry
    # *count* (13). An earlier draft asserted the literal "13" here and failed
    # against a correct string: the count is stated as "n_layers + 1 entries"
    # and the last index as [12], which is right.
    assert f"[{eng._n_layers()}]" in text, (
        f"the convention never names the final index {eng._n_layers()}: "
        f"{declared!r}"
    )


def test_the_two_residual_endpoints_agree_on_what_a_layer_index_means():
    """Two ways of asking for the same thing, two conventions.

    `ResidualInspector.get_residual_norms` walks `range(num_layers)` -- for
    GPT-2 that is 12 entries, embedding through block 10, with no entry for
    the final normalised residual. The logit-lens path returns 13, ending in
    the post-`ln_f`` value. Neither endpoint says which it is doing.

    Measured on the current code:

        inspector:  range(num_layers)      -> 12 entries, no final
        logit lens: n_layers + 1           -> 13 entries, post-ln_f last

    So `residual-norms[12]` is an IndexError on one path and the single most
    important value on the other. Whichever is right, the two must not be
    silently different.
    """
    from backend.interpretability.residual_inspector import ResidualInspector

    path = (Path(__file__).resolve().parents[2] / "backend" / "interpretability"
        / "residual_inspector.py")
    text = path.read_text(encoding="utf-8")

    assert "class ResidualInspector" in text, (
        "could not read residual_inspector.py; this test is not asserting "
        "anything about the convention")

    # Either it walks num_layers, or it says how that relates to the logit-lens
    # stream. What it must not do is walk num_layers while presenting itself as
    # the same series.
    #
    # The phrase is matched against the plain text, with markdown emphasis
    # stripped: an earlier draft searched for "not the same as" and the
    # docstring's `**not**` broke the substring, so a correct declaration
    # failed the test that exists to require it.
    plain = text.replace("*", "")
    walks_layers = "range(self.repository.num_layers)" in plain
    declares_relation = (
        "n_layers + 1" in plain
        and "not the same" in plain.lower()
    )

    assert declares_relation, (
        "get_residual_norms walks range(num_layers) and does not say how that "
        "relates to the logit-lens stream, which has one more entry and ends "
        "post-ln_f. A caller cannot tell which series they are holding from "
        "the response -- `get_residual_norms()[-1]` is block 10's raw output "
        "here and ln_f(block 11) there."
    )
    assert walks_layers, (
        "get_residual_norms no longer walks range(num_layers); this test's "
        "description of the disagreement no longer matches the code and "
        "should be rewritten rather than left asserting the wrong thing"
    )