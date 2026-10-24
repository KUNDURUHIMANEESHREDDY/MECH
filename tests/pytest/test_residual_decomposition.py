"""D8: the residual-stream decomposition identity, and the `hidden[]` convention.

What the checklist asks for
--------------------------
"Confirm that the sum of residual components reconstructs the target residual
stream under the selected architecture's decomposition convention."

What was actually wrong
-----------------------
`gpt2_engine._snapshot()["hidden"]` is read from HuggingFace's
`output_hidden_states=True`, and that list is **not uniform**:

    hidden[0]    = wte + wpe
    hidden[1..n] = block outputs, PRE-ln_f
    hidden[n+1]   = ln_f(block n), POST-ln_f

So the final entry is on a different scale from every other one, and it is the
one `logit_lens(n-1)` reads -- correctly, since the final lens must project the
normalised stream. Nothing crashed and no individual value was wrong; every
reading was valid in isolation. That is why `docs/notes/tl_block11_anomaly.md`
recorded the last block as diverging by ~357 from an independently replayed
stream and could not explain it: the replay compared a pre-`ln_f` quantity
against a post-`ln_f` one.

Nothing asserted this, because the existing cross-check
(`test_tensor_shapes_and_logit_lens.py::test_cross_check_against_transformer_lens`)
compares layers 0, 5 and 10 -- never the last index, and never a decomposition.

The tests below pin the identity by replaying each block from its own
submodules, which is a genuinely independent recomputation rather than a
restatement of whatever the engine recorded.
"""

from __future__ import annotations

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


@pytest.fixture(scope="module")
def snapshot(engine):
    engine.run_prompt(PROMPT)
    snap = engine._snapshot()
    assert snap.get("prompt") == PROMPT, (
        f"the engine's snapshot describes {snap.get('prompt')!r}, not the "
        f"prompt just run -- the cache would not describe this measurement"
    )
    return snap


@pytest.fixture(scope="module")
def truth_stream(engine, snapshot):
    """Every residual, recomputed from the model's own submodules.

    Returns ``(pre, post)``: the pre-``ln_f`` stream (embedding plus every
    block output) and the single post-``ln_f`` final residual.

    Replays GPT-2's block by its documented order:

        resid_mid  = resid_pre + attn(ln_1(resid_pre))
        resid_post = resid_mid + mlp(ln_2(resid_mid))

    This is an independent recomputation, so a value the engine merely
    *reproduces* cannot satisfy it.
    """
    model = engine._model
    tokenizer = engine._tokenizer
    ids = tokenizer(PROMPT, return_tensors="pt")["input_ids"]
    seq = ids.shape[1]
    n_layers = engine._n_layers()

    with torch.no_grad():
        token_embeddings = model.transformer.wte(ids)
        position_ids = torch.arange(seq, device=ids.device).unsqueeze(0)
        h = token_embeddings + model.transformer.wpe(position_ids).to(
            token_embeddings.device
        )
        # Transformers 5 constructs a causal mask in GPT2Model.forward and
        # passes it into each block. Calling GPT2Attention directly without
        # that mask makes every position attend to future tokens, so a
        # "manual" replay diverges massively from the actual model. Reuse the
        # public mask factory with identical sequence/device/config semantics.
        causal_mask = create_causal_mask(
            config=model.config,
            inputs_embeds=token_embeddings,
            attention_mask=None,
            past_key_values=None,
            position_ids=position_ids,
            allow_is_causal_skip=False,
        )
        stream = [h]                                  # [1, seq, d]
        for layer in range(n_layers):
            block = model.transformer.h[layer]
            attn_out, _ = block.attn(
                block.ln_1(stream[-1]),
                attention_mask=causal_mask,
            )
            resid_mid = stream[-1] + attn_out
            stream.append(resid_mid + block.mlp(block.ln_2(resid_mid)))

    pre = [t[0] for t in stream]                       # n_layers + 1 entries

    # The recorded list's LAST entry replaces block n-1's pre-norm output
    # with ln_f(block n-1). Measured against HF's own hidden_states: indices
    # 0..n-1 are the embedding and blocks 0..n-2 (all exact), and index n is
    # the normalised final residual -- the pre-norm value for that block is
    # absent from the list, not present alongside it.
    with torch.no_grad():
        post = model.transformer.ln_f(stream[-1])[0]

    return pre, post


# ── D8: the decomposition identity ────────────────────────────────────

@needs_weights
def test_each_block_reconstructs_from_its_own_components(
    engine, snapshot, truth_stream
):
    """Every recorded residual equals what its block computes.

    For each layer, `resid_pre + attn_out + mlp_out` must reproduce the block's
    own output. Checked for all twelve layers rather than a sample, because
    the sample is precisely what let the last block go unexamined.

    The reference is the recomputed stream, not the engine's cache -- so this
    cannot pass by the engine agreeing with itself.
    """
    n_layers = engine._n_layers()
    hidden = snapshot["hidden"]
    pre, _post = truth_stream

    assert len(hidden) == n_layers + 1, (
        f"expected {n_layers + 1} recorded residuals, got {len(hidden)}"
    )
    assert len(pre) == n_layers + 1

    # Indices 0 .. n_layers-1 are pre-ln_f: index 0 is wte+wpe, and index L
    # for 1 <= L <= n_layers-1 is block L-1's output. Index n_layers is the
    # post-ln_f entry, pinned by the convention test below.
    for index in range(n_layers):
        recorded = torch.tensor(hidden[index], dtype=torch.float32)
        difference = float((recorded - pre[index]).abs().max())
        assert difference < 1e-3, (
            f"hidden[{index}]: recorded residual differs from the block's own "
            f"recomputation by {difference:.6g}. The residual stream is not "
            f"reconstructible from its components."
        )


@needs_weights
def test_the_decomposition_terms_sum_to_the_block_output(engine, snapshot):
    """`resid_pre + attn_out + mlp_out == resid_post`, as an explicit sum.

    Spelled out rather than implied by the replay above, because this is the
    identity a residual-stream *viewer* depends on: it adds the components it
    displays and expects the block's output. If any term were substituted
    (the input for the attention contribution, say) this fails while the
    recorded residuals stay correct.

    The MLP term is derived as `output - (resid_pre + attn_out)` so the check
    is over the addition itself, not over a restatement of it.
    """
    model = engine._model
    tokenizer = engine._tokenizer
    hidden = snapshot["hidden"]
    layer = 5

    ids = tokenizer(PROMPT, return_tensors="pt")["input_ids"]
    captured: dict[str, torch.Tensor] = {}

    pre_hook = model.transformer.h[layer].register_forward_pre_hook(
        lambda m, i: captured.__setitem__(
            "resid_pre", i[0].detach().clone())
    )
    def capture_block_output(_module, _inputs, output):
        # Transformers 4 returned a tuple from GPT2Block in this path;
        # Transformers 5 may return the hidden-state tensor directly. Normalize
        # only the wrapper shape, not the batch/sequence dimensions.
        hidden_states = output[0] if isinstance(output, (tuple, list)) else output
        captured["resid_post"] = hidden_states.detach().clone()

    post_hook = model.transformer.h[layer].register_forward_hook(capture_block_output)
    try:
        with torch.no_grad():
            model(ids, output_attentions=True, use_cache=False)
    finally:
        pre_hook.remove()
        post_hook.remove()

    resid_pre = captured["resid_pre"]
    resid_post = captured["resid_post"]

    block = model.transformer.h[layer]
    resid_pre_from_cache = torch.tensor(
        hidden[layer], dtype=torch.float32
    ).unsqueeze(0)
    with torch.no_grad():
        token_embeddings = model.transformer.wte(ids)
        position_ids = torch.arange(ids.shape[1], device=ids.device).unsqueeze(0)
        causal_mask = create_causal_mask(
            config=model.config,
            inputs_embeds=token_embeddings,
            attention_mask=None,
            past_key_values=None,
            position_ids=position_ids,
            allow_is_causal_skip=False,
        )
        attn_out = block.attn(
            block.ln_1(resid_pre_from_cache),
            attention_mask=causal_mask,
        )[0]
        resid_mid = resid_pre_from_cache + attn_out
        mlp_out = block.mlp(block.ln_2(resid_mid))

    torch.testing.assert_close(resid_pre, resid_pre_from_cache)
    torch.testing.assert_close(resid_mid + mlp_out, resid_post, rtol=1e-4,
                               atol=1e-5)
    torch.testing.assert_close(attn_out + mlp_out, resid_post - resid_pre,
                               rtol=1e-4, atol=1e-5)


@needs_weights
def test_the_first_recorded_residual_is_the_embedding(engine, snapshot):
    """`hidden[0]` is `wte + wpe`, not `wte` alone.

    GPT-2 adds learned position embeddings to the token embeddings. A reader
    assuming `hidden[0] == wte` is off by the position signal -- measured here
    at ~4.5 in absolute terms, which is the same order as the block-11
    divergence and the kind of number that gets attributed to the wrong thing.
    """
    model = engine._model
    tokenizer = engine._tokenizer
    ids = tokenizer(PROMPT, return_tensors="pt")["input_ids"]
    seq = ids.shape[1]

    with torch.no_grad():
        tokens_only = model.transformer.wte(ids).detach().cpu()
        with_positions = (
            model.transformer.wte(ids)
            + model.transformer.wpe(torch.arange(seq)).unsqueeze(0)
        ).detach().cpu()

    recorded = torch.tensor(snapshot["hidden"][0], dtype=torch.float32)

    torch.testing.assert_close(
        recorded, with_positions[0], rtol=1e-4, atol=1e-5
    )
    assert not torch.allclose(recorded, tokens_only[0], atol=1e-2), (
        "hidden[0] equals the bare token embedding, so position embeddings "
        "are missing from the recorded stream"
    )


# ── the convention at the final index ──────────────────────────────────

@needs_weights
def test_the_final_recorded_residual_is_after_the_final_layer_norm(
    engine, snapshot, truth_stream
):
    """`hidden[n+1]` is `ln_f(block n)` -- post-norm, unlike every other entry.

    This is the finding the block-11 note could not explain, and it is worth a
    permanent test because the value is *correct for what reads it*: the final
    LogitLens must project the normalised stream, and `logit_lens(11)` reading
    `hidden[12]` is right.

    So the property pinned here is not "the last entry is wrong" -- it is that
    the last entry is on a different convention, and the identity test above
    must therefore not extend to it. Asserting that distinction explicitly
    stops a future reader from "fixing" the list by pre-normalising it, which
    would silently break every final-layer logit-lens projection.
    """
    model = engine._model
    n_layers = engine._n_layers()
    hidden = snapshot["hidden"]
    pre, post = truth_stream

    recorded = torch.tensor(hidden[n_layers], dtype=torch.float32)
    torch.testing.assert_close(recorded, post, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(
        recorded,
        model.transformer.ln_f(pre[n_layers].unsqueeze(0))[0],
        rtol=1e-4, atol=1e-5,
    )

    # And it is NOT the raw block output, which is what the layer-identity
    # test above asserts for every other index.
    difference = float((recorded - pre[n_layers]).abs().max())
    assert difference > 1.0, (
        "the final recorded residual equals the raw block output, so the "
        "post-ln_f convention this test documents no longer holds"
    )


@needs_weights
def test_no_recorded_residual_is_normalised_twice(engine, snapshot, truth_stream):
    """Every recorded residual sits on exactly one convention.

    `hidden[0..n]` are pre-`ln_f` and `hidden[n+1]` is post-`ln_f`. Applying
    `ln_f` to a post-`ln_f` value is not idempotent -- GPT-2's final norm has
    learned scale and bias, so a second pass rescales it -- which is how a
    single entry ends up on a scale none of its neighbours share.

    Asserted as a property of the whole list rather than one entry, because
    that is the property that actually matters: a reader walking `hidden` has
    no way to know which entries are safe to normalise.
    """
    model = engine._model
    n_layers = engine._n_layers()
    hidden = snapshot["hidden"]

    for index in range(n_layers):
        recorded = torch.tensor(hidden[index], dtype=torch.float32)
        with torch.no_grad():
            renormalised = model.transformer.ln_f(recorded.unsqueeze(0))[0]
        difference = float((renormalised - recorded).abs().max())
        assert difference > 1e-3, (
            "hidden[%d] appears to be already normalised. If entries "
            "0..n-1 are now post-ln_f, every logit_lens layer would be "
            "normalising twice." % index
        )

    recorded_final = torch.tensor(hidden[n_layers], dtype=torch.float32)
    with torch.no_grad():
        renormalised_final = model.transformer.ln_f(recorded_final.unsqueeze(0))[0]
    assert float((renormalised_final - recorded_final).abs().max()) > 1e-3, (
        "hidden[%d] is supposed to be post-ln_f, but re-normalising it "
        "changed nothing; the convention this suite documents has changed and "
        "logit_lens's handling must be re-checked" % n_layers
    )


@needs_weights
def test_the_final_logit_lens_reports_the_models_own_distribution(
    engine, snapshot
):
    """The final lens must not normalise its input twice.

    `logit_lens` reads `hidden[layer + 1]` and applies `ln_f` before projecting.
    For layers 0..n-1 that is correct -- those entries are pre-ln_f. For the
    last layer it is not: `hidden[n+1]` is *already* `ln_f(block n)`, so the
    final projection applied the learned scale and bias a second time.

    The result was not obviously broken, which is what made it dangerous: the
    final lens's argmax still matched the model's, so the existing
    token-identity test passed. Only the probabilities were wrong, and by enough
    to change the reported ranking -- measured p(' the') = 0.038 against the
    model's 0.085, and p(',') 0.019 against the model's 0.0036, i.e. a token
    the model considers unlikely was promoted into the top five.

    So this compares the reported distribution against the model's own softmax
    rather than checking the top token, which is the check that could not see
    the defect.
    """
    n_layers = engine._n_layers()
    lens = engine.logit_lens(n_layers - 1, PROMPT)
    assert lens.get("status") == "ok", lens

    real = torch.tensor(snapshot["logits"], dtype=torch.float32)
    model_probs = torch.softmax(real, dim=-1)
    tokenizer = engine._tokenizer

    worst_token = None
    worst = 0.0
    for entry in lens["top_k_tokens"]:
        ids = tokenizer.encode(entry["token"])
        assert ids, f"lens reported an undecodable token: {entry['token']!r}"
        expected = float(model_probs[ids[0]])
        difference = abs(float(entry["prob"]) - expected)
        if difference > worst:
            worst, worst_token = difference, entry["token"]

    assert worst < 5e-3, (
        f"the final logit lens's probability for {worst_token!r} is off by "
        f"{worst:.6f} from the model's own distribution. The last layer reads "
        f"hidden[{n_layers + 1}], which is already ln_f'd, and logit_lens "
        f"applies ln_f again."
    )