"""D10/D11: position identity, and refusal of indices that do not exist.

The defect
----------
`gpt2_engine` normalised every layer and head index with

    layer = max(0, min(_n_layers() - 1, int(layer)))

at sixteen call sites. So `activations(999)` returned layer 11's data, with
`status: "ok"`, no error, and no flag saying a substitution had happened.
Measured: `activations(999)` is bit-identical to `activations(11)`.

That is the most dangerous shape of wrong answer this project can produce. The
caller asked about a layer that does not exist and received a complete,
plausible, correctly-shaped tensor for a *different* layer. Nothing in the
response distinguishes it from a real measurement -- which is precisely the
condition the project's own rules single out ("an exception is never converted
into a successful result").

The repository already settled how this should behave.
`HFAdapterMixin._layer_error` refuses with "outside this model ... Nothing was
measured", and `test_non_gpt2_adapters_are_real.py::test_a_missing_layer_is_reported_not_clamped`
pins that wording. The non-GPT-2 path refuses; the GPT-2 path clamped. These
tests make the two agree.

D10 is included here because it is the adjacent question and the one an index
error would corrupt silently: if `mlp_post[i]` did not correspond to
`str_tokens[i]`, then every per-token reading in the UI would be attached to
the wrong token, with valid shapes throughout.
"""

from __future__ import annotations

import pytest

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
def primed(engine):
    engine.run_prompt(PROMPT)
    return engine._snapshot()


# ── D11: invalid indices must be refused, not clamped ──────────────────

@needs_weights
@pytest.mark.parametrize("requested", [-1, 12, 17, 999])
def test_an_out_of_range_layer_is_refused_not_clamped(engine, requested):
    """Asking for a layer that does not exist must not return another layer.

    Checked by identity, not just by status: the assertion that matters is that
    the response does not carry a real measurement. An implementation that
    clamped and *also* set `status: "error"` would still be returning data
    describing a different layer, so the tensor itself is checked for absence.
    """
    n_layers = engine._n_layers()
    result = engine.layer_activations(requested, PROMPT)

    assert result.get("status") != "ok", (
        f"layer_activations({requested}) returned status "
        f"{result.get('status')!r} for a {n_layers}-layer model. A layer index "
        f"outside 0..{n_layers - 1} has no activation to report."
    )
    assert not result.get("mlp_post"), (
        f"layer_activations({requested}) returned an mlp_post tensor. That "
        f"tensor describes layer {n_layers - 1}, not a layer that was asked for."
    )
    assert not result.get("resid_post"), (
        f"layer_activations({requested}) returned a resid_post tensor; see "
        f"mlp_post note above."
    )


@needs_weights
@pytest.mark.parametrize("requested", [-1, 12, 500])
def test_an_out_of_range_head_is_refused_not_clamped(engine, requested):
    """Same for heads: no silent substitution of the last head."""
    n_heads = engine._n_heads()
    result = engine.attention_head(engine._n_layers() - 1, requested, PROMPT)

    assert result.get("status") != "ok", (
        f"attention_head(head={requested}) returned status "
        f"{result.get('status')!r} for a {n_heads}-head model."
    )
    assert result.get("pattern") is None, (
        f"attention_head(head={requested}) returned an attention pattern. It "
        f"describes head {n_heads - 1}, which was not the head requested."
    )


@needs_weights
def test_a_refusal_names_the_valid_range_and_measured_nothing(engine):
    """A refusal must be actionable and must claim no measurement.

    The two properties `HFAdapterMixin._layer_error` already guarantees, pinned
    here for the GPT-2 path so the two cannot drift: the message says which
    range is valid, and the response says nothing was measured.
    """
    n_layers = engine._n_layers()
    result = engine.layer_activations(999, PROMPT)

    reason = str(result.get("reason", ""))
    assert reason, "a refusal must say why"
    assert str(n_layers) in reason, (
        f"the refusal does not name the model's layer count ({n_layers}): "
        f"{reason!r}"
    )
    assert result.get("measured") is not True, (
        "a refused request reported measured=True"
    )


@needs_weights
def test_in_range_indices_are_unaffected(engine):
    """The guard must not reject valid requests.

    A refusal added without this check would make the whole layer API
    unusable, which is its own kind of wrong answer. Every index the checklist
    calls valid must still return data.
    """
    for layer in (0, 1, engine._n_layers() - 1):
        result = engine.layer_activations(layer, PROMPT)
        assert result.get("status") == "ok", (
            f"layer {layer} is in range but was refused: {result}"
        )
        assert result.get("mlp_post"), f"layer {layer} returned no activations"

    for head in (0, 1, engine._n_heads() - 1):
        result = engine.attention_head(1, head, PROMPT)
        assert result.get("status") == "ok", (
            f"head {head} is in range but was refused: {result}"
        )


# ── D10: which token each reading belongs to ──────────────────────────

@needs_weights
def test_each_row_of_a_layer_reading_belongs_to_its_own_token(engine, primed):
    """Row `i` must correspond to token `i`, verified against a fresh pass.

    `_forward_locked` builds `mlp_post` from the token tensor and `str_tokens`
    from the id list separately. If those two ever disagreed -- an off-by-one
    from padding, a BOS insertion, a decode applied to the wrong axis -- every
    per-token reading would attach to the wrong token while keeping a valid
    shape, which no shape assertion could detect.

    The reference here is an independent forward pass on the same ids, so the
    correspondence is checked rather than assumed.
    """
    import torch

    layer = 3
    result = engine.layer_activations(layer, PROMPT)
    assert result.get("status") == "ok", result

    model = engine._model
    tokenizer = engine._tokenizer
    ids = tokenizer(PROMPT, return_tensors="pt")["input_ids"]

    with torch.no_grad():
        out = model(
            input_ids=ids,
            output_hidden_states=True,
            attn_implementation="eager",
        )
    # hidden_states[L + 1] is the residual stream after block L.
    reference = out.hidden_states[layer + 1][0].detach().cpu().numpy()

    import numpy as np

    engine_resid = np.asarray(result["resid_post"])
    assert engine_resid.shape == reference.shape, (
        f"engine returned {engine_resid.shape} for a {reference.shape} reading; "
        f"the token axes disagree"
    )
    difference = float(np.abs(engine_resid - reference).max())
    assert difference < 1e-3, (
        f"the engine's residual stream differs from an independent forward pass "
        f"by {difference:.3e}"
    )

    # And the row count must equal the number of tokens, not merely be square.
    tokens = primed["str_tokens"]
    assert engine_resid.shape[0] == len(tokens), (
        f"{engine_resid.shape[0]} activation rows for {len(tokens)} tokens"
    )


@needs_weights
def test_the_token_axis_survives_a_prompt_of_different_length(engine):
    """Row count tracks the prompt, so the correspondence is not incidental.

    Two prompts of clearly different lengths, each checked against its own
    forward pass. If the engine were padding both to some fixed width, the
    shorter prompt would still have the right *shape* relative to its own
    tokens and the mismatch would only show here.
    """
    import numpy as np
    import torch

    model = engine._model
    tokenizer = engine._tokenizer

    for prompt in ("Paris", "The quick brown fox jumps over the lazy dog"):
        engine.run_prompt(prompt)
        layer = 2
        result = engine.layer_activations(layer, prompt)
        assert result.get("status") == "ok", result

        ids = tokenizer(prompt, return_tensors="pt")["input_ids"]
        with torch.no_grad():
            out = model(
                input_ids=ids,
                output_hidden_states=True,
                attn_implementation="eager",
            )
        reference = out.hidden_states[layer + 1][0].detach().cpu().numpy()
        engine_resid = np.asarray(result["resid_post"])

        assert engine_resid.shape == reference.shape, (
            f"prompt {prompt!r}: engine {engine_resid.shape} vs "
            f"independent {reference.shape}"
        )
        assert float(np.abs(engine_resid - reference).max()) < 1e-3, (
            f"prompt {prompt!r}: per-token values differ from an independent "
            f"forward pass"
        )