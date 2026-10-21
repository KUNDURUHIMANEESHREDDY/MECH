"""The token `rank` in the logit lens is a full-vocabulary position.

Spec §13.2 requires token-rank trajectories. A UI can only draw one if the
engine reports where each token sits in the whole next-token distribution --
its position in the truncated top-k is not a rank, and a view that relabelled
an array index as a rank would be inventing the number it plots.

These tests recompute the distribution from the model and require the reported
rank, probability and ordering to agree with it.
"""

from __future__ import annotations

import pytest
import torch

from _weight_guard import skip_reason, weights_available

needs_weights = pytest.mark.skipif(
    not weights_available(), reason=skip_reason(),
)

PROMPT = "The capital of France is"
TOP_K = 5


@pytest.fixture(scope="module")
def engine():
    from backend.services import gpt2_engine as eng

    eng.load()
    return eng


@pytest.fixture(scope="module")
def final_layer(engine):
    return engine._n_layers() - 1


@pytest.fixture(scope="module")
def lens(engine, final_layer):
    out = engine.logit_lens(final_layer, PROMPT, top_k=TOP_K)
    assert out.get("status") == "ok", out
    return out


def test_every_returned_token_carries_rank_logit_and_prob(lens):
    tokens = lens["top_k_tokens"]
    assert len(tokens) == TOP_K
    for entry in tokens:
        assert set(entry) >= {"token", "prob", "logit", "rank"}, entry
        assert isinstance(entry["rank"], int), entry
        assert entry["rank"] >= 1, entry


def test_rank_equals_position_in_the_full_distribution(engine, lens, final_layer):
    """Recompute the distribution; the reported rank must match it exactly.

    The rank is an integer position, so it is compared exactly. The
    probability is not: the lens reports a float32 softmax of the unembedding
    applied to the cached hidden state, and this test recomputes a softmax of
    a separate forward pass's logits. Two float32 passes over the same weights
    do not agree to the last bit -- measured at 1.8e-6 on p(' the') on Linux --
    so the probability is compared at float32 resolution. A real error in the
    lens is not a rounding difference: the double-normalisation this pair of
    tests was written for put p(' the') at 0.038 against the model's 0.085.
    """
    from backend.services import gpt2_engine as eng

    batch = eng._tokenizer(PROMPT, return_tensors="pt")
    input_ids = batch["input_ids"].to(eng._model.device)
    attention_mask = batch.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(eng._model.device)

    config = eng._model.config
    previous = getattr(config, "_attn_implementation", None)
    if previous != "eager":
        config._attn_implementation = "eager"
    try:
        with torch.no_grad():
            logits = eng._model(
                input_ids=input_ids, attention_mask=attention_mask,
            ).logits[0, -1, :].detach().cpu()
    finally:
        if previous is not None:
            config._attn_implementation = previous

    probs = torch.softmax(logits, dim=-1)

    for entry in lens["top_k_tokens"]:
        ids = eng._tokenizer.encode(entry["token"])
        assert len(ids) == 1, (
            f"{entry['token']!r} is not a single token; the label would be "
            "checked against the wrong id"
        )
        token_id = ids[0]

        # True 1-based rank in the full distribution.
        true_rank = int((probs > probs[token_id]).sum().item()) + 1
        assert entry["rank"] == true_rank, (
            f"{entry['token']!r}: reported rank {entry['rank']}, "
            f"distribution says {true_rank}"
        )
        assert entry["prob"] == pytest.approx(
            float(probs[token_id]), rel=1e-3, abs=1e-6
        ), f"{entry['token']!r} probability"

    ranks = [e["rank"] for e in lens["top_k_tokens"]]
    probs_reported = [e["prob"] for e in lens["top_k_tokens"]]

    assert ranks == list(range(1, TOP_K + 1)), (
        f"top-k ranks must be 1..{TOP_K} in order, got {ranks}"
    )
    assert probs_reported == sorted(probs_reported, reverse=True), (
        f"probabilities must fall with rank, got {probs_reported}"
    )
    assert lens["top_token"] == lens["top_k_tokens"][0]["token"]


def test_rank_is_not_the_array_index(engine, final_layer):
    """A token outside top-k must not be reported as if it were top-k.

    Asking for a single token makes rank 1 the argmax; the second entry of a
    top-2 list must therefore be rank 2, never "index 1" relabelled -- the
    distinction is invisible only when the two happen to coincide, which they
    do for top-k, so this checks the truncation boundary instead: the reported
    list length is the truncation, and the contract says so.
    """
    out = engine.logit_lens(final_layer, PROMPT, top_k=1)
    assert out["status"] == "ok", out
    assert out["probability_is_truncated"] is True
    assert out["probabilities_reported"] == 1
    assert out["probabilities_total"] == engine._model.config.vocab_size
    entry = out["top_k_tokens"][0]
    assert entry["rank"] == 1, entry
    assert out["top_token"] == entry["token"]
    assert out["probabilities_reported_mass"] == pytest.approx(
        entry["prob"], abs=1e-6
    )


def test_every_layer_carries_the_truncation_contract(engine):
    out = engine.logit_lens_all(PROMPT, top_k=3)
    assert out["status"] == "ok", out
    assert len(out["layers"]) == engine._n_layers()

    for layer in out["layers"]:
        assert layer["probability_is_truncated"] is True, layer["layer"]
        assert layer["probabilities_reported"] == 3, layer["layer"]
        assert layer["probabilities_total"] == engine._model.config.vocab_size
        assert [t["rank"] for t in layer["top_k_tokens"]] == [1, 2, 3], (
            layer["layer"]
        )
        assert layer["probabilities_reported_mass"] == pytest.approx(
            sum(t["prob"] for t in layer["top_k_tokens"]), abs=1e-6
        )
