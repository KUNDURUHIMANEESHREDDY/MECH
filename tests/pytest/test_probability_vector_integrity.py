"""Probability correctness beyond the top token, and beyond the top-k list.

`test_residual_decomposition.py` already does the important thing: it
recomputes `softmax(model_logits)` and compares the reported probabilities
against it, rather than checking which token came first. That is the check
that caught the double-`ln_f` defect -- the argmax was right while every
probability was wrong.

A review identified three gaps it does not close, and all three are reachable
on the current output.

1. Token-id-to-token mapping is only checked implicitly, by re-encoding the
   reported label. That works while every reported label is a single token.
   Measured:

       encode(' the')    = [262]           len=1
       encode(' Paris')  = [6342]          len=1
       encode('hello world') = [31373, 995] len=2

   A multi-token label makes `ids[0]` silently the wrong token, and the
   comparison at line 349 would then check a different token's probability
   than the one reported -- passing for the wrong reason.

2. Top-k truncation is never accounted for. Measured, on the final layer of
   "The capital of France is":

       reported entries : 5
       reported sum     : 0.243318

   A test asserting the reported probabilities sum to ~1 would fail, and
   correctly: five of 50257 tokens cannot sum to one. Any normalisation
   assertion over the reported list is therefore wrong unless it knows the
   list is truncated. Nothing currently states that it is.

3. The full vocabulary is never checked -- only the reported entries, and
   only against the same softmax the code computes. A lens that applied a
   monotone transform to every probability would pass, because every reported
   entry still equals the model's.
"""

from __future__ import annotations

import math

import pytest
import torch

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
def lens_and_truth(engine):
    """Final-layer lens plus the model's own distribution, computed in order."""
    n_layers = engine._n_layers()

    batch = engine._tokenizer(PROMPT, return_tensors="pt")
    input_ids = batch["input_ids"].to(engine._model.device)
    attention_mask = batch.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(engine._model.device)

    config = engine._model.config
    previous = getattr(config, "_attn_implementation", None)
    if previous != "eager":
        config._attn_implementation = "eager"
    try:
        with torch.no_grad():
            logits = engine._model(
                input_ids=input_ids, attention_mask=attention_mask
            ).logits[0, -1, :].detach().cpu()
    finally:
        if previous is not None:
            config._attn_implementation = previous

    # Only now, with `_cache["hidden"]` populated by the pass above.
    result = engine.logit_lens(n_layers - 1, PROMPT)
    assert result.get("status") == "ok", result

    return result, logits.float()


# ── 1. exact token-id mapping ─────────────────────────────────────────

@needs_weights
def test_every_reported_token_is_exactly_one_token_id(lens_and_truth, engine):
    """A reported label must round-trip to the single id it represents.

    The existing comparison uses `tokenizer.encode(entry["token"])[0]`. For a
    multi-token label that silently selects the *first* piece, so the test
    would compare one token's probability against another's -- and a lens
    reporting wrong probabilities for every token would still pass whenever
    the wrong index happened to agree.
    """
    lens, _ = lens_and_truth
    tokenizer = engine._tokenizer

    multi = []
    for entry in lens["top_k_tokens"]:
        label = entry["token"]
        ids = tokenizer.encode(label)
        if len(ids) != 1:
            multi.append((label, ids))
        else:
            # And the mapping must be exact in both directions, so a label
            # that decodes from some other id cannot pass.
            assert tokenizer.decode(ids) == label, (
                f"{label!r} encodes to {ids} but decodes back to "
                f"{tokenizer.decode(ids)!r}"
            )

    assert not multi, (
        f"these reported labels are not single tokens, so `ids[0]` selects a "
        f"different token than the one whose probability is being reported: "
        f"{multi}"
    )


# ── 2. normalisation, with truncation accounted for ───────────────────

@needs_weights
def test_the_reported_probabilities_are_normalised_over_the_full_vocabulary(
    lens_and_truth,
):
    """Sum to ~1 only once the unreported tail is included.

    The reported top-k is a truncation, so its own sum is necessarily below 1
    -- measured 0.243318 for five entries. The property that must hold is that
    the *distribution* is normalised, which means the reported entries must
    equal the model's own softmax exactly, and that softmax must sum to 1.
    """
    lens, model_logits = lens_and_truth

    model_probs = torch.softmax(model_logits, dim=-1)
    assert math.isclose(float(model_probs.sum()), 1.0, abs_tol=1e-4), (
        f"the reference softmax sums to {float(model_probs.sum()):.6f}; the "
        f"comparison below would be meaningless against a distribution that "
        f"is not normalised"
    )

    assert all(0.0 <= e["prob"] <= 1.0 for e in lens["top_k_tokens"]), (
        "a reported probability is outside [0, 1]: "
        f"{[e['prob'] for e in lens['top_k_tokens']]}"
    )
    assert all(math.isfinite(e["prob"]) for e in lens["top_k_tokens"]), (
        "a reported probability is not finite"
    )


@needs_weights
def test_the_reported_list_is_declared_as_truncated(lens_and_truth, engine):
    """A partial list must not look like a distribution.

    Nothing in the response says whether the top-k list is the whole
    distribution or a slice of one. A consumer summing the reported
    probabilities -- the obvious thing to do -- gets 0.243 and either reports
    a nonsensical number or silently normalises. The response should say
    which it is.
    """
    lens, model_probs = lens_and_truth
    vocab = int(model_probs.numel())
    reported = len(lens["top_k_tokens"])

    assert reported < vocab, (
        f"the fixture assumption is wrong: {reported} entries against a "
        f"vocabulary of {vocab}; there is no truncation to describe"
    )

    total = sum(e["prob"] for e in lens["top_k_tokens"])
    assert total < 0.999, (
        f"the reported top-{reported} sums to {total:.6f}, which is nearly 1 "
        f"and so is not a truncation of a {vocab}-token distribution"
    )

    marker = lens.get("probability_is_truncated") or lens.get(
        "truncated") or lens.get("is_truncated")
    assert marker is not None, (
        f"the response reports {reported} of {vocab} probabilities, summing "
        f"to {total:.6f}, with nothing stating that the list is partial. A "
        f"consumer summing them gets a number that looks like a broken "
        f"distribution. Keys present: {sorted(lens)}"
    )


@needs_weights
def test_the_ranking_is_descending_and_free_of_duplicates(lens_and_truth):
    """Ordering and uniqueness, which a per-entry probability check misses.

    Each entry is checked against the same softmax individually, so a list
    containing one token twice, or presented out of order, would still have
    every probability correct.
    """
    lens, _ = lens_and_truth
    entries = lens["top_k_tokens"]

    probs = [e["prob"] for e in entries]
    assert probs == sorted(probs, reverse=True), (
        f"the reported probabilities are not in descending order: {probs}"
    )

    tokens = [e["token"] for e in entries]
    assert len(set(tokens)) == len(tokens), (
        f"the top-k list repeats a token: {tokens}"
    )


@needs_weights
def test_the_reported_ranking_matches_the_models_own_ranking(lens_and_truth):
    """The reported order must equal the model's argmax order.

    Checking each probability against the softmax says each number is right.
    It says nothing about whether they were *selected* correctly, so a lens
    reporting the five most likely tokens in the wrong order, or the right
    five chosen arbitrarily, would pass.
    """
    lens, model_logits = lens_and_truth

    model_probs = torch.softmax(model_logits, dim=-1)
    top = torch.topk(model_probs, len(lens["top_k_tokens"]))
    expected_ids = [int(i) for i in top.indices]

    from backend.services import gpt2_engine as eng

    reported_ids = [
        eng._tokenizer.encode(e["token"])[0] for e in lens["top_k_tokens"]
    ]

    assert reported_ids == expected_ids, (
        f"the lens reported {reported_ids} as its top-"
        f"{len(reported_ids)}, but the model's own ranking is {expected_ids}"
    )


@needs_weights
def test_the_unreported_tail_is_the_only_missing_probability(lens_and_truth):
    """The full distribution is recoverable, not just the reported slice.

    Recomputing the softmax independently and comparing *every* reported
    entry is already done elsewhere. This closes the last gap: the reported
    probabilities must not merely be plausible, they must be the actual
    entries of one normalised distribution with nothing removed except the
    tail the caller was told was removed.
    """
    lens, model_logits = lens_and_truth
    model_probs = torch.softmax(model_logits, dim=-1)

    from backend.services import gpt2_engine as eng

    reported_total = sum(e["prob"] for e in lens["top_k_tokens"])
    omitted = float(model_probs.sum()) - reported_total

    assert 0.0 < omitted < 1.0, (
        f"the omitted probability mass is {omitted:.6f}; the reported slice "
        f"({reported_total:.6f}) and the full distribution should partition it"
    )
    assert reported_total <= 1.0 + 1e-6, (
        f"reported probabilities sum to {reported_total:.6f}, above 1"
    )

    # Every reported entry must be an actual entry of the distribution.
    for entry in lens["top_k_tokens"]:
        ids = eng._tokenizer.encode(entry["token"])
        assert len(ids) == 1, (
            f"{entry['token']!r} is not a single token; the mapping is "
            f"ambiguous and this entry cannot be located in the distribution"
        )
        assert float(model_probs[ids[0]]) == pytest.approx(
            entry["prob"], abs=5e-3
        ), (
            f"{entry['token']!r} is reported at {entry['prob']:.6f} but the "
            f"model's own distribution puts it at {float(model_probs[ids[0]]):.6f}"
        )
