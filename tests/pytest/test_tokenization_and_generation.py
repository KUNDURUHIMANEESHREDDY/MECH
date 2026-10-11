"""B1-B12: tokenization, generation and inference.

`generate_text` had **zero** test references in a 1900-test suite, and it is
the function that produces the sample text the UI shows. Three defects, found
by running it:

1. **Greedy decoding is unreachable.** `temperature` was clamped with
   `max(0.1, ...)`, so a caller passing `temperature=0.0` -- the standard way
   to ask for deterministic output -- silently got 0.1 and kept sampling. Two
   identical calls with `temperature=0.0` returned different text. The
   reported `temperature` field said `0.1`, so the substitution was visible only
   if a caller compared it against what it asked for.

2. **No seed is taken.** The sampling path calls `torch.multinomial` with no
   seeding, so a run is reproducible only by luck of process-global RNG state.

3. **The engine's tokenizer cannot pad.** GPT-2 defines no pad token and
   `gpt2_engine.load` does not set one, so `tokenizer([...], padding=True)`
   raises. Batched inference is therefore impossible through the engine's own
   tokenizer -- while `model_manager.load_model` *does* set `pad_token` from
   `eos_token` for the same model. Two loaders, same weights, different
   tokenizer configuration.

`tokens_added` was checked and is correct: it is measured against the stripped
primer the model actually consumed, not the caller's original string, so a
primer with surrounding whitespace still reconciles.

Deliberately not asserted: that generation is reproducible *without* a seed.
That is not a property the code has and should not acquire silently -- the
reproducibility requirement in the checklist is about runs being repeatable
within declared tolerances, which is the seeded case.
"""

from __future__ import annotations

import math

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


# ── B1: tokenization is the known one ──────────────────────────────────

@needs_weights
def test_a_known_string_tokenizes_to_the_expected_ids(engine):
    """Asserted against literal ids, not against a re-encode.

    Comparing the tokenizer's output to another call into the same tokenizer
    proves nothing about whether it is right. The ids below are GPT-2's actual
    vocabulary entries and are the fixture: if the tokenizer's vocabulary or
    normalisation changed, this fails rather than quietly redefining itself.
    """
    tokenizer = engine._tokenizer

    assert tokenizer("The capital of France is")["input_ids"] == [
        464, 3139, 286, 4881, 318
    ], "GPT-2 tokenization of the probe sentence changed"
    assert tokenizer(" Paris")["input_ids"] == [6342]
    assert tokenizer("")["input_ids"] == []


@needs_weights
def test_decode_inverts_encode(engine):
    """A round trip must return the original text.

    Run over the awkward cases as well as the plain one, because whitespace
    and newline handling is where a tokenizer's round trip usually breaks.
    """
    tokenizer = engine._tokenizer

    for text in (
        PROMPT,
        " Paris",
        "  ",
        "\n\t",
        "a b c",
    ):
        ids = tokenizer(text)["input_ids"]
        decoded = tokenizer.decode(ids, skip_special_tokens=True)
        assert decoded == text, (
            f"{text!r} round-tripped to {decoded!r} "
            f"(ids {ids})"
        )


@needs_weights
def test_single_token_words_map_to_one_id_each(engine):
    """The names IOI experiments depend on must be single tokens.

    `capture_head_outputs`' docstring records that this was once assumed and
    silently wrong. A two-token name cannot be used as a single `score_token_id`,
    and an off-by-one in `encode(...)[0]` would select the wrong token while
    still returning a plausible id.
    """
    tokenizer = engine._tokenizer

    for word in (" Paris", " Mary", " John"):
        ids = tokenizer(word)["input_ids"]
        assert len(ids) == 1, (
            f"{word!r} is {len(ids)} tokens {ids}; experiments that pass "
            f"token id {ids[0]} would score the wrong one"
        )


# ── B3/B4: special tokens, empty input ────────────────────────────────

@needs_weights
def test_empty_and_whitespace_prompts_are_handled_explicitly(engine):
    """A blank prompt must produce a defined result, not a traceback.

    The behaviour on empty input is a policy choice; what matters is that it is
    explicit and reported, rather than raising from deep inside the model.
    """
    for blank in ("", "   ", "\n\t"):
        result = engine.generate_text(blank, max_new_tokens=3)
        assert isinstance(result, dict), (
            f"generate_text({blank!r}) returned {type(result).__name__}"
        )
        assert result.get("status") == "ok", (
            f"generate_text({blank!r}) -> {result}"
        )
        # And it must say which primer it actually used, since an empty string
        # is replaced.
        assert result["primer"].strip(), (
            f"generate_text({blank!r}) reported an empty primer; the "
            f"substitution must be visible to the caller"
        )
        assert result["tokens_added"] >= 1


# ── B7: determinism ────────────────────────────────────────────────────

@needs_weights
def test_zero_temperature_is_deterministic(engine):
    """`temperature=0.0` must mean greedy, and greedy must repeat.

    The defect: the clamp was `max(0.1, ...)`, so 0.0 became 0.1 and two
    identical calls returned different text. Greedy decoding was unreachable
    through the public API.

    Asserted on the returned text rather than the reported temperature, because
    the reported field said `0.1` and a test comparing it to the request would
    have passed while generation stayed random.
    """
    a = engine.generate_text(PROMPT, max_new_tokens=8, temperature=0.0)
    b = engine.generate_text(PROMPT, max_new_tokens=8, temperature=0.0)

    assert a["status"] == "ok" and b["status"] == "ok"
    assert a["text"] == b["text"], (
        "temperature=0.0 produced different text on two identical calls:\n"
        f"  {a['text']!r}\n  {b['text']!r}\n"
        f"reported temperature: {a['temperature']}"
    )


@needs_weights
def test_a_zero_temperature_request_is_not_silently_raised(engine):
    """The reported temperature must be what the caller asked for.

    The clamp was what hid the defect: the result said `0.1` while the caller
    had asked for `0.0`. Whatever a valid range turns out to be, the response
    must not describe a different sampling temperature than the one applied.
    """
    result = engine.generate_text(PROMPT, max_new_tokens=4, temperature=0.0)
    assert result["temperature"] == 0.0, (
        f"asked for temperature 0.0, the result reports "
        f"{result['temperature']}. The applied temperature was substituted, "
        f"so the caller cannot tell greedy from sampled."
    )


@needs_weights
def test_sampling_is_reproducible_under_an_explicit_seed(engine):
    """Seeded sampling must repeat exactly.

    The reproducibility requirement is about runs being repeatable within
    declared tolerances. With sampling, the only way to declare a tolerance is
    to seed, and this asserts the seed is actually honoured end to end.
    """
    import torch

    torch.manual_seed(1234)
    a = engine.generate_text(PROMPT, max_new_tokens=8, temperature=0.9)
    torch.manual_seed(1234)
    b = engine.generate_text(PROMPT, max_new_tokens=8, temperature=0.9)

    assert a["text"] == b["text"], (
        f"the same seed produced different text:\n  {a['text']!r}\n  {b['text']!r}"
    )
    assert a["token_ids"] == b["token_ids"]


@needs_weights
def test_greedy_output_is_argmax_at_every_step(engine):
    """Greedy must pick the highest-probability token each step.

    Stronger than "the same text twice": a decoder that is deterministic for
    the wrong reason -- a frozen RNG, a cached result -- would pass the
    determinism test while producing a sequence the model never ranked first.

    Replays the continuation without sampling and requires each chosen token to
    be the argmax of the model's own distribution.
    """
    import torch

    result = engine.generate_text(PROMPT, max_new_tokens=6, temperature=0.0)
    chosen = result["token_ids"]

    model = engine._model
    tokenizer = engine._tokenizer
    prefix = tokenizer(PROMPT, return_tensors="pt")["input_ids"][0].tolist()

    for offset, token_id in enumerate(chosen[len(prefix):], start=1):
        # The context for step N is the primer plus the first N-1 generated
        # tokens, so the slice end is primer-relative: `len(prefix) + N - 1`.
        # An earlier draft used `offset` as an absolute index, which fed the
        # primer alone at every step and compared each token against the wrong
        # distribution -- reporting a greedy path that was in fact correct.
        context = torch.tensor(
            [prefix + chosen[len(prefix):len(prefix) + offset - 1]]
        )
        with torch.no_grad():
            logits = model(input_ids=context).logits[0, -1].float()
        assert int(logits.argmax()) == token_id, (
            f"at step {offset} the model ranked token {int(logits.argmax())} "
            f"highest, but greedy generation chose {token_id}. The sequence "
            f"is not the model's own greedy continuation."
        )


# ── B8: max_new_tokens ─────────────────────────────────────────────────

@needs_weights
@pytest.mark.parametrize("requested", [1, 4, 12, 48])
def test_generation_adds_exactly_the_requested_token_count(engine, requested):
    """`tokens_added` must equal the request.

    The clamp is `max(1, min(48, ...))`, so 0 and 200 are mapped to 1 and 48;
    those bounds are checked separately. What is asserted here is the
    middle of the range, where the count must be honoured exactly.
    """
    result = engine.generate_text(
        PROMPT, max_new_tokens=requested, temperature=0.0
    )
    assert result["tokens_added"] == requested, (
        f"asked for {requested} tokens, added {result['tokens_added']}"
    )
    assert len(result["token_ids"]) - len(
        engine._tokenizer.encode(PROMPT)
    ) == requested


@needs_weights
def test_out_of_range_token_counts_are_clamped_and_reported(engine):
    """Out-of-range requests must be bounded, and the bound reported.

    A clamp is reasonable here; an unreported one is not, because the caller
    cannot tell a short generation from a truncated one.
    """
    for requested, floor in ((0, 1), (-5, 1), (500, 48)):
        result = engine.generate_text(
            PROMPT, max_new_tokens=requested, temperature=0.0
        )
        assert result["tokens_added"] == floor, (
            f"requested {requested} tokens, added {result['tokens_added']}, "
            f"expected the clamp floor {floor}"
        )
        assert result["tokens_added"] <= 48, (
            f"requested {requested} and generated {result['tokens_added']}; "
            f"the upper bound is not enforced"
        )


# ── B12: logits are finite ─────────────────────────────────────────────

@needs_weights
def test_run_prompt_logits_are_finite_and_ranked(engine):
    """Every reported logit must be a real number, and the ranking coherent.

    Finiteness is asserted directly because a NaN passes a negated range check
    (`not 0 <= nan <= 1` is True). The ranking check is separate: a payload of
    five NaNs would satisfy "all finite" under a loose comparison.
    """
    result = engine.run_prompt(PROMPT)
    assert result.get("status") == "ok", result

    top5 = result["top5"]
    assert top5, "no top-5 returned"

    values = [float(e["logit"]) for e in top5]
    for entry, value in zip(top5, values):
        assert math.isfinite(value), (
            f"top-5 logit for {entry.get('token')!r} is {value}"
        )
    assert values == sorted(values, reverse=True), (
        f"top-5 logits are not in descending order: {values}"
    )

    assert result["next_token"] == top5[0]["token"], (
        f"next_token is {result['next_token']!r} but the top-1 is "
        f"{top5[0]['token']!r}"
    )


# ── B3/B9/B10: padding and batching ────────────────────────────────────

@needs_weights
def test_the_tokenizer_can_pad_a_batch(engine):
    """Batched inference needs a pad token; this one has none.

    GPT-2 defines no pad token and `gpt2_engine.load` does not supply one, so
    `tokenizer([...], padding=True)` raises `ValueError`. That makes batched
    inference impossible through this engine's tokenizer --
    `model_manager.load_model` sets `pad_token` from `eos_token` for the same
    weights, so two loaders produce different tokenizer configuration for one
    model.

    GPT-2 uses eos as its pad in practice, so that is what is required here.
    """
    tokenizer = engine._tokenizer

    assert tokenizer.pad_token is not None, (
        "the engine's tokenizer has no pad token, so a padded batch cannot be "
        "tokenized at all"
    )

    short = tokenizer(["Paris"], return_tensors="pt")
    long = tokenizer([PROMPT], return_tensors="pt")
    batch = tokenizer(
        [PROMPT, "Paris"], return_tensors="pt", padding=True
    )

    assert batch["input_ids"].shape[1] == long["input_ids"].shape[1], (
        "padding did not extend the short sequence to the longest"
    )
    assert batch["input_ids"][1][-1].item() == tokenizer.pad_token_id, (
        "the padded tail is not the pad token"
    )
    # And the short row must carry its own mask, so padding is distinguishable
    # from real content.
    assert "attention_mask" in batch
    assert batch["attention_mask"][1][-1].item() == 0, (
        "the padded position is not masked out; a model attending to it reads "
        "pad tokens as content"
    )
    assert batch["attention_mask"][0].sum() == long["input_ids"].shape[1]
    assert batch["attention_mask"][1].sum() == short["input_ids"].shape[1]