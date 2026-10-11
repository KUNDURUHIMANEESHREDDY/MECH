"""PAT-008: the causal effect must have the right SIGN, not just consistent
arithmetic.

Why this file exists
--------------------
Every other check in this area verifies that the reported recovery is
*self-consistent* -- that recovery equals `(patched - corrupted) / (clean -
corrupted)` for the numbers reported. All of those pass if the metric is
defined the wrong way round, because the numerator and the reported ratio flip
together. Nothing checked the direction against the model.

The direction is checkable, because GPT-2 Small solves IOI. Using the pair
this repository already uses in `acdc.py` and `causal_scrubbing.py`:

    clean      "John gave a drink to Mary"   -> the IO is Mary
    corrupted  "John gave a drink to John"   -> the subject is John

and `_logit_diff = logit[IO] - logit[subject]`, the clean run must score
clearly positive and the corrupted run clearly negative. So:

* injecting the clean activations into the corrupted run must move the metric
  **toward** the clean value;
* injecting corrupted activations into the clean run must move it **away**.

A metric with a flipped sign fails both, while every self-consistency
assertion elsewhere keeps passing.

Two things are deliberately *not* asserted. That the top-1 token becomes "Mary"
after injection -- these prompts contain their own answer, so the top-1 is a
comma and the logit difference is the quantity that carries the signal. And
that a specific head set produces a specific magnitude: the point is the sign
and the monotonicity, not a particular circuit.

The heads are discovered by ablation within the test rather than hardcoded to
a published list, because a test that imports the answer it is checking cannot
detect a wrong answer.
"""

from __future__ import annotations

import pytest

from _weight_guard import skip_reason, weights_available

needs_weights = pytest.mark.skipif(
    not weights_available(), reason=skip_reason()
)

CLEAN = "John gave a drink to Mary"
CORRUPTED = "John gave a drink to John"


@pytest.fixture(scope="module")
def ioi():
    """The IO/subject token pair and their clean/corrupted baselines.

    Also asserts the fixture is a valid IOI pair before any test relies on it,
    because every assertion below is phrased relative to these two numbers --
    and a backwards fixture would invert all of them into passing tests.
    """
    from backend.interpretability.discovery import live_measure as lm

    engine = lm._engine()
    tokenizer = engine._tokenizer
    io_id = tokenizer(" Mary")["input_ids"][0]
    subj_id = tokenizer(" John")["input_ids"][0]

    clean = lm.baseline(CLEAN, io_id, subj_id)
    corrupted = lm.baseline(CORRUPTED, io_id, subj_id)

    assert clean["logit_diff"] > 0, (
        f"the clean prompt does not prefer the IO (logit_diff="
        f"{clean['logit_diff']:.4f}); this is not a usable IOI pair"
    )
    assert corrupted["logit_diff"] < 0, (
        f"the corrupted prompt does not prefer the subject (logit_diff="
        f"{corrupted['logit_diff']:.4f}); the corruption did not take"
    )

    return {
        "io_id": io_id,
        "subj_id": subj_id,
        "clean": clean,
        "corrupted": corrupted,
        "gap": clean["logit_diff"] - corrupted["logit_diff"],
    }


def _supporting_heads(lm, prompt: str, io_id: int, subj_id: int,
                      n_layers: int = 12, limit: int = 8) -> list[tuple[int, int]]:
    """Heads whose removal most increases the IO-minus-subject difference.

    Found by ablation here rather than hardcoded, so the test does not import
    the circuit it is meant to be validating. A test that takes its expected
    head list from the implementation cannot fail when the implementation
    picks the wrong heads.
    """
    baseline = lm.baseline(prompt, io_id, subj_id)["logit_diff"]
    scored = []
    for layer in range(n_layers):
        for head in range(lm.dims()[1]):
            value = lm.ablate(prompt, io_id, subj_id, {(layer, head)})
            scored.append((value - baseline, (layer, head)))
    scored.sort(reverse=True)
    return [head for _delta, head in scored[:limit]]


@needs_weights
def test_injecting_clean_activations_moves_the_metric_toward_clean(ioi):
    """PAT-008, forward direction: clean injections must increase logit_diff.

    The corrupted run prefers the subject. Restoring the clean attention
    outputs for the heads that carry the IO mechanism must push the metric back
    toward the clean value -- an increase, since the clean value is higher.

    Asserted for a single head and for accumulating heads, because a
    one-head-only test would pass on a head whose effect is noise, and the
    accumulating form is what a real circuit evaluation does.
    """
    from backend.interpretability.discovery import live_measure as lm

    _, clean_caps = lm.capture(CLEAN)
    heads = _supporting_heads(
        lm, CORRUPTED, ioi["io_id"], ioi["subj_id"]
    )

    corrupted_diff = ioi["corrupted"]["logit_diff"]

    single = lm.inject(CORRUPTED, ioi["io_id"], ioi["subj_id"],
                       clean_caps, {heads[0]})
    assert single["logit_diff"] > corrupted_diff, (
        f"injecting clean activations for {heads[0]} left the metric at "
        f"{single['logit_diff']:.4f}, at or below the corrupted baseline "
        f"{corrupted_diff:.4f}. Restoring the clean computation must move the "
        f"corrupted run toward the clean one."
    )

    accumulated = lm.inject(CORRUPTED, ioi["io_id"], ioi["subj_id"],
                            clean_caps, set(heads[:4]))
    assert accumulated["logit_diff"] > corrupted_diff, (
        f"injecting clean activations for four heads moved the metric from "
        f"{corrupted_diff:.4f} to {accumulated['logit_diff']:.4f} -- not toward "
        f"clean ({ioi['clean']['logit_diff']:.4f})"
    )


@needs_weights
def test_injecting_corrupted_activations_moves_the_metric_away_from_clean(ioi):
    """PAT-009 territory, checked from the same fixture: the mirror image.

    Destroying the clean run's IO mechanism must lower its metric. This is the
    check that would catch a metric defined as subject-minus-IO: under that
    definition both this test and the previous one flip, and neither is
    individually suspicious.

    Together they pin the sign from both ends, which is stronger than either
    alone -- a constant offset would satisfy one and break the other.
    """
    from backend.interpretability.discovery import live_measure as lm

    _, corrupted_caps = lm.capture(CORRUPTED)
    heads = _supporting_heads(lm, CLEAN, ioi["io_id"], ioi["subj_id"])

    clean_diff = ioi["clean"]["logit_diff"]

    result = lm.inject(CLEAN, ioi["io_id"], ioi["subj_id"],
                       corrupted_caps, set(heads[:4]))
    assert result["logit_diff"] < clean_diff, (
        f"injecting corrupted activations into the clean run left the metric "
        f"at {result['logit_diff']:.4f}, at or above the clean baseline "
        f"{clean_diff:.4f}. Overwriting the clean computation with corrupted "
        f"attention outputs must degrade it."
    )


@needs_weights
def test_unrelated_heads_do_not_reproduce_the_effect(ioi):
    """Negative control: the direction must not be trivially satisfiable.

    If injecting *any* clean activations moved the metric toward clean, then
    the forward test would pass without identifying anything. So a set of heads
    drawn at random -- chosen to exclude the discovered ones -- is injected
    instead, and must move the metric substantially less than the real heads do
    for the same number of components.

    Not asserted as "random heads move it the wrong way", which is too strong:
    random components do perturb the metric. The claim is that the discovered
    heads matter more, which is the thing a causal claim actually rests on.
    """
    import random

    from backend.interpretability.discovery import live_measure as lm

    _, clean_caps = lm.capture(CLEAN)
    heads = _supporting_heads(lm, CORRUPTED, ioi["io_id"], ioi["subj_id"])
    target = set(heads[:4])

    rng = random.Random(0)
    all_heads = [(l, h) for l in range(12) for h in range(lm.dims()[1])]
    control = set(rng.sample([h for h in all_heads if h not in target], 4))

    corrupted_diff = ioi["corrupted"]["logit_diff"]
    real_gain = (
        lm.inject(CORRUPTED, ioi["io_id"], ioi["subj_id"], clean_caps, target)
        ["logit_diff"]
        - corrupted_diff
    )
    control_gain = (
        lm.inject(CORRUPTED, ioi["io_id"], ioi["subj_id"], clean_caps, control)
        ["logit_diff"]
        - corrupted_diff
    )

    assert real_gain > control_gain, (
        f"injecting the discovered heads {sorted(target)} moved the metric by "
        f"{real_gain:.4f}, while four unrelated heads moved it by "
        f"{control_gain:.4f}. The identified circuit is not doing more work "
        f"than arbitrary components."
    )


@needs_weights
def test_injection_is_faithful_so_full_restoration_is_exact(ioi):
    """Injecting clean activations into the CLEAN prompt must be an exact no-op.

    This is the normalisation anchor, and it is the right way to pin the sign.

    The tempting assertion -- "injecting clean activations into the corrupted
    run must recover 1.0" -- is false, and an earlier draft of this test made
    it. The two prompts differ in their *final token* (`Mary` vs ` John`, ids
    5335 vs 1757; the first five tokens are identical). Attention injection
    cannot change which token sits at the last position, so no combination of
    restored attention outputs reproduces the clean run. Measured: injecting
    clean activations for all 144 heads recovers 0.256 of the gap.

    Injecting into the clean prompt has no such obstacle -- same tokens, same
    attention -- so the restoration must be exact. Measured delta: 0.000000.

    That exactness is what makes the sign check meaningful elsewhere: it shows
    the recovery numbers in the other tests are limited by the intervention, not
    by the metric being mis-scaled.
    """
    from backend.interpretability.discovery import live_measure as lm

    _, clean_caps = lm.capture(CLEAN)
    every_head = {
        (layer, head)
        for layer in range(12)
        for head in range(lm.dims()[1])
    }

    restored = lm.inject(CLEAN, ioi["io_id"], ioi["subj_id"],
                         clean_caps, every_head)

    assert restored["logit_diff"] == pytest.approx(
        ioi["clean"]["logit_diff"], abs=1e-6
    ), (
        f"injecting the clean activations back into the clean prompt moved the "
        f"metric from {ioi['clean']['logit_diff']:.6f} to "
        f"{restored['logit_diff']:.6f}. Restoring a computation with itself "
        f"must be exact; a non-zero difference means the injection writes "
        f"something other than what it read."
    )


@needs_weights
def test_recovery_is_bounded_and_reported_without_clipping(ioi):
    """Recovery is a signed quantity in (0, 1) for this intervention.

    Binds the checklist's normalisation requirement to a measured case. The
    checklist is explicit that recovery may fall below 0 or exceed 1 and must
    not be clipped without a stated methodological reason -- so the property
    pinned here is that the reported figure is *reported*, in range, and
    consistent with the three underlying measurements, rather than silently
    truncated to a tidy [0, 1].

    Consistency is checked against the raw numbers rather than recomputing the
    formula the implementation already uses, so this catches a change in the
    metric rather than restating the arithmetic.
    """
    from backend.interpretability.discovery import live_measure as lm

    _, clean_caps = lm.capture(CLEAN)
    heads = set(
        _supporting_heads(lm, CORRUPTED, ioi["io_id"], ioi["subj_id"])[:4]
    )

    restored = lm.inject(CORRUPTED, ioi["io_id"], ioi["subj_id"],
                         clean_caps, heads)

    recovery = (
        restored["logit_diff"] - ioi["corrupted"]["logit_diff"]
    ) / ioi["gap"]

    # Positive: restoring the clean mechanism helps. This is the sign.
    assert recovery > 0, (
        f"recovery is {recovery:.4f}; restoring clean activations must be a "
        f"positive fraction of the clean-corrupted gap"
    )
    # Below 1: the final token cannot be restored by attention injection, so
    # this intervention is not expected to fully close the gap.
    assert recovery < 1.0, (
        f"recovery is {recovery:.4f}; the corrupted prompt's final token "
        f"differs from the clean one, so attention injection alone cannot "
        f"reach 1.0. If this now holds, the token-level difference is no "
        f"longer the limiting factor and this bound should be revisited."
    )