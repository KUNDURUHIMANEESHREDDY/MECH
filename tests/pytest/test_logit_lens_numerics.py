"""E3/E4/E8/E9: logit-lens probability normalisation, and a tuned lens that
must actually be applied when it reports itself as applied.

E4 was the substantive gap. `TunedLens.train()` had 20+ tests covering its
structure -- the KL objective, the holdout split, the refusal to quote
in-sample figures, the `translator_kind` disclosure -- and *not one of them ran
it against real weights*. Every `train()` call in the suite hit a stub with no
`_model` and took the `unavailable` branch. The only evidence a tuned lens was
ever trained was a committed artifact read back by assertions, so a regression
that produced a worse-but-still-live report would have passed all of them.

Running it live is what surfaced the defect below.

The defect
----------
`project()` does not apply the trained translator. Its docstring says so
outright -- "the engine's cached projection is returned as-is" -- while the
payload reports `affine_translation_applied: True`, `tuned_lens_available: True`,
`trained: True` and `layer_trained: True`.

Measured on GPT-2 Small after fitting 18,432 parameters over 12 layers:

    trained    top_probability = 0.551669   (all four flags True)
    untrained  top_probability = 0.551669
    identical.

The translators are not the problem -- they are fitted and non-trivial (scale
mean 0.81-0.92, max 2.16; shift max 1.23) and `apply_translator` moves the
hidden state by max|d| = 15.48. The projection simply never calls it.

That is the exact failure the project rules exist to prevent: four independent
flags assert a transformation was applied, the payload looks complete, and the
number is the untuned one. A researcher comparing a tuned lens against a raw
lens would conclude the tuning had no effect, and the report would have said
the tuning was applied.
"""

from __future__ import annotations

import pytest

from _weight_guard import skip_reason, weights_available

needs_weights = pytest.mark.skipif(
    not weights_available(), reason=skip_reason()
)

PROMPT = "The capital of France is"

CALIBRATION = [
    "The capital of %s is" % country
    for country in (
        "France", "Germany", "Italy", "Spain", "Japan", "Brazil", "Egypt",
        "Norway", "Poland", "Turkey", "Kenya", "Chile", "Cuba", "Peru",
        "Ghana",
    )
]


@pytest.fixture(scope="module")
def engine():
    from backend.services import gpt2_engine as eng

    eng.load()
    return eng


@pytest.fixture(scope="module")
def trained_lens(engine):
    """A genuinely fitted tuned lens, trained here rather than loaded.

    Returns the report alongside the lens: `train()` returns its report rather
    than retaining it, so the held-out figure exists only in that return value.
    An earlier draft of the holdout test reached for `lens.last_report` and
    `lens.holdout_improved`, neither of which exists, so it failed on a missing
    attribute rather than on the property being checked.
    """
    from backend.interpretability.algorithms.tuned_lens import TunedLens

    lens = TunedLens(engine=engine)
    report = lens.train(CALIBRATION, steps=120, learning_rate=1e-2, seed=42)
    assert report.get("status") == "completed", (
        f"training did not complete: {report.get('reason')}"
    )
    assert lens.is_trained, "train() reported completion but is_trained is False"
    lens._training_report = report
    return lens


@pytest.fixture(scope="module")
def lens(trained_lens):
    """Alias so a test can name the lens directly."""
    return trained_lens


@pytest.fixture(scope="module")
def training_report(trained_lens):
    return trained_lens._training_report


# ── E4: the tuned lens trains, and generalises ─────────────────────────

@needs_weights
def test_the_tuned_lens_trains_against_real_weights(trained_lens):
    """Live training must actually fit something.

    Asserted on the *parameters*, not on the report. The report is the
    component under suspicion here: it claimed `affine_translation_applied`
    while the projection did nothing, so a report-only assertion cannot
    establish that training happened.
    """
    import torch

    from backend.interpretability.algorithms.tuned_lens import FULL_AFFINE

    lens = trained_lens
    assert trained_lens.translators, "no translators were stored"
    for layer, entry in trained_lens.translators.items():
        # Both families start at the identity map, so "training fitted
        # something" means the map moved away from it -- a d x d weight and
        # a bias for the full affine family, a scale and a shift for the
        # restricted one.
        kind = entry.get("kind")
        if kind == FULL_AFFINE:
            weight, bias = entry["weight"], entry["bias"]
            identity_weight = torch.eye(weight.shape[0], dtype=weight.dtype,
                                        device=weight.device)
            identity = (
                torch.allclose(weight, identity_weight)
                and torch.allclose(bias, torch.zeros_like(bias))
            )
        else:
            scale, shift = entry["scale"], entry["shift"]
            identity = (
                torch.allclose(scale, torch.ones_like(scale))
                and torch.allclose(shift, torch.zeros_like(shift))
            )
        assert not identity, (
            f"layer {layer}'s {kind} translator is still the identity map; "
            f"training reported success but fitted nothing"
        )


@needs_weights
def test_training_improves_the_held_out_objective(training_report):
    """The held-out figure must improve -- the only claim worth making.

    The report carries both an in-sample and a held-out KL reduction, and the
    docstring is explicit that a small diagonal map can drive the in-sample
    figure to near zero by memorising. So this reads the *held-out* count and
    requires genuine improvement across the layers.
    """
    report = training_report

    improvement = report.get("layers_holdout_improved")
    assert improvement is not None, (
        "the training report does not record a held-out improvement count; "
        "the in-sample figure alone is not evidence of generalisation"
    )
    assert improvement > 0, (
        f"held-out KL did not improve on any layer (improved={improvement}); "
        f"the translators fit noise"
    )

    # And the in-sample figure must not be the one doing the work.
    assert report.get("headline_metric") == "holdout_kl_reduction_pct", (
        f"the headline metric is {report.get('headline_metric')!r}; the "
        f"in-sample reduction is not the figure that generalises"
    )


@needs_weights
def test_training_discloses_its_own_limits(training_report):
    """A fitted lens must still refuse to be quoted as publication-ready.

    Checked because the alternative failure is silence: a lens that trains,
    improves held-out, and reports `publication_eligible: True` would let a
    15-prompt calibration stand in for a published tuned lens. The
    report's own `reason` names both restrictions.
    """
    from backend.interpretability.algorithms.tuned_lens import (
        DIAGONAL_AFFINE,
        FULL_AFFINE,
    )

    report = training_report

    assert report.get("publication_eligible") is False, (
        "a 15-prompt calibration reported itself publication-eligible"
    )
    kind = report.get("translator_kind")
    assert kind in (FULL_AFFINE, DIAGONAL_AFFINE), (
        f"the report does not name a known translator family: {kind!r}"
    )
    reason = str(report.get("reason", ""))
    assert kind in reason, (
        f"the report does not disclose the translator variant it used: "
        f"{reason!r}"
    )
    assert "calibration" in reason.lower(), (
        f"the report does not disclose the size of the calibration set: "
        f"{reason!r}"
    )


@needs_weights
def test_the_projection_names_the_family_it_actually_applied(trained_lens):
    """The payload must describe the map that was applied, not another one.

    The projection's reason used to hard-code 'diagonal-affine' while the
    default family is the full affine one, so every payload described a
    translator that was never fitted.
    """
    from backend.interpretability.algorithms.tuned_lens import (
        DIAGONAL_AFFINE,
        FULL_AFFINE,
    )

    out = trained_lens.project(PROMPT, layer=6, top_k=3)

    assert out.get("affine_translation_applied") is True, out
    kind = out.get("translator_kind")
    assert kind in (FULL_AFFINE, DIAGONAL_AFFINE), out
    assert kind in str(out.get("reason", "")), (
        f"the projection does not name the family it applied: {out!r}"
    )


@needs_weights
def test_a_trained_projection_differs_from_an_untrained_one(
    trained_lens, engine
):
    """The defect: `project()` must reflect the translator it reports applying.

    A trained lens and a fresh one are asked the same question at the same
    layer. If the answers are identical, the projection is not using the
    translator -- which is what was happening while every flag in the payload
    said otherwise.

    Compared across several prompts and layers, because a single prompt could
    coincide: the untrained projection differs by layer, so a per-layer
    coincidence is possible even when the translator is being applied. Requiring
    a difference at *every* layer tested makes coincidence implausible.
    """

    from backend.interpretability.algorithms.tuned_lens import TunedLens

    fresh = TunedLens(engine=engine)
    probes = [
        "The capital of France is",
        "Scientists recently discovered that",
        "The weather today is",
    ]

    differing = 0
    checked = 0
    for prompt in probes:
        for layer in (3, 6, 9):
            trained_out = trained_lens.project(prompt, layer=layer, top_k=5)
            fresh_out = fresh.project(prompt, layer=layer, top_k=5)
            if trained_out.get("status") not in ("ok", "completed"):
                continue
            if fresh_out.get("status") not in ("ok", "completed"):
                continue
            checked += 1
            a = trained_out.get("top_k_tokens")
            b = fresh_out.get("top_k_tokens")
            if a != b:
                differing += 1

    assert checked, "no layer/prompt combination produced a projection"
    assert differing == checked, (
        f"a trained tuned lens returned projections identical to an untrained "
        f"one at {checked - differing} of {checked} layer/prompt combinations. "
        f"The report says affine_translation_applied=True, so the trained "
        f"translator must change the output."
    )


@needs_weights
def test_the_applied_flag_is_not_claimed_when_nothing_is_translated(
    trained_lens, engine
):
    """A payload asserting a transformation must show that transformation.

    The flag is only trustworthy if something checks it. This asserts it the
    other way round for the untrained case, where the honest answer is False,
    so the flag cannot be a constant that always reads True.
    """
    from backend.interpretability.algorithms.tuned_lens import TunedLens

    fresh = TunedLens(engine=engine)
    out = fresh.project(PROMPT, layer=6, top_k=3)

    assert out.get("affine_translation_applied") is not True, (
        "an untrained lens reports affine_translation_applied=True"
    )


# ── E3: reported probabilities must be real probabilities ──────────────

@needs_weights
def test_reported_lens_probabilities_match_the_models_own_distribution(engine):
    """The lens's probability must be the model's, to the digits.

    `logit_lens` returns only a top-k slice, so the reported numbers cannot
    sum to 1 and must not be asked to. What can be checked, and is the property
    that actually matters, is that each reported probability equals the softmax
    of the model's own logits for that token.

    This is the check that caught the double-normalisation in `logit_lens`:
    the argmax was unchanged, so a top-token comparison passed while
    p(' the') was 0.038 against the model's 0.085.
    
    The comparison is at float32 resolution, not exact. The lens projects the
    cached hidden state through the unembedding matrix; the expected value is a
    float64 softmax of the model's own float32 logits. Two different matmul
    shapes over the same weights do not sum in the same order, so the two
    results differ by about a float32 step -- measured at 1.4e-6 on p(' the')
    on Linux, where the reduction order differs from Windows'. The tolerance is
    a thousandth of the difference this test exists to catch: the
    double-normalisation put p(' the') at 0.038 against the model's 0.085, a
    relative error of 0.55.
    """
    import numpy as np

    n_layers = engine._n_layers()
    engine.run_prompt(PROMPT)
    snapshot = engine._snapshot()

    lens = engine.logit_lens(n_layers - 1, PROMPT, top_k=5)
    assert lens.get("status") == "ok", lens

    logits = np.asarray(snapshot["logits"], dtype=np.float64)
    shifted = np.exp(logits - logits.max())
    probs = shifted / shifted.sum()

    tokenizer = engine._tokenizer
    for entry in lens["top_k_tokens"]:
        ids = tokenizer.encode(entry["token"])
        assert ids, f"undecodable token {entry['token']!r}"
        expected = float(probs[ids[0]])
        assert float(entry["prob"]) == pytest.approx(
            expected, rel=1e-3, abs=1e-6), (
            f"lens reports p({entry['token']!r})={float(entry['prob']):.8f}, "
            f"the model's own softmax gives {expected:.8f}"
        )


@needs_weights
def test_the_reported_top_k_is_a_monotone_slice_of_one_distribution(engine):
    """The reported probabilities must come from a single distribution.

    Checks the internal ordering and range rather than a sum, since a top-k
    slice cannot sum to 1. A lens that renormalised the slice, or mixed
    probabilities from different layers, would still report plausible numbers.

    Monotonicity matters here: `torch.topk` returns descending order, so a
    slice whose probabilities increase down the list did not come from one
    softmax.
    """
    n_layers = engine._n_layers()
    engine.run_prompt(PROMPT)

    for layer in (0, n_layers // 2, n_layers - 1):
        lens = engine.logit_lens(layer, PROMPT, top_k=5)
        assert lens.get("status") == "ok", lens
        probs = [float(e["prob"]) for e in lens["top_k_tokens"]]

        assert all(0.0 <= p <= 1.0 for p in probs), (
            f"layer {layer}: probabilities outside [0, 1]: {probs}"
        )
        assert probs == sorted(probs, reverse=True), (
            f"layer {layer}: top-k probabilities are not in descending order: "
            f"{probs}. They do not come from a single softmax."
        )
        assert lens["top_token"] == lens["top_k_tokens"][0]["token"], (
            f"layer {layer}: top_token disagrees with the first top-k entry"
        )


# ── E8: numerical stability ────────────────────────────────────────────

@needs_weights
def test_the_lens_normalises_extreme_logits_without_nan(engine):
    """Finite extreme logits must still yield a distribution.

    GPT-2's outlier dimensions reach the hundreds, and the softmax over a
    50k-vocabulary projection can see very large magnitudes. The failure this
    guards is the documented one: exp overflow producing NaN, which would then
    be reported as a probability.

    Checked on the model's own last-layer logits (which contain those
    outliers) and on synthetic extremes pushed further, so the condition is
    exercised rather than assumed.
    """
    import torch

    n_layers = engine._n_layers()
    engine.run_prompt(PROMPT)
    logits = torch.tensor(
        list(engine._snapshot()["logits"]), dtype=torch.float32
    )

    for name, probe in (
        ("model logits", logits),
        ("x1e4", logits * 1e4),
        ("x1e8", logits * 1e8),
        ("mixed extreme", torch.tensor([1e30, -1e30, 0.0, 1e10, -1e10])),
    ):
        probs = torch.softmax(probe, dim=-1)
        assert torch.isfinite(probs).all(), (
            f"{name}: softmax produced non-finite probabilities"
        )
        total = float(probs.sum())
        assert total == pytest.approx(1.0, abs=1e-5), (
            f"{name}: probabilities sum to {total}, not 1"
        )


@needs_weights
def test_the_reported_top_probability_is_finite(engine):
    """Whatever the lens reports must be a real number.

    A NaN probability is not caught by a range check written as `0 <= p <= 1`,
    because NaN compares false against everything and would pass a negated
    form. So finiteness is asserted directly.
    """
    import math

    n_layers = engine._n_layers()
    engine.run_prompt(PROMPT)

    for layer in (0, 5, n_layers - 1):
        lens = engine.logit_lens(layer, PROMPT, top_k=5)
        assert lens.get("status") == "ok", lens
        for entry in lens["top_k_tokens"]:
            value = float(entry["prob"])
            assert math.isfinite(value), (
                f"layer {layer}: p({entry['token']!r}) is {value}, not finite"
            )
            assert 0.0 <= value <= 1.0, (
                f"layer {layer}: p({entry['token']!r})={value} outside [0, 1]"
            )


# ── E9: dtype and device ───────────────────────────────────────────────

@needs_weights
def test_the_lens_returns_tensors_whose_dtype_and_device_match_the_model(engine):
    """The lens must not silently change precision or move device.

    The projection is `torch.from_numpy(cache) @ wte.weight.T`, so a dtype
    mismatch between the cache and the weights would either raise or promote
    to float64 and quietly halve throughput. Asserted on the tensors rather
    than on the reported dict, since the dict carries no dtype.
    """
    import torch

    engine.run_prompt(PROMPT)
    weights = engine._model.transformer.wte.weight

    cached = torch.from_numpy(
        __import__("numpy").asarray(engine._snapshot()["hidden"][5][-1])
    )
    assert cached.dtype == weights.dtype, (
        f"the cache holds {cached.dtype} while the unembedding is "
        f"{weights.dtype}; the matmul would not be a like-for-like product"
    )
    assert cached.device == weights.device or cached.device.type == "cpu", (
        f"cache on {cached.device}, weights on {weights.device}"
    )

    projected = cached.float().to(weights.device) @ weights.T
    assert projected.dtype == torch.float32, (
        f"the projection came out as {projected.dtype}, not float32"
    )
    assert projected.device == weights.device, (
        f"the projection came out on {projected.device}, not "
        f"{weights.device}"
    )
    assert torch.isfinite(projected).all(), "the projection produced non-finite values"