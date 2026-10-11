"""G gap 4: `attribution_patching` ranks components without a direction.

The six existing tests in `test_discovery_no_fabrication.py` cover the search
space, fabricated activations, the structural closing edge, score variance,
the gradient-vs-ratio substitution, and prompt shape. All six are about the
*report's honesty*. None asks whether a high score means the component does
what the report implies.

The gap: `attribution_patching.py:345` computes

    attr_score = sum(abs(d * g) for d, g in zip(delta_x, grad))

The absolute value makes the score **direction-blind**. A head that pushes the
metric toward the corrupted answer and one that pushes it toward the clean
answer score identically, provided the magnitudes match. The report ranks by
that score and presents the result as an attribution ranking.

Measured on GPT-2 Small, canonical IOI pair
(`clean = "When John and Mary went to the store, John gave a drink to"`,
`corrupted = "... Mary gave a drink to John"`), metric
`logit(' Mary') - logit(' John')` at the final position:

    clean metric = +2.5000    corrupted metric = -2.1875

Patching each top-attributed head with corrupted activations, into the clean
run. A negative delta means the head carries IOI information -- corrupting it
moves the clean run toward the corrupted answer:

    Head_L8_H10   score=1.9428  patch_delta=-3.50000  carries IOI
    Head_L8_H3    score=1.5918  patch_delta=-1.56250  carries IOI
    Head_L11_H10  score=1.4019  patch_delta=+1.68750  NOT an IOI carrier
    Head_L10_H7   score=1.3504  patch_delta=+4.62500  NOT an IOI carrier
    Head_L8_H6    score=1.1818  patch_delta=-1.43750  carries IOI
    Head_L7_H9    score=1.1586  patch_delta=-3.00000  carries IOI
    Head_L9_H9    score=1.1086  patch_delta=-1.31250  carries IOI
    Head_L9_H6    score=1.0435  patch_delta=+0.50000  NOT an IOI carrier

Three of the top eight are heads whose measured causal effect runs the
opposite way to what "top attribution" implies.

This is **not** a request that the ranking become causal. A first-order
gradient-times-activation score is a legitimate, cheap screening tool and
first-order attribution scores are sign-blind by definition. The defect is
narrower and is the one this project cares about: the report presents a
magnitude ranking with no statement that it carries no direction, so a reader
cannot tell a screening heuristic from a directional causal claim.

The fix is therefore to **declare the ranking's scope**, and to publish the
signed quantity so a reader can do the directional check themselves rather
than having to re-run patches. It is not to reorder anything: changing the
ranking would be a scientific claim this test file has no standing to make.
"""

from __future__ import annotations

import pytest

from _weight_guard import skip_reason, weights_available

needs_weights = pytest.mark.skipif(
    not weights_available(), reason=skip_reason()
)

CLEAN = "When John and Mary went to the store, John gave a drink to"
CORRUPTED = "When John and Mary went to the store, Mary gave a drink to John"
DATASET = {
    "id": "IOI-Canonical-100",
    "clean": CLEAN,
    "corrupted": CORRUPTED,
    "target_token": " Mary",
    "baseline_token": " John",
}

# GPT-2 token ids for the IOI metric.
MARY_ID = 5335
JOHN_ID = 1757


@pytest.fixture(scope="module")
def adapter():
    from backend.science.models.gpt2_adapter import GPT2Adapter

    instance = GPT2Adapter(variant="small", mock_mode=False)
    assert instance._model is not None
    return instance


@pytest.fixture(scope="module")
def report(adapter):
    from backend.interpretability.discovery.algorithms.attribution_patching import (
        AttributionPatchingAlgorithm,
    )

    return AttributionPatchingAlgorithm(adapter).run(DATASET)


def _metric(adapter, text: str) -> float:
    import torch

    inputs = adapter._tokenizer(text, return_tensors="pt").to(
        adapter._model.device
    )
    with torch.no_grad():
        logits = adapter._model(**inputs).logits[0, -1, :]
    return float(logits[MARY_ID] - logits[JOHN_ID])


# ── the precondition, asserted before anything relies on it ───────────

@needs_weights
def test_the_clean_run_prefers_mary_and_the_corrupted_run_prefers_john(
    adapter,
):
    """Fixture guard.

    Every direction claim below is meaningless if the metric does not
    actually separate the two prompts. Asserted first, and separately, so a
    backwards fixture cannot invert the remaining tests into passing ones.
    """
    clean = _metric(adapter, CLEAN)
    corrupted = _metric(adapter, CORRUPTED)

    assert clean > 0, f"the clean run does not prefer ' Mary': {clean}"
    assert corrupted < clean, (
        f"the corrupted run is not worse than the clean run: "
        f"corrupted={corrupted} clean={clean}"
    )


# ── the defect: a direction-blind score presented without that label ───

@needs_weights
def test_the_ranking_is_declared_direction_blind(report):
    """A magnitude ranking must say it carries no direction.

    `attr_score = sum(abs(d * g))` cannot distinguish a component that pushes
    the metric toward clean from one that pushes it toward corrupted. If the
    report does not say so, a reader cannot tell a screening heuristic from a
    directional causal claim -- and three of the top eight are not IOI
    carriers.
    """
    statistics = report.statistics

    marker = statistics.get("attribution_is_direction_blind")
    assert marker is True, (
        "the attribution score is sum(abs(d*g)) and is therefore "
        "direction-blind, but the report does not say so "
        f"(attribution_is_direction_blind={marker!r}). A reader would take "
        f"a magnitude ranking for a directional causal claim."
    )

    assert "direction" in str(statistics.get("metric_definition", "")).lower() or (
        "sign" in str(statistics.get("metric_definition", "")).lower()
    ), (
        "the metric definition must state that the score is direction-blind; "
        f"got: {statistics.get('metric_definition')!r}"
    )


@needs_weights
def test_each_attribution_publishes_its_signed_contribution(report):
    """The signed quantity must be available, so direction can be checked.

    The absolute-value score is not wrong -- it is a legitimate screening
    statistic. What is missing is the signed term it is built from. A reader
    who wants to know which components push toward clean and which push toward
    corrupted should not have to re-run 144 patches to find out.
    """
    tops = report.evidence["top_attributions"]
    assert tops, "no attributions returned"

    for item in tops:
        assert "signed_attribution" in item, (
            f"{item.get('component')} has no signed contribution; only "
            f"attribution_score={item.get('attribution_score')!r}, which is "
            f"sum(abs(d*g)) and cannot express direction"
        )
        signed = item["signed_attribution"]
        assert isinstance(signed, (int, float)), (
            f"{item.get('component')} signed_attribution is "
            f"{type(signed).__name__}, expected a number"
        )

    # The signed and absolute scores must disagree in sign at least once --
    # otherwise this test is not exercising the distinction it exists for.
    signs = {i["signed_attribution"] > 0 for i in tops}
    assert len(signs) == 2, (
        "every signed contribution has the same sign; the signed and "
        "absolute scores are not being distinguished at all"
    )


@needs_weights
def test_a_high_score_does_not_imply_a_causal_carrier(adapter, report):
    """The documented consequence, asserted rather than assumed.

    A first-order score is sign-blind, so a high score may rank a head whose
    measured patch effect runs against the IOI direction. This test records
    that it happens, so that changing the scoring to be directional -- or
    quietly assuming the ranking is causal -- is a deliberate act.

    It asserts only that the magnitude ranking and the measured direction are
    *not* the same ordering. It does not assert that the ranking is wrong:
    the top-ranked head L8_H10 does carry IOI information (delta -3.5), so
    the heuristic works here.
    """
    tops = report.evidence["top_attributions"][:8]
    assert len(tops) >= 4, f"only {len(tops)} attributions to compare"

    directions = []
    for item in tops:
        result = adapter.patch_head_output(
            CLEAN, item["layer"], item["head"], score_token_id=MARY_ID
        )
        # Negative delta: corrupting this head moves the clean run toward the
        # corrupted answer, so the head carries IOI information.
        directions.append(result.delta < 0)

    assert any(directions) and not all(directions), (
        "every top-attributed head moved the metric the same way; either the "
        "scoring has become directional, or the fixture has changed. "
        "Re-read the module docstring before changing this assertion."
    )


@needs_weights
def test_the_score_magnitude_is_unaffected_by_the_signed_field(report):
    """The ranking must not change -- only what it claims about itself.

    Guards against a 'fix' that silently reorders results. The signed field is
    additive information; the ranking by magnitude is unchanged.
    """
    tops = report.evidence["top_attributions"]

    scores = [i["attribution_score"] for i in tops]
    assert scores == sorted(scores, reverse=True), (
        f"attributions are not sorted by descending score: {scores}"
    )

    for item in tops:
        # The two fields are related by the triangle inequality, not by
        # approximation: sum(abs(d*g)) >= abs(sum(d*g)), with equality only
        # when every term shares a sign. An earlier draft asserted
        # approximate equality with a 35% tolerance, which is wrong -- the
        # gap is real and is exactly the cancellation that makes a
        # direction-blind score disagree with a signed one.
        assert item["attribution_score"] >= abs(item["signed_attribution"]) - 1e-6, (
            f"{item.get('component')}: attribution_score="
            f"{item['attribution_score']} is below abs(signed_attribution)="
            f"{abs(item['signed_attribution'])}. The score is sum(abs(d*g)) "
            f"and cannot be smaller than abs(sum(d*g)); if it is, the score "
            f"has been redefined, which is a scientific change to the "
            f"algorithm and belongs in a different review."
        )
        assert item["attribution_score"] == pytest.approx(
            sum(abs(d * g) for d, g in
                zip(item["delta_x"], item["grad_m"])),
            rel=1e-3, abs=1e-3,
        ), (
            f"{item.get('component')}: attribution_score must still be the "
            f"magnitude sum(abs(d*g)) recomputable from the published vectors"
        )