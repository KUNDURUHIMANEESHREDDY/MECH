"""Tests for AI 3 Mechanistic Interpretability Sprint 4 Autonomous Discovery epics."""
from api.dispatcher import build_dispatcher


def test_discovery_engine_run():
    """Discovery must not reach Publication from the discovery orchestrator.

    The removed synthetic pipeline asserted a Confirmed hypothesis, a
    confidence above 0.80 and a circuit name for every input. The live
    orchestrator stops at Validation and hands off to Society validation, so
    those synthetic fields must be absent rather than invented.
    """
    dispatcher = build_dispatcher()
    res = dispatcher["interpretability/discovery/run"]({
        "hypothesis_statement": "L8_N402 mediates IOI capital retrieval"})

    assert res["lifecycle"]["state"] != "Publication"

    if res.get("provenance") == "live" and res.get("status") == "completed":
        # Live evidence may be eligible, but this orchestrator still does not
        # advance the lifecycle past Validation on its own.
        assert res["lifecycle"]["state"] == "Validation"
        assert res["validation_eligible"] is True
    else:
        assert res["validation_eligible"] is False
        assert res["publication_eligible"] is False
        assert res["reason"]
        # No synthetic confirmations alongside a blocked result.
        assert res.get("test_result") is None
        assert res.get("confidence") is None


def test_automatic_hypothesis_tester_states():
    dispatcher = build_dispatcher()
    res_conf = dispatcher["interpretability/hypothesis/test_auto"]({"hypothesis_statement": "L8_N402 induction head"})
    assert res_conf["outcome_state"] == "Confirmed"

    res_inc = dispatcher["interpretability/hypothesis/test_auto"]({"hypothesis_statement": "Random noise hypothesis"})
    assert res_inc["outcome_state"] == "Inconclusive"


def test_circuit_evolution_declares_itself_a_fixture():
    """The evolution trajectory is fixed, so it must not look measured."""
    dispatcher = build_dispatcher()
    evo = dispatcher["interpretability/circuits/evolution"]({"circuit_id": "c_ioi"})
    assert evo["evolution_steps_synthetic"] is True
    assert evo["provenance"] == "seeded"
    assert evo["validation_eligible"] is False
    assert evo["publication_eligible"] is False
    assert "not implemented" in evo["reason"]


def test_cross_model_alignment_is_not_a_causal_claim():
    """No model is loaded, so no cross-model similarity may be asserted."""
    dispatcher = build_dispatcher()
    align = dispatcher["interpretability/circuits/cross_model"]({
        "source_model": "GPT-2 Small",
        "target_model": "Gemma-2B",
        "circuit_type": "IOI",
    })
    assert align["alignment_measured"] is False
    assert align["provenance"] == "reference"
    assert align["validation_eligible"] is False
    assert align["publication_eligible"] is False
    assert "not implemented" in align["reason"]


def test_feature_genealogy_declares_itself_reference():
    dispatcher = build_dispatcher()
    gen = dispatcher["interpretability/features/genealogy"]({"feature_id": 1402})
    assert gen["genealogy_measured"] is False
    assert gen["provenance"] == "reference"
    assert gen["publication_eligible"] is False
    assert "not implemented" in gen["reason"]


def test_auto_circuit_namer_is_not_a_generated_name():
    """A fixed label asserted as 0.96 confidence was a mechanism claim."""
    dispatcher = build_dispatcher()
    name_res = dispatcher["interpretability/circuits/name_auto"]({"circuit_id": "circuit_31"})
    assert name_res["naming_measured"] is False
    assert name_res["confidence"] == 0.0
    assert name_res["validation_eligible"] is False
    assert name_res["publication_eligible"] is False
    assert "not implemented" in name_res["reason"]


def test_evidence_ranker_flags_its_default_sample():
    """Ranking with no input ranks a fixed sample; that must be visible."""
    dispatcher = build_dispatcher()
    ranked = dispatcher["interpretability/evidence/rank"]({})
    assert len(ranked) >= 2
    assert ranked[0]["rank_score"] >= ranked[1]["rank_score"]
    assert all(item["synthetic"] is True for item in ranked)
    assert all(item["publication_eligible"] is False for item in ranked)
    assert all(item["reason"] for item in ranked)


def test_evidence_ranker_scores_supplied_input():
    """Supplied metrics are ranked for real, and marked non-evidence."""
    dispatcher = build_dispatcher()
    ranked = dispatcher["interpretability/evidence/rank"]({"discoveries": [
        {"id": "a", "causal_effect": 0.9, "reproducibility": 0.9, "novelty": 0.9},
        {"id": "b", "causal_effect": 0.1, "reproducibility": 0.1, "novelty": 0.1},
    ]})
    assert [item["id"] for item in ranked] == ["a", "b"]
    assert all(item["synthetic"] is False for item in ranked)
    assert all(item["metrics_complete"] is True for item in ranked)


def test_confidence_scorer_flags_assumed_inputs():
    """No-argument scoring previously reported "High" for a platform that
    had measured nothing."""
    dispatcher = build_dispatcher()
    conf = dispatcher["interpretability/confidence/score"]({})
    assert conf["inputs_assumed"] is True
    assert conf["provenance"] == "unavailable"
    assert conf["validation_eligible"] is False
    assert conf["reason"]


def test_confidence_scorer_with_supplied_metrics():
    """Two of three metrics supplied: the reason must name the third.

    `inputs_assumed` stays True because `variance` was never given, so the
    scorer's own default (0.04) is in the sum. The old blanket reason said
    "inputs were not supplied", which is false here and hides which term is
    fabricated.
    """
    dispatcher = build_dispatcher()
    conf = dispatcher["interpretability/confidence/score"](
        {"evidence_count": 8, "reproducibility_score": 0.98})
    assert conf["inputs_assumed"] is True
    assert conf["assumed_inputs"] == ["variance"]
    assert "variance" in conf["reason"]
    assert conf["evidence_count"] == 8
    assert conf["reproducibility_score"] == 0.98
    assert conf["provenance"] == "unavailable"
    # Still not evidence: a weighted sum is not a calibrated probability.
    assert conf["publication_eligible"] is False


def test_confidence_scorer_with_every_metric_supplied():
    """All three supplied means no default is in the arithmetic."""
    dispatcher = build_dispatcher()
    conf = dispatcher["interpretability/confidence/score"](
        {"evidence_count": 8, "reproducibility_score": 0.98, "variance": 0.04})

    assert conf["inputs_assumed"] is False
    assert conf["assumed_inputs"] == []
    assert conf["confidence_score"] > 0.90
    assert conf["reliability_rating"] == "High"
    assert conf["provenance"] == "reference"
    # Still not evidence: nothing here attests to a run.
    assert conf["publication_eligible"] is False
    assert conf["validation_eligible"] is False
