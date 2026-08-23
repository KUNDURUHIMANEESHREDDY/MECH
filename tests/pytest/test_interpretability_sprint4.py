"""Tests for AI 3 Mechanistic Interpretability Sprint 4 Autonomous Discovery epics."""
from api.dispatcher import build_dispatcher


def test_discovery_engine_run():
    dispatcher = build_dispatcher()
    res = dispatcher["interpretability/discovery/run"]({"hypothesis_statement": "L8_N402 mediates IOI capital retrieval"})
    assert res["lifecycle"]["state"] == "Publication"
    assert res["test_result"]["outcome_state"] == "Confirmed"
    assert res["confidence"]["confidence_score"] > 0.80
    assert "circuit_name" in res


def test_automatic_hypothesis_tester_states():
    dispatcher = build_dispatcher()
    res_conf = dispatcher["interpretability/hypothesis/test_auto"]({"hypothesis_statement": "L8_N402 induction head"})
    assert res_conf["outcome_state"] == "Confirmed"

    res_inc = dispatcher["interpretability/hypothesis/test_auto"]({"hypothesis_statement": "Random noise hypothesis"})
    assert res_inc["outcome_state"] == "Inconclusive"


def test_circuit_evolution():
    dispatcher = build_dispatcher()
    evo = dispatcher["interpretability/circuits/evolution"]({"circuit_id": "c_ioi"})
    assert len(evo["evolution_steps"]) == 3
    assert evo["evolution_steps"][-1]["active_nodes_count"] == 18


def test_cross_model_alignment():
    dispatcher = build_dispatcher()
    align = dispatcher["interpretability/circuits/cross_model"]({
        "source_model": "GPT-2 Small",
        "target_model": "Gemma-2B",
        "circuit_type": "IOI",
    })
    assert align["alignment"]["causal_similarity"] > 0.7
    assert align["alignment"]["functional_similarity"] > 0.80


def test_feature_genealogy_dag():
    dispatcher = build_dispatcher()
    gen = dispatcher["interpretability/features/genealogy"]({"feature_id": 1402})
    assert gen["genealogy"]["feature_id"] == 1402
    assert len(gen["genealogy"]["parents"]) == 2
    assert len(gen["genealogy"]["children"]) == 2


def test_auto_circuit_namer():
    dispatcher = build_dispatcher()
    name_res = dispatcher["interpretability/circuits/name_auto"]({"circuit_id": "circuit_31"})
    assert name_res["name"] == "Name Recognition Circuit"
    assert "Responds strongly to person names" in name_res["description"]


def test_evidence_ranker():
    dispatcher = build_dispatcher()
    ranked = dispatcher["interpretability/evidence/rank"]({})
    assert len(ranked) >= 2
    assert ranked[0]["rank_score"] >= ranked[1]["rank_score"]


def test_confidence_scorer():
    dispatcher = build_dispatcher()
    conf = dispatcher["interpretability/confidence/score"]({"evidence_count": 8, "reproducibility_score": 0.98})
    assert conf["confidence_score"] > 0.90
    assert conf["reliability_rating"] == "High"
