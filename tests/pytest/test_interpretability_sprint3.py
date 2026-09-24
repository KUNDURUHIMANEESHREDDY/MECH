"""Tests for AI 3 Mechanistic Interpretability Sprint 3 Understanding epics."""
from api.dispatcher import build_dispatcher


def test_circuit_discovery():
    dispatcher = build_dispatcher()
    circ = dispatcher["interpretability/circuits/discover"]({
        "prompt": "The capital of France is",
        "target_token": " Paris",
    })
    # Real IOI-pipeline measurements: faithfulness score, nominated heads.
    assert 0.5 < circ["circuit_score"] <= 1.0
    assert len(circ["nodes"]) >= 4
    assert len(circ["edges"]) >= 3
    assert "L10H7" in circ["nodes"]


def test_causal_tracing():
    dispatcher = build_dispatcher()
    trace = dispatcher["interpretability/causal/trace"]({
        "clean_prompt": "The capital of France is",
        "corrupted_prompt": "The capital of Rome is",
    })
    assert trace["max_causal_layer"] == 8
    assert len(trace["layer_effects"]) == 12


def test_attribution_patching():
    dispatcher = build_dispatcher()
    attr = dispatcher["interpretability/attribution/patch"]({
        "clean_prompt": "John gave a book to Mary",
        "corrupted_prompt": "John gave a book to John",
    })
    # Activation patching over IOI-nominated heads: ranked by |delta|.
    assert attr["method"] == "ActivationPatching"
    assert len(attr["top_attributed_nodes"]) >= 3
    assert attr["attribution_score"] > 0


def test_automated_feature_labeling():
    dispatcher = build_dispatcher()
    lbl = dispatcher["interpretability/features/label"]({"feature_id": 1402, "provider": "LocalModel"})
    assert lbl["feature_id"] == 1402
    assert "1402" in lbl["label"]
    assert len(lbl["evidence_prompts"]) >= 1
    assert 0.0 < lbl["confidence_score"] <= 1.0


def test_polysemanticity_detection():
    dispatcher = build_dispatcher()
    poly = dispatcher["interpretability/polysemanticity/detect"]({"target_type": "neuron", "index": 402})
    assert poly["polysemanticity_score"] == 0.28
    assert poly["classification"] == "monosemantic"


def test_feature_clustering():
    dispatcher = build_dispatcher()
    cluster = dispatcher["interpretability/features/cluster"]({"method": "Cosine", "num_clusters": 3})
    assert cluster["method"] == "Cosine"
    assert len(cluster["clusters"]) >= 2


def test_mechanistic_report():
    dispatcher = build_dispatcher()
    report = dispatcher["interpretability/reports/mechanistic"]({
        "prompt": "The capital of France is",
        "target_token": " Paris",
    })
    assert "Explanation" in report["explanation_text"]
    assert len(report["circuit_components"]) >= 3
