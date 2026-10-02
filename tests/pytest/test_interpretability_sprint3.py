"""Tests for AI 3 Mechanistic Interpretability Sprint 3 Understanding epics."""
from api.dispatcher import build_dispatcher


def test_circuit_discovery():
    """A blocked pipeline must not be reported as a measured score.

    The IOI pipeline fails closed when it cannot obtain the comparison tokens.
    The handler used to turn that into `circuit_score: 0.0` with an empty node
    list -- indistinguishable, to a caller, from a real measurement of zero.
    It must now say it is unavailable and why, and must not emit a score.
    """
    dispatcher = build_dispatcher()
    circ = dispatcher["interpretability/circuits/discover"]({
        "prompt": "The capital of France is",
        "target_token": " Paris",
    })

    if circ.get("status") == "completed":
        # Only if the pipeline really measured something may a score appear.
        assert circ["provenance"] == "live"
        assert 0.0 <= circ["circuit_score"] <= 1.0
        assert len(circ["nodes"]) >= 1
    else:
        assert circ["status"] in ("unavailable", "blocked", "error")
        assert circ["provenance"] != "live"
        assert circ["validation_eligible"] is False
        assert circ["publication_eligible"] is False
        assert circ["reason"]
        # Crucially: no fabricated zero standing in for a measurement.
        assert "circuit_score" not in circ


def test_mechanistic_report():
    """The report must not narrate a mechanism that was not measured."""
    dispatcher = build_dispatcher()
    report = dispatcher["interpretability/reports/mechanistic"]({
        "prompt": "The capital of France is",
        "target_token": " Paris",
    })
    assert "Explanation" in report["explanation_text"]

    if report.get("status") == "completed":
        assert len(report["circuit_components"]) >= 3
    else:
        assert report["status"] in ("unavailable", "error")
        assert report["publication_eligible"] is False
        assert report["reason"]
        # Must not assert a faithfulness figure or a head count it never got.
        assert "reaches faithfulness" not in report["explanation_text"]


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
    """The labeller is a fixture; it must not report a confidence."""
    dispatcher = build_dispatcher()
    lbl = dispatcher["interpretability/features/label"](
        {"feature_id": 1402, "provider": "LocalModel"})
    assert lbl["feature_id"] == 1402
    assert "1402" in lbl["label"]
    assert lbl["confidence_score"] == 0.0
    assert lbl["label_measured"] is False
    assert lbl["validation_eligible"] is False
    assert lbl["publication_eligible"] is False
    assert "not implemented" in lbl["reason"]
    # No fabricated activations standing in for interpretation evidence.
    assert lbl["explanation_evidence"] == []


def test_polysemanticity_detection_is_not_a_hardcoded_literal():
    """The detector must report a real classification, not a fixed answer.

    `semantics/feature_labeler.py` carries a hardcoded confidence_score of
    0.94. Confidence is only meaningful with provenance attached, so the
    synthetic-labelling path has to declare itself rather than emit a number
    that reads like a measurement.
    """
    dispatcher = build_dispatcher()
    poly = dispatcher["interpretability/polysemanticity/detect"](
        {"target_type": "neuron", "index": 402})
    assert "polysemanticity_score" in poly
    if poly.get("provenance") == "live":
        assert 0.0 <= poly["polysemanticity_score"] <= 1.0
        assert poly["classification"] in ("monosemantic", "polysemantic")
    else:
        assert poly["provenance"] in ("seeded", "unavailable", "reference")

def test_polysemanticity_detection_declares_itself_synthetic():
    """The detector returns fixed fixtures, and must say so."""
    dispatcher = build_dispatcher()
    poly = dispatcher["interpretability/polysemanticity/detect"]({"target_type": "neuron", "index": 402})
    assert poly["provenance"] == "seeded"
    assert poly["status"] == "unavailable"
    assert poly["validation_eligible"] is False
    assert poly["publication_eligible"] is False
    assert "not implemented" in poly["reason"]
    assert poly["evidence_synthetic"] is True
    # The activations are literals, so each entry must disclaim measurement.
    assert all(e.get("measured") is False for e in poly["evidence"])


def test_feature_clustering():
    dispatcher = build_dispatcher()
    cluster = dispatcher["interpretability/features/cluster"]({"method": "Cosine", "num_clusters": 3})
    assert cluster["method"] == "Cosine"
    assert len(cluster["clusters"]) >= 2
