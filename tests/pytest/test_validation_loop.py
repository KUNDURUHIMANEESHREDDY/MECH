"""Tests for the closed validation loop: live pipeline vs published
(TransformerLens-paper) baselines -> ReproducibilityReport -> 0.85 gate.
"""
from agents.critic import Critic, CONFIDENCE_THRESHOLD


def test_reproduce_live_report_and_gate():
    critic = Critic()
    out = critic.reproduce("ioi", n_prompts=6)
    assert out["status"] == "completed"
    assert out["mock_mode"] is False
    assert out["n_prompts"] == 6

    observed = out["observed_metrics"]
    assert 0.0 < observed["circuit_faithfulness"] <= 1.0
    assert 0.0 < observed["circuit_completeness"] <= 1.0

    report = out["report"]
    assert report["paper_id"] == "ioi"
    assert "overall_fidelity_pct" in report
    assert "overall_tier" in report
    names = {m["name"] for m in report["metric_results"]}
    assert {"circuit_faithfulness", "circuit_completeness",
            "circuit_minimality"} <= names
    # Faithfulness beats its published baseline on live weights.
    faith = next(m for m in report["metric_results"]
                 if m["name"] == "circuit_faithfulness")
    assert faith["passed"] is True
    assert faith["fidelity_pct"] >= 85.0
    # Minimality is honestly unmeasured, never fabricated.
    minimal = next(m for m in report["metric_results"]
                   if m["name"] == "circuit_minimality")
    assert minimal["observed_value"] == 0.0
    assert minimal["passed"] is False

    gate = out["gate"]
    assert gate["threshold"] == CONFIDENCE_THRESHOLD == 0.85
    assert gate["metric"] == "overall_fidelity_pct"
    assert isinstance(gate["value"], (int, float))
    assert isinstance(gate["passed"], bool)


def test_reproduce_unimplemented_paper():
    out = Critic().reproduce("induction_heads")
    assert out["status"] == "unavailable"
    assert "not implemented" in out["reason"]
