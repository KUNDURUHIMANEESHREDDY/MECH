"""Unit and integration tests for Quantitative Circuit Metrics Suite (ACDC Standards).

NOTE: These tests now assert that metrics are REAL (derived from genuine
activation-patching ablations on the loaded GPT-2 model) and fall within valid
[0, 1] ranges. They no longer assert the fabricated >= 0.90 / 0.85 / 0.80
thresholds, since a circuit's true faithfulness/completeness/minimality is an
empirical result, not a guaranteed pass.
"""

import pytest
from backend.science.circuits.acdc_circuit_metrics import (
    ACDCCircuitMetricsEvaluator,
    ACDCCircuitEvaluation,
    NodeSensitivity,
)
from backend.core.capability_registry import CapabilityRegistry
from backend.research_platform.autonomous.ai_research_assistant import AIResearchAssistant
from backend.services import gpt2_engine as _gpt2

pytestmark = pytest.mark.skipif(
    not _gpt2.is_available(),
    reason="ACDCCircuitMetricsEvaluator requires a loaded GPT-2 model for real ablations",
)


def test_acdc_circuit_metrics_evaluation():
    evaluator = ACDCCircuitMetricsEvaluator(
        faithfulness_target=0.90,
        completeness_target=0.85,
        minimality_target=0.80,
    )

    nodes = [
        {"id": "input", "data": {"label": "Input: 'The Eiffel Tower'"}},
        {"id": "node_L6_MLP", "data": {"attribution": 0.48, "layer": 6}},
        {"id": "node_L8_H5", "data": {"attribution": 0.38, "layer": 8}},
        {"id": "node_L9_H3", "data": {"attribution": 0.32, "layer": 9}},
        {"id": "output", "data": {"label": "Target: ' Paris'"}},
    ]
    edges = [
        {"source": "input", "target": "node_L6_MLP"},
        {"source": "node_L6_MLP", "target": "node_L8_H5"},
        {"source": "node_L8_H5", "target": "node_L9_H3"},
        {"source": "node_L9_H3", "target": "output"},
    ]

    res = evaluator.evaluate_circuit(
        nodes=nodes,
        edges=edges,
        clean_prompt="The Eiffel Tower is in the city of",
        corrupted_prompt="The Colosseum is in the city of",
        target_token=" Paris",
        task_name="Fact Retrieval",
    )

    assert isinstance(res, ACDCCircuitEvaluation)
    assert res.num_nodes == 3  # Excludes input/output
    assert res.num_edges == 4

    # Metrics are REAL (activation-patching ablations) and bounded in [0, 1].
    assert 0.0 <= res.faithfulness <= 1.0
    assert isinstance(res.meets_faithfulness_policy, bool)
    assert 0.0 <= res.completeness <= 1.0
    assert isinstance(res.meets_completeness_policy, bool)
    assert 0.0 <= res.minimality_score <= 1.0
    assert isinstance(res.meets_minimality_policy, bool)

    # Underlying deltas are real model logits (not fabricated constants).
    assert res.full_model_clean_delta != 4.35
    assert res.full_model_corrupted_delta != 0.42

    # Minimality specifics.
    assert res.node_necessity_threshold == 0.15
    assert 0.0 <= res.critical_node_fraction <= 1.0
    assert len(res.node_sensitivities) == 3
    for ns in res.node_sensitivities:
        assert 0.0 <= ns.faithfulness_drop_on_ablation <= 1.0

    assert isinstance(res.policy_grade, str) and len(res.policy_grade) > 0


def test_acdc_circuit_metrics_tool_in_capability_registry():
    registry = CapabilityRegistry()
    assert registry.get_tool("evaluate_circuit_metrics") is not None

    tool_res = registry.execute_tool("evaluate_circuit_metrics", {
        "nodes": [
            {"id": "node_L6_MLP", "data": {"attribution": 0.52}},
            {"id": "node_L9_H3", "data": {"attribution": 0.41}},
        ],
        "edges": [{"source": "node_L6_MLP", "target": "node_L9_H3"}],
    })

    assert tool_res["status"] == "success"
    assert "evaluation" in tool_res
    ev = tool_res["evaluation"]
    assert 0.0 <= ev["faithfulness"] <= 1.0
    assert 0.0 <= ev["completeness"] <= 1.0
    assert 0.0 <= ev["minimality"] <= 1.0
    assert "minimality_details" in ev


def test_ai_research_assistant_acdc_integration():
    assistant = AIResearchAssistant()
    res = assistant.investigate(
        goal="Investigate why GPT-2 predicts Paris for the Eiffel Tower",
        model_name="gpt2",
    )

    assert res["status"] == "completed"
    assert "circuit" in res
    c = res["circuit"]
    assert "faithfulness" in c
    assert "completeness" in c
    assert "minimality" in c
    assert "policy_grade" in c
    assert 0.0 <= c["faithfulness"] <= 1.0
    assert 0.0 <= c["completeness"] <= 1.0
