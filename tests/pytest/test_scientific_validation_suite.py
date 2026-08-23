"""Unit and integration tests for ScientificValidationSuite."""

import pytest
from backend.science.validation.scientific_validation_suite import (
    ScientificValidationSuite,
    CircuitValidationReport,
    StatisticalCI,
    HeldOutValidationResult,
    NegativeControlResult,
    MultiInterventionResult,
)
from backend.core.capability_registry import CapabilityRegistry
from backend.research_platform.autonomous.ai_research_assistant import AIResearchAssistant


def test_statistical_ci_computation():
    samples = [0.92, 0.94, 0.95, 0.91, 0.93, 0.96]
    ci = StatisticalCI.compute(samples)
    assert ci.sample_size == 6
    assert ci.mean > 0.90
    assert ci.ci_lower < ci.mean < ci.ci_upper
    assert "95% CI" in ci.format_ci()


def test_scientific_validation_suite():
    suite = ScientificValidationSuite(seed=42)

    dummy_nodes = [
        {"id": "node_L6_MLP", "data": {"label": "L6_MLP", "attribution": 0.48}},
        {"id": "node_L8_H5", "data": {"label": "L8_H5", "attribution": 0.42}},
    ]
    dummy_edges = [
        {"id": "e1", "source": "node_L6_MLP", "target": "node_L8_H5", "data": {"weight": 0.38}},
    ]

    report = suite.validate_circuit(
        circuit_nodes=dummy_nodes,
        circuit_edges=dummy_edges,
        task_name="Factual Recall (Eiffel Tower)",
        hypothesis="L6_MLP and L8_H5 mediate Paris generation",
        model_name="gpt2",
    )

    assert isinstance(report, CircuitValidationReport)
    
    # 1. Held-out validation checks
    assert report.heldout_validation.passed_generalization is True
    assert report.heldout_validation.generalization_ratio >= 0.85
    assert report.heldout_validation.discovery_sample_count > 0
    assert report.heldout_validation.heldout_sample_count > 0

    # 2. Negative controls checks
    assert report.negative_controls.passed_specificity is True
    assert report.negative_controls.specificity_ratio >= 3.0
    assert report.negative_controls.target_component_drop > report.negative_controls.random_component_drop

    # 3. Multi-intervention triangulation
    assert report.multi_interventions.passed_triangulation is True
    assert report.multi_interventions.ablation_drop_pct >= 60.0
    assert report.multi_interventions.restoration_recovery_pct >= 70.0

    # 4. Statistical CIs and Epistemic Tier
    assert report.overall_evidence_tier == "STRONG"
    assert "Strong causal evidence supporting the proposed mechanism" in report.calibrated_scientific_verdict
    assert report.reproducibility_manifest.manifest_id.startswith("exp_")


def test_scientific_validation_in_capability_registry():
    registry = CapabilityRegistry()
    assert registry.get_tool("run_scientific_circuit_validation") is not None

    res = registry.execute_tool("run_scientific_circuit_validation", {
        "circuit_nodes": [{"id": "node_L6_MLP"}],
        "circuit_edges": [],
        "task_name": "Test Task",
    })

    assert res["status"] == "success"
    assert "validation_report" in res
    vr = res["validation_report"]
    assert "held_out_faithfulness" in vr
    assert "ablation_specificity_score" in vr
    assert "overall_evidence_tier" in vr
    assert "manifest" in vr
