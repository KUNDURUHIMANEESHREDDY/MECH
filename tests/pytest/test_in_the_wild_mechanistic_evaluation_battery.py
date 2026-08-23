"""Unit and integration tests for Phase 40: In-the-Wild Mechanistic Stress Battery."""

import pytest

from backend.discovery.claim_dependency_graph import ClaimDependencyGraphEngine, ClaimEpistemicBelief
from backend.discovery.in_the_wild_evaluation_battery import (
    InTheWildEvaluationEngine,
    InTheWildEvaluationReport,
    InTheWildScenarioResult,
    StressScenarioType,
)


def test_multihop_complex_relational_chain_discovery():
    """Verifies that the engine recovers >= 95% of true causal nodes on a complex 4-hop relational reasoning chain."""
    engine = InTheWildEvaluationEngine()
    scenario = engine.evaluate_multihop_relational_chain()

    assert isinstance(scenario, InTheWildScenarioResult)
    assert scenario.scenario_type == StressScenarioType.MULTIHOP_RELATIONAL_CHAIN
    assert scenario.circuit_recovery_rate >= 0.95
    assert scenario.prospective_prediction_error <= 0.05
    assert scenario.is_verified_safe is True


def test_superposition_and_distractor_resilience():
    """Verifies 100% rejection of adversarial distractor shortcuts and calibrated abstention on superposition noise."""
    engine = InTheWildEvaluationEngine()
    distractor_res = engine.evaluate_adversarial_distractors()
    superposition_res = engine.evaluate_polysemantic_superposition()

    assert distractor_res.distractors_rejected_pct == 100.0
    assert "DISTRACTOR_FALSIFICATION_RESISTANT" in distractor_res.epistemic_status

    assert superposition_res.circuit_recovery_rate >= 0.95
    assert "CALIBRATED_ABSTENTION_ENFORCED" in superposition_res.epistemic_status


def test_end_to_end_closed_loop_science_tournament():
    """Verifies that the entire in-the-wild evaluation report passes all strict recovery, rejection, and error bounds."""
    engine = InTheWildEvaluationEngine()
    report = engine.run_full_in_the_wild_battery()

    assert isinstance(report, InTheWildEvaluationReport)
    assert report.is_overall_benchmark_passed is True
    assert report.mean_circuit_recovery_rate >= 0.95
    assert report.mean_distractor_rejection_rate == 100.0
    assert report.mean_prospective_prediction_error <= 0.05
    assert len(report.scenarios) == 3


def test_in_the_wild_certificate_and_dag_audit_lineage():
    """Verifies that the master in-the-wild benchmark certificate is sealed with SHA-256 and integrated into the DAG."""
    claim_graph = ClaimDependencyGraphEngine()
    engine = InTheWildEvaluationEngine(claim_graph=claim_graph)

    report = engine.run_full_in_the_wild_battery()

    assert len(report.sha256_seal) == 64
    claim_id = f"CLAIM_IN_THE_WILD_BENCHMARK_{report.report_id}"
    assert claim_id in claim_graph.claims
    assert claim_graph.claims[claim_id].belief_status == ClaimEpistemicBelief.ACTIVE_SUPPORTED
    assert "In-The-Wild Mechanistic Validation Certified" in claim_graph.claims[claim_id].claim_statement
