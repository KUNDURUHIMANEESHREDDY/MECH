"""Unit and integration tests for Phase 26: Empirical Metascience & Epistemic Validation Benchmark."""

import pytest

from backend.discovery.autonomous_science_benchmark import (
    AbstentionAblationComparison,
    MetascienceBenchmarkEngine,
    MetascienceBenchmarkReport,
    SelectionRegretConvergenceResult,
    StressScenarioType,
    StressTestScenarioResult,
)
from backend.discovery.calibration_policy import CalibrationPolicyConfig
from backend.discovery.model_criticism_engine import CriticEpistemicState


def test_stress_distribution_criticism_faithfulness():
    """Verifies that Model Criticism faithfully maps controlled distributions to their correct epistemic states."""
    engine = MetascienceBenchmarkEngine()
    results = engine.run_distribution_stress_tests()

    assert len(results) == 4
    for res in results:
        assert isinstance(res, StressTestScenarioResult)
        assert res.is_expected_state is True

    # Validate specific mappings
    clean_res = next(r for r in results if r.scenario_type == StressScenarioType.CLEAN_GAUSSIAN)
    assert clean_res.epistemic_state == CriticEpistemicState.VALID_CONFIDENT

    heavy_res = next(r for r in results if r.scenario_type == StressScenarioType.HEAVY_TAILED_CAUCHY)
    assert heavy_res.epistemic_state == CriticEpistemicState.ABSTAIN

    sparse_res = next(r for r in results if r.scenario_type == StressScenarioType.SPARSE_SAMPLE_REGIME)
    assert sparse_res.epistemic_state == CriticEpistemicState.ABSTAIN


def test_abstention_prevents_false_mechanism_certification():
    """Verifies that epistemic abstention eliminates false mechanism certifications under adversarial noise."""
    engine = MetascienceBenchmarkEngine()
    comparison = engine.run_abstention_ablation_benchmark(trials=10)

    assert isinstance(comparison, AbstentionAblationComparison)
    # Naive scientist without abstention should falsely certify adversarial noise
    assert comparison.naive_false_certification_rate_pct > 50.0
    # Calibrated scientist with abstention should achieve 0% false certifications
    assert comparison.calibrated_false_certification_rate_pct == 0.0
    assert comparison.abstention_safety_margin_pct > 50.0


def test_selection_regret_monotonic_decay():
    """Verifies that active learning selection regret decays monotonically across consecutive rounds."""
    engine = MetascienceBenchmarkEngine()
    regret_res = engine.run_selection_regret_decay_test(rounds=5)

    assert isinstance(regret_res, SelectionRegretConvergenceResult)
    assert regret_res.is_regret_decaying is True
    assert regret_res.final_round_regret_bits < regret_res.initial_round_regret_bits
    assert regret_res.regret_decay_reduction_bits > 0.30


def test_metascience_benchmark_suite_end_to_end():
    """Verifies full execution of the Metascience Benchmark Engine and report compilation."""
    engine = MetascienceBenchmarkEngine()
    report = engine.run_full_metascience_benchmark()

    assert isinstance(report, MetascienceBenchmarkReport)
    assert report.is_all_benchmarks_passed is True
    assert len(report.stress_test_results) == 4
    assert report.abstention_ablation.calibrated_false_certification_rate_pct == 0.0
    assert "PASSED" in report.summary_verdict
    assert len(report.to_dict()) > 0
