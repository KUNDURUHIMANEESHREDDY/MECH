"""Unit and integration tests for Phase 27: Out-of-Distribution Metascience & Generalization."""

import pytest

from backend.discovery.ood_metascience_engine import (
    OODAbstentionMetrics,
    OODGeneralizationScorecard,
    OODMetascienceEngine,
    OODMetascienceReport,
    OODRegimeType,
)


def test_ood_abstention_precision_and_recall():
    """Verifies that OOD Abstention achieves high Precision (>=90%) and Recall (>=95%) without false certifications."""
    engine = OODMetascienceEngine()
    metrics = engine.evaluate_ood_abstention_precision_recall(trials=20)

    assert isinstance(metrics, OODAbstentionMetrics)
    assert metrics.abstention_precision_pct >= 90.0
    assert metrics.abstention_recall_pct >= 95.0
    assert metrics.abstention_f1_score >= 0.90
    assert metrics.ood_false_certification_rate_pct == 0.0


def test_ood_generalization_gap_bound():
    """Verifies that the EIG calibration generalization gap between In-Distribution and OOD is <= 0.15 bits."""
    engine = OODMetascienceEngine()
    in_ece, ood_ece, gap, is_bounded = engine.evaluate_ood_generalization_gap()

    assert in_ece > 0.0
    assert ood_ece > in_ece
    assert gap <= 0.15
    assert is_bounded is True


def test_ood_claim_revision_cascade_correctness():
    """Verifies that OOD contradictory evidence correctly cascades status weakening in the Claim DAG."""
    engine = OODMetascienceEngine()
    is_cascade_valid = engine.test_ood_claim_revision_cascade()

    assert is_cascade_valid is True


def test_ood_metascience_suite_end_to_end():
    """Verifies full execution of the OOD Metascience Engine and generation of the certified audit report."""
    engine = OODMetascienceEngine()
    report = engine.run_full_ood_metascience_benchmark()

    assert isinstance(report, OODMetascienceReport)
    assert report.is_generalization_certified is True
    assert report.scorecard.is_generalization_gap_bounded is True
    assert report.scorecard.abstention_metrics.ood_false_certification_rate_pct == 0.0
    assert report.claim_revision_cascade_verified is True
    assert "PASSED" in report.summary_verdict
