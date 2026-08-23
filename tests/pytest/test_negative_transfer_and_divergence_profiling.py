"""Unit and integration tests for Phase 29: Negative Transfer Testing & Architecture-Specific Divergence Profiling."""

import pytest

from backend.discovery.cross_model_alignment_engine import (
    CrossModelAlignmentEngine,
    FunctionalRoleType,
    SubstrateIndependenceClass,
    UniversalCircuitCertificate,
)
from backend.discovery.negative_transfer_engine import (
    NegativeTransferEngine,
    NegativeTransferReport,
    NegativeTransferTrialResult,
)


def test_architecture_specific_discrepancy_reporting():
    """Verifies that Universal Circuit Certificates explicitly report physical implementation discrepancies."""
    engine = CrossModelAlignmentEngine()

    src_comps = {
        "gpt2_L0_N12": FunctionalRoleType.SUBJECT_EXTRACTOR,
        "gpt2_L8_N412": FunctionalRoleType.RELATIONAL_RETRIEVER,
        "gpt2_L10_H7": FunctionalRoleType.NAME_MOVER,
        "gpt2_L11_N900": FunctionalRoleType.OUTPUT_PROJECTOR,
    }
    src_edges = [
        ("gpt2_L0_N12", "gpt2_L8_N412"),
        ("gpt2_L8_N412", "gpt2_L10_H7"),
        ("gpt2_L10_H7", "gpt2_L11_N900"),
    ]

    tgt_comps = {
        "pythia_L1_N30": FunctionalRoleType.SUBJECT_EXTRACTOR,
        "pythia_L7_N384": FunctionalRoleType.RELATIONAL_RETRIEVER,
        "pythia_L9_H4": FunctionalRoleType.NAME_MOVER,
        "pythia_L11_N512": FunctionalRoleType.OUTPUT_PROJECTOR,
    }
    tgt_edges = [
        ("pythia_L1_N30", "pythia_L7_N384"),
        ("pythia_L7_N384", "pythia_L9_H4"),
        ("pythia_L9_H4", "pythia_L11_N512"),
    ]

    scorecard = engine.align_model_circuits(
        source_model_id="gpt2-small",
        target_model_id="pythia-70m",
        behavior_name="country_capital",
        source_circuit_components=src_comps,
        source_edges=src_edges,
        target_circuit_components=tgt_comps,
        target_edges=tgt_edges,
    )

    cert = engine.certify_universal_circuit("country_capital", [scorecard])

    assert len(scorecard.architecture_specific_discrepancies) == 4
    assert any("gpt2-small [gpt2_L8_N412]" in d for d in scorecard.architecture_specific_discrepancies)
    assert any("pythia-70m [pythia_L7_N384]" in d for d in scorecard.architecture_specific_discrepancies)
    assert len(cert.architecture_specific_discrepancies) == 4


def test_negative_transfer_falsification():
    """Verifies that behavior-matched but architecturally divergent models fail causal transfer (R < 0.20)."""
    neg_engine = NegativeTransferEngine()
    trial = neg_engine.execute_negative_transfer_trial(
        behavior_name="country_capital",
        source_model_id="gpt2-small",
        isomorphic_target_id="pythia-70m",
        divergent_target_id="dense_ffn_memorizer",
    )

    assert isinstance(trial, NegativeTransferTrialResult)
    assert trial.is_behavior_matched is True
    assert trial.divergent_transfer_rescue < 0.20
    assert trial.canonical_transfer_rescue > 0.80
    assert trial.is_negative_transfer_successfully_detected is True
    assert trial.falsified_universality is True


def test_divergence_specificity_ratio_bound():
    """Verifies that the Specificity Gap between Canonical and Divergent transfer is >= 0.50."""
    neg_engine = NegativeTransferEngine()
    trial = neg_engine.execute_negative_transfer_trial(
        behavior_name="indirect_object_identification",
        source_model_id="gpt2-small",
        isomorphic_target_id="qwen-0.5b",
        divergent_target_id="recurrent_transformer_variant",
    )

    assert trial.divergence_specificity_gap >= 0.50
    assert trial.divergence_specificity_gap == pytest.approx(0.76, abs=0.05)


def test_negative_transfer_suite_end_to_end():
    """Verifies full execution of the Negative Transfer Suite and False Universality Rate = 0.0%."""
    neg_engine = NegativeTransferEngine()
    report = neg_engine.run_negative_transfer_suite()

    assert isinstance(report, NegativeTransferReport)
    assert report.is_anti_universality_bias_verified is True
    assert report.false_universality_rate_pct == 0.0
    assert report.mean_divergence_gap >= 0.50
    assert len(report.trials) == 3
    assert "PASSED" in report.summary_verdict
