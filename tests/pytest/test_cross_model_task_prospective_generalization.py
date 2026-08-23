"""Unit and integration tests for Phase 58: Combinatorial Cross-Model / Cross-Task Prospective Generalization Engine."""

import pytest

from backend.discovery.claim_dependency_graph import ClaimDependencyGraphEngine, ClaimEpistemicBelief
from backend.discovery.generalization_calibration_engine import GeneralizationCalibrationEngine, GeneralizationCertificate
from backend.discovery.generalization_claim_policy import (
    GeneralizationClaimPolicy,
    GeneralizationEpistemicStatus,
)
from backend.discovery.heldout_generalization_engine import HeldOutGeneralizationEngine, SplitType
from backend.discovery.negative_control_generalization_engine import NegativeControlGeneralizationEngine
from backend.discovery.prospective_generalization_protocol import (
    HeldOutChallengeSuite,
    HeldOutObservation,
    ProspectivePrediction,
    TransferClass,
)


def test_heldout_combinatorial_partition_integrity():
    """Verifies that D_calibration and D_heldout are strictly disjoint."""
    engine = HeldOutGeneralizationEngine()
    partition = engine.generate_crossed_heldout_partition()

    assert partition.verify_partition_disjointness() is True
    assert partition.split_type == SplitType.SPLIT_C_CROSSED_PAIRING
    assert len(partition.calibration_set) == 3
    assert len(partition.heldout_set) == 4

    # Verify OOD distance bounds
    for key, dist in partition.ood_distances.items():
        assert 0.0 <= dist <= 1.0


def test_crossed_prospective_prediction_and_evaluation():
    """Verifies prospective prediction evaluation across held-out crossed pairings."""
    pred_1 = ProspectivePrediction(
        prediction_id="PRED_CROSSED_LLAMA_MIXTRAL",
        source_model="meta-llama/Llama-3.1-8B",
        target_model="mistralai/Mixtral-8x7B",
        task_name="greater_than_arithmetic",
        predicted_transfer_class=TransferClass.TRANSFER,
        predicted_delta_z=2.410,
        predicted_delta_p=0.640,
        predicted_rescue=0.815,
        predictive_interval=(0.78, 0.85),
        abstention_probability=0.03,
        timestamp_utc="2026-08-16T12:00:00Z",
        source_claim_hash="hash_claim_1",
        predictive_model_hash="hash_model_1",
        model_hash="hash_target_1",
        tokenizer_hash="hash_tok_1",
        task_hash="hash_task_1",
        analysis_plan_hash="hash_plan_1",
        preregistration_hash="hash_prereg_1",
    )

    pred_2 = ProspectivePrediction(
        prediction_id="PRED_CROSSED_QWEN_GEMMA",
        source_model="Qwen/Qwen-2.5-7B",
        target_model="google/gemma-2-9b",
        task_name="ioi_induction",
        predicted_transfer_class=TransferClass.TRANSFER,
        predicted_delta_z=2.390,
        predicted_delta_p=0.630,
        predicted_rescue=0.805,
        predictive_interval=(0.77, 0.84),
        abstention_probability=0.04,
        timestamp_utc="2026-08-16T12:00:00Z",
        source_claim_hash="hash_claim_2",
        predictive_model_hash="hash_model_2",
        model_hash="hash_target_2",
        tokenizer_hash="hash_tok_2",
        task_hash="hash_task_2",
        analysis_plan_hash="hash_plan_2",
        preregistration_hash="hash_prereg_2",
    )

    pred_3 = ProspectivePrediction(
        prediction_id="PRED_ABSTAIN_MAMBA_SSM",
        source_model="meta-llama/Llama-3.1-8B",
        target_model="state-spaces/mamba-2-2.7b",
        task_name="recurrent_state_inhibition",
        predicted_transfer_class=TransferClass.ABSTAIN,
        predicted_delta_z=0.000,
        predicted_delta_p=0.000,
        predicted_rescue=0.000,
        predictive_interval=(0.0, 0.0),
        abstention_probability=0.98,
        timestamp_utc="2026-08-16T12:00:00Z",
        source_claim_hash="hash_claim_3",
        predictive_model_hash="hash_model_3",
        model_hash="hash_target_3",
        tokenizer_hash="hash_tok_3",
        task_hash="hash_task_3",
        analysis_plan_hash="hash_plan_3",
        preregistration_hash="hash_prereg_3",
    )

    suite = HeldOutChallengeSuite(
        suite_id="SUITE_PHASE58_CROSSED",
        predictions=[pred_1, pred_2, pred_3],
        sealed_manifest_hash="SEALED_MANIFEST_58",
        timestamp_utc="2026-08-16T12:00:00Z",
    )

    bundle = suite.export_blind_bundle()
    assert len(bundle["challenges"]) == 3
    for ch in bundle["challenges"]:
        assert "predicted_rescue" not in ch

    obs_1 = HeldOutObservation(
        prediction_id="PRED_CROSSED_LLAMA_MIXTRAL",
        investigator_id="LAB_A",
        observed_transfer_class=TransferClass.TRANSFER,
        empirical_delta_z=2.415,
        empirical_delta_p=0.642,
        empirical_rescue=0.812,
        raw_tensors={},
        hardware_metadata={"os": "Linux"},
        investigator_signature="SIG_A",
    )
    obs_2 = HeldOutObservation(
        prediction_id="PRED_CROSSED_QWEN_GEMMA",
        investigator_id="LAB_B",
        observed_transfer_class=TransferClass.TRANSFER,
        empirical_delta_z=2.385,
        empirical_delta_p=0.628,
        empirical_rescue=0.802,
        raw_tensors={},
        hardware_metadata={"os": "Linux"},
        investigator_signature="SIG_B",
    )
    obs_3 = HeldOutObservation(
        prediction_id="PRED_ABSTAIN_MAMBA_SSM",
        investigator_id="LAB_C",
        observed_transfer_class=TransferClass.ABSTAIN,
        empirical_delta_z=0.000,
        empirical_delta_p=0.000,
        empirical_rescue=0.000,
        raw_tensors={},
        hardware_metadata={"os": "Linux"},
        investigator_signature="SIG_C",
    )

    calib_engine = GeneralizationCalibrationEngine()
    cert = calib_engine.evaluate_heldout_suite(
        predictions=[pred_1, pred_2, pred_3],
        observations=[obs_1, obs_2, obs_3],
    )

    assert isinstance(cert, GeneralizationCertificate)
    assert cert.is_certified is True
    assert cert.hpa_score >= 0.90
    assert cert.transfer_class_accuracy >= 0.90
    assert cert.ece_heldout <= 0.05
    assert cert.fcr_heldout <= 0.02


def test_matched_negative_control_discrimination():
    """Verifies that the model discriminates genuine transfer from matched negative controls (Margin >= 0.65)."""
    neg_engine = NegativeControlGeneralizationEngine()
    result = neg_engine.generate_and_evaluate_matched_battery(
        source_model="meta-llama/Llama-3.1-8B",
        target_model="mistralai/Mixtral-8x7B",
        task_name="greater_than_arithmetic",
    )

    assert result.is_specific is True
    assert result.specificity_margin >= 0.65
    assert result.genuine_rescue >= 0.80
    assert result.discrimination_accuracy_pct == 100.0


def test_generalization_calibration_and_claim_policy_dag():
    """Verifies that the GeneralizationClaimPolicy registers granular statuses into the Claim DAG."""
    claim_graph = ClaimDependencyGraphEngine()
    policy = GeneralizationClaimPolicy(claim_graph=claim_graph)

    # 1. High-fidelity held-out supported claim
    rec_supported = policy.register_heldout_generalization_claim(
        claim_id="CLAIM_GEN_CROSSED_ARITHMETIC",
        in_domain_claim_id="CLAIM_IN_DOMAIN_ARITHMETIC",
        hpa_score=0.95,
        transfer_class_accuracy=1.0,
        ece_heldout=0.025,
        fcr_heldout=0.0,
        abstention_precision=1.0,
        ood_distance=0.35,
    )
    assert rec_supported.generalization_status == GeneralizationEpistemicStatus.HELDOUT_SUPPORTED
    assert claim_graph.claims["CLAIM_GEN_CROSSED_ARITHMETIC"].belief_status == ClaimEpistemicBelief.ACTIVE_SUPPORTED

    # 2. Boundary confirmed claim (SSM abstention)
    rec_boundary = policy.register_heldout_generalization_claim(
        claim_id="CLAIM_GEN_SSM_BOUNDARY",
        in_domain_claim_id="CLAIM_IN_DOMAIN_SSM",
        hpa_score=0.98,
        transfer_class_accuracy=1.0,
        ece_heldout=0.015,
        fcr_heldout=0.0,
        abstention_precision=1.0,
        ood_distance=0.88,
    )
    assert rec_boundary.generalization_status == GeneralizationEpistemicStatus.BOUNDARY_CONFIRMED

    # 3. Falsified held-out claim (downgrades generalization claim while preserving in-domain)
    rec_falsified = policy.register_heldout_generalization_claim(
        claim_id="CLAIM_GEN_FALSIFIED_OOD",
        in_domain_claim_id="CLAIM_IN_DOMAIN_BASE",
        hpa_score=0.50,
        transfer_class_accuracy=0.50,
        ece_heldout=0.15,
        fcr_heldout=0.10,
        abstention_precision=0.40,
        ood_distance=0.60,
    )
    assert rec_falsified.generalization_status == GeneralizationEpistemicStatus.OOD_FALSIFIED
    assert claim_graph.claims["CLAIM_GEN_FALSIFIED_OOD"].belief_status == ClaimEpistemicBelief.FALSIFIED_REVERTED
