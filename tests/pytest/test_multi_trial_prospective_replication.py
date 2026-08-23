"""Unit and integration tests for Phase 57: Multi-Trial Prospective Replication Engine."""

import pytest

from backend.discovery.claim_dependency_graph import ClaimDependencyGraphEngine, ClaimEpistemicBelief
from backend.discovery.multi_trial_prospective_replication_engine import (
    MultiTrialExportBundle,
    MultiTrialJointAuditScorecard,
    MultiTrialProspectiveReplicationEngine,
    TrialObservationRecord,
)


def test_multi_trial_bundle_packaging_and_preregistration():
    """Verifies that the multi-trial bundle packages 3 trials cleanly with zero target ground truths."""
    engine = MultiTrialProspectiveReplicationEngine()
    manifest, bundle = engine.generate_multi_trial_bundle()

    assert manifest.verify_integrity() is True
    assert isinstance(bundle, MultiTrialExportBundle)
    assert len(bundle.trials) == 3

    trial_ids = [t["trial_id"] for t in bundle.trials]
    assert "TRIAL_1_POS_QWEN_INDUCTION" in trial_ids
    assert "TRIAL_2_NEG_STARCODER_POLY" in trial_ids
    assert "TRIAL_3_ABSTAIN_MAMBA_SSM" in trial_ids

    for t in bundle.trials:
        assert "empirical_r" not in t
        assert "empirical_delta_z" not in t


def test_joint_tri_regime_execution_and_evaluation():
    """Verifies joint evaluation across Positive, Negative Transfer, and Abstention regimes (MTRF >= 95.0%)."""
    engine = MultiTrialProspectiveReplicationEngine()
    manifest, bundle = engine.generate_multi_trial_bundle()

    obs_1 = TrialObservationRecord(
        trial_id="TRIAL_1_POS_QWEN_INDUCTION",
        investigator_id="LAB_A_BERKELEY",
        empirical_r=0.820,
        empirical_delta_z=2.435,
        empirical_delta_p=0.655,
        is_abstained=False,
        raw_tensors={"delta_z": [2.43, 2.435, 2.44]},
        hardware_metadata={"device": "cuda:0", "os": "Linux"},
        signature="SIG_LAB_A_123",
    )

    obs_2 = TrialObservationRecord(
        trial_id="TRIAL_2_NEG_STARCODER_POLY",
        investigator_id="LAB_B_OXFORD",
        empirical_r=0.125,
        empirical_delta_z=0.345,
        empirical_delta_p=0.088,
        is_abstained=False,
        raw_tensors={"delta_z": [0.34, 0.345, 0.35]},
        hardware_metadata={"device": "cuda:1", "os": "Linux"},
        signature="SIG_LAB_B_456",
    )

    obs_3 = TrialObservationRecord(
        trial_id="TRIAL_3_ABSTAIN_MAMBA_SSM",
        investigator_id="LAB_C_STANFORD",
        empirical_r=0.000,
        empirical_delta_z=0.000,
        empirical_delta_p=0.000,
        is_abstained=True,
        raw_tensors={},
        hardware_metadata={"device": "cpu", "os": "Linux"},
        signature="SIG_LAB_C_789",
    )

    scorecard = engine.audit_and_evaluate_multi_trial_observations(
        manifest=manifest,
        observations=[obs_1, obs_2, obs_3],
    )

    assert isinstance(scorecard, MultiTrialJointAuditScorecard)
    assert scorecard.is_quorum_verified is True
    assert scorecard.mtrf_score >= 95.0
    assert scorecard.cross_trial_heterogeneity <= 0.05
    assert scorecard.pos_fidelity_pct >= 95.0
    assert scorecard.neg_fidelity_pct >= 95.0
    assert scorecard.abstention_fidelity_pct == 100.0


def test_automatic_claim_downgrade_on_unexpected_divergence():
    """Verifies that an uncalibrated divergence automatically triggers a Claim DAG downgrade."""
    claim_graph = ClaimDependencyGraphEngine()
    engine = MultiTrialProspectiveReplicationEngine(claim_graph=claim_graph)
    manifest, bundle = engine.generate_multi_trial_bundle()

    # Divergent observation on Trial 1 (e.g. delta_z off by 30%)
    obs_divergent = TrialObservationRecord(
        trial_id="TRIAL_1_POS_QWEN_INDUCTION",
        investigator_id="LAB_A_BERKELEY",
        empirical_r=0.450,
        empirical_delta_z=1.500,  # Expected 2.450 -> >30% error
        empirical_delta_p=0.300,
        is_abstained=False,
        raw_tensors={"delta_z": [1.50]},
        hardware_metadata={"device": "cuda:0", "os": "Linux"},
        signature="SIG_LAB_A_DIV",
    )

    scorecard = engine.audit_and_evaluate_multi_trial_observations(
        manifest=manifest,
        observations=[obs_divergent],
    )

    assert scorecard.is_quorum_verified is False
    assert "DOWNGRADED" in scorecard.summary_verdict

    claim_id = f"CLAIM_MULTI_TRIAL_PROSPECTIVE_REPLICATION_{manifest.pre_reg_id}"
    assert claim_id in claim_graph.claims
    assert claim_graph.claims[claim_id].belief_status == ClaimEpistemicBelief.EVIDENCE_WEAKENED


def test_dag_registration_for_multi_trial_replication():
    """Verifies that successful multi-trial replication is registered with complete audit lineage."""
    claim_graph = ClaimDependencyGraphEngine()
    engine = MultiTrialProspectiveReplicationEngine(claim_graph=claim_graph)
    manifest, bundle = engine.generate_multi_trial_bundle()

    obs_1 = TrialObservationRecord(
        trial_id="TRIAL_1_POS_QWEN_INDUCTION",
        investigator_id="LAB_A",
        empirical_r=0.820,
        empirical_delta_z=2.435,
        empirical_delta_p=0.655,
        is_abstained=False,
        raw_tensors={},
        hardware_metadata={},
        signature="SIG_A",
    )
    obs_2 = TrialObservationRecord(
        trial_id="TRIAL_2_NEG_STARCODER_POLY",
        investigator_id="LAB_B",
        empirical_r=0.125,
        empirical_delta_z=0.345,
        empirical_delta_p=0.088,
        is_abstained=False,
        raw_tensors={},
        hardware_metadata={},
        signature="SIG_B",
    )
    obs_3 = TrialObservationRecord(
        trial_id="TRIAL_3_ABSTAIN_MAMBA_SSM",
        investigator_id="LAB_C",
        empirical_r=0.000,
        empirical_delta_z=0.000,
        empirical_delta_p=0.000,
        is_abstained=True,
        raw_tensors={},
        hardware_metadata={},
        signature="SIG_C",
    )

    scorecard = engine.audit_and_evaluate_multi_trial_observations(
        manifest=manifest,
        observations=[obs_1, obs_2, obs_3],
    )

    claim_id = f"CLAIM_MULTI_TRIAL_PROSPECTIVE_REPLICATION_{manifest.pre_reg_id}"
    assert claim_id in claim_graph.claims
    assert claim_graph.claims[claim_id].belief_status == ClaimEpistemicBelief.ACTIVE_SUPPORTED
    assert "Multi-Trial Prospective Replication Certified" in claim_graph.claims[claim_id].claim_statement
