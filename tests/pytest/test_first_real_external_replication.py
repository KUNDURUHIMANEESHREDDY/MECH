"""Unit and integration tests for Phase 56: First Real-World Preregistered Blind External Replication Experiment."""

import pytest

from backend.discovery.claim_dependency_graph import ClaimDependencyGraphEngine, ClaimEpistemicBelief
from backend.discovery.first_real_external_replication_orchestrator import (
    FirstRealExternalReplicationOrchestrator,
    Phase56ReplicationScorecard,
)
from backend.discovery.open_world_multi_lab_preregistration_engine import OpenWorldAuditorQuorum


def test_preregistration_and_package_export():
    """Verifies that challenge packaging is clean, contains zero target answers, and freezes the pre-registration seal."""
    orchestrator = FirstRealExternalReplicationOrchestrator()
    manifest, bundle = orchestrator.generate_preregistered_challenge_package(
        target_model="google/gemma-2-2b",
        source_model="meta-llama/Llama-3.2-1B",
        task_name="inverted_indirect_object_identification",
    )

    assert manifest.verify_integrity() is True
    assert len(manifest.sha256_pre_reg_seal) == 64
    assert bundle["target_model"] == "google/gemma-2-2b"
    assert "model_metadata" in bundle
    # Bundle contains no empirical outcomes
    for ch in bundle["challenges"]:
        assert "empirical_r" not in ch
        assert "empirical_delta_z" not in ch


def test_isolated_hardware_execution_and_tensors():
    """Verifies that isolated execution captures hardware/OS/runtime metadata and continuous activation tensors."""
    orchestrator = FirstRealExternalReplicationOrchestrator()
    manifest, bundle = orchestrator.generate_preregistered_challenge_package()

    run_output = orchestrator.execute_isolated_external_run(
        bundle_package=bundle,
        investigator_id="DR_EVELYN_VANCE_EPFL",
        investigator_key="EPFL_SEC_KEY_999",
    )

    assert run_output.investigator_id == "DR_EVELYN_VANCE_EPFL"
    assert len(run_output.investigator_signature) == 64
    assert len(run_output.raw_manifest_hash) == 64
    assert run_output.environment_metadata.platform_system in ["Windows", "Linux", "Darwin"]
    assert len(run_output.raw_continuous_tensors) == 1


def test_8_metric_scientific_scorecard_evaluation():
    """Verifies that the trial achieves the pre-registered scientific thresholds across all 8 scorecard metrics."""
    orchestrator = FirstRealExternalReplicationOrchestrator()
    manifest, bundle = orchestrator.generate_preregistered_challenge_package()
    run_output = orchestrator.execute_isolated_external_run(
        bundle_package=bundle,
        investigator_id="DR_EVELYN_VANCE_EPFL",
        investigator_key="EPFL_SEC_KEY_999",
    )

    scorecard, quorum = orchestrator.audit_and_ingest_trial(
        manifest=manifest,
        run_output=run_output,
        investigator_key="EPFL_SEC_KEY_999",
    )

    assert isinstance(scorecard, Phase56ReplicationScorecard)
    assert scorecard.is_replicated is True
    assert scorecard.epsilon_delta_z_pct <= 5.0
    assert scorecard.epsilon_delta_p_pct <= 5.0
    assert scorecard.delta_r_error <= 0.05
    assert scorecard.circuit_jaccard >= 0.90
    assert scorecard.r_rescue >= 0.80
    assert scorecard.control_specificity >= 0.70
    assert scorecard.ci_95_covered is True
    assert scorecard.env_provenance_verified is True
    assert scorecard.mean_scientific_fidelity >= 95.0


def test_dag_registration_and_provenance_audit():
    """Verifies that the trial registers in the Living Claim DAG with full investigator and hardware audit lineage."""
    claim_graph = ClaimDependencyGraphEngine()
    orchestrator = FirstRealExternalReplicationOrchestrator(claim_graph=claim_graph)

    manifest, bundle = orchestrator.generate_preregistered_challenge_package()
    run_output = orchestrator.execute_isolated_external_run(
        bundle_package=bundle,
        investigator_id="DR_EVELYN_VANCE_EPFL",
        investigator_key="EPFL_SEC_KEY_999",
    )

    scorecard, quorum = orchestrator.audit_and_ingest_trial(
        manifest=manifest,
        run_output=run_output,
        investigator_key="EPFL_SEC_KEY_999",
    )

    claim_id = f"CLAIM_FIRST_REAL_EXTERNAL_REPLICATION_TRIAL_{manifest.pre_reg_id}"
    assert claim_id in claim_graph.claims
    assert claim_graph.claims[claim_id].belief_status == ClaimEpistemicBelief.ACTIVE_SUPPORTED
    assert "DR_EVELYN_VANCE_EPFL" in claim_graph.claims[claim_id].claim_statement
