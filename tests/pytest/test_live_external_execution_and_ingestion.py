"""Unit and integration tests for Phase 55: Live Open-World External Execution & Ingestion Workflow."""

import pytest

from backend.discovery.claim_dependency_graph import ClaimDependencyGraphEngine, ClaimEpistemicBelief
from backend.discovery.external_runner_cli import (
    ExternalExperimentRunOutput,
    HardwareEnvironmentMetadata,
    execute_external_challenge_bundle,
)
from backend.discovery.open_world_ingestion_pipeline import LiveObservationIngestionPipeline
from backend.discovery.open_world_multi_lab_preregistration_engine import (
    MetaScientificRegime,
    OpenWorldAuditorQuorum,
    OpenWorldMultiLabEngine,
    PreRegistrationManifest,
)


def test_external_runner_cli_bundle_execution():
    """Verifies that the standalone CLI runner executes challenge bundles and captures hardware metadata and raw tensors."""
    bundle_data = {
        "bundle_id": "BUNDLE_EXTERNAL_CHALLENGE_001",
        "challenges": [
            {
                "challenge_id": "CHALLENGE_LLAMA3_INDUCTION",
                "functional_role_similarity": 0.90,
            },
            {
                "challenge_id": "CHALLENGE_MAMBA_SSM",
                "is_ssm": True,
            },
            {
                "challenge_id": "CHALLENGE_STARCODER_POLY",
                "is_negative_transfer": True,
            },
        ],
    }

    run_output = execute_external_challenge_bundle(
        bundle_data=bundle_data,
        investigator_id="EXTERNAL_RESEARCHER_ALICE",
        investigator_private_key="ALICE_SECRET_KEY_123",
    )

    assert isinstance(run_output, ExternalExperimentRunOutput)
    assert run_output.investigator_id == "EXTERNAL_RESEARCHER_ALICE"
    assert len(run_output.challenge_results) == 3
    assert len(run_output.raw_continuous_tensors) == 3
    assert len(run_output.raw_manifest_hash) == 64
    assert len(run_output.investigator_signature) == 64
    assert run_output.environment_metadata.platform_system in ["Windows", "Linux", "Darwin"]


def test_ingestion_signature_and_prereg_validation():
    """Verifies that the ingestion pipeline validates investigator signatures and rejects tampered manifests."""
    meta_engine = OpenWorldMultiLabEngine()
    manifest = meta_engine.create_preregistration_manifest(
        claim_statement="Causal transportability across diverse architectures.",
        models=["Llama-3.1-8B", "StarCoder-7B"],
        tasks=["Induction", "Distractor"],
        predicted_rescues=[0.81, 0.125],
    )

    bundle_data = {
        "bundle_id": "BUNDLE_VALIDATION",
        "challenges": [
            {"challenge_id": "CHALLENGE_LLAMA3", "functional_role_similarity": 0.85},
        ],
    }

    run_output = execute_external_challenge_bundle(
        bundle_data=bundle_data,
        investigator_id="RESEARCHER_BOB",
        investigator_private_key="BOB_SECRET_KEY",
    )

    pipeline = LiveObservationIngestionPipeline()

    # Valid ingestion
    quorum = pipeline.verify_and_ingest_external_run(
        run_output=run_output,
        manifest=manifest,
        expected_investigator_key="BOB_SECRET_KEY",
    )
    assert isinstance(quorum, OpenWorldAuditorQuorum)

    # Ingestion fails if key or signature is invalid
    with pytest.raises(ValueError, match="Invalid investigator signature"):
        pipeline.verify_and_ingest_external_run(
            run_output=run_output,
            manifest=manifest,
            expected_investigator_key="WRONG_SECRET_KEY",
        )


def test_end_to_end_hierarchical_unsealing_and_scoring():
    """Verifies that live observation ingestion performs Hierarchical Bayesian Meta-Analysis and assigns regime."""
    meta_engine = OpenWorldMultiLabEngine()
    manifest = meta_engine.create_preregistration_manifest(
        claim_statement="Live external execution meta-analysis validation.",
        models=["Llama-3.1-8B"],
        tasks=["Induction"],
        predicted_rescues=[0.825],
    )

    bundle_data = {
        "bundle_id": "BUNDLE_DENSE_INDUCTION",
        "challenges": [
            {"challenge_id": "CHALLENGE_1", "functional_role_similarity": 0.90},
            {"challenge_id": "CHALLENGE_2", "functional_role_similarity": 0.88},
        ],
    }

    run_output = execute_external_challenge_bundle(
        bundle_data=bundle_data,
        investigator_id="LAB_CAMBRIDGE",
        investigator_private_key="CAMBRIDGE_KEY",
    )

    pipeline = LiveObservationIngestionPipeline()
    quorum = pipeline.verify_and_ingest_external_run(
        run_output=run_output,
        manifest=manifest,
        expected_investigator_key="CAMBRIDGE_KEY",
    )

    assert quorum.meta_analysis.posterior_belief_h >= 0.95
    assert quorum.meta_analysis.regime == MetaScientificRegime.CONSENSUS
    assert quorum.meta_analysis.is_verified is True


def test_living_claim_dag_provenance_audit():
    """Verifies that ingested runs enrich the Living Claim DAG with full hardware/environment provenance."""
    claim_graph = ClaimDependencyGraphEngine()
    pipeline = LiveObservationIngestionPipeline(claim_graph=claim_graph)

    manifest = pipeline.meta_engine.create_preregistration_manifest(
        claim_statement="Hardware provenance audit validation.",
        models=["Llama-3.1-8B"],
        tasks=["Induction"],
        predicted_rescues=[0.81],
    )

    bundle_data = {
        "bundle_id": "BUNDLE_PROVENANCE",
        "challenges": [{"challenge_id": "CHALLENGE_A", "functional_role_similarity": 0.85}],
    }

    run_output = execute_external_challenge_bundle(
        bundle_data=bundle_data,
        investigator_id="LAB_STANFORD_LIVE",
        investigator_private_key="STANFORD_KEY",
    )

    pipeline.verify_and_ingest_external_run(
        run_output=run_output,
        manifest=manifest,
        expected_investigator_key="STANFORD_KEY",
    )

    claim_id = f"CLAIM_OPEN_WORLD_PREREG_{manifest.pre_reg_id}"
    assert claim_id in claim_graph.claims
    assert claim_graph.claims[claim_id].belief_status == ClaimEpistemicBelief.ACTIVE_SUPPORTED
    assert "Provenance: LAB_STANFORD_LIVE" in claim_graph.claims[claim_id].claim_statement
