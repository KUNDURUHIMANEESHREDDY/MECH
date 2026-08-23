"""Unit and integration tests for Phase 54: Real Open-World Multi-Lab Replication & Pre-Registration Engine."""

import pytest

from backend.discovery.claim_dependency_graph import ClaimDependencyGraphEngine, ClaimEpistemicBelief
from backend.discovery.open_world_multi_lab_preregistration_engine import (
    HierarchicalMetaAnalysisResult,
    MetaScientificRegime,
    OpenWorldAuditorQuorum,
    OpenWorldMultiLabEngine,
    PreRegistrationManifest,
    VarianceDecomposition,
)


def test_cryptographic_preregistration_manifest_freeze():
    """Verifies that the pre-registration manifest is cryptographically sealed and detects any tampering."""
    engine = OpenWorldMultiLabEngine()
    manifest = engine.create_preregistration_manifest(
        claim_statement="Causal transportability law holds across dense transformer architectures.",
        models=["Llama-3.1-8B", "Qwen-2.5-7B"],
        tasks=["Multi-Hop Induction", "Greater-Than Arithmetic"],
        predicted_rescues=[0.82, 0.835],
    )

    assert isinstance(manifest, PreRegistrationManifest)
    assert len(manifest.sha256_pre_reg_seal) == 64
    assert manifest.verify_integrity() is True

    # Tampering with models triggers validation failure
    tampered_manifest = PreRegistrationManifest(
        pre_reg_id=manifest.pre_reg_id,
        claim_hash=manifest.claim_hash,
        protocol_hash=manifest.protocol_hash,
        model_hashes=manifest.model_hashes + ["tampered_hash"],
        task_hashes=manifest.task_hashes,
        prediction_hashes=manifest.prediction_hashes,
        analysis_plan_hash=manifest.analysis_plan_hash,
        sha256_pre_reg_seal=manifest.sha256_pre_reg_seal,
        timestamp_utc=manifest.timestamp_utc,
    )
    assert tampered_manifest.verify_integrity() is False


def test_hierarchical_bayesian_variance_decomposition():
    """Verifies that Hierarchical Bayesian Meta-Analysis computes P(H|E) >= 0.95 and valid variance decomposition."""
    engine = OpenWorldMultiLabEngine()
    manifest = engine.create_preregistration_manifest(
        claim_statement="Dense transformer causal transportability verification.",
        models=["Llama-3.1-8B", "Qwen-2.5-7B"],
        tasks=["Induction", "Arithmetic"],
        predicted_rescues=[0.81, 0.83],
    )

    observations = [
        {"lab_id": "LAB_BERKELEY", "model": "Llama-3.1-8B", "empirical_r": 0.810},
        {"lab_id": "LAB_OXFORD", "model": "Qwen-2.5-7B", "empirical_r": 0.830},
        {"lab_id": "LAB_STANFORD", "model": "Llama-3.1-8B", "empirical_r": 0.812},
    ]

    res = engine.execute_hierarchical_meta_analysis(manifest, observations)

    assert isinstance(res, HierarchicalMetaAnalysisResult)
    assert res.posterior_belief_h >= 0.95
    assert res.is_verified is True
    assert res.variance_decomp.total_variance <= 0.05
    assert res.credible_interval_95[0] < res.mean_effect_size < res.credible_interval_95[1]
    assert res.regime == MetaScientificRegime.CONSENSUS


def test_5_regime_metascientific_classification():
    """Verifies that distinct observation profiles trigger CONSENSUS, NEGATIVE_TRANSFER, and UNRESOLVED."""
    engine = OpenWorldMultiLabEngine()
    manifest = engine.create_preregistration_manifest(
        claim_statement="Multi-regime scientific evaluation.",
        models=["Llama-3.1-8B", "StarCoder-7B", "Mamba-2-SSM"],
        tasks=["Induction", "Distractor", "IOI"],
        predicted_rescues=[0.81, 0.13, 0.00],
    )

    # Negative Transfer Case
    obs_neg = [
        {"lab_id": "LAB_STANFORD", "model": "StarCoder-7B", "empirical_r": 0.130, "is_negative_transfer": True},
    ]
    res_neg = engine.execute_hierarchical_meta_analysis(manifest, obs_neg)
    assert res_neg.regime == MetaScientificRegime.NEGATIVE_TRANSFER

    # Unresolved / Abstention Case
    obs_unresolved = [
        {"lab_id": "LAB_STANFORD", "model": "Mamba-2-SSM", "is_abstained": True},
    ]
    res_unresolved = engine.execute_hierarchical_meta_analysis(manifest, obs_unresolved)
    assert res_unresolved.regime == MetaScientificRegime.UNRESOLVED


def test_open_world_audit_and_dag_certification():
    """Verifies that the Independent Auditor Quorum signs the certificate and registers it in the Claim DAG."""
    claim_graph = ClaimDependencyGraphEngine()
    engine = OpenWorldMultiLabEngine(claim_graph=claim_graph)

    manifest = engine.create_preregistration_manifest(
        claim_statement="Pre-registered open-world multi-lab audit.",
        models=["Llama-3.1-8B", "Qwen-2.5-7B"],
        tasks=["Induction", "Arithmetic"],
        predicted_rescues=[0.81, 0.83],
    )

    observations = [
        {"lab_id": "LAB_BERKELEY", "model": "Llama-3.1-8B", "empirical_r": 0.810},
        {"lab_id": "LAB_OXFORD", "model": "Qwen-2.5-7B", "empirical_r": 0.830},
    ]

    quorum = engine.run_open_world_auditor_quorum(manifest, observations)

    assert isinstance(quorum, OpenWorldAuditorQuorum)
    assert quorum.is_pre_reg_intact is True
    assert len(quorum.sha256_audit_seal) == 64
    assert len(quorum.auditor_signature) == 64

    claim_id = f"CLAIM_OPEN_WORLD_PREREG_{manifest.pre_reg_id}"
    assert claim_id in claim_graph.claims
    assert claim_graph.claims[claim_id].belief_status == ClaimEpistemicBelief.ACTIVE_SUPPORTED
    assert "Open-World Pre-Registered Multi-Lab Replication Certified" in claim_graph.claims[claim_id].claim_statement
