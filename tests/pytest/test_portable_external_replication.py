"""Unit and integration tests for Phase 52: Portable Standalone External Replication Package & Protocol."""

import pytest

from backend.discovery.claim_dependency_graph import ClaimDependencyGraphEngine, ClaimEpistemicBelief
from backend.discovery.portable_external_replication_package import (
    EpistemicValidationTier,
    PortableChallengeBundle,
    PortableChallengeSpec,
    PortablePredictionSubmission,
    PortableReplicationAuditLedger,
    PortableReplicationEngine,
    PortableReplicationEvaluation,
)


def test_portable_package_schema_and_signature_validation():
    """Verifies that external challenge bundles are properly formatted, signed, and hide empirical truth in public export."""
    bundle = PortableReplicationEngine.create_canonical_external_bundle(lab_id="LAB_OXFORD_ALIGNMENT")
    pub_dict = bundle.to_public_dict()

    assert pub_dict["origin_lab_id"] == "LAB_OXFORD_ALIGNMENT"
    assert len(pub_dict["challenges"]) == 4
    for ch in pub_dict["challenges"]:
        assert "challenge_id" in ch
        assert "probe_role" in ch
        assert "empirical_true_r" not in ch  # Zero knowledge isolation


def test_zero_knowledge_prediction_submission_and_tamper_rejection():
    """Verifies that predictions are signed and tampering with predictions raises a validation error."""
    bundle = PortableReplicationEngine.create_canonical_external_bundle(lab_id="LAB_STANFORD_HAI")
    engine = PortableReplicationEngine()

    submission = engine.generate_predictions_for_bundle(bundle.to_public_dict())
    assert len(submission.predictions) == 4
    assert len(submission.mech_signature) == 64

    # Legitimate verification works
    ledger = engine.verify_and_audit_unsealed_bundle(bundle, submission)
    assert ledger.is_overall_certified is True

    # Tampering with prediction payload raises ValueError
    tampered_submission = PortablePredictionSubmission(
        bundle_id=submission.bundle_id,
        predictions=[{**p, "predicted_r": 0.99} for p in submission.predictions],
        mech_public_key=submission.mech_public_key,
        mech_signature=submission.mech_signature,  # Signature now invalid for altered payload
        timestamp_utc=submission.timestamp_utc,
    )
    with pytest.raises(ValueError, match="Tampering detected"):
        engine.verify_and_audit_unsealed_bundle(bundle, tampered_submission)


def test_portable_unsealing_and_prf_scoring():
    """Verifies that unsealing achieves PRF >= 95.0%, relative error <= 5.0%, and 100% abstention accuracy."""
    bundle = PortableReplicationEngine.create_canonical_external_bundle(lab_id="LAB_MIT_CSAIL")
    engine = PortableReplicationEngine()

    submission = engine.generate_predictions_for_bundle(bundle.to_public_dict())
    ledger = engine.verify_and_audit_unsealed_bundle(bundle, submission)

    assert isinstance(ledger, PortableReplicationAuditLedger)
    assert ledger.prf_score >= 0.95
    assert ledger.mean_relative_error_pct <= 5.0
    assert ledger.abstention_accuracy_pct == 100.0
    assert len(ledger.dual_signature_quorum) == 2


def test_epistemic_hierarchy_and_dag_certification():
    """Verifies that the claim is certified strictly under the PORTABLE_EXTERNAL_LAB_REPLICATED tier in the Claim DAG."""
    claim_graph = ClaimDependencyGraphEngine()
    engine = PortableReplicationEngine(claim_graph=claim_graph)
    bundle = PortableReplicationEngine.create_canonical_external_bundle(lab_id="LAB_CAMBRIDGE_CSI")

    submission = engine.generate_predictions_for_bundle(bundle.to_public_dict())
    ledger = engine.verify_and_audit_unsealed_bundle(bundle, submission)

    assert ledger.epistemic_tier == EpistemicValidationTier.PORTABLE_EXTERNAL_LAB_REPLICATED

    claim_id = f"CLAIM_PORTABLE_REPLICATION_{bundle.bundle_id}"
    assert claim_id in claim_graph.claims
    assert claim_graph.claims[claim_id].belief_status == ClaimEpistemicBelief.ACTIVE_SUPPORTED
    assert "PORTABLE_EXTERNAL_LAB_REPLICATED" in claim_graph.claims[claim_id].claim_statement
