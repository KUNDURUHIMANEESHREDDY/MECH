import pytest
from backend.science.scientific_envelope import (
    wrap_scientific_success,
    wrap_unexecuted_failure,
    wrap_tamper_failure,
    compute_manifest_hash,
    verify_manifest_integrity,
)
from backend.storage.scientific_entities import KnowledgeType


def test_scientific_response_envelope_success_and_hash():
    payload = {"delta_logit": 3.42, "component": "L9H9", "effect": "POSITIVE"}
    env = wrap_scientific_success(
        data=payload,
        manifest_id="man_test_123",
        manifest_payload=payload,
        knowledge_type=KnowledgeType.CAUSAL_EVIDENCE,
    )

    assert env.status == "SUCCESS"
    assert env.integrity_status == "VERIFIED"
    assert env.manifest_sha256 != ""
    assert env.provenance.manifest_sha256 == env.manifest_sha256

    # Verify manifest integrity
    is_valid = verify_manifest_integrity(payload, env.manifest_sha256)
    assert is_valid is True


def test_scientific_response_envelope_tamper_detection():
    payload = {"delta_logit": 3.42, "component": "L9H9"}
    original_hash = compute_manifest_hash(payload)

    # Tampered payload
    tampered_payload = {"delta_logit": 9.99, "component": "L9H9"}
    is_valid = verify_manifest_integrity(tampered_payload, original_hash)
    assert is_valid is False


def test_unexecuted_experiment_fail_closed():
    env = wrap_unexecuted_failure("Model forward pass was not executed on live PyTorch weights.")
    assert env.status == "UNEXECUTED_EXPERIMENT"
    assert env.integrity_status == "UNVERIFIED"
    assert env.data is None
    assert "Epistemic Gate Violation" in env.error
