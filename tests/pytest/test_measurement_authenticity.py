"""P1: Attestation checks must establish cryptographic authenticity.

A structurally plausible record (attested=True plus four binding fields)
must NOT satisfy the evidence gate. Verification must prove:
1. The record carries a valid Ed25519 signature (`signature_valid`).
2. The signing key is in the configured trusted set (`key_trusted`).
   A self-signed record from an untrusted key is a forgery, not evidence.
3. The model content hash is derived from actual weight artifacts, not
   from hashing the model name (`content_hash_source` must be
   "model_weights" or caller-supplied "explicit", never
   "model_name_fallback").
4. The input/output commitments match the payload they travel with:
   altering an output after attestation must fail verification.
5. Caller-supplied attestation fields are never trusted: `withhold`
   strips forged records, and `attest_measurement` overwrites them.
"""
import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.agents.evidence_policy import _is_attested  # noqa: E402
from backend.core.provenance import (  # noqa: E402
    attest_measurement,
    reset_trusted_key_ids_cache,
    verify_measurement_record,
    verify_payload_commitments,
    withhold,
)


def _make_key_file(directory: str) -> str:
    from cryptography.hazmat.primitives.asymmetric.ed25519 import (
        Ed25519PrivateKey,
    )

    seed = Ed25519PrivateKey.generate().private_bytes_raw()
    key_file = Path(directory) / f"signing_{os.urandom(4).hex()}.key"
    key_file.write_bytes(seed)
    return str(key_file)


@pytest.fixture
def two_keys(tmp_path, monkeypatch):
    """Two independent signing keys; only the second is trusted."""
    untrusted = _make_key_file(str(tmp_path))
    trusted = _make_key_file(str(tmp_path))
    monkeypatch.setenv("MECH_SIGNING_KEY_PATH", untrusted)
    monkeypatch.setenv("MECH_TRUSTED_SIGNING_KEY_PATH", trusted)
    reset_trusted_key_ids_cache()
    yield {"untrusted": untrusted, "trusted": trusted}
    reset_trusted_key_ids_cache()


@pytest.fixture
def one_key(tmp_path, monkeypatch):
    """A single key used for both signing and trust."""
    key = _make_key_file(str(tmp_path))
    monkeypatch.setenv("MECH_SIGNING_KEY_PATH", key)
    monkeypatch.setenv("MECH_TRUSTED_SIGNING_KEY_PATH", key)
    reset_trusted_key_ids_cache()
    yield key
    reset_trusted_key_ids_cache()


def _genuine(signing_key: str, **overrides):
    """A genuinely measured, weight-bound, signed payload."""
    base = {"status": "ok", "output": {"value": 42}}
    out = attest_measurement(
        base,
        model_loaded="gpt2",
        model_requested="gpt2",
        # Fingerprint of actually-loaded weight artifacts, supplied by the
        # measurement layer (here a fixed test value standing in for one).
        model_content_hash="sha256:test-weight-fingerprint",
        signing_key=signing_key,
    )
    out.update(overrides)
    return out


# ── 1. bare boolean ──────────────────────────────────────────────────────

def test_forged_boolean_without_record_fails():
    forged = {"attested": True, "provenance": "live", "output": "fake"}
    assert _is_attested(forged) is False


# ── 2. self-signed with an untrusted key ─────────────────────────────────

def test_self_signed_untrusted_key_fails(two_keys):
    """A valid signature under a key nobody trusts is still a forgery."""
    out = attest_measurement(
        {"status": "ok", "output": {"value": 42}},
        model_loaded="gpt2",
        model_content_hash="sha256:test-weight-fingerprint",
        signing_key=two_keys["untrusted"],
    )
    mr = out["measurement_record"]
    # The signature itself verifies (it was really made by that key)...
    from backend.core.provenance import load_trusted_key_ids

    verification = verify_measurement_record(mr, load_trusted_key_ids())
    assert verification["checks"]["signature_valid"] is True
    assert verification["checks"]["key_trusted"] is False
    # ...but the gate must refuse it.
    assert _is_attested(out) is False


# ── 3. tampered signature ────────────────────────────────────────────────

def test_tampered_signature_fails(one_key):
    out = _genuine(one_key)
    out["measurement_record"] = dict(
        out["measurement_record"], signature="00" * 64
    )
    assert _is_attested(out) is False


# ── 4. output altered after attestation ──────────────────────────────────

def test_altered_output_after_attestation_fails(one_key):
    """Commitments must be recomputed against the payload they travel with."""
    out = _genuine(one_key)
    assert _is_attested(out) is True
    out["output"] = {"value": 43}
    assert _is_attested(out) is False
    assert verify_payload_commitments(out)["valid"] is False


def test_altered_input_after_attestation_fails(one_key):
    out = attest_measurement(
        {"status": "ok", "input": "hello", "output": "world"},
        model_loaded="gpt2",
        model_content_hash="sha256:test-weight-fingerprint",
        signing_key=one_key,
    )
    assert _is_attested(out) is True
    out["input"] = "goodbye"
    assert _is_attested(out) is False


# ── 5. substituted model identity ────────────────────────────────────────

def test_substituted_model_identity_fails(one_key):
    """Rewriting who measured it breaks the signature."""
    out = _genuine(one_key)
    assert _is_attested(out) is True
    mr = dict(out["measurement_record"])
    mr["model_identity"] = {"name": "gpt2-large", "content_hash": "f" * 32,
                            "content_hash_source": "explicit"}
    out["measurement_record"] = mr
    assert _is_attested(out) is False


# ── 6. name-derived content hash ─────────────────────────────────────────

def test_name_derived_content_hash_rejected_even_when_signed(one_key):
    """Hashing the model NAME proves nothing about the weights."""
    out = attest_measurement(
        {"status": "ok", "output": {"value": 42}},
        model_loaded="gpt2",
        signing_key=one_key,
    )
    mr = out["measurement_record"]
    assert mr["model_identity"]["content_hash_source"] == "model_name_fallback"
    verification = verify_measurement_record(
        mr, {"ed25519:" + "0" * 16} | {mr.get("key_id")},
    )
    assert verification["checks"]["signature_valid"] is True
    assert _is_attested(out) is False


# ── 7. genuine record passes (guards overcorrection) ─────────────────────

def test_genuine_weight_bound_record_passes(one_key):
    out = _genuine(one_key)
    assert _is_attested(out) is True


def test_adding_non_basis_fields_keeps_validity(one_key):
    """Post-attest additions (eligibility, verdicts) must not break the
    binding — commitments cover only their recorded basis keys."""
    out = _genuine(one_key)
    out.update({"validation_eligible": True, "publication_eligible": True,
                "validated": True})
    assert _is_attested(out) is True


# ── 8/9. caller-supplied attestation fields are never trusted ────────────

def test_withhold_strips_forged_measurement_record():
    forged_mr = {
        "model_identity": {"name": "gpt2", "content_hash": "a" * 32},
        "input_commitment": "b" * 32,
        "output_commitment": "c" * 32,
        "execution_context": {"measurement_boundary_id": "d00d"},
    }
    out = withhold({"attested": True, "measurement_record": forged_mr,
                    "output": "fake"}, reason="r")
    assert out.get("attested") is False
    assert "measurement_record" not in out


def test_attest_overwrites_caller_supplied_record(one_key):
    forged_mr = {
        "model_identity": {"name": "evil", "content_hash": "a" * 32},
        "input_commitment": "b" * 32,
        "output_commitment": "c" * 32,
        "execution_context": {"measurement_boundary_id": "d00d"},
    }
    out = attest_measurement(
        {"status": "ok", "output": {"value": 1},
         "measurement_record": forged_mr},
        model_loaded="gpt2",
        model_content_hash="sha256:test-weight-fingerprint",
        signing_key=one_key,
    )
    assert out["measurement_record"] != forged_mr
    assert out["measurement_record"]["model_identity"]["name"] == "gpt2"
    assert _is_attested(out) is True


# ── basis is recorded for audit ──────────────────────────────────────────

def test_commitment_basis_keys_are_recorded(one_key):
    out = attest_measurement(
        {"status": "ok", "input": "hello", "output": "world"},
        model_loaded="gpt2",
        model_content_hash="sha256:test-weight-fingerprint",
        signing_key=one_key,
        input_fields=["input"],
        output_fields=["output"],
    )
    mr = out["measurement_record"]
    assert mr["input_keys"] == ["input"]
    assert mr["output_keys"] == ["output"]
    assert _is_attested(out) is True
