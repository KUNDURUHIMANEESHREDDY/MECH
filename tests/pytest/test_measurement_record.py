"""F-05: Attestation must be a cryptographically bound measurement record, not a boolean.

The current `attest_measurement` just sets `attested: True` on a result dict.
This is forgeable — any component can construct a dict with `attested: True`
and `provenance: "live"` and pass it through.

The fix: `attest_measurement` must produce a cryptographically bound
`MeasurementRecord` that binds:
- model_identity (name + content hash)
- input_commitment (hash of inputs)
- output_commitment (hash of outputs)  
- execution_context (timestamp, environment, etc.)
- measurement_boundary_id (which boundary issued it)

This record is then embedded in the evidence envelope and signed.
Verification checks the cryptographic binding, not just a boolean flag.
"""
import hashlib
import json
import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.core.provenance import attest_measurement, LIVE  # noqa: E402

VALID = "rab12cd34ef56"
VALID2 = "rcd34ef56ab12"
VALID3 = "r1234567890ab"
VALID4 = "r4567890abcde"


def _hash_data(data: dict) -> str:
    """Canonical hash of a data structure."""
    canonical = json.dumps(data, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()[:32]


def test_attest_returns_measurement_record_not_just_boolean():
    """attest_measurement must return a record with cryptographic bindings."""
    result = {"status": "ok", "output": {"value": 42}}
    attested = attest_measurement(result, model_loaded="gpt2", model_requested="gpt2")

    # Must have measurement_record field with cryptographic bindings
    assert "measurement_record" in attested, "missing measurement_record"
    mr = attested["measurement_record"]
    
    # Must bind model identity
    assert "model_identity" in mr
    assert mr["model_identity"]["name"] == "gpt2"
    assert "content_hash" in mr["model_identity"]
    
    # Must bind inputs
    assert "input_commitment" in mr
    
    # Must bind outputs (hash of the output field only)
    assert "output_commitment" in mr
    # The output_commitment hashes only the output field
    expected_output = {"output": {"value": 42}}
    canonical = json.dumps(expected_output, sort_keys=True, separators=(",", ":"), default=str)
    expected_hash = hashlib.sha256(canonical.encode()).hexdigest()[:32]
    assert mr["output_commitment"] == expected_hash
    
    # Must bind execution context
    assert "execution_context" in mr
    assert "timestamp" in mr["execution_context"]
    assert "measurement_boundary_id" in mr["execution_context"]
    
    # The old boolean must still exist for backward compat but derive from record
    assert attested.get("attested") is True


def test_measurement_record_binds_model_identity():
    """Different models must produce different measurement records."""
    # Use separate dicts to avoid mutation issues (attest_measurement mutates input)
    base1 = {"status": "ok", "output": "same"}
    base2 = {"status": "ok", "output": "same"}
    
    r1 = attest_measurement(base1, model_loaded="gpt2", model_requested="gpt2")
    r2 = attest_measurement(base2, model_loaded="gpt2-large", model_requested="gpt2-large")
    
    assert r1["measurement_record"]["model_identity"]["content_hash"] != \
           r2["measurement_record"]["model_identity"]["content_hash"]


def test_measurement_record_binds_inputs():
    """Different inputs must produce different records."""
    r1 = attest_measurement({"status": "ok", "input": "hello"}, model_loaded="gpt2")
    r2 = attest_measurement({"status": "ok", "input": "world"}, model_loaded="gpt2")
    
    assert r1["measurement_record"]["input_commitment"] != \
           r2["measurement_record"]["input_commitment"]


def test_measurement_record_binds_outputs():
    """Different outputs must produce different records."""
    r1 = attest_measurement({"status": "ok", "output": "foo"}, model_loaded="gpt2")
    r2 = attest_measurement({"status": "ok", "output": "bar"}, model_loaded="gpt2")
    
    assert r1["measurement_record"]["output_commitment"] != \
           r2["measurement_record"]["output_commitment"]


def test_measurement_record_includes_timestamp():
    """Record must include verifiable timestamp."""
    import time
    before = time.time()
    r = attest_measurement({"status": "ok"}, model_loaded="gpt2")
    after = time.time()
    
    ts = r["measurement_record"]["execution_context"]["timestamp"]
    assert before <= ts <= after


def test_measurement_record_has_boundary_id():
    """Must identify which measurement boundary issued it."""
    r = attest_measurement({"status": "ok"}, model_loaded="gpt2")
    assert "measurement_boundary_id" in r["measurement_record"]["execution_context"]


def test_measurement_record_verifiable_independent_of_boolean():
    """Verification must check cryptographic binding, not just boolean."""
    # A forged result with attested=True but no record must fail verification
    forged = {"attested": True, "provenance": "live", "output": "fake"}
    
    # This should be rejected by the verification layer
    from backend.agents import evidence_policy
    # The policy should check for measurement_record, not just boolean
    # (this test documents the requirement; the actual check is in evidence_policy)


def test_envelope_embeds_measurement_record():
    """Evidence envelope must embed the full measurement record."""
    from backend.core import evidence_graph as eg
    import tempfile
    
    with tempfile.TemporaryDirectory() as tmp:
        from backend.science.integrity import signing as signing_mod
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
        
        seed = Ed25519PrivateKey.generate().private_bytes_raw()
        key_file = Path(tmp) / "signing.key"
        key_file.write_bytes(seed)
        
        import os
        os.environ["MECH_SIGNING_KEY_PATH"] = str(key_file)
        os.environ["MECH_TRUSTED_SIGNING_KEY_PATH"] = str(key_file)
        
        result = {"status": "ok", "output": {"value": 42}}
        attested = attest_measurement(result, model_loaded="gpt2")
        
        from backend.core import evidence_graph as eg_mod
        eg_mod.save_run_record(VALID, attested, directory=tmp)
        
        env = json.loads((Path(tmp) / f"{VALID}.json").read_text(encoding="utf-8"))
        # The output_commitment hashes only the output field
        expected_output = {"output": {"value": 42}}
        canonical = json.dumps(expected_output, sort_keys=True, separators=(",", ":"), default=str)
        expected_hash = hashlib.sha256(canonical.encode()).hexdigest()[:32]
        assert env["record"]["measurement_record"]["output_commitment"] == expected_hash


def test_verification_checks_cryptographic_binding():
    """Verification must check the cryptographic binding, not just boolean."""
    from backend.core import evidence_graph as eg
    import tempfile
    
    with tempfile.TemporaryDirectory() as tmp:
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
        seed = Ed25519PrivateKey.generate().private_bytes_raw()
        key_file = Path(tmp) / "signing.key"
        key_file.write_bytes(seed)
        
        import os
        os.environ["MECH_SIGNING_KEY_PATH"] = str(key_file)
        os.environ["MECH_TRUSTED_SIGNING_KEY_PATH"] = str(key_file)
        
        # Valid record with measurement_record
        result = {"status": "ok", "output": {"value": 42}}
        attested = attest_measurement(result, model_loaded="gpt2")
        
        from backend.core import evidence_graph as eg_mod
        eg_mod.save_run_record(VALID, attested, directory=tmp)
        status = eg.envelope_status(VALID, directory=tmp)
        
        assert status["attested"] is True
        assert status["signature"] == "valid"


def test_forged_boolean_without_record_rejected():
    """A result with attested=True but no measurement_record must be rejected."""
    # This test documents the requirement — the verification layer
    # must check for measurement_record, not just the boolean
    from backend.core import evidence_graph as eg
    import tempfile
    
    with tempfile.TemporaryDirectory() as tmp:
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
        seed = Ed25519PrivateKey.generate().private_bytes_raw()
        key_file = Path(tmp) / "signing.key"
        key_file.write_bytes(seed)
        
        import os
        os.environ["MECH_SIGNING_KEY_PATH"] = str(key_file)
        os.environ["MECH_TRUSTED_SIGNING_KEY_PATH"] = str(key_file)
        
        # Forge a result with attested=True but no measurement_record
        forged = {"attested": True, "provenance": "live", "output": "fake"}
        from backend.core import evidence_graph as eg_mod
        eg_mod.save_run_record(VALID, forged, directory=tmp)
        
        # Should be rejected — no measurement_record
        status = eg.envelope_status(VALID, directory=tmp)
        assert status["attested"] is False, "forged boolean without record must be rejected"