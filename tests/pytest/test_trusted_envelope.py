"""F-03: Envelope must be signed by a TRUSTED key, not just any key.

The current `envelope_status` verifies the Ed25519 signature using the public key
embedded in the envelope. But ANY key pair can sign its own envelope perfectly.
A malicious actor can generate a key pair, sign a fabricated envelope, and put
their own public key in it — the signature will verify perfectly.

The fix: enforce a TRUSTED signing key identity. The system must be configured
with a trusted signing key (via env var or config), and `envelope_status` must
compare the envelope's `key_id` against the configured trusted key(s).
Only envelopes signed by a trusted key get `attested: True`.
"""
import json
import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.core import evidence_graph as eg  # noqa: E402


@pytest.fixture(autouse=True)
def _reset_trusted_key_cache():
    """Reset the trusted key IDs cache before each test."""
    eg._reset_trusted_key_ids_cache()
    yield
    eg._reset_trusted_key_ids_cache()


def _seed_key_file(tmp_path):
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    seed = Ed25519PrivateKey.generate().private_bytes_raw()
    key_file = tmp_path / "signing.key"
    key_file.write_bytes(seed)
    return str(key_file)


def _trusted_key_file(tmp_path):
    """A different key file representing the TRUSTED signing key."""
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    seed = Ed25519PrivateKey.generate().private_bytes_raw()
    key_file = tmp_path / "trusted.key"
    key_file.write_bytes(seed)
    return str(key_file)


def _untrusted_key_file(tmp_path):
    """A key that is NOT the trusted one."""
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    seed = Ed25519PrivateKey.generate().private_bytes_raw()
    key_file = tmp_path / "untrusted.key"
    key_file.write_bytes(seed)
    return str(key_file)


VALID = "rab12cd34ef56"
VALID2 = "rcd34ef56ab12"
VALID3 = "r1234567890ab"
VALID4 = "r4567890abcde"


# ── The core defect: an envelope signed by an untrusted key passes ────── #

def test_envelope_signed_by_untrusted_key_is_rejected(tmp_path, monkeypatch):
    """The core defect: an envelope signed with an untrusted key should be rejected.

    This test demonstrates the current bug: a malicious actor can generate their
    own key pair, sign a fabricated envelope with their private key, and the
    signature will verify perfectly. The envelope_status should REJECT this
    because the key is not in the trusted set.
    """
    # Configure the trusted key
    trusted_key_path = _trusted_key_file(tmp_path)
    monkeypatch.setenv("MECH_TRUSTED_SIGNING_KEY_PATH", trusted_key_path)

    # But the envelope was signed with a DIFFERENT (untrusted) key
    untrusted_key_path = _untrusted_key_file(tmp_path)
    monkeypatch.setenv("MECH_SIGNING_KEY_PATH", untrusted_key_path)

    eg.save_run_record(VALID, {"run_id": VALID, "goal": "malicious"}, directory=str(tmp_path))
    status = eg.envelope_status(VALID, directory=str(tmp_path))

    # CURRENTLY this passes (bug) - after fix it should FAIL with attested=False
    assert status["attested"] is False, (
        "envelope signed by untrusted key was accepted; "
        f"key_id={status.get('key_id')}, status={status}"
    )
    assert status["signature"] == "invalid" or status["signature"] == "untrusted"


# ── Trusted key passes ────────────────────────────────────────────────── #

def test_envelope_signed_by_trusted_key_is_accepted(tmp_path, monkeypatch):
    """Envelope signed by the configured trusted key must be accepted."""
    trusted_key_path = _trusted_key_file(tmp_path)
    monkeypatch.setenv("MECH_TRUSTED_SIGNING_KEY_PATH", trusted_key_path)
    monkeypatch.setenv("MECH_SIGNING_KEY_PATH", trusted_key_path)

    eg.save_run_record(VALID, {"run_id": VALID, "goal": "ok"}, directory=str(tmp_path))
    status = eg.envelope_status(VALID, directory=str(tmp_path))

    assert status["attested"] is True
    assert status["signature"] == "valid"


# ── Multiple trusted keys ─────────────────────────────────────────────── #

def test_multiple_trusted_keys_accepted(tmp_path, monkeypatch):
    """Support multiple trusted keys (comma-separated paths)."""
    trusted1 = _trusted_key_file(tmp_path)
    trusted2 = _trusted_key_file(tmp_path)
    monkeypatch.setenv("MECH_TRUSTED_SIGNING_KEY_PATH", f"{trusted1},{trusted2}")
    monkeypatch.setenv("MECH_SIGNING_KEY_PATH", trusted1)

    eg.save_run_record(VALID, {"run_id": VALID}, directory=str(tmp_path))
    status = eg.envelope_status(VALID, directory=str(tmp_path))
    assert status["attested"] is True

    # Also works with second trusted key
    monkeypatch.setenv("MECH_SIGNING_KEY_PATH", trusted2)
    eg.save_run_record(VALID2, {"run_id": VALID2}, directory=str(tmp_path))
    status2 = eg.envelope_status(VALID2, directory=str(tmp_path))
    assert status2["attested"] is True


# ── No trusted key configured = all envelopes untrusted ───────────────── #

def test_no_trusted_key_configured_all_rejected(tmp_path, monkeypatch):
    """If no trusted key is configured, no envelope should be trusted."""
    monkeypatch.delenv("MECH_TRUSTED_SIGNING_KEY_PATH", raising=False)
    # Even if signing key is configured, without trusted key list nothing is trusted
    monkeypatch.setenv("MECH_SIGNING_KEY_PATH", _trusted_key_file(tmp_path))

    eg.save_run_record(VALID, {"run_id": VALID}, directory=str(tmp_path))
    status = eg.envelope_status(VALID, directory=str(tmp_path))

    # Should be treated as unsigned/untrusted
    assert status["attested"] is False


# ── Key ID comparison ─────────────────────────────────────────────────── #

def test_key_id_comparison_is_stable(tmp_path, monkeypatch):
    """Trusted key comparison uses stable key_id, not raw public key."""
    trusted_key_path = _trusted_key_file(tmp_path)
    monkeypatch.setenv("MECH_TRUSTED_SIGNING_KEY_PATH", trusted_key_path)
    monkeypatch.setenv("MECH_SIGNING_KEY_PATH", trusted_key_path)

    eg.save_run_record(VALID, {"run_id": VALID}, directory=str(tmp_path))
    status = eg.envelope_status(VALID, directory=str(tmp_path))

    # The envelope should carry a key_id that matches our trusted key
    assert status["attested"] is True
    assert status.get("key_id") is not None
    assert isinstance(status["key_id"], str)
    assert status["key_id"].startswith("ed25519:")


def test_key_id_computed_from_public_key(tmp_path, monkeypatch):
    """key_id is derived from public key, not from private key material."""
    from backend.science.integrity import signing as signing_mod

    trusted_key_path = _trusted_key_file(tmp_path)
    monkeypatch.setenv("MECH_TRUSTED_SIGNING_KEY_PATH", trusted_key_path)
    monkeypatch.setenv("MECH_SIGNING_KEY_PATH", trusted_key_path)

    # Compute the expected key_id from the trusted key
    pub_hex = signing_mod.public_key_hex(trusted_key_path)
    expected_key_id = signing_mod.key_id_for(pub_hex)

    eg.save_run_record(VALID, {"run_id": VALID}, directory=str(tmp_path))
    status = eg.envelope_status(VALID, directory=str(tmp_path))

    assert status["key_id"] == expected_key_id, (
        f"envelope key_id {status.get('key_id')} != expected {expected_key_id}"
    )


# ── Backward compatibility: no trusted key env = warn but work (for now) ─ #

def test_missing_trusted_key_env_does_not_crash(tmp_path, monkeypatch):
    """Absence of MECH_TRUSTED_SIGNING_KEY_PATH should not crash; just mark untrusted."""
    monkeypatch.delenv("MECH_TRUSTED_SIGNING_KEY_PATH", raising=False)
    monkeypatch.setenv("MECH_SIGNING_KEY_PATH", _trusted_key_file(tmp_path))

    eg.save_run_record(VALID, {"run_id": VALID}, directory=str(tmp_path))
    status = eg.envelope_status(VALID, directory=str(tmp_path))

    # Should not crash, should mark as untrusted
    assert "attested" in status
    assert status["attested"] is False


def test_envelope_with_missing_key_id_treated_as_untrusted(tmp_path, monkeypatch):
    """If envelope has no key_id (legacy), it's untrusted."""
    # Create envelope without key_id (simulate old envelope)
    monkeypatch.setenv("MECH_SIGNING_KEY_PATH", _trusted_key_file(tmp_path))
    eg.save_run_record(VALID, {"run_id": VALID}, directory=str(tmp_path))

    # Manually remove key_id from envelope
    import json
    path = tmp_path / f"{VALID}.json"
    env = json.loads(path.read_text(encoding="utf-8"))
    env["attestation"].pop("key_id", None)
    path.write_text(json.dumps(env), encoding="utf-8")

    # Configure trusted key
    monkeypatch.setenv("MECH_TRUSTED_SIGNING_KEY_PATH", _trusted_key_file(tmp_path))

    status = eg.envelope_status(VALID, directory=str(tmp_path))
    assert status["attested"] is False, "envelope missing key_id should be untrusted"


def test_envelope_unsigned_is_untrusted_even_with_trusted_key(tmp_path, monkeypatch):
    """Unsigned envelope is untrusted even if key would match."""
    # Create envelope WITHOUT signing (no MECH_SIGNING_KEY_PATH)
    monkeypatch.delenv("MECH_SIGNING_KEY_PATH", raising=False)
    eg.save_run_record(VALID, {"run_id": VALID}, directory=str(tmp_path))

    # Now configure trusted key
    monkeypatch.setenv("MECH_TRUSTED_SIGNING_KEY_PATH", _trusted_key_file(tmp_path))

    status = eg.envelope_status(VALID, directory=str(tmp_path))
    assert status["attested"] is False
    assert status["signature"] == "missing"