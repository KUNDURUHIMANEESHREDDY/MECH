"""P1-5: persisted Society run records are tamper-evident.

Seam: persistence boundary in backend.core.evidence_graph
(save wraps in a signed v1 envelope; load unwraps; envelope_status verifies)
plus the persisted fallback in dispatcher.society_run_status.
"""
import json
import os
import sys

import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from backend.core import evidence_graph as eg  # noqa: E402

VALID = "rab12cd34ef56"
VALID2 = "rcd34ef56ab12"


def _seed_key_file(tmp_path):
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    seed = Ed25519PrivateKey.generate().private_bytes_raw()
    key_file = tmp_path / "signing.key"
    key_file.write_bytes(seed)
    return str(key_file)


def _read_envelope(tmp_path, run_id=VALID):
    return json.loads((tmp_path / f"{run_id}.json").read_text(encoding="utf-8"))


def test_signed_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setenv("MECH_SIGNING_KEY_PATH", _seed_key_file(tmp_path))
    record = {"run_id": VALID, "goal": "g", "status": "completed"}
    eg.save_run_record(VALID, record, directory=str(tmp_path))
    env = _read_envelope(tmp_path)
    assert env["schema_version"] == 1
    assert env["run_id"] == VALID
    assert env["created_at"]
    assert env["record"] == record
    assert env["attestation"]["attested"] is True
    assert len(env["attestation"]["signature"]) == 128
    assert env["attestation"]["public_key"]
    assert eg.load_run_record(VALID, directory=str(tmp_path)) == record
    status = eg.envelope_status(VALID, directory=str(tmp_path))
    assert status["envelope"] == "v1" and status["attested"] is True
    assert status["signature"] == "valid"


def test_unsigned_when_no_key(tmp_path, monkeypatch):
    monkeypatch.delenv("MECH_SIGNING_KEY_PATH", raising=False)
    record = {"run_id": VALID, "goal": "g"}
    eg.save_run_record(VALID, record, directory=str(tmp_path))
    env = _read_envelope(tmp_path)
    assert env["attestation"]["attested"] is False
    assert env["attestation"]["signature"] is None
    assert "signing" in str(env["attestation"]["reason"]).lower() or \
        "key" in str(env["attestation"]["reason"]).lower()
    assert eg.load_run_record(VALID, directory=str(tmp_path)) == record
    status = eg.envelope_status(VALID, directory=str(tmp_path))
    assert status["attested"] is False and status["signature"] == "missing"


def test_tampered_record_detected(tmp_path, monkeypatch):
    monkeypatch.setenv("MECH_SIGNING_KEY_PATH", _seed_key_file(tmp_path))
    eg.save_run_record(VALID, {"run_id": VALID, "score": 0.5}, directory=str(tmp_path))
    path = tmp_path / f"{VALID}.json"
    env = json.loads(path.read_text(encoding="utf-8"))
    env["record"]["score"] = 0.99
    path.write_text(json.dumps(env), encoding="utf-8")
    status = eg.envelope_status(VALID, directory=str(tmp_path))
    assert status["signature"] == "invalid" and status["attested"] is False


def test_run_id_mismatch_detected(tmp_path, monkeypatch):
    monkeypatch.setenv("MECH_SIGNING_KEY_PATH", _seed_key_file(tmp_path))
    eg.save_run_record(VALID, {"run_id": VALID}, directory=str(tmp_path))
    path = tmp_path / f"{VALID}.json"
    env = json.loads(path.read_text(encoding="utf-8"))
    env["run_id"] = VALID2
    path.write_text(json.dumps(env), encoding="utf-8")
    status = eg.envelope_status(VALID, directory=str(tmp_path))
    assert status["signature"] == "invalid"


def test_swapped_key_detected(tmp_path, monkeypatch):
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    monkeypatch.setenv("MECH_SIGNING_KEY_PATH", _seed_key_file(tmp_path))
    eg.save_run_record(VALID, {"run_id": VALID}, directory=str(tmp_path))
    other_pub = Ed25519PrivateKey.generate().public_key().public_bytes_raw().hex()
    path = tmp_path / f"{VALID}.json"
    env = json.loads(path.read_text(encoding="utf-8"))
    env["attestation"]["public_key"] = other_pub
    path.write_text(json.dumps(env), encoding="utf-8")
    status = eg.envelope_status(VALID, directory=str(tmp_path))
    assert status["signature"] == "invalid"


def test_legacy_file_served_but_flagged(tmp_path):
    legacy = {"run_id": VALID, "goal": "old", "status": "completed"}
    (tmp_path / f"{VALID}.json").write_text(json.dumps(legacy), encoding="utf-8")
    assert eg.load_run_record(VALID, directory=str(tmp_path)) == legacy
    status = eg.envelope_status(VALID, directory=str(tmp_path))
    assert status["envelope"] == "legacy" and status["attested"] is False


def test_corrupt_and_missing_files(tmp_path):
    (tmp_path / f"{VALID}.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(Exception):
        eg.load_run_record(VALID, directory=str(tmp_path))
    assert eg.envelope_status(VALID, directory=str(tmp_path))["envelope"] == "corrupt"
    assert eg.envelope_status(VALID2, directory=str(tmp_path))["envelope"] == "missing"
    assert eg.envelope_status("../secret", directory=str(tmp_path))["envelope"] == "invalid"


def test_save_rejects_non_dict(tmp_path):
    with pytest.raises(ValueError):
        eg.save_run_record(VALID, ["not", "a", "dict"], directory=str(tmp_path))


def test_list_reads_through_envelopes(tmp_path, monkeypatch):
    monkeypatch.setenv("MECH_SIGNING_KEY_PATH", _seed_key_file(tmp_path))
    eg.save_run_record(VALID, {"run_id": VALID, "goal": "new",
                               "status": "completed",
                               "publication": {"steps_completed": 3}},
                       directory=str(tmp_path))
    (tmp_path / f"{VALID2}.json").write_text(json.dumps(
        {"run_id": VALID2, "goal": "old", "status": "completed"}),
        encoding="utf-8")
    summaries = {s["run_id"]: s for s in eg.list_run_records(directory=str(tmp_path))}
    assert summaries[VALID]["goal"] == "new"
    assert summaries[VALID]["steps_completed"] == 3
    assert summaries[VALID2]["goal"] == "old"


def test_algorithm_downgrade_rejected(tmp_path, monkeypatch):
    monkeypatch.setenv("MECH_SIGNING_KEY_PATH", _seed_key_file(tmp_path))
    eg.save_run_record(VALID, {"run_id": VALID}, directory=str(tmp_path))
    path = tmp_path / f"{VALID}.json"
    env = json.loads(path.read_text(encoding="utf-8"))
    env["attestation"]["algorithm"] = "none"
    path.write_text(json.dumps(env), encoding="utf-8")
    status = eg.envelope_status(VALID, directory=str(tmp_path))
    assert status["signature"] == "invalid" and status["attested"] is False


def test_metadata_rewrite_detected(tmp_path, monkeypatch):
    monkeypatch.setenv("MECH_SIGNING_KEY_PATH", _seed_key_file(tmp_path))
    eg.save_run_record(VALID, {"run_id": VALID}, directory=str(tmp_path))
    path = tmp_path / f"{VALID}.json"
    env = json.loads(path.read_text(encoding="utf-8"))
    env["created_at"] = "2000-01-01T00:00:00+00:00"
    path.write_text(json.dumps(env), encoding="utf-8")
    status = eg.envelope_status(VALID, directory=str(tmp_path))
    assert status["signature"] == "invalid" and status["attested"] is False


def test_list_uses_envelope_run_id(tmp_path, monkeypatch):
    monkeypatch.setenv("MECH_SIGNING_KEY_PATH", _seed_key_file(tmp_path))
    eg.save_run_record(VALID, {"run_id": "rdeadbeef1234"}, directory=str(tmp_path))
    summaries = eg.list_run_records(directory=str(tmp_path))
    assert summaries and summaries[0]["run_id"] == VALID


def test_route_fallback_carries_envelope_status(tmp_path, monkeypatch):
    import backend.core.evidence_graph as live_eg
    monkeypatch.setattr(live_eg, "EVIDENCE_DIR", str(tmp_path))
    monkeypatch.setenv("MECH_SIGNING_KEY_PATH", _seed_key_file(tmp_path))
    from backend.api import dispatcher as core
    record = {"run_id": VALID, "goal": "g", "status": "completed"}
    live_eg.save_run_record(VALID, record)
    try:
        body = core.society_run_status(VALID)
        assert body["goal"] == "g"
        assert body["envelope_status"]["signature"] == "valid"
        legacy = {"run_id": VALID2, "goal": "old"}
        (tmp_path / f"{VALID2}.json").write_text(json.dumps(legacy), encoding="utf-8")
        body2 = core.society_run_status(VALID2)
        assert body2["envelope_status"]["envelope"] == "legacy"
    finally:
        for name in (f"{VALID}.json", f"{VALID2}.json"):
            try:
                os.unlink(tmp_path / name)
            except OSError:
                pass
