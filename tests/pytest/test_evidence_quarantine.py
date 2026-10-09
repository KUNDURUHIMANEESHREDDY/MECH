"""P1-7: pre-envelope records are quarantined out of the authoritative dir.

Seam: quarantine_legacy_records in backend.core.evidence_graph + the
lifespan hook in backend.main (function-tested here; wiring asserted below).
"""
import json
import os
import sys

import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from backend.core import evidence_graph as eg  # noqa: E402

LIVE = "rab12cd34ef56"
BLOCKED = "rcd34ef56ab12"
ENVELOPED = "rdeadbeef1234"


def _seed_key_file(tmp_path):
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    seed = Ed25519PrivateKey.generate().private_bytes_raw()
    key_file = tmp_path / "signing.key"
    key_file.write_bytes(seed)
    return str(key_file)


def _seed_evidence(evidence):
    live = {"run_id": LIVE, "goal": "old-live", "status": "completed",
            "result": {"status": "completed", "provenance": "live",
                       "validation_eligible": True, "publication_eligible": True}}
    (evidence / f"{LIVE}.json").write_text(json.dumps(live), encoding="utf-8")
    blocked = {"run_id": BLOCKED, "goal": "old-blocked", "status": "blocked",
               "result": {"status": "blocked", "provenance": "unavailable",
                          "validation_eligible": False,
                          "publication_eligible": False}}
    (evidence / f"{BLOCKED}.json").write_text(json.dumps(blocked), encoding="utf-8")
    (evidence / "r_persist1.json").write_text(
        json.dumps({"run_id": "r_persist1", "goal": "odd"}), encoding="utf-8")
    (evidence / "corrupt.json").write_text("{torn", encoding="utf-8")
    (evidence / f"{LIVE}.json.tmp.123").write_text("{torn", encoding="utf-8")
    (evidence / "notes.txt").write_text("not evidence", encoding="utf-8")


def test_quarantine_moves_and_stamps(tmp_path, monkeypatch):
    monkeypatch.setenv("MECH_SIGNING_KEY_PATH", _seed_key_file(tmp_path))
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    historical = tmp_path / "evidence_historical"
    _seed_evidence(evidence)
    eg.save_run_record(ENVELOPED, {"run_id": ENVELOPED}, directory=str(evidence))

    report = eg.quarantine_legacy_records(directory=str(evidence),
                                          historical_directory=str(historical))
    assert report["moved"] == 4  # live, blocked, odd-name, corrupt
    assert report["stamped"] == 3
    assert report["skipped"] == 1  # the envelope

    remaining = sorted(p.name for p in evidence.iterdir())
    assert set(remaining) == {f"{ENVELOPED}.json", f"{LIVE}.json.tmp.123", "notes.txt"}

    moved = json.loads((historical / f"{LIVE}.json").read_text(encoding="utf-8"))
    assert moved["origin"] == "historical_fixture"
    for flag in ("scientific_eligible", "validation_eligible", "publication_eligible"):
        assert moved[flag] is False
        assert moved["result"][flag] is False

    # Authoritative surface no longer serves the legacy record ...
    assert eg.list_run_records(directory=str(evidence))[0]["run_id"] == ENVELOPED
    with pytest.raises(Exception):
        eg.load_run_record(LIVE, directory=str(evidence))
    # ... but audit can still read it, flagged.
    assert eg.load_quarantined_record(LIVE, directory=str(evidence))["origin"] == \
        "historical_fixture"
    origins = {s["run_id"]: s["origin"]
               for s in eg.list_quarantined_records(directory=str(evidence))}
    assert origins[LIVE] == "historical_fixture"

    # Idempotent: nothing left to move.
    again = eg.quarantine_legacy_records(directory=str(evidence),
                                        historical_directory=str(historical))
    assert again["moved"] == 0 and again["skipped"] == 1


def test_quarantine_closes_eligibility_gates(tmp_path):
    from backend.agents import evidence_policy as pol
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    _seed_evidence(evidence)
    eg.quarantine_legacy_records(directory=str(evidence),
                                 historical_directory=str(tmp_path / "evidence_historical"))
    result = eg.load_quarantined_record(LIVE, directory=str(evidence))["result"]
    assert pol.discovery_is_live(result) is False
    assert pol.validation_is_live(result) is False


def test_lifespan_runs_quarantine():
    import pathlib
    src = (pathlib.Path(__file__).resolve().parents[2] / "backend" / "main.py"
           ).read_text(encoding="utf-8")
    assert "quarantine_legacy_records()" in src


def test_save_refuses_historical_origin(tmp_path):
    stamped = {"run_id": LIVE, "origin": "historical_fixture",
               "validation_eligible": False, "publication_eligible": False}
    with pytest.raises(ValueError, match="quarantined"):
        eg.save_run_record(LIVE, stamped, directory=str(tmp_path))


def test_copied_back_stamp_not_served(tmp_path, monkeypatch):
    monkeypatch.setenv("MECH_SIGNING_KEY_PATH", _seed_key_file(tmp_path))
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    _seed_evidence(evidence)
    eg.quarantine_legacy_records(directory=str(evidence),
                                 historical_directory=str(tmp_path / "evidence_historical"))
    # Copy the stamped file back into the authoritative dir.
    import shutil
    shutil.copy(str(tmp_path / "evidence_historical" / f"{LIVE}.json"),
                str(evidence / f"{LIVE}.json"))
    with pytest.raises(ValueError, match="quarantined"):
        eg.load_run_record(LIVE, directory=str(evidence))
    assert eg.list_run_records(directory=str(evidence)) == []
    assert eg.envelope_status(LIVE, directory=str(evidence))["attested"] is False


def test_self_destruct_config_refused(tmp_path):
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    _seed_evidence(evidence)
    report = eg.quarantine_legacy_records(directory=str(evidence),
                                          historical_directory=str(evidence))
    assert report["moved"] == 0 and "refusing" in report.get("reason", "")
    assert (evidence / f"{LIVE}.json").exists()


def test_odd_named_file_auditable(tmp_path):
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    _seed_evidence(evidence)
    historical = tmp_path / "evidence_historical"
    eg.quarantine_legacy_records(directory=str(evidence),
                                 historical_directory=str(historical))
    body = eg.load_quarantined_file("r_persist1.json", directory=str(evidence),
                                    historical_directory=str(historical))
    assert body["goal"] == "odd"
    with pytest.raises(ValueError):
        eg.load_quarantined_file("../evidence/notes.txt", directory=str(evidence),
                                 historical_directory=str(historical))
    with pytest.raises(ValueError):
        eg.load_quarantined_record("r_persist1", directory=str(evidence),
                                   historical_directory=str(historical))


def test_corrupt_bytes_preserved_verbatim(tmp_path):
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    raw = b"\xff\xfe{torn binary"
    (evidence / "badbytes.json").write_bytes(raw)
    historical = tmp_path / "evidence_historical"
    report = eg.quarantine_legacy_records(directory=str(evidence),
                                          historical_directory=str(historical))
    assert report["moved"] == 1
    assert (historical / "badbytes.json").read_bytes() == raw
    assert not (evidence / "badbytes.json").exists()
    listed = {s["run_id"]: s
              for s in eg.list_quarantined_records(directory=str(evidence),
                                                   historical_directory=str(historical))}
    assert listed["badbytes"]["origin"] == "unparseable"


def test_stamped_envelope_reports_quarantined(tmp_path, monkeypatch):
    """A valid signature over stamped bytes must still report quarantined."""
    import json as _json
    from backend.science.integrity import signing as _signing
    key_file = _seed_key_file(tmp_path)
    monkeypatch.setenv("MECH_SIGNING_KEY_PATH", key_file)
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    record = {"run_id": LIVE, "origin": "historical_fixture",
              "scientific_eligible": False, "validation_eligible": False,
              "publication_eligible": False}
    created_at = "2026-01-01T00:00:00+00:00"
    public_key = _signing.public_key_hex(key_file)
    env = {"schema_version": 1, "run_id": LIVE, "created_at": created_at,
           "record": record,
           "attestation": {
               "attested": True, "algorithm": "Ed25519",
               "public_key": public_key,
               "key_id": _signing.key_id_for(public_key),
               "signature": _signing.sign(
                   eg._signing_payload(LIVE, record, created_at=created_at,
                                       public_key=public_key), key_file),
               "reason": None}}
    (evidence / f"{LIVE}.json").write_text(_json.dumps(env), encoding="utf-8")
    status = eg.envelope_status(LIVE, directory=str(evidence))
    assert status["signature"] == "invalid" and status["attested"] is False
    assert "quarantine" in status["reason"]
    with pytest.raises(ValueError, match="quarantined"):
        eg.load_run_record(LIVE, directory=str(evidence))
