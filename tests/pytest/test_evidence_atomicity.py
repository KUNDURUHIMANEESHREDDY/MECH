"""P1-6: evidence writes are atomic — readers never see a torn record.

Seam: save_run_record in backend.core.evidence_graph (tmp + fsync + replace).
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


def test_no_tmp_sidecar_after_save(tmp_path):
    eg.save_run_record(VALID, {"run_id": VALID}, directory=str(tmp_path))
    assert sorted(p.name for p in tmp_path.iterdir()) == [f"{VALID}.json"]
    # The committed file is a complete, parseable envelope.
    data = json.loads((tmp_path / f"{VALID}.json").read_text(encoding="utf-8"))
    assert data["schema_version"] == 1 and data["run_id"] == VALID


def test_stale_tmp_ignored_and_overwritten(tmp_path):
    """A crash orphan sidecar is invisible to readers; resave wins atomically."""
    (tmp_path / f"{VALID}.json.tmp.12345").write_text("{torn", encoding="utf-8")
    assert eg.list_run_records(directory=str(tmp_path)) == []
    eg.save_run_record(VALID, {"run_id": VALID, "goal": "g"}, directory=str(tmp_path))
    assert eg.load_run_record(VALID, directory=str(tmp_path))["goal"] == "g"
    assert (tmp_path / f"{VALID}.json").exists()


def test_concurrent_saves_stay_complete(tmp_path):
    import threading
    errors = []

    def save(rid, n):
        try:
            eg.save_run_record(rid, {"run_id": rid, "n": n,
                                     "pad": "x" * 10_000},
                               directory=str(tmp_path))
        except Exception as exc:  # noqa: BLE001 - collected, then asserted
            errors.append(exc)

    threads = [threading.Thread(target=save, args=(rid, n))
               for n, rid in enumerate([VALID, VALID2] * 5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    for rid in (VALID, VALID2):
        body = eg.load_run_record(rid, directory=str(tmp_path))
        assert body["run_id"] == rid and len(body["pad"]) == 10_000


def test_large_record_survives(tmp_path):
    big = {"run_id": VALID, "blob": "y" * 500_000}
    eg.save_run_record(VALID, big, directory=str(tmp_path))
    assert eg.load_run_record(VALID, directory=str(tmp_path)) == big


def test_crash_mid_write_keeps_old_record(tmp_path, monkeypatch):
    """Fault during commit: the previous complete record survives intact."""
    import json as _json
    old = {"run_id": VALID, "goal": "old"}
    eg.save_run_record(VALID, old, directory=str(tmp_path))
    before = (tmp_path / f"{VALID}.json").read_bytes()

    def _boom(*args, **kwargs):
        raise RuntimeError("simulated crash during serialization")
    monkeypatch.setattr(_json, "dump", _boom)
    with pytest.raises(RuntimeError, match="simulated crash"):
        eg.save_run_record(VALID, {"run_id": VALID, "goal": "new"},
                           directory=str(tmp_path))
    assert (tmp_path / f"{VALID}.json").read_bytes() == before
    assert eg.load_run_record(VALID, directory=str(tmp_path)) == old
    leftovers = [p for p in tmp_path.iterdir() if ".tmp." in p.name]
    assert leftovers == []
