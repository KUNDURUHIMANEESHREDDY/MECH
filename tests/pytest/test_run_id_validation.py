"""P1-4: Society run IDs cannot escape the evidence directory.

Seam: persistence boundary in backend.core.evidence_graph
(validate_run_id + _record_path used by save/load_run_record) plus the
route-level fallback in dispatcher.society_run_status.
"""
import os
import sys

import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from backend.core import evidence_graph as eg  # noqa: E402

VALID = "rab12cd34ef56"


def test_generator_format_validates():
    import uuid
    for _ in range(100):
        rid = "r" + uuid.uuid4().hex[:12]
        assert eg.validate_run_id(rid) == rid


def test_roundtrip_in_tmp_dir(tmp_path):
    path = eg.save_run_record(VALID, {"run_id": VALID, "goal": "g"}, directory=str(tmp_path))
    assert path.endswith(f"{VALID}.json")
    assert eg.load_run_record(VALID, directory=str(tmp_path))["goal"] == "g"


@pytest.mark.parametrize("bad", [
    "../secret",
    "../../etc/passwd",
    "..\\..\\secret",
    "/etc/passwd",
    "/nonexistent/x.json",
    "C:\\Windows\\secret",
    "C:/Windows/secret",
    "C:evil",
    "\\\\server\\share\\x",
    "%2e%2e%2fsecret",
    "..%2fsecret",
    "",
    ".",
    "..",
    "run_id",
    "r123",
    "r_test_cancel_probe",
    "r_does_not_exist_123",
    "RAB12CD34EF56",
    "rab12cd34ef5",
    "rab12cd34ef567",
    "rab12cd34ef5g",
    " rab12cd34ef56",
    "rab12cd34ef56 ",
    "s_ab12cd34ef56",
])
def test_load_rejects_non_ids(tmp_path, bad):
    with pytest.raises(ValueError):
        eg.load_run_record(bad, directory=str(tmp_path))


@pytest.mark.parametrize("bad", ["../secret", "/etc/passwd", "C:\\Windows\\x", "", "r_test_cancel_probe"])
def test_save_rejects_non_ids(tmp_path, bad):
    with pytest.raises(ValueError):
        eg.save_run_record(bad, {"run_id": bad}, directory=str(tmp_path))


@pytest.mark.parametrize("bad", [None, 123, b"rab12cd34ef56", ["rab12cd34ef56"]])
def test_non_string_ids_rejected(tmp_path, bad):
    with pytest.raises(ValueError):
        eg.load_run_record(bad, directory=str(tmp_path))


def test_route_never_serves_sibling_json(monkeypatch, tmp_path):
    """A traversal runId must read as unknown, not as the sibling file."""
    import json
    monkeypatch.setattr(eg, "EVIDENCE_DIR", str(tmp_path))
    (tmp_path / "secret.json").write_text(json.dumps({"marker": "S3CR3T"}), encoding="utf-8")
    from backend.api import dispatcher as core
    for bad in ("../secret", "secret", "/etc/passwd", "C:\\Windows\\x"):
        body = core.society_run_status(bad)
        assert body.get("status") == "error", body
        assert "S3CR3T" not in str(body)
