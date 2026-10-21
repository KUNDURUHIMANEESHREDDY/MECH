"""The project export route, and the honesty defects reachability exposes.

`backend/api/project_export.py` used to be reachable only from the test
suite. It is now a route and a CLI command, and that turns three quiet
defects into observable ones:

* an export of a project that does not exist used to succeed, because
  nothing checked that any session actually named the project;
* the size of an export was never estimated, so a large archive could be
  written without the confirmation spec 19.3 asks for;
* the archive was written straight onto its final name, so a failure left a
  truncated file where a finished archive was expected (spec 20.1).

Each test below pins one of those, against a real temporary export root
and a real storage authority. Nothing here is mocked except the one
deliberate write failure, which is the point of that test.
"""

from __future__ import annotations

import hashlib
import json
import os
import zipfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import backend.api.project_export as project_export
from backend.core import auth as auth_mod

AUTH_HEADERS = auth_mod.auth_headers()
INVALID_AUTH_HEADERS = {"Authorization": "Bearer invalid-token-12345"}

REPO_ROOT = Path(__file__).resolve().parents[2]

#: The documents that describe the export. The export is reachable now, so
#: none of them may still claim it is not.
EXPORT_DOCUMENTS = (
    REPO_ROOT / "backend" / "api" / "project_export.py",
    REPO_ROOT / "backend" / "api" / "export_contracts.py",
    REPO_ROOT / "backend" / "cli" / "main.py",
    REPO_ROOT / "CHANGELOG.md",
    REPO_ROOT / "docs" / "experiments.md",
    REPO_ROOT / "docs" / "reproducibility.md",
)

#: Claims that the export is not wired up. A phrase split across lines is
#: still a claim, so the documents are compared with their whitespace
#: flattened.
STALE_CLAIMS = ("not reachable", "unreachable", "not available today")


@pytest.fixture
def export_root(tmp_path, monkeypatch):
    """A real export root and a real storage authority, both under tmp."""
    root = tmp_path / "exports"
    monkeypatch.setenv("MECH_EXPORT_ROOT", str(root))
    monkeypatch.setenv("MECH_STORAGE_DB", str(tmp_path / "authority.db"))
    monkeypatch.delenv("MECH_EXPORT_CONFIRM_BYTES", raising=False)
    return root


@pytest.fixture
def client():
    from backend.main import app

    return TestClient(app, raise_server_exceptions=False)


def _authority(tmp_path):
    """The storage authority the export reads, with its schema created."""
    from backend.storage import DesktopStorage

    store = DesktopStorage(tmp_path / "authority.db")
    store.initialize()
    return store


def _archives(root: Path):
    """Every ``.interp-project`` archive under the export root."""
    if not root.exists():
        return []
    return sorted(root.rglob("*.interp-project"))


# ---------------------------------------------------------------------------
# authentication
# ---------------------------------------------------------------------------


def test_an_anonymous_export_is_refused(export_root, client):
    response = client.post("/api/exports/alpha", json={})
    assert response.status_code == 401
    assert _archives(export_root) == []


def test_an_export_with_a_wrong_token_is_refused(export_root, client):
    response = client.post("/api/exports/alpha", headers=INVALID_AUTH_HEADERS,
                            json={})
    assert response.status_code == 401
    assert _archives(export_root) == []


def test_get_on_the_export_path_is_not_allowed(export_root, client):
    """A GET must not export anything. The route is POST-only.

    The 405 is checked on the router itself, because the assembled app
    mounts the built frontend at "/" as a catch-all: there, a request whose
    method no API route accepts falls through to the single-page app and
    comes back 404. That is how every other POST-only route in this
    platform answers a GET as well. The route contract -- no GET on this
    path -- is the router's, so that is where it is checked.
    """
    from fastapi import FastAPI

    from backend.api import export_contracts

    router_only = FastAPI()
    router_only.include_router(export_contracts.router)
    router_client = TestClient(router_only, raise_server_exceptions=False)

    response = router_client.get("/exports/alpha", headers=AUTH_HEADERS)
    assert response.status_code == 405

    # And through the real app, a GET still exports nothing.
    response = client.get("/api/exports/alpha", headers=AUTH_HEADERS)
    assert response.status_code < 500
    assert response.status_code not in (200, 201)
    assert _archives(export_root) == []


# ---------------------------------------------------------------------------
# the request may not choose a destination
# ---------------------------------------------------------------------------


def test_a_body_that_names_a_destination_is_refused(export_root, client):
    """The destination is the server's export root and nothing else."""
    response = client.post("/api/exports/alpha", headers=AUTH_HEADERS,
                           json={"dest_dir": str(export_root)})
    assert response.status_code == 422
    assert not export_root.exists()


# ---------------------------------------------------------------------------
# refusals
# ---------------------------------------------------------------------------


def test_an_invalid_project_id_is_refused_and_creates_nothing(export_root,
                                                               client):
    """The slug is checked before any I/O, so nothing is created."""
    response = client.post("/api/exports/not a slug", headers=AUTH_HEADERS,
                           json={})
    assert 400 <= response.status_code < 500
    assert not export_root.exists(), (
        "an invalid project id brought the export root into existence")


def test_an_unknown_project_is_not_found_and_writes_nothing(export_root,
                                                             client):
    """A project no stored session names does not exist."""
    response = client.post("/api/exports/ghost", headers=AUTH_HEADERS, json={})
    assert response.status_code == 404
    detail = response.json()["detail"]
    assert detail["code"] == "project_not_found"
    assert _archives(export_root) == []


# ---------------------------------------------------------------------------
# a real export
# ---------------------------------------------------------------------------


def test_a_real_project_exports_its_own_sessions_only(export_root, client,
                                                      tmp_path):
    store = _authority(tmp_path)
    store.add_session({"id": "s-alpha", "project_id": "alpha",
                       "note": "alpha session"})
    store.add_session({"id": "s-beta", "project_id": "beta",
                       "note": "beta session"})

    response = client.post("/api/exports/alpha", headers=AUTH_HEADERS, json={})
    assert response.status_code == 200, response.text
    body = response.json()

    assert body["status"] == "ok"
    assert body["project_id"] == "alpha"
    assert body["archive_name"] == "alpha.interp-project"
    assert body["archive_path_relative"] == "alpha.interp-project"
    assert body["sessions_included"] == 1
    assert body["sessions_omitted_unattributed"] == 0
    assert body["sessions_omitted_other_projects"] == 1
    assert body["estimate_basis"] == "serialized_sessions_json"
    assert body["threshold_bytes"] == 100 * 1024 * 1024
    assert body["estimated_bytes"] > 0
    assert body["bytes_written"] > 0
    assert body["export_version"] == "1.0"
    assert body["exported_at"]

    archive = export_root / "alpha.interp-project"
    assert archive.is_file()
    assert archive.stat().st_size == body["bytes_written"]

    # Another project's session must not be in the archive, in any form.
    raw = archive.read_bytes()
    assert b"s-beta" not in raw

    with zipfile.ZipFile(archive) as opened:
        names = set(opened.namelist())
        metadata = json.loads(opened.read("metadata.json"))
        sessions = json.loads(opened.read("sessions.json"))
        members = {name: opened.read(name) for name in names}

    assert names == {"metadata.json", "sessions.json"}
    assert "reports.json" not in names
    assert [session["id"] for session in sessions] == ["s-alpha"]

    assert metadata["project_id"] == "alpha"
    assert metadata["exported_at"] == body["exported_at"]
    assert metadata["sessions_included"] == 1
    assert metadata["sessions_omitted_unattributed"] == 0
    assert metadata["sessions_omitted_other_projects"] == 1
    assert metadata["reports_included"] == 0
    assert "project_id" in metadata["reports_omitted_reason"], (
        "the omission must state its reason inside the archive, so a "
        "recipient can tell it apart from an empty project")
    assert metadata["source"], "the archive must name its source"
    assert not os.path.isabs(metadata["source"]), (
        "the source identifier must not be an absolute path: the archive "
        "travels to machines where that path means nothing")

    manifest = metadata["files"]
    assert manifest, "the archive must carry a manifest of its members"
    for entry in manifest:
        assert set(entry) == {"path", "sha256", "bytes"}
        assert entry["path"] in members
        assert entry["bytes"] == len(members[entry["path"]])
        assert entry["sha256"] == hashlib.sha256(
            members[entry["path"]]).hexdigest()


def test_the_route_is_mounted_at_both_prefixes(export_root, client, tmp_path):
    store = _authority(tmp_path)
    store.add_session({"id": "s-alpha", "project_id": "alpha"})

    for prefix in ("/api", "/api/v1"):
        response = client.post(f"{prefix}/exports/alpha", headers=AUTH_HEADERS,
                               json={})
        assert response.status_code == 200, f"{prefix}: {response.text}"


def test_the_answer_carries_no_absolute_server_path(export_root, client,
                                                    tmp_path):
    store = _authority(tmp_path)
    store.add_session({"id": "s-alpha", "project_id": "alpha"})

    response = client.post("/api/exports/alpha", headers=AUTH_HEADERS, json={})
    assert response.status_code == 200
    body = response.json()

    assert str(export_root) not in json.dumps(body)
    for key, value in body.items():
        assert not os.path.isabs(str(value)), f"{key} is an absolute path"


# ---------------------------------------------------------------------------
# the size threshold
# ---------------------------------------------------------------------------


def test_an_export_over_the_threshold_is_refused_until_confirmed(export_root,
                                                                 client,
                                                                 tmp_path,
                                                                 monkeypatch):
    store = _authority(tmp_path)
    store.add_session({"id": "s-alpha", "project_id": "alpha"})
    monkeypatch.setenv("MECH_EXPORT_CONFIRM_BYTES", "1")

    response = client.post("/api/exports/alpha", headers=AUTH_HEADERS, json={})
    assert response.status_code == 409
    detail = response.json()["detail"]
    assert detail["code"] == "confirmation_required"
    assert detail["threshold_bytes"] == 1
    assert detail["estimate"]["estimated_bytes"] > 1
    assert _archives(export_root) == []


def test_an_export_over_the_threshold_succeeds_with_confirmation(export_root,
                                                                 client,
                                                                 tmp_path,
                                                                 monkeypatch):
    store = _authority(tmp_path)
    store.add_session({"id": "s-alpha", "project_id": "alpha"})
    monkeypatch.setenv("MECH_EXPORT_CONFIRM_BYTES", "1")

    response = client.post("/api/exports/alpha", headers=AUTH_HEADERS,
                           json={"confirm": True})
    assert response.status_code == 200, response.text
    assert (export_root / "alpha.interp-project").is_file()


def test_a_malformed_threshold_falls_back_to_the_default(export_root, client,
                                                          tmp_path,
                                                          monkeypatch):
    store = _authority(tmp_path)
    store.add_session({"id": "s-alpha", "project_id": "alpha"})
    monkeypatch.setenv("MECH_EXPORT_CONFIRM_BYTES", "not a number")

    response = client.post("/api/exports/alpha", headers=AUTH_HEADERS, json={})
    assert response.status_code == 200
    assert response.json()["threshold_bytes"] == 100 * 1024 * 1024


# ---------------------------------------------------------------------------
# a failure while writing
# ---------------------------------------------------------------------------


def test_a_failure_while_writing_leaves_no_archive(export_root, client,
                                                    tmp_path, monkeypatch):
    """A half-written archive must never be left where a finished one was."""
    store = _authority(tmp_path)
    store.add_session({"id": "s-alpha", "project_id": "alpha"})

    def boom(*args, **kwargs):
        raise RuntimeError("the disk said no")

    monkeypatch.setattr(project_export.zipfile, "ZipFile", boom)

    response = client.post("/api/exports/alpha", headers=AUTH_HEADERS, json={})
    assert response.status_code >= 400
    assert not (export_root / "alpha.interp-project").exists()
    assert list(export_root.iterdir()) == [], (
        "the temporary file was left behind")


# ---------------------------------------------------------------------------
# the estimate follows the data
# ---------------------------------------------------------------------------


def test_the_estimate_follows_the_session_content(export_root, tmp_path):
    from backend.api.project_export import estimate_export

    store = _authority(tmp_path)
    store.add_session({"id": "s1", "project_id": "alpha", "note": "x"})
    first = estimate_export("alpha")
    store.add_session({"id": "s2", "project_id": "alpha", "note": "y"})
    second = estimate_export("alpha")

    assert first["estimate_basis"] == "serialized_sessions_json"
    assert first["sessions_included"] == 1
    assert second["sessions_included"] == 2
    assert second["estimated_bytes"] > first["estimated_bytes"]


def test_an_unknown_project_has_no_sessions_in_the_estimate(export_root,
                                                            tmp_path):
    from backend.api.project_export import estimate_export

    store = _authority(tmp_path)
    store.add_session({"id": "s1", "project_id": "alpha"})

    estimate = estimate_export("ghost")
    assert estimate["sessions_included"] == 0
    assert estimate["requires_confirmation"] is False


# ---------------------------------------------------------------------------
# the documentation keeps up
# ---------------------------------------------------------------------------


def test_the_export_is_no_longer_documented_as_unreachable():
    """The anti-rot guard for the claim this change removed.

    The exporter used to be described as unreachable in five places. When it
    became a route and a command, those descriptions became false, and a
    false claim in a document is how the next reader decides not to wire it
    up. So the documents are checked, not trusted.
    """
    stale = []
    for path in EXPORT_DOCUMENTS:
        text = path.read_text(encoding="utf-8")
        flat = " ".join(text.lower().split())
        for phrase in STALE_CLAIMS:
            if phrase in flat:
                stale.append(f"{path.relative_to(REPO_ROOT)}: {phrase!r}")

    assert not stale, (
        "the export is reachable now, so these documents still claim it is "
        f"not:\n  " + "\n  ".join(stale))
