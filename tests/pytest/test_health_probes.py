"""Subsystem health must be probed, not assumed.

The old `/health` returned `{"status": "healthy"}` unconditionally and
`/runtime/status` returned `"connected"` with five engine names, none of which
were ever checked. These tests pin the replacement: every subsystem is actually
exercised, and nothing is reported ok without a probe having run.
"""
import os
import pytest
from fastapi.testclient import TestClient

from backend.core import auth as auth_mod
from backend.core import health as health_mod
from backend.main import app


@pytest.fixture(autouse=True)
def _test_token(monkeypatch):
    monkeypatch.setenv("MECH_API_TOKEN", "health-probe-token")
    auth_mod.reset_cache()
    yield
    auth_mod.reset_cache()


def _h():
    return auth_mod.auth_headers("health-probe-token")


def test_health_endpoint_stays_a_liveness_probe():
    """/health must not become slow: it answers 'am I alive', nothing else."""
    with TestClient(app) as client:
        r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "healthy"}


def test_subsystem_endpoint_returns_a_report_for_every_subsystem():
    with TestClient(app) as client:
        assert client.get("/health/subsystems").status_code == 401
        r = client.get("/health/subsystems", headers=_h())
    assert r.status_code == 200
    body = r.json()

    assert set(body["subsystems"]) == {
        "process", "dependency", "model", "executor", "storage", "evidence"
    }
    for name, probe in body["subsystems"].items():
        assert "status" in probe, name
        assert probe["status"] in {"ok", "degraded", "unavailable", "unknown"}


def test_subsystem_report_never_claims_healthy():
    """'healthy' is not in the vocabulary: nothing here proves health."""
    with TestClient(app) as client:
        body = client.get("/health/subsystems", headers=_h()).json()
    assert body["status"] != "healthy"
    for probe in body["subsystems"].values():
        assert probe["status"] != "healthy"


def test_can_capable_of_live_evidence_depends_on_the_probes():
    """The headline flag must follow the probes, not default to True."""
    with TestClient(app) as client:
        body = client.get("/health/subsystems", headers=_h()).json()
    worst = body["subsystems"]
    assert body["capable_of_live_evidence"] == (
        all(p["status"] == "ok" for p in worst.values())
    )


def test_subsystem_filter_runs_only_what_was_asked_for():
    with TestClient(app) as client:
        body = client.get("/health/subsystems?subsystem=process,evidence", headers=_h()).json()
    assert set(body["subsystems"]) == {"process", "evidence"}


def test_a_probe_that_raises_is_reported_not_propagated():
    def boom():
        raise RuntimeError("probe exploded")

    results = health_mod.run_probes()
    assert isinstance(results, dict)
    # Overlay a failing probe to prove the guard works.
    original = health_mod.PROBES["process"]
    health_mod.PROBES["process"] = boom
    try:
        out = health_mod.run_probes(["process"])
    finally:
        health_mod.PROBES["process"] = original
    assert out["process"]["status"] == "unavailable"
    assert "probe exploded" in out["process"]["reason"]


def test_overall_is_worst_status_wins():
    assert health_mod.overall({"a": {"status": "ok"}}) == "ok"
    assert health_mod.overall(
        {"a": {"status": "ok"}, "b": {"status": "degraded"}}) == "degraded"
    assert health_mod.overall(
        {"a": {"status": "degraded"}, "b": {"status": "unavailable"}}
    ) == "unavailable"


def test_storage_probe_actually_writes():
    """""ok" for storage must mean a file was written, not a path check."""
    probe = health_mod._probe_storage()
    assert "status" in probe
    if probe["status"] == "ok":
        assert "writable" in probe.get("detail", "")


def test_evidence_probe_reports_the_evidence_modules_that_exist():
    """The evidence probe checks the evidence modules, and only those.

    `boundary` used to be asserted here. `backend/core/evidence_boundary.py` was
    deleted in 624bd3f as a duplicate load path, and the probe was corrected to
    stop importing it — but this test kept expecting the component, so it failed
    with `Extra items in the right set: 'boundary'` for a probe that was
    reporting the truth.

    Kept distinct from a bare status check on purpose: `probe["status"] == "ok"`
    would also pass if the probe silently dropped a component, which is the
    failure this test exists to catch. The component *set* is the assertion
    that a healthy evidence pipeline has exactly graph and policy in it.
    """
    probe = health_mod._probe_evidence()
    assert probe["status"] == "ok", probe
    assert set(probe["components"]) == {
        "evidence_graph", "evidence_policy"
    }, (
        "the evidence probe's component set changed; if a module was added, "
        "assert it, and if one was removed, say here why")


def test_runtime_status_does_not_claim_unreachable_engines():
    """Slurm and Ray must not be reported as connected on a laptop."""
    with TestClient(app) as client:
        assert client.post("/api/runtime/status", json={}).status_code == 401
        r = client.post("/api/runtime/status", json={}, headers=_h())
    assert r.status_code == 200
    body = r.json()

    assert body["status"] in ("connected", "unavailable")
    assert set(body["engine_detail"]) == {
        "local", "distributed", "kubernetes", "slurm", "ray"
    }
    if body["status"] == "connected":
        assert body["engines"], "claimed connected with no reachable engine"
    else:
        assert body["engines"] == []
        assert body["reason"]

    # Each engine carries a probed verdict, never a bare name.
    for name, detail in body["engine_detail"].items():
        assert isinstance(detail["reachable"], bool), name
        assert detail["detail"], name
