"""Subsystem health must be probed, not assumed.

The old `/health` returned `{"status": "healthy"}` unconditionally and
`/runtime/status` returned `"connected"` with five engine names, none of which
were ever checked. These tests pin the replacement: every subsystem is actually
exercised, and nothing is reported ok without a probe having run.
"""
import pytest
from fastapi.testclient import TestClient

from backend.core import health as health_mod
from backend.main import app


def test_health_endpoint_stays_a_liveness_probe():
    """/health must not become slow: it answers 'am I alive', nothing else."""
    with TestClient(app) as client:
        r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "healthy"}


def test_subsystem_endpoint_returns_a_report_for_every_subsystem():
    with TestClient(app) as client:
        r = client.get("/health/subsystems")
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
        body = client.get("/health/subsystems").json()
    assert body["status"] != "healthy"
    for probe in body["subsystems"].values():
        assert probe["status"] != "healthy"


def test_can_capable_of_live_evidence_depends_on_the_probes():
    """The headline flag must follow the probes, not default to True."""
    with TestClient(app) as client:
        body = client.get("/health/subsystems").json()
    worst = body["subsystems"]
    assert body["capable_of_live_evidence"] == (
        all(p["status"] == "ok" for p in worst.values())
    )


def test_subsystem_filter_runs_only_what_was_asked_for():
    with TestClient(app) as client:
        body = client.get("/health/subsystems?subsystem=process,evidence").json()
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


def test_evidence_probe_checks_the_boundary_imports():
    probe = health_mod._probe_evidence()
    assert probe["status"] == "ok", probe
    assert set(probe["components"]) == {
        "boundary", "evidence_graph", "evidence_policy"
    }


def test_runtime_status_does_not_claim_unreachable_engines():
    """Slurm and Ray must not be reported as connected on a laptop."""
    with TestClient(app) as client:
        r = client.post("/api/runtime/status", json={})
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
