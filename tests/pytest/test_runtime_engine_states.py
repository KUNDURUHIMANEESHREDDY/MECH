"""Execution engines must report a capability ladder, not one boolean.

`/runtime/status` used to collapse four different questions into a single
`reachable` flag:

    installed       the client library or binary is on this machine
    configured      it names an endpoint / holds credentials
    reachable       something at that endpoint answered
    execution_ready it has capacity to actually run a job

`importlib.util.find_spec("ray") is not None` answered all four with `True`.
A developer laptop with `ray` pip-installed and no cluster at all was reported
`connected`, and a Slurm install with no `sbatch` controller was reported the
same. Nothing distinguished "I could import a library" from "a job would run".

The probe is now a ladder, and `_ladder` clamps it: no engine can hold a rung
unless every rung below it holds. These tests pin that discipline.
"""
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.api import dispatcher as d  # noqa: E402

ENGINES = ("local", "distributed", "kubernetes", "slurm", "ray")


# ── the ladder itself ────────────────────────────────────────────────── #

def test_the_ladder_declares_four_rungs_in_order():
    assert d.ENGINE_STATES == (
        "installed", "configured", "reachable", "execution_ready")


@pytest.mark.parametrize("rungs,expected_state", [
    ((False, False, False, False), "absent"),
    ((True, False, False, False), "installed"),
    ((True, True, False, False), "configured"),
    ((True, True, True, False), "reachable"),
    ((True, True, True, True), "execution_ready"),
])
def test_state_names_the_highest_rung_held(rungs, expected_state):
    report = d._ladder(*rungs, "why")
    assert report["state"] == expected_state


@pytest.mark.parametrize("rungs", [
    # Each case claims a higher rung without the rungs beneath it. A probe
    # that got its own logic wrong must not be able to over-claim.
    (False, True, True, True),
    (True, False, True, True),
    (True, True, False, True),
    (False, False, False, True),
])
def test_the_ladder_clamps_over_claims(rungs):
    report = d._ladder(*rungs, "why")
    held = [report[s] for s in d.ENGINE_STATES]
    # Monotone: once a rung is False, every rung above it is too.
    for i in range(1, len(held)):
        assert held[i] <= held[i - 1], (
            f"{d.ENGINE_STATES[i]} claimed without {d.ENGINE_STATES[i - 1]}")
    if not report["installed"]:
        assert report["state"] == "absent"


def test_the_ladder_reports_bools_and_a_reason():
    report = d._ladder(True, True, False, False, "controller silent")
    for rung in d.ENGINE_STATES:
        assert isinstance(report[rung], bool)
    assert report["detail"] == "controller silent"


# ── installed is not reachable ───────────────────────────────────────── #
# The actual defect. These force the "library present" half to be true and
# the endpoint half to be false, which is exactly a developer machine.

def test_ray_installed_but_unconfigured_is_not_reachable(monkeypatch):
    monkeypatch.setattr(d, "_module_present", lambda m: True)
    monkeypatch.delenv("RAY_ADDRESS", raising=False)
    report = d._probe_ray()
    assert report["installed"] is True
    assert report["configured"] is False
    assert report["reachable"] is False
    assert report["state"] == "installed"


def test_ray_configured_but_silent_is_not_reachable(monkeypatch):
    monkeypatch.setattr(d, "_module_present", lambda m: True)
    monkeypatch.setenv("RAY_ADDRESS", "ray://10.255.255.1:6379")
    monkeypatch.setattr(d, "_tcp_reachable", lambda *a, **k: False)
    report = d._probe_ray()
    assert report["installed"] is True
    assert report["configured"] is True
    assert report["reachable"] is False
    assert report["state"] == "configured"


def test_kubernetes_installed_but_unconfigured_is_not_reachable(monkeypatch):
    monkeypatch.setattr(d, "_module_present", lambda m: True)
    monkeypatch.setattr(d, "_kubeconfig_endpoint", lambda: (None, None, None))
    report = d._probe_kubernetes()
    assert report["installed"] is True
    assert report["reachable"] is False
    assert report["state"] == "installed"


def test_slurm_on_path_but_unnamed_is_not_reachable(monkeypatch):
    monkeypatch.setattr(d, "_binary_on_path", lambda n: f"/usr/bin/{n}")
    for var in ("SLURM_CLUSTER_NAME", "SCRATCH", "SLURM_CONF"):
        monkeypatch.delenv(var, raising=False)
    report = d._probe_slurm()
    assert report["installed"] is True
    assert report["reachable"] is False
    assert report["state"] == "installed"


def test_a_silent_controller_is_reachable_but_not_execution_ready(monkeypatch):
    """Answering `sinfo` with nothing is a reach, not a runnable cluster."""
    monkeypatch.setattr(d, "_binary_on_path", lambda n: f"/usr/bin/{n}")
    monkeypatch.setenv("SLURM_CLUSTER_NAME", "hpc")
    monkeypatch.setattr(d, "_run_probe", lambda *a, **k: (0, ""))
    report = d._probe_slurm()
    assert report["reachable"] is True
    assert report["execution_ready"] is False
    assert report["state"] == "reachable"


def test_a_controller_reporting_partitions_is_execution_ready(monkeypatch):
    monkeypatch.setattr(d, "_binary_on_path", lambda n: f"/usr/bin/{n}")
    monkeypatch.setenv("SLURM_CLUSTER_NAME", "hpc")
    monkeypatch.setattr(d, "_run_probe", lambda *a, **k: (0, "gpu\ncpu\n"))
    report = d._probe_slurm()
    assert report["execution_ready"] is True
    assert report["state"] == "execution_ready"


def test_a_failing_controller_is_not_even_reachable(monkeypatch):
    monkeypatch.setattr(d, "_binary_on_path", lambda n: f"/usr/bin/{n}")
    monkeypatch.setenv("SLURM_CLUSTER_NAME", "hpc")
    monkeypatch.setattr(d, "_run_probe", lambda *a, **k: (1, "slurm_load_partitions failed"))
    report = d._probe_slurm()
    assert report["reachable"] is False
    assert report["state"] == "configured"


# ── distributed must not be reachable from a placeholder registry ─────── #

def test_distributed_does_not_claim_reachability(monkeypatch):
    """`ResourceManager` seeds three hardcoded GPUs and a fake EPYC cluster.

    Reading that registry as evidence of a reachable cluster is how a laptop
    gets reported as a datacentre. There is no transport in
    `DistributedWorker`, so there is no endpoint to reach, and the probe has
    to say so rather than infer one from the placeholder profiles.
    """
    report = d._probe_distributed()
    assert report["reachable"] is False
    assert report["execution_ready"] is False
    assert report["state"] in ("installed", "absent")


def test_distributed_explains_why_it_stops_where_it_does():
    report = d._probe_distributed()
    if report["installed"]:
        assert "transport" in report["detail"] or "configure" in report["detail"]


# ── the aggregate ────────────────────────────────────────────────────── #

def test_every_engine_reports_all_four_rungs():
    for name in ENGINES:
        report = d._probe_engine(name)
        for rung in d.ENGINE_STATES:
            assert isinstance(report[rung], bool), f"{name}.{rung}"
        assert report["state"] in d.ENGINE_STATES + ("absent",), name
        assert report["detail"], name


def test_an_exploding_probe_fails_closed(monkeypatch):
    """A probe that raises must report nothing achieved, and say why."""
    def boom():
        raise RuntimeError("probe exploded")
    monkeypatch.setattr(d, "_ENGINE_PROBES", {
        name: (boom if name == "ray" else _ok_report)
        for name in ENGINES})
    report = d._probe_engine("ray")
    for rung in d.ENGINE_STATES:
        assert report[rung] is False
    assert report["state"] == "absent"
    assert "probe exploded" in report["detail"]


def test_an_unnamed_engine_fails_closed():
    report = d._probe_engine("quantum-grid")
    assert report["state"] == "absent"
    assert all(report[r] is False for r in d.ENGINE_STATES)
    assert "quantum-grid" in report["detail"]


def _ok_report():
    return d._ladder(True, True, True, True, "fine")


def test_connected_status_requires_a_reachable_engine():
    report = d.runtime_status()
    reachable = [n for n, v in report["engine_detail"].items() if v["reachable"]]
    ready = [n for n, v in report["engine_detail"].items()
             if v["execution_ready"]]
    assert report["engines"] == reachable
    assert report["execution_ready_engines"] == ready
    if report["status"] == "connected":
        assert report["engines"]
    else:
        assert not report["engines"]
        assert report["reason"]


def test_execution_ready_never_exceeds_reachable():
    report = d.runtime_status()
    for name, detail in report["engine_detail"].items():
        if detail["execution_ready"]:
            assert detail["reachable"], name
        if name in report["execution_ready_engines"]:
            assert name in report["engines"], name


def test_the_route_reports_the_ladder_over_http():
    from fastapi.testclient import TestClient
    from backend.core import auth as auth_mod
    from backend.main import app

    with TestClient(app) as client:
        r = client.post("/api/runtime/status", json={},
                        headers=auth_mod.auth_headers())
    assert r.status_code == 200
    body = r.json()
    assert set(body["engine_detail"]) == set(ENGINES)
    assert "execution_ready_engines" in body
    for name, detail in body["engine_detail"].items():
        for rung in d.ENGINE_STATES:
            assert isinstance(detail[rung], bool), f"{name}.{rung}"
