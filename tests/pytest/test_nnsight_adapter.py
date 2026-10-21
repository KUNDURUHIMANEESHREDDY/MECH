"""The NNsight adapter must report its dependency's absence, not work around it.

``nnsight`` is not installed in this environment, and AGENTS.md is explicit:
absence is reported as ``NOT_EXECUTED`` with no data, never as a partially
working engine. These tests pin both halves -- the real absent state, and a
package injected into ``sys.modules`` to prove the adapter would notice one
without claiming anything it has not done.

The injected module is not an execution double. It must never run, and the
tests assert it was never called. The adapter's execution path follows
NNsight's documented trace API but has not been exercised against a real
install here, so no test pretends otherwise: the only execution-path assertions
are about failing closed (``FAILED``, no data, no leftover job scope).
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.core.execution_status import (  # noqa: E402
    DATA_FIELDS,
    FAILED,
    IMPLEMENTED,
    NOT_EXECUTED,
)
from backend.interpretability import nnsight_adapter as adapter  # noqa: E402


@pytest.fixture(autouse=True)
def _fresh_probe():
    """A cached import must not leak between tests, in either direction."""
    adapter._CACHE.clear()
    adapter._ACTIVE.clear()
    yield
    adapter._CACHE.clear()
    adapter._ACTIVE.clear()


def _require_absent():
    """Skip rather than lie if this environment ever gains the package."""
    if importlib.util.find_spec("nnsight") is not None:
        pytest.skip("nnsight is installed; the absent-package path cannot be exercised")


def _assert_no_measurement(payload):
    leaked = sorted(set(payload) & DATA_FIELDS)
    assert not leaked, f"payload carried data fields: {leaked}"


def _fake_nnsight(monkeypatch, *, version="0.4.1", entry_point=True, calls=None):
    """Inject a bare module. Nothing in the adapter may call into it here."""
    module = types.ModuleType("nnsight")
    if version is not None:
        module.__version__ = version
    if entry_point:
        class LanguageModel:
            def __init__(self, *args, **kwargs):
                if calls is not None:
                    calls.append((args, kwargs))

        module.LanguageModel = LanguageModel
    monkeypatch.setitem(sys.modules, "nnsight", module)
    return module


# ── absent: the state this build is actually in ──────────────────────

def test_absent_package_is_not_executed_with_a_reason_and_no_version():
    _require_absent()
    report = adapter.describe()

    assert report["available"] is False
    assert report["status"] == NOT_EXECUTED
    assert report["version"] is None
    assert report["reason"], "an unavailable adapter must say why"
    assert "nnsight" in report["reason"]
    assert report["adapter"] == "nnsight"
    assert report["package"] == "nnsight"


def test_no_operation_is_claimed_while_the_package_is_absent():
    _require_absent()
    report = adapter.capabilities()

    assert report["supported_operations"] == []
    assert set(report["operations"]) == set(adapter.OPERATIONS)
    assert not any(report["operations"].values()), (
        "a capability cannot be claimed without the package behind it"
    )


def test_run_without_the_package_is_not_executed_and_carries_no_data():
    _require_absent()
    request = adapter.build_trace_request(
        "The capital of France is",
        ["transformer.h[0].output"],
        interventions=[{"site_id": "transformer.h[0].output", "op": "zero"}],
    )

    # The model is touched last on purpose: the absent dependency is reported
    # before anything is asked of a model.
    result = adapter.run_trace(request, model=object())

    assert result["status"] == NOT_EXECUTED
    assert result["status"] != FAILED
    assert "nnsight" in result["reason"]
    assert result["version"] is None
    assert result["measured"] is False
    _assert_no_measurement(result)
    for key in ("saves", "activations", "sites", "interventions", "provenance"):
        assert key not in result, f"a NOT_EXECUTED payload may not carry {key!r}"


# ── request construction: pure data, no execution ────────────────────

def test_request_construction_works_without_the_package_and_is_not_an_execution():
    _require_absent()
    request = adapter.build_trace_request(
        "The capital of France is",
        [
            {
                "site_id": "resid_0",
                "canonical_path": "transformer.h[0].output",
                "component": "residual",
                "layer": 0,
                "axis_names": ["batch", "sequence", "hidden"],
            },
            "transformer.h[1].resid_post",
        ],
        interventions=[{"site_id": "resid_0", "op": "set", "value": 0.0}],
        job_id="job-1",
    )

    assert request["executed"] is False
    assert request["status"] == NOT_EXECUTED
    assert request["job_id"] == "job-1"
    assert request["available"] is False
    assert request["version"] is None
    assert request["saves"] == {
        "resid_0": "transformer.h[0].output",
        "transformer.h[1].resid_post": "transformer.h[1].resid_post",
    }
    assert request["sites"][0]["component"] == "residual"
    assert request["sites"][0]["axis_names"] == ["batch", "sequence", "hidden"]
    assert request["interventions"] == [
        {
            "site_id": "resid_0",
            "canonical_path": "transformer.h[0].output",
            "op": "set",
            "value": 0.0,
        }
    ]


def test_a_request_must_declare_sites_and_each_intervention_must_name_one():
    with pytest.raises(adapter.TraceRequestError):
        adapter.build_trace_request("hi", [])

    with pytest.raises(adapter.TraceRequestError):
        # names no declared site and supplies no path
        adapter.build_trace_request(
            "hi", ["transformer.h[0].output"],
            interventions=[{"site_id": "not_a_site", "op": "zero"}],
        )

    with pytest.raises(adapter.TraceRequestError):
        adapter.build_trace_request(
            "hi", ["transformer.h[0].output"],
            interventions=[{"site_id": "transformer.h[0].output", "op": "replace"}],
        )

    with pytest.raises(adapter.TraceRequestError):
        adapter.build_trace_request(
            "hi", ["transformer.h[0].output"],
            interventions=[{"site_id": "transformer.h[0].output", "op": "set"}],
        )

    with pytest.raises(adapter.TraceRequestError):
        adapter.build_trace_request("", ["transformer.h[0].output"])


def test_site_paths_resolve_attributes_and_indices():
    """The resolver is the one piece of real logic that runs without nnsight."""

    class Block:
        output = "block-output"

    class Transformer:
        h = [Block(), Block()]

    class Model:
        transformer = Transformer()

    assert adapter._resolve_site(Model(), "transformer.h[1].output") == "block-output"
    assert adapter._parse_site_path("transformer.h[0].output") == [
        "transformer", "h", 0, "output",
    ]

    with pytest.raises(adapter.TraceRequestError):
        adapter._resolve_site(Model(), "transformer.h[9].output")
    with pytest.raises(adapter.TraceRequestError):
        adapter._parse_site_path("transformer..h")


# ── injected package: noticed, versioned, still not executed ─────────

def test_an_injected_package_is_reported_available_and_still_not_executed(monkeypatch):
    calls: list = []
    _fake_nnsight(monkeypatch, version="0.4.1", calls=calls)

    report = adapter.describe()
    assert report["available"] is True
    assert report["status"] == IMPLEMENTED
    assert report["version"] == "0.4.1"
    assert report["entry_point"] == "LanguageModel"
    assert all(report["operations"].values())

    request = adapter.build_trace_request("hello", ["transformer.h[0].output"])
    assert request["available"] is True
    assert request["version"] == "0.4.1"
    assert request["executed"] is False
    assert request["status"] == NOT_EXECUTED

    assert calls == [], "building a request must not construct or run a model"


def test_a_module_without_a_trace_entry_point_is_still_not_available(monkeypatch):
    _fake_nnsight(monkeypatch, version="9.9.9", entry_point=False)

    report = adapter.describe()
    assert report["available"] is False
    assert report["status"] == NOT_EXECUTED
    assert report["supported_operations"] == []
    assert not any(report["operations"].values())
    assert "entry point" in report["reason"]


def test_a_failed_trace_is_failed_not_a_capture_and_leaves_no_scope(monkeypatch):
    _fake_nnsight(monkeypatch, version="0.4.1")

    class Model:
        def trace(self, prompt):
            raise RuntimeError("dispatch failed")

    request = adapter.build_trace_request(
        "hello",
        ["transformer.h[0].output"],
        interventions=[{"site_id": "transformer.h[0].output", "op": "zero"}],
    )
    result = adapter.run_trace(request, model=Model())

    assert result["status"] == FAILED
    assert result["status"] != NOT_EXECUTED
    assert "dispatch failed" in result["reason"]
    assert result["measured"] is False
    _assert_no_measurement(result)
    assert "saves" not in result
    assert adapter.active_traces() == [], (
        "a job's hook scope must be dropped even when the trace throws"
    )
