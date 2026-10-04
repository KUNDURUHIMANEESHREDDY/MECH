"""Provenance must originate from the measurement layer, never from a wrapper.

The rule under test
------------------
**A wrapper may not label a result `live` because the function exists, imported
successfully, or was reachable.** Only the layer that did the measurement may say
`live`.

Two defects motivated this file, both in the same trust boundary:

* `backend/agents/executor.py:_call` did `res.setdefault("provenance", "live")`
  and then defaulted *every* field to `live`. A returned
  `{"status": "unavailable"}` became a live measurement, and its `reason` — the
  explanation of why nothing was measured — was itself marked live.
* `backend/api/dispatcher.py:/infer` did the same, gated only on
  `engine.is_available()`, which reports that torch and transformers *imported*.
  It does not report that a forward pass ran.

And the second was demonstrably wrong rather than merely fragile, because
`gpt2_engine` loads exactly one model and `infer()` echoed the caller's
`model_name` back verbatim. Measured before the fix:

    infer("Hello", "gpt2-large")["model_name"] -> "gpt2-large"
    infer("Hello", "gpt2-large")["d_model"]    -> 768      # gpt2-large is 1280

so a caller asking for gpt2-large received a response labelled `live` naming
gpt2-large, while every tensor came from gpt2-small.
"""

from __future__ import annotations

import inspect
from typing import Any, Dict

import pytest

from backend.core.provenance import (
    LIVE,
    UNAVAILABLE,
    attest_measurement,
    is_failure_status,
    pass_through,
    withhold,
)


# ── the primitives ─────────────────────────────────────────────────────────

@pytest.mark.parametrize("status", [
    "error", "failed", "failure", "unavailable", "not_run", "blocked",
    "skipped", "timeout", "invalid", "rejected", "declined", "ERROR", "Failed",
])
def test_failure_statuses_are_recognised(status):
    assert is_failure_status(status)


@pytest.mark.parametrize("status", [
    "ok", "completed", "live", "measured", "success", None, "", 0,
])
def test_success_and_absent_statuses_are_not_failures(status):
    assert not is_failure_status(status)


def test_attest_refuses_to_stamp_live_on_a_failure():
    """The guard that stops a mistaken caller manufacturing a live claim."""
    for status in ("error", "unavailable", "failed", "not_run"):
        result = attest_measurement(
            {"status": status, "value": 1.0}, model_loaded="gpt2")
        assert result["provenance"] != LIVE, status
        assert result["provenance"] == UNAVAILABLE
        assert "nothing was measured" in result["reason"]


def test_attest_records_which_weights_ran():
    result = attest_measurement(
        {"value": 1.0}, model_loaded="gpt2", model_requested="gpt2-large")
    assert result["provenance"] == LIVE
    assert result["model_loaded"] == "gpt2"
    assert result["model_requested"] == "gpt2-large"
    assert result["model_mismatch"] is True

    same = attest_measurement(
        {"value": 1.0}, model_loaded="gpt2", model_requested="gpt2")
    assert same["model_mismatch"] is False


def test_attest_never_overwrites_a_deliberate_label():
    """A measurement layer that chose `seeded` keeps it."""
    result = attest_measurement({"provenance": "seeded", "value": 1.0})
    assert result["provenance"] == "seeded"
    assert "model_mismatch" not in result


def test_attest_does_not_mark_framing_fields_as_measured():
    """`reason` and `status` describe the record, not a measured quantity."""
    result = attest_measurement(
        {"tokens": [1], "status": "ok", "reason": "ok", "elapsed_ms": 3})
    assert result["field_provenance"]["tokens"] == LIVE
    for framing in ("status", "reason", "elapsed_ms"):
        assert framing not in result["field_provenance"], framing


def test_pass_through_withholds_an_unattested_record():
    """The core of the fix: absence of a label means absence of a claim."""
    result = pass_through({"status": "unavailable", "value": 1.0})
    assert result["provenance"] != LIVE
    assert result["provenance"] == UNAVAILABLE
    assert "cannot be presented as a measurement" in result["reason"]


def test_pass_through_preserves_an_existing_label():
    for label in (LIVE, "seeded", "reference"):
        assert pass_through({"provenance": label})["provenance"] == label


def test_withhold_does_not_downgrade_a_more_specific_label():
    result = withhold({"provenance": "seeded"}, reason="later")
    assert result["provenance"] == "seeded"
    assert result["reason"] == "later"


# ── the wrappers ───────────────────────────────────────────────────────────

def test_executor_never_upgrades_an_unattested_engine_result():
    """`executor._call` used `setdefault("provenance", "live")`.

    Driven through a stub engine so the vulnerable inputs are exercised directly:
    a dict with no provenance, a dict reporting unavailability, and a non-dict.
    """
    import asyncio

    from backend.agents import executor as executor_mod

    class StubEngine:
        """Stands in for gpt2_engine, returning whatever the test dictates."""

        def __init__(self, value):
            self._value = value

        @staticmethod
        def is_available():
            return True

        def __getattr__(self, name):
            def call(*a, **k):
                return self._value
            return call

    async def call(value):
        ex = executor_mod.Executor()
        ex._engine = lambda: StubEngine(value)
        return await ex._call("some_op")

    # 1. A dict with no provenance must not become live.
    out = asyncio.run(call({"value": 1.0}))
    assert out["provenance"] != LIVE, out
    assert out["provenance"] == UNAVAILABLE

    # 2. An explicit failure must stay a failure.
    out = asyncio.run(call({"status": "unavailable", "reason": "no weights"}))
    assert out["provenance"] != LIVE
    assert out["reason"] == "no weights"

    # 3. A bare value must not become a live record.
    out = asyncio.run(call(None))
    assert out["provenance"] != LIVE
    assert out["status"] == "unmeasured"

    # 4. An engine that raises must not become live.
    class Boom(StubEngine):
        def __getattr__(self, name):
            def call(*a, **k):
                raise RuntimeError("forward pass failed")
            return call
    ex = executor_mod.Executor()
    ex._engine = lambda: Boom(None)
    out = asyncio.run(ex._call("some_op"))
    assert out["provenance"] != LIVE
    assert out["status"] == "error"


def _provenance_setdefault_calls(func) -> list:
    """`setdefault("<provenance key>", "live")` calls inside a function.

    Matched on the AST rather than the source text: the comments explaining this
    fix quote the removed expression verbatim, so a text search reports the
    explanation as if it were the defect.
    """
    import ast

    try:
        tree = ast.parse(inspect.getsource(func).lstrip())
    except (OSError, TypeError, SyntaxError):
        return []
    found = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "setdefault"):
            continue
        first = node.args[0] if node.args else None
        if not (isinstance(first, ast.Constant)
                and isinstance(first.value, str)
                and "provenance" in first.value):
            continue
        second = node.args[1] if len(node.args) > 1 else None
        if isinstance(second, ast.Constant) and second.value == LIVE:
            found.append(node.lineno)
    return found


def test_dispatcher_infer_does_not_infer_live():
    """`/infer` did `res.setdefault("provenance", "live")` behind
    `engine.is_available()`."""
    from backend.api import dispatcher

    assert not _provenance_setdefault_calls(dispatcher.infer), (
        "infer must not default provenance; it may only pass through what the "
        "measurement layer attested")

    source = inspect.getsource(dispatcher.infer)
    assert "pass_through" in source


def test_no_module_defaults_provenance_to_live_via_setdefault():
    """A repo-wide guard on the exact anti-pattern.

    This is what found the two instances the audit did not name --
    `agents/critic.py` and `agents/inspector.py` -- alongside the two it did.
    """
    import ast
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    offenders = []
    for path in (root / "backend").rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        try:
            tree = ast.parse(text)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "setdefault"):
                continue
            first = node.args[0] if node.args else None
            if not (isinstance(first, ast.Constant)
                    and isinstance(first.value, str)
                    and "provenance" in first.value):
                continue
            second = node.args[1] if len(node.args) > 1 else None
            if isinstance(second, ast.Constant) and second.value == LIVE:
                offenders.append(f"{path.relative_to(root)}:{node.lineno}")
    assert not offenders, (
        "provenance defaulted to live, which invents a measurement label: "
        + "; ".join(offenders))


# ── the real engine ────────────────────────────────────────────────────────

def test_engine_infer_reports_the_weights_that_actually_ran():
    """The concrete defect: `infer()` echoed the caller's model_name while the
    forward pass used the single loaded checkpoint."""
    pytest.importorskip("torch")
    from backend.services import gpt2_engine

    assert gpt2_engine.is_available(), "weights unavailable; nothing to attest"

    requested = "gpt2-some-model-that-is-not-loaded"
    result: Dict[str, Any] = gpt2_engine.infer("Hello", requested)

    assert result["provenance"] == LIVE
    assert result["model_loaded"], "a live response must name its weights"
    assert result["model_requested"] == requested
    assert result["model_mismatch"] is True, (
        "asking for an unloaded model must be visible in the record")

    # The response must not claim to be the model that was asked for.
    assert result.get("model_name") is None, (
        "model_name must not echo the requested model; it described weights "
        "that never loaded")

    # Its real geometry must be self-consistent with the model it names.
    layers = result["n_layers"]
    assert layers == 12 and result["d_model"] == 768, (
        "the loaded checkpoint is gpt2 (12 layers, d_model 768)")

    # A matching request is not a mismatch.
    matching = gpt2_engine.infer("Hello", result["model_loaded"])
    assert matching["model_mismatch"] is False


def test_engine_measures_and_attests_consistently():
    """A live response's field_provenance must actually cover its data fields."""
    pytest.importorskip("torch")
    from backend.services import gpt2_engine

    result = gpt2_engine.infer("Hello there", "gpt2")
    assert result["provenance"] == LIVE

    fields = result["field_provenance"]
    for field in ("tokens", "generated_text", "attention_maps",
                  "neuron_activations"):
        assert field in result, field
        assert fields.get(field) == LIVE, (
            f"{field} is present in a live response but not marked live")