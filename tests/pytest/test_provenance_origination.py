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

import ast
import inspect
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import pytest

ROOT = Path(__file__).resolve().parents[2]

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


# ── Classification of every direct `provenance: "live"` stamp ──────────────
#
# The rule is that provenance is originated by the layer that did the
# measurement. Widening the scanner from `setdefault` alone to all three shapes
# it can take took the site count from 1 to 36 across 16 modules, so the tables
# below are the real review surface, not a convention.
#
# Measured: 36 sites, and all 36 are dict literals. The `setdefault` shape is
# now empty -- the P0 fix eliminated it -- which is why the blind spot survived
# so long: the shape that remained was the one shape nobody was looking at.
#
# The categories differ in *what supports the claim*, and the distinction is the
# whole point:
#
#   MEASUREMENT_LAYERS   knows because it measured.
#   GUARDED_WRAPPERS     knows because it refused to emit `live` unless
#                        something upstream had already claimed it, so the label
#                        is inherited under a check rather than invented.
#   LIVE_BY_CONSTRUCTION knows because the value cannot be stale -- it came out
#                        of the running system on this very call.
#   KNOWN_OVERCLAIM      does not know. Kept as its own category rather than
#                        quietly allowed, so the residual risk stays visible.
#
# The first three are defensible. The fourth is not, and listing it separately is
# the point: a guard that cannot tell "justified" from "not justified" is not
# making a judgement.
#
# Each entry is keyed by module, never by line number. Line keys break the first
# time anyone edits the file above the stamp, and a guard that breaks on
# formatting gets deleted rather than obeyed.
#
# Every reason was checked against the code, not inferred from the module name.

#: Modules that computed the value they are labelling.
MEASUREMENT_LAYERS: Dict[str, str] = {
    "backend/science/models/gpt2_adapter.py":
        "runs GPT-2 forward passes and captures per-head and per-neuron "
        "activations; the numbers it labels are the ones it just produced",
    "backend/science/reproducibility/ioi_pipeline.py":
        "runs the IOI circuit reproduction against loaded weights and measures "
        "faithfulness, minimality and recovery itself",
    "backend/science/reproducibility/logit_lens_pipeline.py":
        "computes the logit lens over real hidden states at each layer",
    "backend/science/reproducibility/greater_than_pipeline.py":
        "computes per-layer importances for the greater-than task by ablating "
        "layers and re-measuring",
    "backend/interpretability/sae/loader.py":
        "loads a real SAE checkpoint and computes feature activations from it",
    "backend/interpretability/discovery/live_discovery.py":
        "accumulates per-prompt faithfulness and completeness from real "
        "ablations; guards its own mean_faithfulness with _adequacy()",
    "backend/interpretability/discovery/algorithms/causal_scrubbing.py":
        "runs real causal-scrubbing interventions. Its stamp is a nested dict "
        "with an inner provenance key, which is deliberate -- the evidence "
        "policy resolves dict provenance for this documented reason",
    "backend/interpretability/discovery/algorithms/path_patching.py":
        "runs real path patching; same nested-dict provenance shape as "
        "causal_scrubbing, for the same documented reason",
    "backend/validation/live_validation.py":
        "derives confidence from measured replication_rate, recovery and "
        "minimality; it runs the passes it reports on",
    "backend/validation/benchmark_runner.py":
        "computes accuracy and robustness from the per-prompt clean/corrupted "
        "runs it just executed",
    "backend/analysis/experiment_runner.py":
        "calls engine.list_neurons and derives max/mean/variance/sparsity from "
        "the returned neurons; returns provenance='unavailable', measured=False "
        "when the engine reports non-ok, so its 'live' is earned by a "
        "measurement it made (the container's own defaults are closed)",
    "backend/experiments/attention_experiment.py":
        "calls engine.layer_detail and computes per-head importance plus "
        "max/mean/min statistics from the returned heads; returns status "
        "'unavailable' rather than stamping 'live' when the engine cannot "
        "answer",
}

#: Wrappers that stamp `live` only after verifying an inner claim of liveness.
#: The label is inherited under a check, not invented.
GUARDED_WRAPPERS: Dict[str, str] = {
    "backend/agents/executor.py":
        "returns early unless res['provenance'] == 'live' and mock_mode is not "
        "True, so the echoed label is one it checked rather than one it chose",
    "backend/agents/critic.py":
        "gated by validation_is_live(res), and its two other stamps sit inside "
        "the path that already required reproduction_is_live(run). On the "
        "metric-mapping failure it reports status=unavailable with "
        "provenance=live, which is coherent: the run was live, the mapping was "
        "not. field_provenance marks the unmapped fields unavailable and "
        "publication_eligible stays False",
    "backend/agents/scribe.py":
        "two of its four stamps are behind publication_block_reason(), which "
        "returns non-empty unless every discovery and validation step clears "
        "discovery_is_live/validation_is_live; the other two follow a write to "
        "the live evidence graph that already returned a node or edge count",
    "backend/agents/discoverer.py":
        "stamps live only after discovery_is_live(res) passes, and the guarded "
        "keys are now spread first so a later **res cannot override the verdict",
}

#: Operational endpoints and writers. `live` here means "not a fixture" -- the "
#: value was read from or written to the running system on this call. It does "
#: not mean a scientific measurement was made, and it cannot be mistaken for
#: one: nothing in evidence_policy promotes on provenance alone
#: (discovery_is_live additionally requires _opted_in), and
#: test_operational_reads_cannot_opt_into_science makes that executable.
LIVE_BY_CONSTRUCTION: Dict[str, str] = {
    "backend/api/dispatcher.py":
        "12 stamps, all on operational endpoints -- /settings, /projects, "
        "/logs, /recent-files, /build, /portal/summary and the compute-engine "
        "probes. Each returns live application state, so `live` distinguishes "
        "it from a fixture, which is the only distinction this field can make "
        "here. None of these endpoints measures anything scientific and none "
        "opts into validation or publication",
    "backend/mcp_server/server.py":
        "every stamp flows through the module `_ok()` helper on MCP tool "
        "responses -- file listings, loop status, runtime probes, git and "
        "allowlisted test/build results. Each reports an action this process "
        "just performed, so `live` distinguishes it from a fixture. The "
        "module sets no eligibility flag anywhere, so nothing here can "
        "promote into validation or publication",
    "backend/mcp_server/allowed_commands.py":
        "same `_ok()` shape as the MCP server module above, on allowlisted "
        "test/build execution results. Reports what the spawned command "
        "returned; sets no eligibility flag anywhere",
}

#: Modules whose stamp is a claim the code cannot actually support. Kept as its
#: own category rather than quietly allowed, so the residual risk stays visible
#: and this test fails if one of them is deleted rather than fixed.
KNOWN_OVERCLAIM: Dict[str, str] = {
    "backend/interpretability/discovery/discovery_quality_score.py":
        "a weighted arithmetic aggregate over six caller-supplied floats. It "
        "verified that the inputs are PRESENT, not that they were MEASURED, so "
        "it stamps live on a claim it cannot support: any six numbers produce "
        "an A+ grade labelled live. Bounded by two facts -- no production caller "
        "exists (discovery_engine.py only instantiates the scorer), and the "
        "only exercise of the claim is a unit test that supplies the six old "
        "fabricated defaults. The fix needs a provenance value meaning "
        "'computed', which does not exist in the vocabulary yet; inventing one "
        "has blast radius across evidence_policy, the API and the UI, so it is "
        "tracked as a backlog item rather than decided here",
}

#: Excluded from the scan entirely, with the reason. Exemption is the strongest
#: statement this guard can make, so each one has to justify itself.
SCAN_EXEMPT: Dict[str, str] = {
    "backend/core/provenance.py":
        "is the helper that stamps live; it is the rule, not a violation of it",
    "scripts/capture_results.py":
        "a reporting script. Its meta provenance is now DERIVED after the "
        "captures run -- it compares which of the 8 expected sections are "
        "present and withholds the label if any is missing -- instead of being "
        "asserted up front as it was, which is how the README's own results "
        "artifact came to claim `live` before a single forward pass",
}

#: Every classified module, for disjointness and coverage checks.
ALL_CLASSIFIED = (
    set(MEASUREMENT_LAYERS) | set(GUARDED_WRAPPERS)
    | set(LIVE_BY_CONSTRUCTION) | set(KNOWN_OVERCLAIM) | set(SCAN_EXEMPT)
)

SETDEFAULT = "setdefault"
DICT_LITERAL = "dict-literal"
SUBSCRIPT = "subscript-assign"


def _live_stamp_sites() -> List[Tuple[str, int, str]]:
    """Every site that writes `provenance: "live"` without the helper.

    Returned as ``(relative path, line, shape)``. Three shapes, because the
    defect takes three shapes:

      * ``res.setdefault("provenance", "live")``
      * ``{"provenance": "live", ...}``          -- a dict literal
      * ``obj["provenance"] = "live"``           -- a subscript assignment

    The original guard matched only the first. A dict literal is the most
    natural way to write this bug and was completely invisible to it -- which is
    how `scripts/capture_results.py` came to stamp every artifact it wrote,
    including the one the README's Results section is generated from.
    """
    sites: List[Tuple[str, int, str]] = []
    for path in sorted((ROOT / "backend").rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        relative = path.relative_to(ROOT).as_posix()
        try:
            tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
        except SyntaxError:
            continue

        for node in ast.walk(tree):
            found: List[Tuple[int, str]] = []
            if isinstance(node, ast.Call):
                func = node.func
                name = (func.attr if isinstance(func, ast.Attribute)
                        else func.id if isinstance(func, ast.Name) else None)
                if name == "setdefault" and len(node.args) >= 2:
                    if (_is_key(node.args[0]) and _is_live(node.args[1])):
                        found.append((node.lineno, SETDEFAULT))
            elif isinstance(node, ast.Dict):
                for key, value in zip(node.keys, node.values):
                    if _is_key(key) and _is_live(value):
                        found.append((node.lineno, DICT_LITERAL))
            elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = (node.targets if isinstance(node, ast.Assign)
                           else [node.target])
                for target in targets:
                    if (isinstance(target, ast.Subscript)
                            and _is_key(target.slice) and _is_live(node.value)):
                        found.append((node.lineno, SUBSCRIPT))
            sites.extend((relative, line, shape) for line, shape in found)
    return sites


def _is_key(node: Any) -> bool:
    return isinstance(node, ast.Constant) and node.value == "provenance"


def _is_live(node: Any) -> bool:
    return isinstance(node, ast.Constant) and node.value == LIVE


def _paths_stamping_live() -> Set[str]:
    return {path for path, _, _ in _live_stamp_sites()}


def test_no_module_defaults_provenance_to_live_via_setdefault():
    """The original defect shape, kept as its own test so its reason survives.

    `setdefault` is the one shape that is always wrong: it fires when the record
    has no label at all, so the wrapper supplies the claim.
    """
    offenders = [f"{p}:{line}" for p, line, shape in _live_stamp_sites()
                 if shape == SETDEFAULT]
    assert not offenders, (
        "provenance defaulted to live, which invents a measurement label: "
        + "; ".join(offenders))


def test_every_module_stamping_live_is_classified():
    """The blind spot, closed.

    The original guard saw 1 site. It now sees every shape the defect takes, so
    a new module stamping `live` has to say which of the four reasons it is --
    measured, guarded, read from the running system, or a known overclaim.
    """
    unclassified = sorted(_paths_stamping_live() - ALL_CLASSIFIED)
    assert not unclassified, (
        "these modules stamp provenance='live' directly and are not "
        "classified. If this one measures, add it to MEASUREMENT_LAYERS with a "
        "reason. If it only relays a label, route it through "
        "backend.core.provenance or list it as GUARDED_WRAPPERS naming the "
        "guard. Do not add it just to make the test pass:\n  "
        + "\n  ".join(unclassified))


def test_measurement_layers_still_measure():
    """An allowlist that only ever shrinks into fiction is not a guard.

    Every declared measurement layer must still exist and must still contain a
    direct stamp, so a refactor that moves the measurement elsewhere cannot
    leave a stale entry claiming cover for code that no longer measures.
    """
    stamped = _paths_stamping_live()
    for path in MEASUREMENT_LAYERS:
        assert (ROOT / path).exists(), (
            f"MEASUREMENT_LAYERS lists {path}, which no longer exists")
        assert path in stamped, (
            f"MEASUREMENT_LAYERS lists {path} but it no longer stamps live "
            f"directly; remove it rather than leaving a stale justification")


def test_exemptions_are_disjoint_justified_and_real():
    """Exemption is the strongest statement this guard can make."""
    buckets = [set(MEASUREMENT_LAYERS), set(GUARDED_WRAPPERS),
               set(LIVE_BY_CONSTRUCTION), set(KNOWN_OVERCLAIM), set(SCAN_EXEMPT)]
    for i, a in enumerate(buckets):
        for b in buckets[i + 1:]:
            assert not (a & b), f"a module appears in two categories: {sorted(a & b)}"

    for name, table in (("MEASUREMENT_LAYERS", MEASUREMENT_LAYERS),
                        ("GUARDED_WRAPPERS", GUARDED_WRAPPERS),
                        ("LIVE_BY_CONSTRUCTION", LIVE_BY_CONSTRUCTION),
                        ("KNOWN_OVERCLAIM", KNOWN_OVERCLAIM),
                        ("SCAN_EXEMPT", SCAN_EXEMPT)):
        for path, reason in table.items():
            assert (ROOT / path).exists(), (
                f"{name} lists {path}, which no longer exists")
            assert len(reason) > 60, (
                f"{name}[{path!r}] needs a real justification ({len(reason)} "
                f"chars), not a label")


def test_operational_reads_cannot_opt_into_science():
    """Make the LIVE_BY_CONSTRUCTION justification executable.

    Those modules are allowed to say `live` on the grounds that the value came
    out of the running system rather than a fixture. That is only safe while
    `live` cannot promote anything into validation or publication -- and it
    cannot, because `discovery_is_live` also requires `_opted_in`, i.e. both
    `validation_eligible is True` and `publication_eligible is True`.

    If an operational endpoint ever sets either flag, that safety argument is
    void: a project list would start looking like a publishable measurement.
    So this asserts the absence rather than trusting the prose above.
    """
    opted_in = ("validation_eligible", "publication_eligible")

    for path in LIVE_BY_CONSTRUCTION:
        tree = ast.parse((ROOT / path).read_text(encoding="utf-8", errors="replace"))
        offenders = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Dict):
                for key, value in zip(node.keys, node.values):
                    if (isinstance(key, ast.Constant)
                            and key.value in opted_in
                            and isinstance(value, ast.Constant)
                            and value.value is True):
                        offenders.append(f"{key.value}=True at line {node.lineno}")
        assert not offenders, (
            f"{path} is classified as an operational read, but it opts into "
            f"science: {'; '.join(offenders)}. `live` on an operational value "
            f"only means 'not a fixture' -- if this can also set the "
            f"eligibility flags, it can promote a settings list into a "
            f"publishable measurement.")


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