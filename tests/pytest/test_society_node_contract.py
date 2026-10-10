"""The Society plan must be a plan the supervisor can actually run.

The defect
----------
`backend/agents/planner.py` emitted its unconditional anchor node as::

    {"id": "load", "agent": "executor", "op": "reproduce",
     "args": {"model_name": "gpt2"}, ...}

`Executor.reproduce(paper_id, n_prompts)` cannot accept `model_name`, so every
Society run died at its first step with::

    TypeError: Executor.reproduce() got an unexpected keyword argument 'model_name'

The anchor is unconditional -- nothing can be measured without weights -- so this
failed for *every* goal, not just IOI. The UI reported "Society run failed", and
the two committed Society screenshots turned out to be byte-identical: the old
capture script photographed the same error page twice and called it success.

Why it survived
---------------
`Planner.offer()` checked that the named op *exists* on the agent. `reproduce`
does exist, so the node passed. Nothing checked that the op *accepted the
arguments*. The guard tested dispatchability, not the call signature.

These tests cover both halves of the fix: the corrected `op`, and the missing
argument check that would have caught this without it.

That check deliberately has two strengths. `_argument_mismatch` requires the call
to be *complete*; `_supplied_keys_rejected` only asks whether a supplied key is
unacceptable. The weaker one is what `plan()` uses, because the supervisor fills
in context the planner cannot know (`discovery_id` for `critic.validate`), and
using the strict form there dropped the validation gate from every plan --
silently removing the step that gates publication. Both are tested, including a
test that they still differ.

Note on the diagnosis: the initial report described this as a
`model_name` -> `model_variant` mapping problem. That is wrong.
`model_variant` ("small"/"medium", composed as `gpt2-{variant}`) is an
`IOIPipeline.run` parameter -- a different concept from a loaded model identity
-- and is not reachable from this call site at all. `Executor.ensure_model` is
the method whose signature takes `model_name`. See
`docs/audit/society-contract-mismatch.md`.
"""

from __future__ import annotations

import asyncio
import inspect
from pathlib import Path
from typing import Any, Dict, List

import pytest

ROOT = Path(__file__).resolve().parents[2]
import sys  # noqa: E402

sys.path.insert(0, str(ROOT))

from backend.agents.planner import (  # noqa: E402
    Planner,
    _argument_mismatch,
    _supplied_keys_rejected,
)
from backend.agents.society import ResearchSocietyV2  # noqa: E402

GOAL = "Reproduce IOI on gpt2-small and find causally important heads"


def plan(goal: str = GOAL) -> Dict[str, Any]:
    return Planner().plan(goal)


def node_by_id(nodes: List[Dict[str, Any]], node_id: str) -> Dict[str, Any]:
    for node in nodes:
        if node["id"] == node_id:
            return node
    raise AssertionError(f"no node {node_id!r} in {[n['id'] for n in nodes]}")


def bound_method(agent: str, op: str):
    supervisor = ResearchSocietyV2()
    return getattr(getattr(supervisor, agent), op, None)


# ── The anchor node ─────────────────────────────────────────────────────

def test_the_anchor_node_loads_the_model():
    """The node's own id, rationale and args all describe a model load.

    `'id': 'load'`, *"Every downstream step needs live weights"*, and
    `args={"model_name": "gpt2"}` are all consistent with each other and with
    `Executor.ensure_model`. Only the `op` disagreed -- it had been carried over
    from the `reproduce` node that follows it.
    """
    anchor = node_by_id(plan()["nodes"], "load")

    assert anchor["op"] == "ensure_model", (
        f"the anchor node calls {anchor['op']!r}, but its id, rationale and args "
        f"describe a model load")
    assert anchor["agent"] == "executor"
    assert anchor["args"] == {"model_name": "gpt2"}


def test_the_anchor_nodes_arguments_bind_against_its_method():
    """The exact call that raised, as a direct assertion."""
    signature = inspect.signature(bound_method("executor", "ensure_model"))

    signature.bind(**node_by_id(plan()["nodes"], "load")["args"])


def test_the_previous_op_is_rejected_by_that_check():
    """The negative control: the old plan does not bind.

    Without this, `test_the_anchor_nodes_arguments_bind` could be passing for a
    reason unrelated to the fix.
    """
    signature = inspect.signature(bound_method("executor", "ensure_model"))

    with pytest.raises(TypeError) as excinfo:
        signature.bind(model_name="gpt2")  # binds -- this is the control
        signature.bind(paper_id="ioi")     # and this is what must not

    assert "paper_id" in str(excinfo.value)


def test_the_anchor_runs_and_reports_the_model_it_loaded(monkeypatch):
    """Execution, not just planning.

    `ensure_model` is a coroutine, so the supervisor awaits it directly rather
    than dispatching it to a thread; this exercises that path.
    """
    supervisor = ResearchSocietyV2()

    async def fake_load(model_name: str = "gpt2"):
        return {"status": "loaded", "model_name": model_name}

    monkeypatch.setattr(supervisor.executor, "ensure_model", fake_load)
    anchor = node_by_id(plan()["nodes"], "load")

    result = asyncio.run(
        supervisor._run_node(anchor, {"goal": GOAL, "trace": []}))

    assert result["status"] == "loaded"
    assert result["model_name"] == "gpt2", (
        "the model name the node asked for did not reach ensure_model")


def test_the_reproduce_node_is_unchanged_and_still_binds():
    """The fix must not have altered the node it was confused with."""
    node = node_by_id(plan()["nodes"], "reproduce")

    assert node["op"] == "reproduce"
    assert node["args"] == {"paper_id": "ioi", "n_prompts": 4}
    inspect.signature(bound_method("executor", "reproduce")).bind(**node["args"])


# ── Every emitted node must be runnable ─────────────────────────────────

@pytest.mark.parametrize("goal", [
    GOAL,
    "study protein folding",
    "measure induction heads",
    "logit lens behaviour",
])
def test_no_node_supplies_an_argument_its_method_cannot_accept(goal):
    """The general guard, which is what would have caught the original defect.

    Deliberately the *weak* check: a node may be incomplete, because the
    supervisor fills in run context. It may not hand the target a keyword that
    has nowhere to go. That is precisely what `model_name` on `reproduce` did.
    """
    offenders = [
        f"{n['id']} -> {n['agent']}.{n['op']}: "
        f"{problem}"
        for n in plan(goal)["nodes"]
        for problem in [_supplied_keys_rejected(
            n["agent"], n["op"], n.get("args", {}))]
        if problem is not None
    ]

    assert not offenders, "unrunnable nodes:\n  " + "\n  ".join(offenders)


@pytest.mark.parametrize("goal", [
    GOAL,
    "study protein folding",
])
def test_every_node_the_planner_emits_dispatches_to_a_real_method(goal):
    """The other half: the named op exists at all."""
    offenders = [
        f"{n['id']} -> {n['agent']}.{n['op']}: no such method"
        for n in plan(goal)["nodes"]
        if bound_method(n["agent"], n["op"]) is None
    ]

    assert not offenders, "\n  ".join(offenders)


def test_the_anchor_is_present_for_every_goal():
    """Dropping the anchor would make the plan pass this file while measuring
    nothing, so its presence is asserted alongside its correctness."""
    for goal in ("study protein folding", "measure induction heads"):
        ids = [n["id"] for n in plan(goal)["nodes"]]
        assert "load" in ids, f"{goal!r} has no weights step"


# ── The guard itself ────────────────────────────────────────────────────

def test_the_mismatch_check_reports_a_bad_call():
    """The guard must be capable of failing, or it protects nothing."""
    problem = _argument_mismatch("executor", "reproduce", {"model_name": "gpt2"})

    assert problem is not None, (
        "the guard accepted `model_name` for reproduce() -- the original defect "
        "would pass through it")
    assert "model_name" in problem


def test_the_planning_check_rejects_the_original_defect():
    """The weaker, planning-time check must still catch this specific bug.

    `_supplied_keys_rejected` asks only whether a *supplied* key is unacceptable,
    not whether the call is complete. That relaxation exists because the
    supervisor fills in context the planner cannot know (see the next test), but
    it would be useless if it also let the original defect through.
    """
    problem = _supplied_keys_rejected("executor", "reproduce",
                                      {"model_name": "gpt2"})

    assert problem is not None, (
        "the planning check accepted `model_name` for reproduce()")
    assert "model_name" in problem
    assert "reproduce accepts" in problem, (
        "the message must name what the method does accept")


def test_the_planning_check_tolerates_incomplete_but_valid_nodes():
    """`validate` supplies only `hypothesis`; the supervisor adds the rest.

    A strict bind here would have dropped the validation gate from every plan --
    silently removing the step that gates publication. That is worse than the
    defect being guarded against, so the incompleteness is allowed deliberately.
    """
    assert _supplied_keys_rejected("critic", "validate",
                                   {"hypothesis": "x"}) is None
    assert _argument_mismatch("critic", "validate", {"hypothesis": "x"}) is not None, (
        "the strict check should still report validate as incomplete -- the two "
        "checks are meant to differ, and if they stopped differing this test "
        "would be vacuous")


def test_the_validation_gate_stays_in_every_plan():
    """Directly asserted, because the failure mode was silent removal."""
    for goal in ("reproduce the IOI circuit", "study protein folding"):
        ids = [n["id"] for n in plan(goal)["nodes"]]
        assert "validate" in ids, (
            f"{goal!r} lost its validation step: {ids}")
        assert "publish" in ids, f"{goal!r} lost its publication step: {ids}"


def test_the_mismatch_check_passes_a_good_call():
    assert _argument_mismatch("executor", "ensure_model",
                              {"model_name": "gpt2"}) is None
    assert _argument_mismatch("executor", "reproduce",
                              {"paper_id": "ioi", "n_prompts": 4}) is None


def test_the_mismatch_check_names_what_is_accepted():
    """The message has to be actionable, not just a boolean."""
    problem = _argument_mismatch("executor", "reproduce", {"nonsense": 1})

    assert problem and "nonsense" in problem


def test_an_unreadable_signature_is_not_treated_as_a_mismatch():
    """Refusing to plan because a signature could not be read would be worse
    than the bug this prevents."""
    assert _argument_mismatch("executor", "no_such_op", {"a": 1}) is None


def test_a_node_with_unbindable_args_is_reported_rather_than_dropped_silently(
        monkeypatch):
    """A broken plan must be visible.

    `offer()` already records unknown ops in `dropped_nodes`; a node supplying an
    unacceptable keyword belongs there too, so the omission is reported rather
    than deferred to a runtime error.
    """
    import backend.agents.planner as planner_module

    monkeypatch.setattr(
        planner_module, "_supplied_keys_rejected",
        lambda agent, op, args: "simulated mismatch")

    result = Planner().plan(GOAL)

    assert not [n for n in result["nodes"] if n["id"] == "load"], (
        "the node was kept despite the mismatch")
    dropped = {d["id"]: d for d in result["dropped_nodes"]}
    assert "load" in dropped, f"dropped nodes: {sorted(dropped)}"
    assert "simulated mismatch" in dropped["load"]["why"], (
        "the drop must say what was wrong with it")


# ── The supervisor reports rather than raising ──────────────────────────

def test_a_bad_node_fails_the_step_instead_of_the_whole_run():
    """`fn(**args)` used to raise TypeError out of the entire workflow.

    One malformed step destroyed the trace of every step that came before it.
    Now it is a failed step, carrying the node, the op and what was accepted.
    """
    supervisor = ResearchSocietyV2()
    bad = {"id": "load", "agent": "executor", "op": "reproduce",
           "args": {"model_name": "gpt2"}}

    result = asyncio.run(supervisor._run_node(bad, {"goal": GOAL, "trace": []}))

    assert result["status"] == "error"
    assert "load" in result["error"]
    assert "reproduce" in result["error"]
    assert "model_name" in result["error"], (
        "the error must name the offending keyword")
    assert result["accepted"], (
        "the error must say what the method would have accepted")


def test_an_unknown_op_is_still_reported_as_before():
    result = asyncio.run(ResearchSocietyV2()._run_node(
        {"id": "x", "agent": "executor", "op": "nope", "args": {}},
        {"goal": GOAL, "trace": []}))

    assert result["status"] == "error"
    assert "unknown op" in result["error"]


# ── Capture exit codes distinguish success from failure ─────────────────

CAPTURE = ROOT / "scripts" / "capture_screenshots.cjs"


def _capture_source() -> str:
    return CAPTURE.read_text(encoding="utf-8")


def test_the_capture_script_does_not_treat_the_society_route_as_optional():
    """The acceptance criterion: the capture must be able to tell them apart.

    The Society route previously failed and the script screenshotted the error
    anyway, producing two identical "progress" images. It must assert on
    progress *or* report the error, and fail.
    """
    source = _capture_source()

    assert 'data-testid="society-steps"' in source, (
        "the Society route no longer asserts that the run progressed")
    assert 'data-testid="society-error"' in source, (
        "the Society route no longer watches for a reported error")


def test_the_society_verdict_is_the_terminal_phase_not_the_presence_of_a_trace():
    """A defect found by running the fixed capture.

    With only the planner fix, the route passed and exited 0 over a workflow that
    had reported `Society run failed`. The trace panel and the error banner render
    independently, so "steps appeared" and "the run failed" are both true at the
    same time -- waiting for steps alone reports success on a failed run.

    That is the same class of defect as the original one: a screenshot captured
    as progress that was not progress.
    """
    source = _capture_source()

    assert 'phase !== "done"' in source, (
        "the Society route must judge the run on its terminal phase; waiting "
        "for trace steps alone passes on a run that failed after producing them")
    assert "not \"done\"" in source, (
        "a non-done phase must produce a failure message naming the phase")
    assert "never reached a terminal phase" in source, (
        "a run that never settles is a failure, not something to photograph")


def test_a_failed_assertion_still_sets_a_nonzero_exit_code():
    """The property the whole capture rewrite rests on."""
    source = _capture_source()

    assert "process.exitCode" in source
    assert "route(s) FAILED their assertions" in source


def test_the_finding_is_documented_separately_from_the_scripts_work():
    """Kept apart on purpose, so test results distinguish the original mismatch
    from anything this fix introduced."""
    writeup = ROOT / "docs" / "audit" / "society-contract-mismatch.md"

    assert writeup.exists()
    text = writeup.read_text(encoding="utf-8")
    assert "unexpected keyword argument 'model_name'" in text
    assert "model_variant" in text, (
        "the write-up must record that model_variant was considered and ruled "
        "out, so the next reader does not re-derive it")


def test_a_publish_node_cannot_skip_the_fidelity_gate():
    """A plan node reaching `publish` must carry the gate.

    Found by an independent review. `_run_node`'s scribe branch called
    `publish(goal, trace, reflection)` with no `reproducibility` and no `gate`,
    so `publication_block_reason(trace, None, None)` checked only the discovery
    and validation chain -- the fidelity gate was never consulted. The
    authoritative call in `run()` does pass both, so the final publication was
    still gated, but a node arriving here could return a publication-shaped
    result that had never been measured.

    The mutation test for this matters: removing the forwarding from the node
    path passes the rest of this file unchanged.
    """
    supervisor = ResearchSocietyV2()
    node = {"id": "publish", "agent": "scribe", "op": "publish", "args": {}}
    ctx = {"goal": "any", "trace": [], "reflection": {}, "run_id": "run_x",
           "reproducibility": {"marker": "repro"}, "gate": {"marker": "gate"}}

    seen = {}
    original = supervisor.scribe.publish

    def spy(goal, trace, reflection, run_id="", reproducibility=None,
            gate=None):
        seen["reproducibility"] = reproducibility
        seen["gate"] = gate
        return original(goal, trace, reflection, run_id, reproducibility, gate)

    supervisor.scribe.publish = spy
    try:
        asyncio.run(supervisor._run_node(node, ctx))
    finally:
        supervisor.scribe.publish = original

    assert seen.get("gate") == {"marker": "gate"}, (
        "the node path did not forward the gate, so a publish node can be "
        "reached without the fidelity gate ever being consulted")
    assert seen.get("reproducibility") == {"marker": "repro"}, (
        "the node path did not forward reproducibility")
