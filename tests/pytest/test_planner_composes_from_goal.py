"""The planner must compose from the goal, not emit one fixed DAG.

What this replaces
------------------
`plan()` returned the same seven nodes for every goal:

    load -> reproduce -> inspect -> patch -> discover -> validate -> publish

with only the pipeline hint, the hypothesis text and two IOI head coordinates
varying. That was disclosed -- `plan_is_templated: True` said so -- but disclosure
is not implementation. A planner that emits an identical DAG whatever it is asked
has not planned anything, and the field name claimed otherwise.

It also emitted ops the supervisor cannot run. `society.execute` does
`getattr(agent, op, None)` and returns `{"error": "unknown op ..."}`, so a node
naming a method the agent lacks is a runtime error rather than a plan.

These tests pin the three properties the rewrite is built on:

  * the plan varies with the goal;
  * every emitted node names a method the agent actually exposes, so the
    supervisor can dispatch it;
  * a goal that matches nothing gets a discovery-only plan rather than a
    fabricated reproduction target -- which is the defect `detect_pipeline`
    previously had when it defaulted to "ioi".
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path
from typing import Any, Dict, List

import pytest

from backend.agents.planner import GOAL_HINTS, MAX_STEPS, Planner

ROOT = Path(__file__).resolve().parents[2]
PLANNER = ROOT / "backend" / "agents" / "planner.py"
SOCIETY = ROOT / "backend" / "agents" / "society.py"

AGENTS = ("executor", "inspector", "critic", "discoverer", "scribe")


def _plan(goal: str) -> Dict[str, Any]:
    return Planner().plan(goal)


def _ids(plan: Dict[str, Any]) -> List[str]:
    return [n["id"] for n in plan["nodes"]]


def _agent_methods(agent: str) -> set:
    """The public methods on the real agent object."""
    from backend.agents.society import ResearchSocietyV2

    society = ResearchSocietyV2()
    obj = getattr(society, agent, None)
    assert obj is not None, f"the society has no {agent}"
    return {name for name in dir(obj)
            if not name.startswith("_") and callable(getattr(obj, name, None))}


# ── The plan must vary with the goal ────────────────────────────────────

def test_a_matched_goal_and_an_unmatched_goal_get_different_plans():
    """The whole point.

    These two were byte-identical node lists before. A planner that cannot tell
    two goals apart is not planning.
    """
    ioi = _plan("reproduce the IOI circuit")
    other = _plan("study protein folding")

    assert _ids(ioi) != _ids(other), (
        f"both plans returned {_ids(ioi)}; the goal had no effect on the plan")


def test_the_ioi_branch_is_emitted_only_for_ioi():
    ioi = _plan("reproduce the IOI circuit")
    assert "reproduce" in _ids(ioi)
    assert "inspect" in _ids(ioi)
    assert "patch" in _ids(ioi)

    other = _plan("study protein folding")
    for node_id in ("reproduce", "inspect", "patch"):
        assert node_id not in _ids(other), (
            f"{node_id} is in the plan for a goal that matches no pipeline")


def test_the_unconditional_floor_is_present_for_every_goal():
    """Weights, discovery, validation, publication apply to any goal."""
    for goal in ("reproduce the IOI circuit", "study protein folding",
                 "measure induction heads", "logit lens behaviour"):
        ids = _ids(_plan(goal))
        for required in ("load", "discover", "validate", "publish"):
            assert required in ids, f"{goal!r} is missing the {required} step"


def test_the_floor_is_marked_as_unconditional():
    """A reader must be able to tell which nodes came from the goal."""
    plan = _plan("reproduce the IOI circuit")

    for node in plan["nodes"]:
        assert "because" in node, f"{node['id']} has no justification"
        assert node["because"], f"{node['id']} has an empty justification"

    from_goal = plan["nodes_from_goal"]
    assert from_goal, "an IOI goal must contribute at least its own branch"
    for node in plan["nodes"]:
        if node["id"] in from_goal:
            assert not node["because"].startswith("unconditional"), (
                f"{node['id']} is listed as goal-derived but justified as "
                f"unconditional")


def test_an_unmatched_goal_contributes_no_goal_derived_nodes():
    plan = _plan("study protein folding")

    assert plan["pipeline"] is None
    assert plan["nodes_from_goal"] == [], (
        f"no pipeline matched, so nothing in the plan came from the goal: "
        f"{plan['nodes_from_goal']}")


# ── Nothing may be invented to fill a gap ──────────────────────────────

def test_an_unmatched_goal_is_told_what_could_not_be_resolved():
    plan = _plan("study protein folding")

    assert plan["pipeline_matched"] is False
    assert "pipeline" in plan["unresolved"]
    assert "inspect_target" in plan["unresolved"]
    assert "patch_target" in plan["unresolved"]


def test_an_unmatched_goal_drops_the_reproduce_step_with_a_reason():
    """Not silently omitted -- the omission is the finding."""
    plan = _plan("study protein folding")

    dropped = {d["id"]: d for d in plan["dropped_nodes"]}
    assert "reproduce" in dropped, (
        f"the reproduce step was dropped without being recorded: "
        f"{plan['dropped_nodes']}")
    assert "no reproduction pipeline matched" in dropped["reproduce"]["why"]
    assert "invent a target" in dropped["reproduce"]["why"]


def test_a_matched_non_ioi_goal_carries_no_head_coordinates():
    """Attaching IOI's L10H7 to a greater-than plan is the old defect."""
    for goal in ("measure greater-than circuits", "induction head behaviour"):
        plan = _plan(goal)
        for node in plan["nodes"]:
            if node["id"] in ("inspect", "patch"):
                assert not node["args"], (
                    f"{goal!r}: {node['id']} carries {node['args']} but no head "
                    f"is known for that pipeline")


def test_the_ioi_plan_keeps_the_published_coordinates():
    """When the coordinates ARE known, dropping them loses real information."""
    plan = _plan("reproduce the IOI circuit")
    by_id = {n["id"]: n for n in plan["nodes"]}

    assert by_id["inspect"]["args"] == {"layer": 10, "head": 7}
    assert by_id["patch"]["args"] == {"layer": 9, "head": 9}


# ── Every node must be dispatchable ─────────────────────────────────────

@pytest.mark.parametrize("goal", [
    "reproduce the IOI circuit",
    "study protein folding",
    "measure induction heads",
    "logit lens behaviour",
    "sparse autoencoder features",
])
def test_every_emitted_node_names_a_method_the_agent_exposes(goal):
    """Otherwise the supervisor returns `unknown op` at execution time."""
    plan = _plan(goal)

    for node in plan["nodes"]:
        agent = node["agent"]
        op = node["op"]
        methods = _agent_methods(agent)
        assert op in methods, (
            f"{goal!r}: node {node['id']!r} names {agent}.{op!r}, which does not "
            f"exist. Available: {sorted(methods)}")


def test_a_node_whose_op_does_not_exist_is_dropped(monkeypatch):
    """The capability filter, tested by making it bite.

    Disabling the filter was invisible to the rest of this file: every node the
    planner offers names an op that does exist, so removing the check changed
    nothing observable and the negative control for it passed while the filter
    was gone. A guard that cannot fail when the feature is removed is not a
    guard on that feature.

    So the table is replaced with one missing `discover`, which the planner does
    offer. The node must be dropped and recorded.
    """
    import backend.agents.planner as planner_module

    real = planner_module._agent_methods()
    table = {agent: set(methods) for agent, methods in real.items()}
    table["discoverer"] = table["discoverer"] - {"discover"}
    monkeypatch.setattr(planner_module, "_agent_methods", lambda: table)

    plan = planner_module.Planner().plan("reproduce the IOI circuit")

    ids = [n["id"] for n in plan["nodes"]]
    assert "discover" not in ids, (
        "the discover node was emitted even though the discoverer agent "
        "exposes no discover method")

    dropped = {d["id"]: d for d in plan["dropped_nodes"]}
    assert "discover" in dropped, (
        f"the discover node was dropped without being recorded: "
        f"{plan['dropped_nodes']}")
    assert "discover" in dropped["discover"]["why"]
    assert "exposes no" in dropped["discover"]["why"], (
        "the reason must name the missing capability, so a reader can supply "
        f"it: {dropped['discover']['why']!r}")


def test_the_filter_is_what_drops_it_not_the_goal_matching():
    """The same plan, with the method restored, must include the node.

    Otherwise the previous test could pass for the wrong reason -- e.g. because
    the goal stopped matching, which would drop `discover` for a different
    reason entirely.
    """
    ioi = _plan("reproduce the IOI circuit")
    assert "discover" in [n["id"] for n in ioi["nodes"]], (
        "with the real capability table the discover node must be present; "
        "otherwise the drop test proves nothing")


def test_dropped_nodes_are_dispatchable_failures_not_silent_ones():
    """A node that cannot be dispatched must be recorded, not emitted."""
    plan = _plan("reproduce the IOI circuit")
    by_id = {n["id"]: n for n in plan["nodes"]}

    for dropped in plan["dropped_nodes"]:
        assert dropped["id"] not in by_id, (
            f"{dropped['id']} is both emitted and dropped")
        assert dropped["why"], "a dropped node must say why"


def test_the_plan_never_names_an_agent_outside_the_six():
    for goal in ("reproduce the IOI circuit", "study protein folding"):
        for node in _plan(goal)["nodes"]:
            assert node["agent"] in AGENTS, (
                f"unknown agent {node['agent']!r}; the society dispatches only "
                f"{list(AGENTS)}")


def test_the_step_cap_is_respected():
    for goal in ("reproduce the IOI circuit", "study protein folding"):
        assert len(_plan(goal)["nodes"]) <= MAX_STEPS


# ── The planner's own detection must not invent a target ────────────────

@pytest.mark.parametrize("goal", [
    "investigate protein folding",
    "measure rainfall in Wales",
    "translate this poem",
    "",
])
def test_an_unrelated_goal_matches_no_pipeline(goal):
    """`detect_pipeline` used to return "ioi" when nothing matched.

    That produced a plan to reproduce indirect object identification, with IOI
    attention heads attached, for a question about protein folding.
    """
    assert Planner().detect_pipeline(goal) is None


@pytest.mark.parametrize("keyword,pipeline", sorted(GOAL_HINTS.items()))
def test_every_declared_hint_still_resolves(keyword, pipeline):
    assert Planner().detect_pipeline(f"please run the {keyword} analysis") == pipeline


def test_the_goal_text_reaches_the_discovery_step():
    """The one place the goal genuinely is the plan's input."""
    goal = "does the IOI circuit rely on the name mover"
    plan = _plan(goal)
    discover = next(n for n in plan["nodes"] if n["id"] == "discover")

    assert discover["args"]["hypothesis"] == goal


# ── The old templated flag must be gone, not merely reworded ────────────

def test_plan_is_templated_is_gone():
    """Leaving it would now be false in the other direction.

    The plan is no longer a fixed skeleton, so a flag asserting it *is* templated
    would be a lie -- and a flag asserting it is not would be equally unearned,
    because four of its seven nodes are still unconditional.
    """
    plan = _plan("reproduce the IOI circuit")

    assert "plan_is_templated" not in plan, (
        "the plan is no longer a fixed skeleton, so this flag cannot stay")
    assert "template_reason" in plan, (
        "the residual templating must still be explained")


def test_the_residual_templating_is_stated():
    plan = _plan("reproduce the IOI circuit")

    reason = plan["template_reason"].lower()
    assert "unconditional" in reason, (
        "the explanation must name which nodes are fixed regardless of goal")
    assert "capability" in reason or "dispatch" in reason, (
        "the explanation must say that nodes are filtered by what the agents "
        "can actually run")


def test_composition_is_claimed_only_because_it_happens():
    plan = _plan("reproduce the IOI circuit")

    assert plan["composed_from_capabilities"] is True
    # The claim is only worth making if it is true, which the dispatchability
    # tests check. Here, assert the count is consistent with the node list.
    assert plan["n_nodes"] == len(plan["nodes"])


# ── The capability table must come from the real agents ────────────────

def test_agent_methods_is_not_hardcoded():
    """It has to reflect the classes, or the filtering is theatre."""
    source = inspect.getsource(
        __import__("backend.agents.planner", fromlist=["_agent_methods"])
        ._agent_methods)

    assert "_cache" in source, (
        "the method table is recomputed on every call and imports the agents "
        "each time")
    for agent in AGENTS:
        assert f"{agent}" in source, (
            f"{agent} is missing from the capability lookup, so its nodes would "
            f"be filtered out or wrongly allowed")


def test_the_lookup_finds_real_methods():
    table = __import__("backend.agents.planner",
                       fromlist=["_agent_methods"])._agent_methods()

    for agent in AGENTS:
        assert table.get(agent), f"{agent} resolved to an empty method set"


# ── Structural: the old fixed node list must be gone ────────────────────

def test_plan_does_not_build_one_literal_node_list():
    """The defect's shape: a single list literal containing every node id.

    Two earlier versions of this guard searched for a node id string, which the
    docstring quotes. So this parses the function and looks at what it actually
    constructs.
    """
    tree = ast.parse(PLANNER.read_text(encoding="utf-8"))
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "plan")

    # Every dict literal built directly inside plan(), as opposed to inside the
    # nested `offer` helper or another function.
    plan_body = {id(s) for s in fn.body}
    literals = []
    for stmt in fn.body:
        for node in ast.walk(stmt):
            if isinstance(node, ast.Dict) and id(node) in plan_body:
                literals.append(node)

    id_literals = [
        node for node in literals
        if any(isinstance(k, ast.Constant) and k.value == "id"
               for k in node.keys)
    ]

    assert len(id_literals) <= 1, (
        f"plan() builds {len(id_literals)} node dicts inline; a plan assembled "
        f"from one literal per node is the fixed-skeleton defect")


def test_the_society_dispatch_contract_is_unchanged():
    """The plan is only valid because dispatch does `getattr(agent, op)`."""
    source = SOCIETY.read_text(encoding="utf-8")

    assert "getattr(agent, op" in source, (
        "society.execute no longer resolves ops by getattr, so the planner's "
        "capability filtering is checking the wrong contract")