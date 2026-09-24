"""Tests for AI 1 Chief Architect Sprint 4 Autonomous Research Platform epics."""
from api.dispatcher import build_dispatcher


def test_autonomous_research_agent_run():
    dispatcher = build_dispatcher()
    res = dispatcher["platform/autonomous/agent_run"]({"goal": "Investigate IOI Circuit in GPT-2"})
    # Real Society v2 run contract (replaces the hardcoded stub society).
    for key in ("status", "goal", "workflow", "trace", "reflection",
                "publication", "events"):
        assert key in res
    assert isinstance(res["trace"], list)
    assert isinstance(res["events"], list)


def test_typed_research_graph():
    dispatcher = build_dispatcher()
    graph = dispatcher["platform/autonomous/graph_get"]({})
    assert graph["nodes_count"] >= 6
    assert graph["edges_count"] >= 5


def test_knowledge_base_query():
    dispatcher = build_dispatcher()
    facts = dispatcher["platform/autonomous/knowledge_query"]({"query": "Neuron"})
    assert len(facts) >= 1
    assert "Neuron L8_N402" in facts[0]["entity"]


def test_research_memory_store():
    dispatcher = build_dispatcher()
    mem = dispatcher["platform/autonomous/memory_store"]({
        "category": "successful_intervention",
        "description": "Zeroing L8_N402 worked",
        "utility_score": 0.98,
    })
    assert mem["category"] == "successful_intervention"
    assert mem["utility_score"] == 0.98


def test_hypothesis_generator():
    dispatcher = build_dispatcher()
    hyps = dispatcher["platform/autonomous/hypotheses_generate"]({"prompt": "The capital of France is"})
    assert len(hyps) >= 2
    assert "suggested_experiment" in hyps[0]


def test_autonomous_planner():
    dispatcher = build_dispatcher()
    plan = dispatcher["platform/autonomous/plan_create"]({"goal": "Discover Circuit"})
    assert plan["status"] == "planned"
    assert len(plan["experiment_stages"]) >= 3


def test_research_dashboard_summary():
    dispatcher = build_dispatcher()
    dash = dispatcher["platform/autonomous/dashboard_summary"]({})
    # Real registry/telemetry counts (small on a fresh checkout).
    for key in ("active_goals_count", "active_campaigns", "discoveries",
                "circuits_discovered_count"):
        assert key in dash
        assert isinstance(dash[key], int)
        assert dash[key] >= 0
