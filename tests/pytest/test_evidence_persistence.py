"""Tests for persistent evidence: per-run EvidenceGraph (demo-free) and
Society knowledge-graph write-back compounding across sessions.
"""
import os

from core.evidence_graph import (
    TraceableEvidenceGraph,
    list_run_records,
    load_run_record,
    save_run_record,
)

TRACE = [
    {"node": "load", "agent": "executor", "op": "ensure_model",
     "status": "loaded"},
    {"node": "reproduce", "agent": "executor", "op": "reproduce",
     "status": "completed"},
    {"node": "inspect", "agent": "inspector", "op": "attention",
     "status": "ok"},
    {"node": "patch", "agent": "executor", "op": "patch_head",
     "status": "ok", "layer": 9, "head": 9, "delta": -1.4},
    {"node": "discover", "agent": "discoverer", "op": "discover",
     "status": "completed",
     "result": {"discovery_id": "disc_abc", "circuit_score": 0.88}},
    {"node": "validate", "agent": "critic", "op": "validate",
     "status": "completed",
     "result": {"validated": True,
                "confidence": {"confidence_score": 0.96}}},
    {"node": "publish", "agent": "scribe", "op": "publish",
     "status": "completed"},
]


def test_demo_seed_preserved_for_legacy_route():
    graph = TraceableEvidenceGraph()
    assert len(graph.nodes) == 7
    assert len(graph.edges) == 6


def test_from_run_has_no_demo_nodes():
    graph = TraceableEvidenceGraph.from_run("r_test1", "probe goal", TRACE)
    ids = {n["id"] for n in graph.nodes}
    assert "n_402" not in ids and "c_ioi" not in ids
    assert len(graph.nodes) == 1 + 7 + 3  # goal + steps + evidence nodes
    types = {n["type"] for n in graph.nodes}
    assert {"Goal", "Execution", "Observation", "Discovery",
            "Validation", "Publication", "Evidence"} <= types
    ev_labels = [n["label"] for n in graph.nodes if n["type"] == "Evidence"]
    assert any("delta" in label for label in ev_labels)
    assert any("circuit_score" in label for label in ev_labels)
    assert any("confidence_score" in label for label in ev_labels)


def test_run_record_roundtrip(tmp_path):
    record = {"run_id": "r_rt1", "goal": "g", "status": "completed",
              "result": {"trace": TRACE}}
    path = save_run_record("r_rt1", record, directory=str(tmp_path))
    assert os.path.exists(path)
    assert load_run_record("r_rt1", directory=str(tmp_path)) == record
    summaries = list_run_records(directory=str(tmp_path))
    assert len(summaries) == 1
    assert summaries[0]["run_id"] == "r_rt1"


def test_list_run_records_missing_dir():
    assert list_run_records(directory="/nonexistent-dir-xyz") == []


def test_kg_writeback_compounds(tmp_path):
    from agents.scribe import Scribe
    from knowledge_graph.graph_store import GraphStore

    store_path = str(tmp_path / "kg.json")
    publication = {"status": "completed", "experiment_id": "exp_1",
                   "steps_completed": "7/7"}
    scribe = Scribe()
    first = scribe.write_back("r_a", "goal A", TRACE, {},
                              publication, store=GraphStore(store_path))
    assert len(first.get("stored", [])) >= 5
    assert any("circuit_disc_abc" in nid for nid in first["stored"])
    assert any("claim_r_a" in nid for nid in first["stored"])

    # Second run compounds: reload from disk sees both campaigns.
    reloaded = GraphStore(store_path)
    assert reloaded.nodes.get("camp_r_a") is not None
    second = scribe.write_back("r_b", "goal B", TRACE, {},
                               publication, store=reloaded)
    assert any("camp_r_b" in nid for nid in second["stored"])
    assert reloaded.nodes.get("camp_r_a") is not None
    assert reloaded.nodes.get("camp_r_b") is not None


def test_kg_writeback_never_fails_run():
    from agents.scribe import Scribe
    out = Scribe().write_back("r_x", "g", [], {}, {},
                              store="not-a-store")
    assert out["stored"] == []
    assert "error" in out
