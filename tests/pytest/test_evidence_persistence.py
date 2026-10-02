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


def _live_trace():
    """The same trace with discover/validate results explicitly opted in.

    `_step_allows_evidence` refuses scientific numbers from steps that are not
    completed + live + explicitly eligible, so the ineligible TRACE above must
    not contribute evidence nodes. This variant carries that opt-in and is used
    to prove the eligible path still works.
    """
    trace = [dict(step) for step in TRACE]
    for step in trace:
        if step["node"] == "discover":
            step["result"] = dict(step["result"], status="completed",
                                   provenance="live",
                                   validation_eligible=True,
                                   publication_eligible=True)
        elif step["node"] == "validate":
            step["result"] = dict(step["result"], status="completed",
                                   provenance="live",
                                   validation_eligible=True,
                                   publication_eligible=True)
    return trace


def test_demo_seed_preserved_for_legacy_route():
    graph = TraceableEvidenceGraph()
    assert len(graph.nodes) == 7
    assert len(graph.edges) == 6


def test_from_run_has_no_demo_nodes():
    graph = TraceableEvidenceGraph.from_run("r_test1", "probe goal", TRACE)
    ids = {n["id"] for n in graph.nodes}
    assert "n_402" not in ids and "c_ioi" not in ids
    # goal + one node per step + the patch delta. The trace's discover and
    # validate results carry no live provenance or eligibility, so their
    # circuit_score / confidence_score numbers must NOT become evidence.
    assert len(graph.nodes) == 1 + len(TRACE) + 1
    types = {n["type"] for n in graph.nodes}
    assert {"Goal", "Execution", "Observation", "Discovery",
            "Validation", "Publication", "Evidence"} <= types

    ev_labels = [n["label"] for n in graph.nodes if n["type"] == "Evidence"]
    # The patch step is not a scientific gate, so its measured delta stands.
    assert any("delta" in label for label in ev_labels)
    assert not any("circuit_score" in label for label in ev_labels)
    assert not any("confidence_score" in label for label in ev_labels)


def test_from_run_promotes_evidence_when_live_and_eligible():
    """Control: an explicitly live, eligible trace does produce evidence nodes."""
    graph = TraceableEvidenceGraph.from_run("r_live", "probe goal", _live_trace())
    ev_labels = [n["label"] for n in graph.nodes if n["type"] == "Evidence"]
    assert any("delta" in label for label in ev_labels)
    assert any("circuit_score" in label for label in ev_labels)
    assert any("confidence_score" in label for label in ev_labels)
    assert len(graph.nodes) == 1 + len(TRACE) + 3


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
    first = scribe.write_back("r_a", "goal A", _live_trace(), {},
                              publication, store=GraphStore(store_path))
    assert first["status"] == "completed", first.get("reason")
    assert len(first.get("stored", [])) >= 5
    assert any("circuit_disc_abc" in nid for nid in first["stored"])
    assert any("claim_r_a" in nid for nid in first["stored"])

    # Second run compounds: reload from disk sees both campaigns.
    reloaded = GraphStore(store_path)
    assert reloaded.nodes.get("camp_r_a") is not None
    second = scribe.write_back("r_b", "goal B", _live_trace(), {},
                               publication, store=reloaded)
    assert any("camp_r_b" in nid for nid in second["stored"])
    assert reloaded.nodes.get("camp_r_a") is not None
    assert reloaded.nodes.get("camp_r_b") is not None


def test_kg_writeback_blocks_ineligible_trace():
    """An ineligible trace must not reach the knowledge graph.

    The write-back gate is the last line of defence: without live, eligible
    discover/validate results, nothing is persisted and the reason is stated.
    """
    from agents.scribe import Scribe

    out = Scribe().write_back("r_blocked", "goal", TRACE, {},
                              {"status": "completed"})
    assert out["status"] == "blocked"
    assert out["stored"] == []
    assert out["provenance"] == "unavailable"
    assert out["reason"]


def test_kg_writeback_never_fails_run():
    """A broken store surfaces as an error, never as an exception.

    The run itself must complete even when persistence fails, and the partial
    node list is still reported.
    """
    from agents.scribe import Scribe

    out = Scribe().write_back("r_x", "g", _live_trace(), {},
                              {"status": "completed"},
                              store="not-a-store")
    assert out["status"] == "error"
    assert "error" in out
    assert out["stored"] == []


# ── Provenance is derived, never defaulted to live ──────────────────────────

def _evidence_nodes(graph):
    return [n for n in graph.nodes if n["type"] == "Evidence"]


def test_absent_provenance_does_not_default_to_live():
    """The regression: evidence_provenance used to be initialised to "live".

    It was only overwritten when a result happened to carry a provenance key,
    so any step whose result omitted one produced a live-labelled evidence
    node.
    """
    graph = TraceableEvidenceGraph.from_run("r_prov", "g", TRACE)
    assert _evidence_nodes(graph), "expected at least the patch delta node"
    for node in _evidence_nodes(graph):
        assert node["payload"]["provenance"] != "live"
        assert node["payload"]["provenance"] == "unavailable"


def test_declared_live_provenance_is_preserved_but_marked_unattested():
    graph = TraceableEvidenceGraph.from_run("r_live2", "g", _live_trace())
    live_nodes = [n for n in _evidence_nodes(graph)
                  if n["payload"]["provenance"] == "live"]
    assert live_nodes, "a declared-live step should keep its label"
    # Declared is not proved: no RunAttestation backs these nodes.
    for node in live_nodes:
        assert node["payload"]["attested"] is False
        assert node["payload"]["reason"]


def test_seeded_step_provenance_is_preserved():
    trace = [
        {"node": "patch", "agent": "executor", "op": "patch_head",
         "status": "ok",
         "result": {"provenance": "seeded", "delta": -1.4}},
    ]
    graph = TraceableEvidenceGraph.from_run("r_seeded", "g", trace)
    nodes = _evidence_nodes(graph)
    assert nodes
    assert nodes[0]["payload"]["provenance"] == "seeded"
    assert nodes[0]["payload"]["field_provenance"]["value"] == "seeded"


def test_unknown_provenance_string_is_not_trusted():
    trace = [
        {"node": "patch", "agent": "executor", "op": "patch_head",
         "status": "ok",
         "result": {"provenance": "definitely-real", "delta": 1.0}},
    ]
    graph = TraceableEvidenceGraph.from_run("r_bogus", "g", trace)
    nodes = _evidence_nodes(graph)
    assert nodes[0]["payload"]["provenance"] == "unavailable"

