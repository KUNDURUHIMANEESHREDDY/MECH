"""The knowledge graph must not record a fabricated score as evidence.

The rule under test
-------------------
**A fixture score is not evidence.** It may not create an edge asserting that a
mechanism claim is *supported*, nor that a result was *validated against* a
published reference. Both are scientific claims, and both were being made from
numbers that were never measured.

What was wrong
--------------
`backend/benchmarking/kg_integrator.py` did two things at once.

1. It could not run at all. `_ingest_result` opened with

       from ..benchmarks.benchmark_tasks import TASK_CATALOGUE

   `backend/benchmarks` does not exist -- the package is `backend/benchmarking`,
   so the sibling module is `.benchmark_tasks`, not `..benchmarks.benchmark_tasks`.
   The wanted symbol has existed the whole time at `benchmark_tasks.py:184`.

   The import was function-local, so `import backend.benchmarking` *succeeded* --
   `__init__.py` re-exports `BenchmarkKGIntegrator` -- and the ModuleNotFoundError
   only appeared when a result was actually ingested. `ingest_report` on any
   non-empty report could not complete.

2. It wrote fabricated evidence. Given a score, it created:

       EdgeType.SUPPORTS if result.fidelity_pct > 80 else EdgeType.CONTRADICTS
       EdgeType.VALIDATED_AGAINST          # unconditionally

   Neither consulted `is_fixture` or `mode`, both of which were on the result the
   whole time. And it was reachable *by default*: `BenchmarkRunner.run_full_suite`
   defaults to `ExecutionMode.MOCK`.

This is not theoretical. The runtime graph store on this machine,
`backend/research_datasets/knowledge_graph_index.json` (gitignored), contains 29
`supports` edges and 16 `validated_against` edges whose source nodes are
`BenchmarkResult` with `"mode": "mock"` and fidelity values like 99.78, 98.89 and
99.44. The defect had already fired and persisted.

Those nodes predate the `is_fixture` field, so `is_fixture` reads as absent on
them and checking only that field would not have caught the historical records.
The mode is recorded on all of them, so the guard tests `is_fixture or mode` --
either signal is enough, and one of them is always present.

Isolation
---------
These tests build a `GraphStore` on a tmp path and assert that they did. The
default is a **relative** path, so a test that constructs `GraphStore()` writes
into whatever directory pytest happens to be in. That is how
`tests/pytest/backend/datasets/knowledge_graph_index.json` came to exist as a
committed file.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from backend.benchmarking.benchmark_runner import (
    BenchmarkReport,
    ModelBenchmarkSuite,
)
from backend.benchmarking.benchmark_tasks import (
    BenchmarkResult,
    BenchmarkTask,
    ExecutionMode,
)
from backend.benchmarking.kg_integrator import (
    FIDELITY_SUPPORTS_THRESHOLD,
    BenchmarkKGIntegrator,
)
from backend.knowledge_graph.graph_store import GraphStore
from backend.knowledge_graph.ontology import EdgeType

#: Edge types that assert something about the world rather than describe structure.
EPISTEMIC = (EdgeType.SUPPORTS, EdgeType.CONTRADICTS, EdgeType.VALIDATED_AGAINST)


def _result(*, mode: ExecutionMode, fidelity: float,
            is_fixture: bool = False) -> BenchmarkResult:
    return BenchmarkResult(
        task_id=BenchmarkTask.IOI,
        model_id="gpt2",
        backend="hf",
        mode=mode,
        primary_score=fidelity,
        reference_score=88.0,
        fidelity_pct=fidelity,
        runtime_s=1.0,
        peak_memory_mb=None,
        confidence_interval_low=None,
        confidence_interval_high=None,
        is_fixture=is_fixture,
        backend_effective="transformer_lens",
        run_id="r1",
    )


def _ingest(tmp_path: Path, result: BenchmarkResult):
    """Ingest one result into a graph store isolated to `tmp_path`."""
    store = GraphStore(storage_path=str(tmp_path / "kg.json"))
    integrator = BenchmarkKGIntegrator(graph_store=store)
    report = BenchmarkReport(
        suites=[ModelBenchmarkSuite(
            model_id="gpt2",
            backend="hf",
            mode=result.mode,
            n_layers=12,
            n_params_b=0.124,
            task_results=[result],
            coverage_pct=100.0,
            mean_fidelity_pct=result.fidelity_pct,
            run_id="r1",
        )],
        overall_coverage_pct=100.0,
        overall_fidelity_pct=result.fidelity_pct,
        models_tested=1,
        tasks_tested=1,
    )
    before = _node_count(store)
    reported = integrator.ingest_report(report)
    return store, reported, before


def _values(container):
    """`GraphStore` keeps dicts keyed by id; the JSON form is a list."""
    return container.values() if isinstance(container, dict) else container


#: `GraphStore._load_or_initialize` seeds a baseline graph on first use -- 13
#: nodes including a `supports` edge and a `contradicts` edge, with properties
#: like `{"provenance": "seeded", "measured": False}`. Asserting over the whole
#: edge set therefore measures the baseline, not the code under test, so edges
#: are scoped to those this ingest created -- exactly the ones whose `source_id`
#: starts with `res_`.
RESULT_PREFIX = "res_"


def _ingested_edges(store: GraphStore) -> list:
    return [e for e in _values(store.edges)
            if str(getattr(e, "source_id", "")).startswith(RESULT_PREFIX)]


def _edge_types(store: GraphStore) -> set:
    # `KGEdge` is a dataclass, not a dict -- `e.get("edge_type")` raises
    # AttributeError. First draft of this helper assumed dicts.
    return {e.edge_type for e in _ingested_edges(store)}


def _node_count(store: GraphStore) -> int:
    """Total nodes.

    Not prefix-filtered: one ingest creates a `run_` node, a `res_` node, and
    possibly `model_`, `claim_` and `paper_` nodes, and `ingest_report` reports
    all of them. Filtering to `res_` counted 1 against a reported 5.
    """
    return len(store.nodes)


def _node_with_withheld_verdict(store: GraphStore) -> list:
    return [n for n in _values(store.nodes)
            if str(getattr(n, "node_id", "")).startswith(RESULT_PREFIX)
            and (getattr(n, "properties", None) or {}).get(
                "epistemic_edge_withheld")]


# â”€â”€ the reported defect â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def test_backend_benchmarking_imports():
    """The dead import was function-local, so this passed while ingest could not."""
    import backend.benchmarking as benchmarking

    assert hasattr(benchmarking, "BenchmarkKGIntegrator")
    for name in benchmarking.__all__:
        assert hasattr(benchmarking, name), name


def test_ingest_report_completes_on_a_populated_report(tmp_path):
    """`ingest_report` used to raise ModuleNotFoundError on any real report."""
    store, reported, _ = _ingest(
        tmp_path, _result(mode=ExecutionMode.REFERENCE, fidelity=95.0))

    assert reported > 0
    assert len(store.nodes) > 0
    assert store.storage_path == str(tmp_path / "kg.json")


def test_no_module_imports_the_nonexistent_benchmarks_package():
    from tests.pytest.test_entry_points import _unresolved_imports

    offenders = [
        f"{f}:{ln} -> {mod}"
        for f, ln, mod in _unresolved_imports()
        if mod.startswith("backend.benchmarks")
        and f == "backend/benchmarking/kg_integrator.py"
    ]
    assert not offenders, offenders


# â”€â”€ fixtures must not become evidence â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def test_a_fixture_score_creates_no_epistemic_edge(tmp_path):
    """The exact historical failure: a mock score at 99.78% fidelity."""
    store, _, _ = _ingest(
        tmp_path, _result(mode=ExecutionMode.MOCK, fidelity=99.78, is_fixture=True))

    present = _edge_types(store) & set(EPISTEMIC)
    assert EdgeType.SUPPORTS not in present
    assert EdgeType.CONTRADICTS not in present
    assert EdgeType.VALIDATED_AGAINST not in present


def test_mock_mode_alone_withholds_the_edge_even_without_is_fixture(tmp_path):
    """Historical nodes recorded `mode` but predate `is_fixture`.

    Checking only `is_fixture` would have passed straight over every fabricated
    edge already sitting in the store, because the field reads as absent on them.
    """
    result = _result(mode=ExecutionMode.MOCK, fidelity=99.78, is_fixture=False)
    assert result.is_fixture is False, "precondition: the old signal is absent"

    store, _, _ = _ingest(tmp_path, result)
    present = _edge_types(store) & set(EPISTEMIC)
    assert present == set(), (
        f"a mock-mode score produced epistemic edges {present} while is_fixture "
        f"was False -- so mode is the only signal that catches the historical "
        f"records")


def test_a_fixture_result_node_records_why_it_has_no_verdict(tmp_path):
    """An absent edge is invisible. The node has to say the edge was withheld."""
    store, _, _ = _ingest(
        tmp_path, _result(mode=ExecutionMode.MOCK, fidelity=95.0, is_fixture=True))

    withheld = _node_with_withheld_verdict(store)
    assert len(withheld) == 1, "the withheld verdict must be recorded on the node"
    assert "not evidence" in withheld[0].properties["epistemic_edge_withheld"]


def test_a_fixture_does_not_claim_validation_against_the_paper(tmp_path):
    """`VALIDATED_AGAINST` says a comparison happened. For a fixture none did."""
    store, _, _ = _ingest(
        tmp_path, _result(mode=ExecutionMode.MOCK, fidelity=95.0, is_fixture=True))

    assert EdgeType.VALIDATED_AGAINST not in _edge_types(store)
    # It still records that the paper exists, which is true.
    assert EdgeType.REFERENCES in _edge_types(store)


# â”€â”€ real measurements keep their edges â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def test_a_measurement_above_the_threshold_supports_the_claim(tmp_path):
    store, _, _ = _ingest(
        tmp_path, _result(mode=ExecutionMode.REFERENCE, fidelity=95.0))
    assert EdgeType.SUPPORTS in _edge_types(store)
    assert EdgeType.VALIDATED_AGAINST in _edge_types(store)


def test_a_measurement_below_the_threshold_contradicts_the_claim(tmp_path):
    store, _, _ = _ingest(
        tmp_path, _result(mode=ExecutionMode.REFERENCE, fidelity=40.0))
    assert EdgeType.CONTRADICTS in _edge_types(store)
    assert EdgeType.SUPPORTS not in _edge_types(store)


def test_the_threshold_is_named_and_its_meaning_recorded(tmp_path):
    """It was a bare `> 80`. An edge type is a claim; the number must be legible."""
    store, _, _ = _ingest(
        tmp_path, _result(mode=ExecutionMode.REFERENCE, fidelity=95.0))

    support = [e for e in _ingested_edges(store)
               if e.edge_type == EdgeType.SUPPORTS]
    assert support, "expected a SUPPORTS edge on a 95% measurement"

    props = support[0].properties or {}
    assert props.get("threshold_pct") == FIDELITY_SUPPORTS_THRESHOLD
    assert "reproduction check" in props.get("threshold_meaning", ""), (
        "the edge must say what the threshold means, so nobody reads 80% "
        "agreement as a verdict on whether the mechanism is true")


# â”€â”€ bookkeeping â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def test_reported_node_count_matches_the_store(tmp_path):
    """It was `return 2  # approx`, summed and logged as "ingested N nodes"."""
    store, reported, before = _ingest(
        tmp_path, _result(mode=ExecutionMode.REFERENCE, fidelity=95.0))
    assert reported == _node_count(store) - before


def test_repeat_ingest_creates_no_duplicate_result_nodes(tmp_path):
    """The old fixed `2` was wrong in both directions.

    Re-ingesting the same result reuses every result node id, so the second pass
    creates no result nodes. The old code returned 2 regardless, and
    `ingest_report` summed it into "ingested N nodes" and logged that as fact.

    One node *is* legitimately created: each `ingest_report` call is a distinct
    run and gets its own `BenchmarkRun` node. That is the correct count, and it
    is 1 rather than 2 or 5.
    """
    result = _result(mode=ExecutionMode.REFERENCE, fidelity=95.0)
    store, first_reported, before = _ingest(tmp_path, result)
    assert first_reported == _node_count(store) - before

    integrator = BenchmarkKGIntegrator(graph_store=store)
    report = BenchmarkReport(
        suites=[ModelBenchmarkSuite(
            model_id="gpt2", backend="hf", mode=result.mode, n_layers=12,
            n_params_b=0.124, task_results=[result], coverage_pct=100.0,
            mean_fidelity_pct=result.fidelity_pct, run_id="r1",
        )],
        overall_coverage_pct=100.0, overall_fidelity_pct=result.fidelity_pct,
        models_tested=1, tasks_tested=1,
    )
    nodes_before = _node_count(store)
    again = integrator.ingest_report(report)
    grew_by = _node_count(store) - nodes_before

    assert grew_by == 1, (
        f"a repeat ingest grew the store by {grew_by}; it should add exactly one "
        f"BenchmarkRun node and no result nodes")
    assert again == grew_by == 1, (
        f"reported {again} new nodes but the store grew by {grew_by}")

    # The result node itself was reused, not duplicated.
    result_nodes = [n for n in _values(store.nodes)
                    if str(getattr(n, "node_id", "")).startswith(RESULT_PREFIX)]
    assert len(result_nodes) == 1
    run_nodes = [n for n in _values(store.nodes)
                 if str(getattr(n, "node_id", "")).startswith("run_")]
    assert len(run_nodes) == 2, "two ingest calls are two runs"


def test_two_ingests_in_the_same_tick_get_different_run_ids(tmp_path):
    """`run_{int(time.time())}` collided.

    Truncating to whole seconds merges two runs into one BenchmarkRun node, and
    the graph then asserts they were the same run. On Windows the clock ticks at
    ~15.6 ms, so a tight loop hits this reliably.
    """
    result = _result(mode=ExecutionMode.REFERENCE, fidelity=95.0)
    store = GraphStore(storage_path=str(tmp_path / "kg.json"))
    integrator = BenchmarkKGIntegrator(graph_store=store)

    def report():
        return BenchmarkReport(
            suites=[ModelBenchmarkSuite(
                model_id="gpt2", backend="hf", mode=result.mode, n_layers=12,
                n_params_b=0.124, task_results=[result], coverage_pct=100.0,
                mean_fidelity_pct=result.fidelity_pct, run_id="r1",
            )],
            overall_coverage_pct=100.0,
            overall_fidelity_pct=result.fidelity_pct,
            models_tested=1, tasks_tested=1,
        )

    for _ in range(5):
        integrator.ingest_report(report())

    run_nodes = [n for n in _values(store.nodes)
                 if str(getattr(n, "node_id", "")).startswith("run_")]
    assert len(run_nodes) == 5, (
        f"5 ingests produced {len(run_nodes)} BenchmarkRun nodes; distinct runs "
        f"must not share a run id")


# â”€â”€ isolation â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def test_these_tests_do_not_write_the_real_graph_store(tmp_path):
    """The default store path is relative, so a test writes into its own cwd."""
    root = Path(__file__).resolve().parents[2]
    _ingest(tmp_path, _result(mode=ExecutionMode.REFERENCE, fidelity=95.0))

    stray = root / "tests" / "pytest" / "backend" / "datasets" / \
        "knowledge_graph_index.json"
    assert not stray.exists(), (
        f"{stray} exists -- something is still constructing GraphStore() with "
        f"the default relative path during tests")


def test_the_committed_stray_copy_of_the_graph_store_is_gone():
    """It was committed by accident and nothing referenced it.

    `tests/pytest/backend/datasets/knowledge_graph_index.json` -- 8 nodes, 7 edges
    -- is test output written under the old `backend/datasets/` path before the
    rename. It reached git in b46fba0.
    """
    root = Path(__file__).resolve().parents[2]
    assert not (root / "tests" / "pytest" / "backend").exists(), (
        "a committed copy of the knowledge graph store survived under "
        "tests/pytest/backend/. It is test output, not a fixture.")


def test_the_real_store_is_gitignored():
    """Runtime accumulation must never be committed."""
    root = Path(__file__).resolve().parents[2]
    ignore = (root / ".gitignore").read_text(encoding="utf-8")
    assert "backend/research_datasets/knowledge_graph_index.json" in ignore
