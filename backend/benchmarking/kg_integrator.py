"""KG Integrator — Populates the Scientific Knowledge Graph with benchmark evidence.

Every benchmark result becomes a KG node of type Experiment linked to:
  - the ModelSpec (as a Feature node)
  - the BenchmarkTask (as a Claim node)
  - the published reference (as a Paper node)
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import List

from .benchmark_runner import BenchmarkReport, BenchmarkResult
from .benchmark_tasks import ExecutionMode, TASK_CATALOGUE
from backend.knowledge_graph.graph_store import GraphStore, KGNode, KGEdge
from backend.knowledge_graph.ontology import NodeType, EdgeType

logger = logging.getLogger(__name__)

#: Agreement with the published reference figure, in percent, above which a
#: measurement is recorded as supporting a mechanism claim.
#:
#: This was a bare `> 80` inline. It was not arbitrary -- `benchmark_tasks`
#: defines fidelity as agreement with a published metric -- but an edge type in a
#: scientific knowledge graph *is* a scientific claim, and burying the deciding
#: number in a conditional expression hid both its meaning and its provenance.
#: Naming it states what it measures: reproduction agreement, not the truth of the
#: mechanism. A result can agree with a published number and still fail to
#: support the claim, and nothing here can tell those apart.
FIDELITY_SUPPORTS_THRESHOLD = 80.0


class BenchmarkKGIntegrator:
    """Writes all benchmark results into the Scientific Knowledge Graph."""

    def __init__(self, graph_store: GraphStore | None = None) -> None:
        self._store = graph_store or GraphStore()

    def ingest_report(self, report: BenchmarkReport) -> int:
        """Ingest all suites in a BenchmarkReport. Returns count of nodes created."""
        count = 0
        # Was `f"run_{int(time.time())}"`. `time.time()` truncated to whole
        # seconds collides, and two ingests inside one tick share a run id -- so
        # results from two separate runs were filed under one BenchmarkRun node
        # and the graph claimed they were the same run. On Windows the system
        # clock ticks at ~15.6 ms, so a loop over a small report hits this
        # reliably rather than rarely. (The same defect was fixed in
        # discovery_planner._claim_id, where a test that passed in isolation
        # failed in a full-suite run.)
        run_id = f"run_{uuid.uuid4().hex[:12]}"

        # Create a central "Benchmark Run" node
        run_node = KGNode(
            node_id=run_id,
            node_type=NodeType.BENCHMARK_RUN,
            label=f"Benchmark Suite {report.generated_at}",
            properties={
                "generated_at": report.generated_at,
                "overall_coverage": report.overall_coverage_pct,
                "overall_fidelity": report.overall_fidelity_pct,
                "models_tested": report.models_tested,
            }
        )
        # `add_node` inserts *or updates*, so count only a genuine insertion.
        is_new_run_node = run_id not in self._store.nodes
        self._store.add_node(run_node)
        if is_new_run_node:
            count += 1

        for suite in report.suites:
            for result in suite.task_results:
                count += self._ingest_result(result, run_id)
        logger.info("BenchmarkKGIntegrator: ingested %d nodes from benchmark report", count)
        return count

    def _ingest_result(self, result: BenchmarkResult, run_id: str) -> int:
        """Create structured nodes and links for a single benchmark result."""
        # Was `from ..benchmarks.benchmark_tasks import TASK_CATALOGUE`.
        #
        # `backend/benchmarks` does not exist; the package is
        # `backend/benchmarking`, so this file's sibling module was one level up
        # and to the left rather than one level up from here. The catalogue it
        # wanted has existed the whole time at `.benchmark_tasks.TASK_CATALOGUE`
        # (benchmark_tasks.py:184).
        #
        # The import was function-local, so `import backend.benchmarking`
        # succeeded -- `backend/benchmarking/__init__.py` re-exports
        # `BenchmarkKGIntegrator` -- and the ModuleNotFoundError only surfaced
        # when a result was actually ingested. `ingest_report` on a report with
        # any result in it could not complete.
        res_id = f"res_{result.model_id}_{result.task_id.value}_{result.run_id or int(time.time())}"
        task_spec = TASK_CATALOGUE[result.task_id]

        # Nodes actually created, counted as they are added. `ingest_report` sums
        # this and logs it as "ingested %d nodes", so a fixed return reported a
        # number unrelated to what happened. It was `return 2  # approx`, which is
        # 0 when every shared node already exists and 4 on a cold graph.
        nodes_added = 0

        # A fixture score is not evidence. Both epistemic edges below are withheld
        # for one, and the node records that they were.
        is_fixture = result.is_fixture or result.mode == ExecutionMode.MOCK

        # 1. Benchmark Result node
        res_node = KGNode(
            node_id=res_id,
            node_type=NodeType.BENCHMARK_RESULT,
            label=f"{task_spec.name} Fidelity: {result.fidelity_pct:.1f}%",
            properties=result.to_dict()
        )
        # `add_node` is documented as "Inserts or updates" and assigns by id, so
        # re-ingesting a result replaces the node rather than adding one. The
        # count has to test for that, or a repeat run reports nodes it did not
        # create. The three shared nodes below already guard on membership; this
        # one did not, which is why a second pass over the same result claimed 2
        # new nodes while the store grew by 0.
        is_new_result_node = res_id not in self._store.nodes
        self._store.add_node(res_node)
        if is_new_result_node:
            nodes_added += 1

        # 2. Link Result to Run
        self._store.add_edge(KGEdge(
            edge_id=f"edge_run_res_{res_id}",
            source_id=run_id,
            target_id=res_id,
            edge_type=EdgeType.CONTAINS,
            properties={}
        ))

        # 3. Model node (Feature type)
        model_id = f"model_{result.model_id}"
        if model_id not in self._store.nodes:
            model_node = KGNode(
                node_id=model_id,
                node_type=NodeType.FEATURE,
                label=f"Model: {result.model_id}",
                properties={"model_id": result.model_id}
            )
            self._store.add_node(model_node)
            nodes_added += 1

        self._store.add_edge(KGEdge(
            edge_id=f"edge_res_model_{res_id}",
            source_id=res_id,
            target_id=model_id,
            edge_type=EdgeType.EVALUATED,
            properties={}
        ))

        # 4. Mechanism Claim node
        claim_id = f"claim_{result.task_id.value}"
        if claim_id not in self._store.nodes:
            claim_node = KGNode(
                node_id=claim_id,
                node_type=NodeType.MECHANISM_CLAIM,
                label=f"Mechanism Claim: {task_spec.name}",
                properties={"task": result.task_id.value, "description": task_spec.description}
            )
            self._store.add_node(claim_node)
            nodes_added += 1

        # The epistemic edge: does this result support or contradict the claim?
        #
        # This edge was written unconditionally, including for fixtures. That is
        # the same fabrication path as the score itself, one step downstream: a
        # fixture score with fidelity_pct above the threshold created a SUPPORTS
        # edge asserting that hand-picked numbers support a mechanism claim, in a
        # graph that other code reads as evidence. It was reachable by default,
        # because `BenchmarkRunner.run_full_suite` defaults to
        # `ExecutionMode.MOCK` -- so the *default* path wrote fabricated
        # evidence into the knowledge graph.
        #
        # `is_fixture` was on the result the whole time, alongside `mode`, and
        # nothing consulted either.
        #
        # So a fixture result now gets no epistemic edge at all. The result node
        # is still created, still carries its full record, and says why it carries
        # no verdict.
        if is_fixture:
            res_node.properties["epistemic_edge_withheld"] = (
                "fixture score: this result is not evidence, so it neither "
                "supports nor contradicts the claim"
            )
            logger.info(
                "BenchmarkKGIntegrator: no SUPPORTS/CONTRADICTS edge for %s -- "
                "the score is a fixture, not a measurement",
                result.task_id.value,
            )
        else:
            edge_type = (
                EdgeType.SUPPORTS
                if result.fidelity_pct > FIDELITY_SUPPORTS_THRESHOLD
                else EdgeType.CONTRADICTS
            )
            self._store.add_edge(KGEdge(
                edge_id=f"edge_res_claim_{res_id}",
                source_id=res_id,
                target_id=claim_id,
                edge_type=edge_type,
                properties={
                    "fidelity_pct": result.fidelity_pct,
                    "threshold_pct": FIDELITY_SUPPORTS_THRESHOLD,
                    "threshold_meaning": (
                        "agreement with the published reference figure; this is "
                        "a reproduction check, not a verdict on the mechanism"
                    ),
                }
            ))

        # 5. Paper node (published reference)
        ref = task_spec.reference
        paper_id = f"paper_{ref.doi_or_arxiv.replace('/', '_').replace(':', '_')}"
        if paper_id not in self._store.nodes:
            paper_node = KGNode(
                node_id=paper_id,
                node_type=NodeType.PAPER,
                label=ref.source_paper,
                properties={
                    "doi_or_arxiv": ref.doi_or_arxiv,
                    "metric_name": ref.metric_name,
                    "metric_value": ref.metric_value,
                }
            )
            self._store.add_node(paper_node)
            nodes_added += 1

        # `VALIDATED_AGAINST` asserts that this result was validated by comparison
        # with the published figure. A fixture score was not compared with
        # anything, so the edge asserted a validation that never happened -- the
        # same defect as the edge above, and equally reachable by default. For a
        # fixture it becomes a plain `REFERENCES` link, which claims only that the
        # paper exists.
        if is_fixture:
            self._store.add_edge(KGEdge(
                edge_id=f"edge_res_paper_{res_id}",
                source_id=res_id,
                target_id=paper_id,
                edge_type=EdgeType.REFERENCES,
                properties={
                    "ref_value": ref.metric_value,
                    "validation_withheld": (
                        "fixture score: no comparison with the reference was "
                        "performed"
                    ),
                }
            ))
        else:
            self._store.add_edge(KGEdge(
                edge_id=f"edge_res_paper_{res_id}",
                source_id=res_id,
                target_id=paper_id,
                edge_type=EdgeType.VALIDATED_AGAINST,
                properties={
                    "ref_value": ref.metric_value,
                    "fidelity_pct": result.fidelity_pct,
                }
            ))

        return nodes_added
