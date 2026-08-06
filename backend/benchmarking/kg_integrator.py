"""KG Integrator — Populates the Scientific Knowledge Graph with benchmark evidence.

Every benchmark result becomes a KG node of type Experiment linked to:
  - the ModelSpec (as a Feature node)
  - the BenchmarkTask (as a Claim node)
  - the published reference (as a Paper node)
"""

from __future__ import annotations

import logging
import time
from typing import List

from .benchmark_runner import BenchmarkReport, BenchmarkResult
from backend.knowledge_graph.graph_store import GraphStore, KGNode, KGEdge
from backend.knowledge_graph.ontology import NodeType, EdgeType

logger = logging.getLogger(__name__)


class BenchmarkKGIntegrator:
    """Writes all benchmark results into the Scientific Knowledge Graph."""

    def __init__(self, graph_store: GraphStore | None = None) -> None:
        self._store = graph_store or GraphStore()

    def ingest_report(self, report: BenchmarkReport) -> int:
        """Ingest all suites in a BenchmarkReport. Returns count of nodes created."""
        count = 0
        run_id = f"run_{int(time.time())}"

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
        self._store.add_node(run_node)
        count += 1

        for suite in report.suites:
            for result in suite.task_results:
                count += self._ingest_result(result, run_id)
        logger.info("BenchmarkKGIntegrator: ingested %d nodes from benchmark report", count)
        return count

    def _ingest_result(self, result: BenchmarkResult, run_id: str) -> int:
        """Create structured nodes and links for a single benchmark result."""
        from ..benchmarks.benchmark_tasks import TASK_CATALOGUE

        res_id = f"res_{result.model_id}_{result.task_id.value}_{result.run_id or int(time.time())}"
        task_spec = TASK_CATALOGUE[result.task_id]

        # 1. Benchmark Result node
        res_node = KGNode(
            node_id=res_id,
            node_type=NodeType.BENCHMARK_RESULT,
            label=f"{task_spec.name} Fidelity: {result.fidelity_pct:.1f}%",
            properties=result.to_dict()
        )
        self._store.add_node(res_node)

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

        self._store.add_edge(KGEdge(
            edge_id=f"edge_res_claim_{res_id}",
            source_id=res_id,
            target_id=claim_id,
            edge_type=EdgeType.SUPPORTS if result.fidelity_pct > 80 else EdgeType.CONTRADICTS,
            properties={"fidelity_pct": result.fidelity_pct}
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

        self._store.add_edge(KGEdge(
            edge_id=f"edge_res_paper_{res_id}",
            source_id=res_id,
            target_id=paper_id,
            edge_type=EdgeType.VALIDATED_AGAINST,
            properties={"ref_value": ref.metric_value}
        ))

        return 2 # new nodes created (approx)
