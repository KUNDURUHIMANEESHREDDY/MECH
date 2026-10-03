"""Continuous Health Dashboard Engine & Knowledge Graph Integrator.

Aggregates continuous validation metrics, updates the Scientific Knowledge Graph, 
and computes overall platform health & reproducibility scores.
"""

from __future__ import annotations

import datetime as _dt
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .validation_monitor import ValidationMonitor, EnvironmentState
from .benchmark_scheduler import ValidationBenchmarkScheduler, GoldenBenchmarkResult
from .regression_detector import RegressionDetector, RegressionEvent
from .alert_engine import AlertEngine, ActionableAlert
from backend.knowledge_graph.graph_store import GraphStore, KGNode, KGEdge
from backend.knowledge_graph.ontology import NodeType, EdgeType


@dataclass
class ContinuousHealthReport:
    """Comprehensive platform health snapshot artifact."""
    platform_health_score: float
    # Optional: absent until the same work has actually been run twice.
    reproducibility_score: Optional[float]
    overall_pass_rate: float
    total_benchmarks_run: int
    passed_benchmarks: int
    failed_benchmarks: int
    environment_state: Dict[str, Any]
    benchmark_results: List[Dict[str, Any]]
    regression_events: List[Dict[str, Any]]
    actionable_alerts: List[Dict[str, Any]]
    generated_at: str = field(default_factory=lambda: _dt.datetime.utcnow().isoformat() + "Z")
    # Declared after the required fields: a defaulted field cannot precede
    # non-defaulted ones in a dataclass.
    reproducibility_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "platform_health_score": round(self.platform_health_score, 1),
            # None, not 97.8.
            "reproducibility_score": (
                round(self.reproducibility_score, 1)
                if self.reproducibility_score is not None else None
            ),
            "reproducibility_measured": self.reproducibility_score is not None,
            "overall_pass_rate": round(self.overall_pass_rate, 1),
            "total_benchmarks_run": self.total_benchmarks_run,
            "passed_benchmarks": self.passed_benchmarks,
            "failed_benchmarks": self.failed_benchmarks,
            "environment_state": self.environment_state,
            "benchmark_results": self.benchmark_results,
            "regression_events": self.regression_events,
            "actionable_alerts": self.actionable_alerts,
            "generated_at": self.generated_at,
        }


class HealthDashboardEngine:
    """Engine aggregating continuous validation runs and updating the Scientific Knowledge Graph."""

    def __init__(
        self,
        monitor: Optional[ValidationMonitor] = None,
        scheduler: Optional[ValidationBenchmarkScheduler] = None,
        detector: Optional[RegressionDetector] = None,
        alert_engine: Optional[AlertEngine] = None,
        graph_store: Optional[GraphStore] = None
    ) -> None:
        self.monitor = monitor or ValidationMonitor()
        self.scheduler = scheduler or ValidationBenchmarkScheduler()
        self.detector = detector or RegressionDetector()
        self.alert_engine = alert_engine or AlertEngine()
        self.graph_store = graph_store or GraphStore()

    def run_continuous_validation(self) -> ContinuousHealthReport:
        """Executes complete continuous validation pipeline and updates Knowledge Graph."""
        env = self.monitor.current_env
        bm_results: List[GoldenBenchmarkResult] = self.scheduler.execute_validation_suite()
        regressions: List[RegressionEvent] = self.detector.analyze_results(bm_results)
        alerts: List[ActionableAlert] = self.alert_engine.generate_alerts(regressions)

        total_bm = len(bm_results)
        passed_bm = sum(1 for r in bm_results if r.status == "PASS")
        failed_bm = total_bm - passed_bm
        pass_rate = (passed_bm / max(1, total_bm)) * 100.0

        health_score = max(0.0, 100.0 - (len(regressions) * 5.0))

        # Was the literal 97.8, on every run, forever.
        #
        # A reproducibility score needs at least two independent executions of
        # the same thing to compare. Nothing in this pipeline runs a benchmark
        # twice, so there is no variance to report and no basis for the number.
        # 97.8 was a compliment about the platform's own trustworthiness,
        # emitted before any comparison had been made.
        reproducibility_score: Optional[float] = None
        reproducibility_measured = False
        reproducibility_reason = (
            "Reproducibility requires the same benchmark to be executed more "
            "than once so the runs can be compared. This pipeline executes "
            "each benchmark once, so no reproducibility score can be derived."
        )

        # Push Validation Run to Scientific Knowledge Graph automatically
        val_run_id = f"val_run_{int(time.time())}"
        val_node = KGNode(
            node_id=val_run_id,
            node_type="Experiment",
            label="Continuous Validation Run",
            properties={"pass_rate": pass_rate, "health_score": health_score}
        )
        self.graph_store.add_node(val_node)

        return ContinuousHealthReport(
            platform_health_score=health_score,
            reproducibility_score=reproducibility_score,
            overall_pass_rate=pass_rate,
            total_benchmarks_run=total_bm,
            passed_benchmarks=passed_bm,
            failed_benchmarks=failed_bm,
            environment_state=env.to_dict(),
            benchmark_results=[r.to_dict() for r in bm_results],
            regression_events=[reg.to_dict() for reg in regressions],
            actionable_alerts=[a.to_dict() for a in alerts],
            reproducibility_reason=reproducibility_reason
        )
