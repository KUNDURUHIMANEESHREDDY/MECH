"""Continuous Health Dashboard Engine & Knowledge Graph Integrator.

Aggregates continuous validation metrics, updates the Scientific Knowledge Graph, 
and computes overall platform health & reproducibility scores.
"""

from __future__ import annotations

import datetime as _dt
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .validation_monitor import ValidationMonitor, EnvironmentState
from .benchmark_scheduler import ValidationBenchmarkScheduler, GoldenBenchmarkResult
from .benchmark_status import MEASURED, NOT_RUN, PASS, REGRESSION, is_comparable
from .regression_detector import RegressionDetector, RegressionEvent
from .alert_engine import AlertEngine, ActionableAlert
from backend.knowledge_graph.graph_store import GraphStore, KGNode, KGEdge
from backend.knowledge_graph.ontology import NodeType, EdgeType


@dataclass
class ContinuousHealthReport:
    """Comprehensive platform health snapshot artifact."""
    # Optional, and None when no benchmark had a comparable baseline.
    #
    # It was a plain `float`, derived as `100 - (regressions * 5)`. That formula
    # cannot tell "nothing regressed" from "nothing could be compared", so it
    # reported a perfect 100 for a suite in which every comparison had been
    # declined -- the mirror image of the false 80 it produced before the
    # comparability contract was honoured. Neither number was a measurement.
    #
    # Same treatment as `reproducibility_score` and `overall_pass_rate` in this
    # dataclass, which are already None when they cannot be derived.
    platform_health_score: Optional[float]
    # Optional: absent until the same work has actually been run twice.
    reproducibility_score: Optional[float]
    overall_pass_rate: Optional[float]
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
    # False when no benchmark could be scored, in which case overall_pass_rate
    # is None rather than 0.0.
    pass_rate_measured: bool = False
    # False when no benchmark had a comparable baseline, in which case
    # platform_health_score is None rather than 100.0.
    health_score_measured: bool = False
    health_score_reason: Optional[str] = None
    # Benchmarks that ran without a comparable baseline, plus those with no
    # implementation. Neither a pass nor a failure.
    benchmarks_unscored: int = 0
    # The comparisons the detector declined, and why. Recorded rather than
    # dropped, so an empty `regression_events` list is legible as "nothing
    # regressed" instead of being ambiguous between that and "nothing compared".
    declined_comparisons: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "platform_health_score": (
                round(self.platform_health_score, 1)
                if self.platform_health_score is not None else None
            ),
            "health_score_measured": self.health_score_measured,
            "health_score_reason": self.health_score_reason,
            # None, not 97.8.
            "reproducibility_score": (
                round(self.reproducibility_score, 1)
                if self.reproducibility_score is not None else None
            ),
            "reproducibility_measured": self.reproducibility_score is not None,
            "overall_pass_rate": (round(self.overall_pass_rate, 1)
                                  if self.overall_pass_rate is not None
                                  else None),
            "pass_rate_measured": self.pass_rate_measured,
            "benchmarks_unscored": self.benchmarks_unscored,
            "declined_comparisons": self.declined_comparisons,
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

        total_bm = len(bm_results)
        # The vocabulary is imported rather than re-spelled as literals. This
        # method was the only consumer that had to know all four names, and it
        # knew them as bare strings, so a fifth status added anywhere would have
        # been silently counted as unscored.
        passed_bm = sum(1 for r in bm_results if r.status == PASS)
        failed_bm = sum(1 for r in bm_results if r.status == REGRESSION)
        # Benchmarks that ran but could not be scored against their published
        # baseline, and benchmarks with no implementation at all. Neither is a
        # pass or a failure, and folding them into one of those two is how
        # "100% healthy" and "0% healthy" both become lies about the same suite.
        unscored_bm = sum(1 for r in bm_results
                          if r.status in (MEASURED, NOT_RUN))
        scorable_bm = passed_bm + failed_bm
        # A pass rate over a suite where nothing is scorable is undefined, not
        # 0% and not 100%.
        pass_rate: Optional[float] = (
            (passed_bm / scorable_bm) * 100.0 if scorable_bm else None
        )
        pass_rate_measured = pass_rate is not None

        # Comparisons the detector declined, recorded so that an empty
        # `regression_events` is legible as "nothing regressed" rather than
        # ambiguous between that and "nothing could be compared".
        declined = self.detector.incomparable_results(bm_results)

        # Whether any comparison was possible at all.
        #
        # Deliberately derived from the comparability flags rather than from
        # `scorable_bm` (PASS + REGRESSION). Both agree in practice, because the
        # scheduler only scores a benchmark it has established to be comparable
        # -- but gating on the score status makes the answer depend on two
        # fields agreeing with each other. If they ever diverge, a suite that did
        # compare things would report no health score at all. Comparability is the
        # actual precondition for the formula, so it is what is asked.
        comparable_any = any(
            is_comparable(r.baseline_is_comparable)
            or is_comparable(r.runtime_is_comparable)
            for r in bm_results
        )
        # Alerts are generated once the counts are known, and told both. Without
        # `scorable_count` the alert engine could not distinguish an empty event
        # list meaning "nothing regressed" from one meaning "nothing was checked",
        # and it chose the optimistic reading: it emitted "Platform Health
        # Optimal -- all golden benchmarks passing baseline fidelity and latency
        # specs" for a suite in which two benchmarks were incomparable and three
        # had no implementation at all.
        alerts: List[ActionableAlert] = self.alert_engine.generate_alerts(
            regressions,
            scorable_count=scorable_bm,
            declined_count=len(declined),
        )

        # Health is a penalty for regressions, so it can only be derived when at
        # least one comparison was actually possible.
        #
        # `100 - (regressions * 5)` was evaluated unconditionally. With every
        # baseline incomparable that yields 100.0 -- a perfect score for a suite in
        # which nothing was checked. Before the comparability contract was
        # honoured the same formula returned 80.0, from regressions raised against
        # baselines the scheduler had already declared invalid. 80 was a lie about
        # the code; 100 is a lie about the coverage. Neither was a measurement.
        health_score: Optional[float] = None
        health_score_measured = False
        health_score_reason: Optional[str] = None
        if comparable_any:
            health_score = max(0.0, 100.0 - (len(regressions) * 5.0))
            health_score_measured = True
        elif declined:
            health_score_reason = (
                f"No benchmark had a comparable baseline, so no regression could "
                f"be detected. {len(declined)} comparison(s) were declined: "
                + "; ".join(
                    f"{d['benchmark_id']} ({', '.join(d['declined_comparisons'])})"
                    for d in declined
                )
            )
        else:
            health_score_reason = (
                "No benchmark was scored and none was declined: the suite "
                "produced no comparable measurement to judge."
            )

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

        # Push Validation Run to Scientific Knowledge Graph automatically.
        # The node records what was actually measured. It used to carry
        # pass_rate=100.0 from a suite that computed baseline*0.995 and scored
        # it PASS against baseline*0.95, so every benchmark passed without any
        # of them running.
        # Was `f"val_run_{int(time.time())}"`.
        #
        # Truncating to whole seconds collides, so two validation runs inside one
        # tick share a node id and the second overwrites the first -- the graph
        # then holds one run where two happened. On Windows the system clock ticks
        # at ~15.6 ms, so consecutive calls hit this reliably rather than rarely.
        #
        # Third instance of this exact defect in this codebase, after
        # discovery_planner._claim_id and BenchmarkKGIntegrator's run_id. A
        # timestamp truncated to a tick is not an identifier.
        val_run_id = f"val_run_{uuid.uuid4().hex[:12]}"
        val_node = KGNode(
            node_id=val_run_id,
            node_type="Experiment",
            label="Continuous Validation Run",
            properties={
                # None when nothing was scorable, rather than a round number
                # that reads as a measurement.
                "pass_rate": pass_rate,
                "pass_rate_measured": pass_rate_measured,
                "health_score": health_score,
                "health_score_measured": health_score_measured,
                "benchmarks_total": total_bm,
                "benchmarks_passed": passed_bm,
                "benchmarks_failed": failed_bm,
                "benchmarks_unscored": unscored_bm,
                "declined_comparisons": len(declined),
                "measured_by": self.scheduler.__class__.__name__,
            }
        )
        self.graph_store.add_node(val_node)

        return ContinuousHealthReport(
            platform_health_score=health_score,
            health_score_measured=health_score_measured,
            health_score_reason=health_score_reason,
            declined_comparisons=declined,
            reproducibility_score=reproducibility_score,
            overall_pass_rate=pass_rate,
            pass_rate_measured=pass_rate_measured,
            benchmarks_unscored=unscored_bm,
            total_benchmarks_run=total_bm,
            passed_benchmarks=passed_bm,
            failed_benchmarks=failed_bm,
            environment_state=env.to_dict(),
            benchmark_results=[r.to_dict() for r in bm_results],
            regression_events=[reg.to_dict() for reg in regressions],
            actionable_alerts=[a.to_dict() for a in alerts],
            reproducibility_reason=reproducibility_reason
        )
