"""Regression Detector Engine.

Computes quantitative regression metrics across fidelity, runtime, VRAM, 
and Bayesian belief confidence intervals.
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .benchmark_scheduler import GoldenBenchmarkResult
from .benchmark_status import is_comparable


@dataclass
class RegressionEvent:
    """Quantitative regression event artifact."""
    event_id: str
    benchmark_id: str
    benchmark_name: str
    metric_type: str  # Fidelity, Runtime, VRAM, Belief
    baseline_value: float
    current_value: float
    percentage_change: float
    severity: str  # LOW, MEDIUM, HIGH, CRITICAL
    detected_at: str = field(default_factory=lambda: _dt.datetime.utcnow().isoformat() + "Z")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "benchmark_id": self.benchmark_id,
            "benchmark_name": self.benchmark_name,
            "metric_type": self.metric_type,
            "baseline_value": self.baseline_value,
            "current_value": self.current_value,
            "percentage_change": round(self.percentage_change, 2),
            "severity": self.severity,
            "detected_at": self.detected_at,
        }


class RegressionDetector:
    """Quantitative regression detection engine."""

    def __init__(self, fidelity_drop_pct: float = -3.0,
                 runtime_slowdown_pct: float = 20.0) -> None:
        # Named rather than inline. These decide whether a measurement is
        # reported as a regression, and both were bare literals at the point of
        # use where nothing recorded what they were.
        self.fidelity_drop_pct = fidelity_drop_pct
        self.runtime_slowdown_pct = runtime_slowdown_pct

    def incomparable_results(
        self, results: List[GoldenBenchmarkResult],
    ) -> List[Dict[str, Any]]:
        """Which comparisons were declined, and why.

        Returned rather than merely logged, so a caller can tell the difference
        between "nothing regressed" and "nothing could be compared". Those are
        opposite situations and reporting them the same way is how a suite with
        no comparable baseline ends up looking perfectly healthy.
        """
        skipped: List[Dict[str, Any]] = []
        for r in results:
            reasons = []
            if (r.current_fidelity is not None
                    and r.published_baseline_fidelity
                    and not is_comparable(r.baseline_is_comparable)):
                reasons.append(
                    "fidelity: the published baseline is not the same "
                    "measurement this harness computes"
                )
            if (r.current_runtime_ms is not None
                    and r.published_baseline_runtime_ms
                    and not is_comparable(r.runtime_is_comparable)):
                reasons.append(
                    "runtime: the published baseline was measured on other "
                    "hardware"
                )
            if reasons:
                skipped.append({
                    "benchmark_id": r.benchmark_id,
                    "name": r.name,
                    "status": r.status,
                    "measured": r.measured,
                    "declined_comparisons": reasons,
                    "reason": r.reason,
                })
        return skipped

    def analyze_results(self, results: List[GoldenBenchmarkResult]) -> List[RegressionEvent]:
        """Analyzes benchmark execution results to detect performance/fidelity regressions.

        A benchmark that did not produce a current value cannot have regressed.
        `current_fidelity` and `current_runtime_ms` are None for benchmarks with
        no implementation, and subtracting None from the baseline raised
        TypeError. Those are now skipped rather than treated as zero, which
        would have manufactured a 100% regression for every unrun benchmark.

        And a benchmark whose baseline is **not comparable** cannot have
        regressed either. This method never read `status` or
        `baseline_is_comparable`; it keyed off `current_fidelity` alone. So while
        the scheduler was correctly declining to score IOI and induction-heads --
        reporting `status=MEASURED`, with the reason "the published baseline is
        not the same measurement" -- this method went ahead and scored them:

            reg_fid_01  bm_ioi  Fidelity  change=-17.73%  severity=HIGH
            reg_fid_02  bm_ind  Fidelity  change=-28.78%  severity=HIGH
            reg_rt_01   bm_ioi  Runtime   change=+6391.35%  severity=MEDIUM
            reg_rt_02   bm_ind  Runtime   change=+716.40%  severity=MEDIUM

        Two components in one pipeline disagreeing about whether a comparison is
        meaningful, with the disagreeing one driving `platform_health_score` down
        to 80 and firing HIGH-severity alerts. It also turned the two numbers this
        project is most careful about -- IOI measured 0.724 against a published
        0.880 that is a different measurement -- into a "HIGH severity
        regression", which is precisely the misreading the comparability flag
        exists to prevent.

        Comparability is now required, and required to be *established*:
        `benchmark_status.is_comparable` treats `None` as not comparable, so a
        baseline nobody has vouched for cannot produce a verdict.
        """
        events: List[RegressionEvent] = []

        for idx, r in enumerate(results):
            # 1. Fidelity Regression Test (> 3.0% drop)
            if (r.current_fidelity is not None
                    and r.published_baseline_fidelity
                    and is_comparable(r.baseline_is_comparable)):
                fid_diff_pct = (
                    (r.current_fidelity - r.published_baseline_fidelity)
                    / r.published_baseline_fidelity
                ) * 100.0
                if fid_diff_pct < self.fidelity_drop_pct:
                    events.append(RegressionEvent(
                        event_id=f"reg_fid_{idx + 1:02d}",
                        benchmark_id=r.benchmark_id,
                        benchmark_name=r.name,
                        metric_type="Fidelity",
                        baseline_value=r.published_baseline_fidelity,
                        current_value=r.current_fidelity,
                        percentage_change=fid_diff_pct,
                        severity="HIGH" if fid_diff_pct < -5.0 else "MEDIUM"
                    ))

            # 2. Runtime Latency Regression Test (> 20.0% slow-down)
            #
            # Gated on `runtime_is_comparable` for the same reason as fidelity,
            # and it matters more here: these baselines are wall-clock figures
            # from papers, measured on the authors' machines. Comparing an RTX
            # 3050 run against them produced "+6391.35% latency regression" and a
            # MEDIUM alert, which is not a finding about this platform at all.
            if (r.current_runtime_ms is not None
                    and r.published_baseline_runtime_ms
                    and is_comparable(r.runtime_is_comparable)):
                rt_diff_pct = (
                    (r.current_runtime_ms - r.published_baseline_runtime_ms)
                    / r.published_baseline_runtime_ms
                ) * 100.0
                if rt_diff_pct > self.runtime_slowdown_pct:
                    events.append(RegressionEvent(
                        event_id=f"reg_rt_{idx + 1:02d}",
                        benchmark_id=r.benchmark_id,
                        benchmark_name=r.name,
                        metric_type="Runtime Latency",
                        baseline_value=r.published_baseline_runtime_ms,
                        current_value=r.current_runtime_ms,
                        percentage_change=rt_diff_pct,
                        severity="MEDIUM"
                    ))

        return events
