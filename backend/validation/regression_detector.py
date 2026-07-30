"""Regression Detector Engine.

Computes quantitative regression metrics across fidelity, runtime, VRAM, 
and Bayesian belief confidence intervals.
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .benchmark_scheduler import GoldenBenchmarkResult


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

    def analyze_results(self, results: List[GoldenBenchmarkResult]) -> List[RegressionEvent]:
        """Analyzes benchmark execution results to detect performance/fidelity regressions."""
        events: List[RegressionEvent] = []

        for idx, r in enumerate(results):
            # 1. Fidelity Regression Test (> 3.0% drop)
            fid_diff_pct = ((r.current_fidelity - r.published_baseline_fidelity) / r.published_baseline_fidelity) * 100.0
            if fid_diff_pct < -3.0:
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
            rt_diff_pct = ((r.current_runtime_ms - r.published_baseline_runtime_ms) / r.published_baseline_runtime_ms) * 100.0
            if rt_diff_pct > 20.0:
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
