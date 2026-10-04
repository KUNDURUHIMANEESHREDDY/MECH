"""Actionable Scientific Alert Engine.

Transforms raw regression events into actionable scientific alerts with root cause 
hypotheses and recommended remediation steps.
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .regression_detector import RegressionEvent


@dataclass
class ActionableAlert:
    """Actionable scientific alert object."""
    alert_id: str
    title: str
    severity: str  # WARNING, CRITICAL, INFO
    description: str
    # Optional. These were `str` and both fields were filled with a fixed
    # sentence regardless of what had actually happened -- see
    # `generate_alerts`. A cause that was not determined should read as
    # undetermined, not as a guess with a version number in it.
    likely_cause: Optional[str] = None
    recommended_action: Optional[str] = None
    timestamp: str = field(default_factory=lambda: _dt.datetime.utcnow().isoformat() + "Z")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "alert_id": self.alert_id,
            "title": self.title,
            "severity": self.severity,
            "description": self.description,
            "likely_cause": self.likely_cause,
            "recommended_action": self.recommended_action,
            "timestamp": self.timestamp,
        }


class AlertEngine:
    """Actionable Scientific Alert Engine."""

    def generate_alerts(
        self,
        regression_events: List["RegressionEvent"],
        *,
        scorable_count: int = 0,
        declined_count: int = 0,
    ) -> List["ActionableAlert"]:
        """Converts regression events into actionable scientific alerts.

        `scorable_count` and `declined_count` say how many benchmarks could be
        compared and how many comparisons were declined. They are needed to tell
        "nothing regressed" from "nothing was checked", and the empty-events
        branch cannot answer that question without them.

        Two things were wrong here.

        The empty-events branch announced success:

            title="Platform Health Optimal"
            description="All golden benchmarks passing baseline fidelity and
                         latency specs."

        An empty event list does not mean the benchmarks passed their specs. It
        means no comparison produced an event -- which includes the case where
        every comparison was declined because no baseline was comparable. On this
        repository that branch asserted that all five golden benchmarks were
        passing fidelity and latency specs while two were measured-but-
        incomparable and three had no implementation at all.

        And every regression alert carried the same invented diagnosis:

            likely_cause="Transformers package upgrade from 4.38.2 to 4.39.0
                          altered activation caching."
            recommended_action="Re-run activation cache calibration suite or pin
                                transformers==4.38.2."

        Fixed text, for every benchmark and every metric. Nothing in the pipeline
        observes package versions or activation caching, so this named a specific
        root cause that no measurement supported -- and named a version
        (4.38.2 -> 4.39.0) that is not the installed one (4.57.6). A remediation
        that says "pin transformers==4.38.2" is an instruction, not a hypothesis.

        So the cause is reported as undetermined unless something determined it,
        and the recommended action is derived from which metric moved rather than
        from a fixed sentence.
        """
        alerts: List[ActionableAlert] = []

        if not regression_events:
            if scorable_count == 0:
                # The dangerous branch: silence here is not health.
                alerts.append(ActionableAlert(
                    alert_id="alert_00",
                    title="No comparisons were possible",
                    severity="WARNING",
                    description=(
                        f"No benchmark produced a regression event because none "
                        f"could be compared against a baseline "
                        f"({declined_count} comparison(s) declined). This is not "
                        f"a clean result: it means the suite could not check "
                        f"anything."
                    ),
                    likely_cause=(
                        "No comparable baseline is available for these "
                        "benchmarks. The published figures were produced by "
                        "different measurements and different hardware."
                    ),
                    recommended_action=(
                        "Establish a baseline this harness can reproduce -- a "
                        "golden record captured on this machine via "
                        "RegressionSuite.define_golden -- before treating the "
                        "absence of regressions as a result."
                    ),
                ))
            else:
                alerts.append(ActionableAlert(
                    alert_id="alert_00",
                    title="No regressions detected",
                    severity="INFO",
                    description=(
                        f"{scorable_count} benchmark comparison(s) were made "
                        f"and none exceeded its threshold."
                        + (f" {declined_count} further comparison(s) were declined."
                           if declined_count else "")
                    ),
                    likely_cause=None,
                    recommended_action=None,
                ))
            return alerts

        for idx, event in enumerate(regression_events):
            alerts.append(ActionableAlert(
                alert_id=f"alert_{idx + 1:02d}",
                title=(f"{event.severity} {event.benchmark_name} "
                       f"{event.metric_type} Regression"),
                severity=("CRITICAL" if event.severity in ("HIGH", "CRITICAL")
                          else "WARNING"),
                description=(
                    f"{event.metric_type} moved {event.percentage_change:+.1f}% "
                    f"from the comparable baseline "
                    f"({event.baseline_value} -> {event.current_value})."
                ),
                # Nothing in this pipeline observes package versions, activation
                # caching, or any other cause. The cause is not determined by this
                # measurement, and saying so is more useful than naming one.
                likely_cause=(
                    "Not determined by this measurement. A threshold was "
                    "exceeded; the detector compares values and does not "
                    "investigate why they moved."
                ),
                recommended_action=_recommendation(event),
            ))

        return alerts


def _recommendation(event: "RegressionEvent") -> str:
    """A next step derived from which metric moved, not from a fixed sentence."""
    if event.metric_type == "Runtime Latency":
        return (
            "Confirm the comparison was made against a baseline captured on "
            "this hardware. If it was not, the comparison should have been "
            "declined; RegressionSuite can define a local golden record."
        )
    if event.metric_type == "Fidelity":
        return (
            "Re-run the pipeline and confirm the baseline is the same "
            "measurement this harness computes "
            "(GoldenBenchmarkResult.baseline_is_comparable) before treating "
            "this as a change in model behaviour."
        )
    return (
        "Inspect the metric directly. No metric-specific guidance is defined "
        "for this metric type."
    )
