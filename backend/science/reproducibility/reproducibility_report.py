"""Reproducibility Report Engine — Gold / Silver / Bronze / Needs Investigation.

Aggregates pipeline metric results, computes fidelity scores, and emits
structured ReproducibilityReport objects with per-metric pass/fail tiers.
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from .paper_registry import BenchmarkRegistry, ExpectedMetric


# Fidelity tier boundaries (percentage of published value achieved)
GOLD_PCT   = 95.0
SILVER_PCT = 90.0
BRONZE_PCT = 85.0


def _fidelity_tier(observed: float, expected: float) -> str:
    if expected == 0:
        return "Needs Investigation"
    pct = (observed / expected) * 100.0
    if pct >= GOLD_PCT:
        return "Gold"
    if pct >= SILVER_PCT:
        return "Silver"
    if pct >= BRONZE_PCT:
        return "Bronze"
    return "Needs Investigation"


@dataclass
class MetricResult:
    name: str
    expected_value: float
    observed_value: float
    difference: float
    fidelity_pct: float
    confidence_interval: str
    tier: str                     # "Gold", "Silver", "Bronze", "Needs Investigation"
    unit: str
    description: str
    passed: bool
    explanation: str


@dataclass
class ReproducibilityReport:
    report_id: str
    paper_id: str
    paper_title: str
    pipeline_name: str
    model_id: str
    dataset_manifest_id: str
    metric_results: List[MetricResult]
    overall_fidelity_pct: float
    overall_tier: str
    summary: str
    explanation_of_diffs: List[str]
    generated_at: str = field(default_factory=lambda: _dt.datetime.now(_dt.timezone.utc).isoformat())


class ReproducibilityReportEngine:
    """Generates structured reproducibility reports from pipeline metric results."""

    def __init__(self) -> None:
        self._registry = BenchmarkRegistry()
        self._reports: Dict[str, ReproducibilityReport] = {}

    def generate_report(
        self,
        paper_id: str,
        pipeline_name: str,
        model_id: str,
        dataset_manifest_id: str,
        observed_metrics: Dict[str, float],
        explanation_of_diffs: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        paper = self._registry.get(paper_id)
        if paper is None:
            return {"error": f"Unknown paper_id '{paper_id}'"}

        metric_results: List[MetricResult] = []
        for expected in paper.required_metrics:
            obs = observed_metrics.get(expected.name, 0.0)
            fidelity_pct = round((obs / expected.published_value) * 100.0, 2) if expected.published_value else 0.0
            tier = _fidelity_tier(obs, expected.published_value)
            diff = obs - expected.published_value
            explanation = next((exp for exp in (explanation_of_diffs or []) if expected.name in exp), "No explanation provided.")
            
            metric_results.append(MetricResult(
                name=expected.name,
                expected_value=expected.published_value,
                observed_value=obs,
                difference=diff,
                fidelity_pct=fidelity_pct,
                confidence_interval="+/- 0.01",  # Placeholder for true CI math
                tier=tier,
                unit=expected.unit,
                description=expected.description,
                passed=tier in ("Gold", "Silver", "Bronze"),
                explanation=explanation,
            ))

        overall_fidelity = round(
            sum(m.fidelity_pct for m in metric_results) / len(metric_results), 2
        ) if metric_results else 0.0

        if overall_fidelity >= GOLD_PCT:
            overall_tier = "Gold"
        elif overall_fidelity >= SILVER_PCT:
            overall_tier = "Silver"
        elif overall_fidelity >= BRONZE_PCT:
            overall_tier = "Bronze"
        else:
            overall_tier = "Needs Investigation"

        passed_count = sum(1 for m in metric_results if m.passed)
        summary = (
            f"Reproduced {passed_count}/{len(metric_results)} metrics at {overall_tier} tier "
            f"({overall_fidelity:.1f}% overall fidelity)."
        )

        report_id = f"report_{paper_id}_{int(_dt.datetime.now(_dt.timezone.utc).timestamp())}"
        report = ReproducibilityReport(
            report_id=report_id,
            paper_id=paper_id,
            paper_title=paper.title,
            pipeline_name=pipeline_name,
            model_id=model_id,
            dataset_manifest_id=dataset_manifest_id,
            metric_results=metric_results,
            overall_fidelity_pct=overall_fidelity,
            overall_tier=overall_tier,
            summary=summary,
            explanation_of_diffs=explanation_of_diffs or [],
        )
        self._reports[report_id] = report
        return asdict(report)

    def list_reports(self) -> List[Dict[str, Any]]:
        return [asdict(r) for r in self._reports.values()]

    def get_report(self, report_id: str) -> Optional[Dict[str, Any]]:
        r = self._reports.get(report_id)
        return asdict(r) if r else None
