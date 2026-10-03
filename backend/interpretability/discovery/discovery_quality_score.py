"""Multi-Dimensional Discovery Quality Score Engine with Configurable Weighting Schemes & Separated Metric Dimensions."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict


@dataclass
class QualityScoreWeights:
    """Configurable weight factors for multi-dimensional discovery evaluation."""

    novelty: float = 0.25
    reproducibility: float = 0.25
    confidence: float = 0.20
    benchmark: float = 0.15
    interpretability: float = 0.10
    cross_model: float = 0.05

    def normalize(self) -> QualityScoreWeights:
        total = self.novelty + self.reproducibility + self.confidence + self.benchmark + self.interpretability + self.cross_model
        if total <= 0:
            return QualityScoreWeights()
        return QualityScoreWeights(
            novelty=self.novelty / total,
            reproducibility=self.reproducibility / total,
            confidence=self.confidence / total,
            benchmark=self.benchmark / total,
            interpretability=self.interpretability / total,
            cross_model=self.cross_model / total,
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any] | None) -> QualityScoreWeights:
        if not data:
            return cls()
        return cls(
            novelty=data.get("novelty", 0.25),
            reproducibility=data.get("reproducibility", 0.25),
            confidence=data.get("confidence", 0.20),
            benchmark=data.get("benchmark", 0.15),
            interpretability=data.get("interpretability", 0.10),
            cross_model=data.get("cross_model", 0.05),
        ).normalize()


class DiscoveryQualityScoreEngine:
    """Summarizes discovery quality across 6 key dimensions using configurable weighting schemes

    and produces distinct Confidence, Scientific Quality, and Expected Impact metrics.
    """

    def __init__(self, default_weights: QualityScoreWeights | Dict[str, Any] | None = None) -> None:
        self.weights = default_weights if isinstance(default_weights, QualityScoreWeights) else QualityScoreWeights.from_dict(default_weights)

    def compute_quality_score(
        self,
        novelty: float | None = None,
        reproducibility: float | None = None,
        confidence: float | None = None,
        benchmark_performance: float | None = None,
        interpretability: float | None = None,
        cross_model_support: float | None = None,
        custom_weights: QualityScoreWeights | Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        """Score a discovery across six weighted dimensions.

        Every dimension defaults to None, meaning *not measured*. They previously
        defaulted to `0.90 / 0.95 / 0.92 / 0.94 / 0.88 / 0.89`, which meant
        `compute_quality_score()` called with no arguments at all returned:

            overall_quality_score  ~0.91
            scientific_quality_score ~0.94
            expected_impact_score    ~0.89
            quality_grade            "A+"

        An A+ grade, from six numbers nobody measured, produced by a function
        whose only argument is the caller's choice to supply none. Each default
        was individually unremarkable; together they composed into a top grade.

        When any dimension is absent the score cannot be computed, and this
        returns the same shape with every score `None` and `quality_grade`
        `None`, plus a `reason`. That keeps the documented return shape intact
        while making an unmeasured discovery unscored rather than graded.

        There were no callers of this method, so the signature change is free.
        """
        w = custom_weights if isinstance(custom_weights, QualityScoreWeights) else (QualityScoreWeights.from_dict(custom_weights) if custom_weights else self.weights)

        dimensions = {
            "novelty": novelty,
            "reproducibility": reproducibility,
            "confidence": confidence,
            "benchmark_performance": benchmark_performance,
            "interpretability": interpretability,
            "cross_model_support": cross_model_support,
        }

        absent = sorted(name for name, value in dimensions.items() if value is None)
        if absent:
            return {
                "overall_quality_score": None,
                "confidence_score": None,
                "scientific_quality_score": None,
                "expected_impact_score": None,
                "dimensions": dimensions,
                "weights": asdict(w),
                "quality_grade": None,
                "provenance": "unavailable",
                "measured": False,
                "reason": (
                    "Not scored: no measurement supplied for "
                    + ", ".join(absent)
                    + ". A quality grade computed from default values would be a "
                    "grade of nothing."
                ),
            }

        weighted_sum = (
            novelty * w.novelty
            + reproducibility * w.reproducibility
            + confidence * w.confidence
            + benchmark_performance * w.benchmark
            + interpretability * w.interpretability
            + cross_model_support * w.cross_model
        )

        overall = round(weighted_sum, 4)
        scientific_quality = round(0.50 * reproducibility + 0.30 * benchmark_performance + 0.20 * interpretability, 4)
        expected_impact = round(0.40 * novelty + 0.35 * cross_model_support + 0.25 * confidence, 4)

        return {
            "overall_quality_score": overall,
            "confidence_score": round(confidence, 4),
            "scientific_quality_score": scientific_quality,
            "expected_impact_score": expected_impact,
            "dimensions": dimensions,
            "weights": asdict(w),
            "quality_grade": "A+" if overall >= 0.90 else "A",
            "provenance": "live",
            "measured": True,
            "reason": None,
        }
