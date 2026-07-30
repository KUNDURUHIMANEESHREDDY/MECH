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
        novelty: float = 0.90,
        reproducibility: float = 0.95,
        confidence: float = 0.92,
        benchmark_performance: float = 0.94,
        interpretability: float = 0.88,
        cross_model_support: float = 0.89,
        custom_weights: QualityScoreWeights | Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        w = custom_weights if isinstance(custom_weights, QualityScoreWeights) else (QualityScoreWeights.from_dict(custom_weights) if custom_weights else self.weights)

        dimensions = {
            "novelty": novelty,
            "reproducibility": reproducibility,
            "confidence": confidence,
            "benchmark_performance": benchmark_performance,
            "interpretability": interpretability,
            "cross_model_support": cross_model_support,
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
        }
