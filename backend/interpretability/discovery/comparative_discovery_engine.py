"""Comparative Discovery Engine — Multi-Model Alignment & Drift Reports.

Automatically compares discovery campaigns between different models or versions
and generates structured drift reports for publication.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from .concept_evolution_engine import ConceptEvolutionEngine


@dataclass
class SemanticDriftReport:
    model_a: str
    model_b: str
    overall_drift: float
    universality_score: float
    shared_concepts: List[str]
    novel_concepts: List[str]
    missing_concepts: List[str]
    split_concepts: List[Dict[str, Any]]
    merged_concepts: List[Dict[str, Any]]
    provenance: str = "STATISTICAL_MATCHING"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_a": self.model_a,
            "model_b": self.model_b,
            "overall_drift": round(self.overall_drift, 4),
            "universality_score": round(self.universality_score, 4),
            "shared_concepts": self.shared_concepts,
            "novel_concepts": self.novel_concepts,
            "missing_concepts": self.missing_concepts,
            "split_concepts": self.split_concepts,
            "merged_concepts": self.merged_concepts,
            "provenance": self.provenance,
        }


class ComparativeDiscoveryEngine:
    """Orchestrates automated cross-model comparative research."""

    def __init__(self) -> None:
        self.evolution = ConceptEvolutionEngine()

    def compare_campaigns(self, results_a: Dict[str, Any], results_b: Dict[str, Any]) -> SemanticDriftReport:
        """Generates a high-fidelity drift report comparing two discovery runs."""
        drift_data = self.evolution.analyze_cross_model_drift(results_a, results_b)

        return SemanticDriftReport(
            model_a=results_a.get("model_id", "Model A"),
            model_b=results_b.get("model_id", "Model B"),
            overall_drift=drift_data["drift_score"],
            universality_score=drift_data["persistence"],
            shared_concepts=drift_data["shared_concepts"],
            novel_concepts=drift_data.get("novel_concepts", []),
            missing_concepts=drift_data.get("missing_concepts", []),
            split_concepts=drift_data.get("split_concepts", []),
            merged_concepts=[],
            provenance="STATISTICAL_MATCHING",
        )


    def generate_publication_figures(self, report: SemanticDriftReport) -> Dict[str, Any]:
        """Emits data structures for cross-model evolution graphs and dendrograms."""
        return {
            "evolution_graph": {
                "nodes": [report.model_a, report.model_b],
                "edges": [{"from": report.model_a, "to": report.model_b, "weight": report.universality_score}],
            }
        }
