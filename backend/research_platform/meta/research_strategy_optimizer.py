"""Epic 3 — Research Strategy Optimizer.

Learns task-specific execution sequences based on historical effectiveness.
"""

from __future__ import annotations

from typing import Any, Dict, List


class ResearchStrategyOptimizer:
    """Optimizes execution pipeline steps for specific scientific research domains."""

    def __init__(self) -> None:
        self.domain_strategies: Dict[str, List[str]] = {
            "circuit_discovery": ["Run SAE Inspection", "Skip Attribution Patching", "Run Causal Tracing", "Verify Cross-Model Alignment"],
            "hypothesis_testing": ["Verify Minimum Sample Density", "Run Causal Intervention", "Compute Uncertainty Bounds", "Debate Anomalies"],
            "feature_analysis": ["Load SAE Checkpoint", "Cluster Polysemantic Features", "Extract Monosemantic Directions", "Annotate Semantic Label"],
        }
        self.strategy_scores: Dict[str, float] = {
            "circuit_discovery": 0.94,
            "hypothesis_testing": 0.89,
            "feature_analysis": 0.91,
        }

    def recommend_strategy(self, domain: str = "circuit_discovery") -> Dict[str, Any]:
        sequence = self.domain_strategies.get(domain, ["Run SAE Inspection", "Run Causal Tracing"])
        score = self.strategy_scores.get(domain, 0.85)

        # The score is a literal in a dict, not the output of a meta-learning
        # analysis. It previously reported "historical_effectiveness_score"
        # with reasoning claiming it was "derived from meta-learning empirical
        # success analysis" -- no analysis exists, and several steps in these
        # sequences (SAE inspection, path patching) are not implemented at all.
        assumed = domain not in self.strategy_scores
        return {
            "domain": domain,
            "recommended_workflow": sequence,
            "historical_effectiveness_score": None if assumed else score,
            "score_is_assumed": True,
            "reasoning": (
                f"No effectiveness was ever measured for '{domain}'. This is a "
                "static template sequence, and several of its steps (SAE "
                "inspection, attribution patching) are not implemented in this "
                "repository."
            ),
            "provenance": "reference",
            "validation_eligible": False,
            "publication_eligible": False,
        }

    def update_strategy(self, domain: str, new_sequence: List[str], observed_effectiveness: float) -> Dict[str, Any]:
        self.domain_strategies[domain] = new_sequence
        self.strategy_scores[domain] = round(observed_effectiveness, 4)

        return {
            "status": "StrategyUpdated",
            "domain": domain,
            "updated_workflow": new_sequence,
            "effectiveness_score": observed_effectiveness,
        }
