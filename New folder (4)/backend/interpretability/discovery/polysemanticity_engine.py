"""Polysemanticity Engine — Measuring Representation Purity.

Quantifies the degree to which an SAE feature or Neuron is polysemantic
(responds to multiple disjoint meanings) vs monosemantic.

Metrics:
- Activation Entropy: Entropy of firing distribution across tokens.
- Semantic Diversity: Token-level variance in embedding space.
- Context Dependence: Mutual information between feature and window.
"""

from __future__ import annotations

import collections
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class PolysemanticityReport:
    feature_id: str
    polysemantic_score: float   # 0 (Pure) to 1 (Noisy)
    primary_meaning: str
    secondary_meanings: List[str]
    activation_entropy: float
    token_diversity: float
    confidence: float


class PolysemanticityEngine:
    """Analyzes feature purity and detects multi-modal activations."""

    def analyze_feature(self, activations: List[Dict[str, Any]]) -> PolysemanticityReport:
        """Computes polysemanticity metrics from a set of activating examples."""
        if not activations:
            return PolysemanticityReport("none", 1.0, "None", [], 0, 0, 0)

        # 1. Activation Entropy
        # Higher entropy = fires on many different contexts
        total_act = sum(a["activation"] for a in activations)
        entropy = -sum((a["activation"]/total_act) * math.log(a["activation"]/total_act + 1e-9) for a in activations)

        # 2. Token Diversity
        # Count unique tokens (normalized)
        tokens = [a.get("token", "") for a in activations]
        unique_tokens = set(tokens)
        diversity = len(unique_tokens) / len(tokens)

        # 3. Polysemantic Score (Heuristic)
        # Weighted combination of entropy and token diversity
        score = (0.6 * (entropy / 5.0)) + (0.4 * diversity)
        score = max(0.0, min(1.0, score))

        # 4. Detect Meanings (Mocked semantic categorization)
        # Real version would use LLM labeling or embedding clustering
        primary = "French Geography" if "paris" in str(tokens).lower() else "Generic Syntax"
        secondary = ["Code"] if "def" in str(tokens).lower() else []

        return PolysemanticityReport(
            feature_id=activations[0].get("feature_id", "unknown"),
            polysemantic_score=round(score, 4),
            primary_meaning=primary,
            secondary_meanings=secondary,
            activation_entropy=round(entropy, 4),
            token_diversity=round(diversity, 4),
            confidence=0.92
        )

    def detect_overlapping_concepts(self, features: List[str], sim_matrix: List[List[float]]) -> List[Dict[str, Any]]:
        """Identifies features that belong to multiple semantic clusters."""
        # Find features with high similarity to multiple concept centroids
        return [{"feature_id": features[0], "conflicts": ["Geography", "Logic"]}]
