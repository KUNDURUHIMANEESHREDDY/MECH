"""Belief Entropy Engine for MECH.

Quantifies scientific uncertainty over the competing hypothesis space:
1. Calculates Shannon Entropy: H(P) = -sum(p_i * log2(p_i))
2. Calculates Normalized Uncertainty Fraction: H_norm = H(P) / log2(N)
3. Identifies the highest-uncertainty competing hypothesis pairs that require discriminating experimental resolution.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Tuple


@dataclass(frozen=True)
class HypothesisEntropyProfile:
    """Entropy and uncertainty profile over a set of competing hypotheses."""
    shannon_entropy_bits: float
    max_possible_entropy_bits: float
    normalized_uncertainty_fraction: float
    is_uncertainty_resolved: bool
    dominant_hypothesis_id: str
    dominant_hypothesis_probability: float
    highest_uncertainty_pairs: List[Tuple[str, str, float]]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class BeliefEntropyEngine:
    """Calculates belief entropy and identifies pairwise uncertainty bottlenecks."""

    def __init__(self, resolution_threshold_bits: float = 0.45) -> None:
        self.resolution_threshold_bits = resolution_threshold_bits


    def compute_entropy_profile(
        self,
        hypothesis_probabilities: Dict[str, float],
    ) -> HypothesisEntropyProfile:
        """Computes Shannon entropy, normalized uncertainty, and highest-uncertainty pairs."""
        probs = {k: max(v, 1e-9) for k, v in hypothesis_probabilities.items()}
        total_p = sum(probs.values())
        norm_probs = {k: v / total_p for k, v in probs.items()}

        n = len(norm_probs)
        if n <= 1:
            return HypothesisEntropyProfile(
                shannon_entropy_bits=0.0,
                max_possible_entropy_bits=0.0,
                normalized_uncertainty_fraction=0.0,
                is_uncertainty_resolved=True,
                dominant_hypothesis_id=list(norm_probs.keys())[0] if n == 1 else "NONE",
                dominant_hypothesis_probability=1.0,
                highest_uncertainty_pairs=[],
            )

        # 1. Shannon Entropy
        entropy = -sum(p * math.log2(p) for p in norm_probs.values())
        max_entropy = math.log2(n)
        norm_fraction = min(1.0, max(0.0, entropy / max_entropy))

        # 2. Dominant Hypothesis
        dominant_id, dominant_prob = max(norm_probs.items(), key=lambda x: x[1])

        # 3. Pairwise Uncertainty (Product of probabilities indicates ambiguity between pairs)
        pairs: List[Tuple[str, str, float]] = []
        keys = list(norm_probs.keys())
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                k1, k2 = keys[i], keys[j]
                ambiguity_score = norm_probs[k1] * norm_probs[k2] * 4.0  # Normalized so 0.5*0.5 -> 1.0
                pairs.append((k1, k2, round(ambiguity_score, 4)))

        pairs.sort(key=lambda x: x[2], reverse=True)

        is_resolved = entropy <= self.resolution_threshold_bits and dominant_prob >= 0.85

        return HypothesisEntropyProfile(
            shannon_entropy_bits=round(entropy, 4),
            max_possible_entropy_bits=round(max_entropy, 4),
            normalized_uncertainty_fraction=round(norm_fraction, 4),
            is_uncertainty_resolved=is_resolved,
            dominant_hypothesis_id=dominant_id,
            dominant_hypothesis_probability=round(dominant_prob, 4),
            highest_uncertainty_pairs=pairs[:3],
        )
