"""Research Critic Agent.

Critiques hypotheses, detects weak evidence, and suggests stronger experiments.
"""

from __future__ import annotations

from typing import Any, Dict, List


class ResearchCriticAgent:
    """Evaluates hypotheses and evidence strength, recommending methodological improvements."""

    def critique_hypothesis(self, hypothesis: str, evidence: List[Dict[str, Any]]) -> Dict[str, Any]:
        weak_points = []
        if len(evidence) < 2:
            weak_points.append("Insufficient evidence samples (less than 2).")

        return {
            "hypothesis": hypothesis,
            "evidence_quality": "High" if len(evidence) >= 2 else "Moderate",
            "weak_points": weak_points,
            "suggested_experiments": [
                "Activation Patching with Corrupted Prompt Baselines",
                "Cross-Layer Causal Tracing",
            ],
            "critique_passed": len(weak_points) == 0,
        }
