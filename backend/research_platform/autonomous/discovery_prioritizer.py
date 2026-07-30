"""Discovery Prioritizer Engine."""

from __future__ import annotations

from typing import Any, Dict, List


class DiscoveryPrioritizerEngine:
    """Scores and ranks discoveries by novelty, reproducibility, impact, and statistical confidence."""

    def prioritize_discoveries(self, discoveries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        prioritized = []
        for d in discoveries:
            novelty = d.get("novelty", 0.9)
            repro = d.get("reproducibility", 0.95)
            impact = d.get("impact", 0.88)
            conf = d.get("confidence", 0.92)
            composite_score = round(novelty * 0.3 + repro * 0.3 + impact * 0.2 + conf * 0.2, 4)

            prioritized.append({
                **d,
                "composite_score": composite_score,
            })

        return sorted(prioritized, key=lambda x: x["composite_score"], reverse=True)
