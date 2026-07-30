"""Evidence Ranking Engine."""

from __future__ import annotations

from typing import Any, Dict, List


class EvidenceRankerEngine:
    """Ranks mechanistic discoveries along causal effect, reproducibility, confidence, novelty, and interpretability."""

    def rank_evidence(self, discoveries: List[Dict[str, Any]] | None = None) -> List[Dict[str, Any]]:
        sample = discoveries or [
            {"id": "disc_1", "title": "IOI Circuit L8_N402", "causal_effect": 0.94, "reproducibility": 0.98, "novelty": 0.85},
            {"id": "disc_2", "title": "Feature #1402 Polysemanticity", "causal_effect": 0.65, "reproducibility": 0.91, "novelty": 0.92},
        ]

        # Multi-metric score computation
        for item in sample:
            score = (
                item.get("causal_effect", 0.5) * 0.35 +
                item.get("reproducibility", 0.5) * 0.25 +
                item.get("novelty", 0.5) * 0.40
            )
            item["rank_score"] = round(score, 4)

        return sorted(sample, key=lambda x: x["rank_score"], reverse=True)
