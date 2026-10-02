"""Evidence Ranking Engine.

Ranks discoveries along causal effect, reproducibility, confidence, novelty, and
interpretability.

The *ranking arithmetic* is real: given caller-supplied metrics it computes a
weighted score and sorts. What is not real is the default input. With no
discoveries supplied it invents two plausible ones (causal_effect 0.94,
reproducibility 0.98, novelty 0.85) and ranks them as though they were results.

That fixture is retained for catalog inspection but is now labelled, so a
caller can never mistake the ranked list for evidence it measured.
"""

from __future__ import annotations

from typing import Any, Dict, List


# Fixed, illustrative input. Marked synthetic on every path that uses it.
_SAMPLE: List[Dict[str, Any]] = [
    {"id": "disc_1", "title": "IOI Circuit L8_N402",
     "causal_effect": 0.94, "reproducibility": 0.98, "novelty": 0.85},
    {"id": "disc_2", "title": "Feature #1402 Polysemanticity",
     "causal_effect": 0.65, "reproducibility": 0.91, "novelty": 0.92},
]


class EvidenceRankerEngine:
    """Weighted ranking over caller-supplied metrics.

    Supply real discoveries. The built-in sample is synthetic and says so.
    """

    def rank_evidence(self, discoveries: List[Dict[str, Any]] | None = None) -> List[Dict[str, Any]]:
        synthetic = not discoveries
        sample = [dict(item) for item in (discoveries or _SAMPLE)]

        for item in sample:
            score = (
                item.get("causal_effect", 0.5) * 0.35 +
                item.get("reproducibility", 0.5) * 0.25 +
                item.get("novelty", 0.5) * 0.40
            )
            item["rank_score"] = round(score, 4)
            # A missing metric is being defaulted to 0.5, so the score is not a
            # measurement even when the other inputs are real.
            item["metrics_complete"] = all(
                isinstance(item.get(k), (int, float))
                for k in ("causal_effect", "reproducibility", "novelty")
            )

        ranked = sorted(sample, key=lambda x: x["rank_score"], reverse=True)
        for rank, item in enumerate(ranked, start=1):
            item["rank"] = rank
            item["provenance"] = "seeded" if synthetic else "reference"
            item["synthetic"] = synthetic
            item["validation_eligible"] = False
            item["publication_eligible"] = False
            if synthetic:
                item["reason"] = (
                    "No discoveries were supplied. These are fixed sample "
                    "entries and the ranking is not evidence."
                )
        return ranked
