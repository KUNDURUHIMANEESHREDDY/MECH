"""Platform Research Confidence Engine.

The scoring arithmetic is real and operates on caller-supplied inputs. What is
not real is the *default*: calling ``score_confidence()`` with no arguments used
to assume 5 pieces of evidence and a reproducibility of 0.95, producing a
confidence above 0.90 and a "High" reliability rating for a platform that had
measured nothing.

The defaults are kept for the formula's shape, but the response now says
whether its inputs were supplied or assumed.
"""

from __future__ import annotations

from typing import Any, Dict, Optional


class PlatformConfidenceEngine:
    """Weighted confidence from supplied metrics. Assumed inputs are labelled."""

    def score_confidence(
        self,
        evidence_count: Optional[int] = None,
        reproducibility_score: Optional[float] = None,
        variance: Optional[float] = None,
    ) -> Dict[str, Any]:
        assumed = (
            evidence_count is None
            or reproducibility_score is None
            or variance is None
        )
        # Former defaults, used only when the caller supplies nothing. Retained
        # so the formula's shape is unchanged and testable.
        n = 5 if evidence_count is None else evidence_count
        repro = 0.95 if reproducibility_score is None else reproducibility_score
        var = 0.04 if variance is None else variance

        conf = min(0.99, 0.70 + (n * 0.04) + (repro * 0.10) - (var * 0.5))
        unc = round(1.0 - conf, 4)
        rel = "High" if conf > 0.85 else "Moderate" if conf > 0.65 else "Low"

        return {
            "confidence_score": round(conf, 4),
            "uncertainty": unc,
            "reliability_rating": rel,
            "evidence_count": n,
            "reproducibility_score": repro,
            "variance": var,
            "inputs_assumed": assumed,
            "provenance": "unavailable" if assumed else "reference",
            "validation_eligible": False,
            "publication_eligible": False,
            "reason": (
                "Inputs were not supplied, so evidence_count, "
                "reproducibility_score and variance are assumed defaults "
                "rather than measurements. This score does not describe the "
                "state of any real evidence."
                if assumed else
                "Scored from supplied metrics. The score is a weighted sum, "
                "not a calibrated probability."
            ),
        }
