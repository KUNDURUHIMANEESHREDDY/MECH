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

    # Former defaults, used only for the inputs the caller omits. Retained so
    # the formula's shape is unchanged and testable.
    ASSUMED_DEFAULTS = {
        "evidence_count": 5,
        "reproducibility_score": 0.95,
        "variance": 0.04,
    }

    def score_confidence(
        self,
        evidence_count: Optional[int] = None,
        reproducibility_score: Optional[float] = None,
        variance: Optional[float] = None,
    ) -> Dict[str, Any]:
        supplied = {
            "evidence_count": evidence_count,
            "reproducibility_score": reproducibility_score,
            "variance": variance,
        }
        # Name the specific fields that were defaulted. A blanket
        # "inputs were not supplied" is false whenever a caller supplies some
        # of them, and it hides which number is doing the work.
        assumed_inputs = sorted(
            name for name, value in supplied.items() if value is None
        )
        assumed = bool(assumed_inputs)

        n = (self.ASSUMED_DEFAULTS["evidence_count"]
             if evidence_count is None else evidence_count)
        repro = (self.ASSUMED_DEFAULTS["reproducibility_score"]
                 if reproducibility_score is None else reproducibility_score)
        var = (self.ASSUMED_DEFAULTS["variance"]
               if variance is None else variance)

        conf = min(0.99, 0.70 + (n * 0.04) + (repro * 0.10) - (var * 0.5))
        unc = round(1.0 - conf, 4)
        rel = "High" if conf > 0.85 else "Moderate" if conf > 0.65 else "Low"

        if assumed:
            reason = (
                f"No value was supplied for {', '.join(assumed_inputs)}, so "
                "those terms use assumed defaults rather than measurements. "
                "This score does not describe the state of any real evidence."
            )
        else:
            reason = (
                "Scored from supplied metrics. The score is a weighted sum, "
                "not a calibrated probability."
            )

        return {
            "confidence_score": round(conf, 4),
            "uncertainty": unc,
            "reliability_rating": rel,
            "evidence_count": n,
            "reproducibility_score": repro,
            "variance": var,
            "inputs_assumed": assumed,
            "assumed_inputs": assumed_inputs,
            "provenance": "unavailable" if assumed else "reference",
            "validation_eligible": False,
            # A weighted sum over supplied numbers is not evidence, whichever
            # path produced the inputs. There is no route here to eligible.
            "publication_eligible": False,
            "reason": reason,
        }
