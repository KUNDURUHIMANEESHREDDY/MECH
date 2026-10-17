"""Scientific Confidence Engine.

NOT IMPLEMENTED as a measurement. ``compute_confidence`` is a weighted sum over
two caller-supplied numbers, but both had defaults (``reproducibility_score``
0.95, ``p_value`` 0.001) and the response hardcoded
``statistical_rigor: "High"``. Calling it with no arguments therefore reported
high statistical rigor for a platform that had measured nothing.

The ``uncertainty_interval`` was not an interval either: it was ``conf - 0.04``
and ``conf + 0.03``, two fixed offsets applied to the score itself, so it moved
with the weighted sum and carried no information the score did not already
carry.

The assumed defaults are removed rather than relabelled, because there is no
confidence to report without inputs. ``statistical_rigor`` is no longer returned
as a rating: no input to this method can establish it.
"""

from __future__ import annotations

from typing import Any, Dict, Optional


class ScientificConfidenceEngine:
    """Confidence from caller-supplied metrics. Refuses to invent one."""

    def compute_confidence(
        self,
        reproducibility_score: Optional[float] = None,
        p_value: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Weighted sum over supplied metrics, or a refusal if any is missing.

        A partial input set is not enough. The score is a sum, so a missing
        term would silently become a specific number with no evidence behind
        it -- which is the defect this replaces, not a lesser version of it.
        """
        supplied = {
            "reproducibility_score": reproducibility_score,
            "p_value": p_value,
        }
        assumed_inputs = sorted(
            name for name, value in supplied.items() if value is None
        )
        if assumed_inputs:
            return {
                "confidence_score": None,
                "uncertainty_interval": None,
                "p_value": p_value,
                "statistical_rigor": None,
                "status": "unavailable",
                "provenance": "unavailable",
                "inputs_assumed": True,
                "assumed_inputs": assumed_inputs,
                "measured": False,
                "validation_eligible": False,
                "publication_eligible": False,
                "reason": (
                    f"No value was supplied for {', '.join(assumed_inputs)}. "
                    "Both terms are required, so no confidence score is "
                    "reported. This method previously defaulted them (0.95 and "
                    "0.001) and returned 'High' statistical rigor for a "
                    "platform that had measured nothing."
                ),
            }

        conf = min(
            0.99,
            reproducibility_score * 0.95 + (0.05 if p_value < 0.01 else 0.0),
        )
        return {
            "confidence_score": round(conf, 4),
            # No distribution or sample stands behind this score, so no
            # interval can be derived from it. The previous interval was two
            # fixed offsets on the score itself.
            "uncertainty_interval": None,
            "p_value": p_value,
            # Not a rating. These two inputs cannot establish statistical rigor.
            "statistical_rigor": None,
            "status": "computed",
            "provenance": "reference",
            "inputs_assumed": False,
            "assumed_inputs": [],
            "measured": False,
            "validation_eligible": False,
            "publication_eligible": False,
            "reason": (
                "Weighted sum over the two supplied metrics. The score is not a "
                "calibrated probability and no uncertainty interval is "
                "reported, because one cannot be derived from these inputs."
            ),
        }
