"""Induction Head Detector.

Would detect induction heads (A B ... A ➔ B) across attention layers by
measuring prefix-matching attention on repeated-block prompts.

Status: NOT IMPLEMENTED
-----------------------
This module previously returned, for every caller and with no model::

    return [
        {"layer": 5, "head": 1, "prefix_score": 0.94, "is_induction_head": True},
        {"layer": 5, "head": 5, "prefix_score": 0.91, "is_induction_head": True},
        {"layer": 6, "head": 9, "prefix_score": 0.88, "is_induction_head": True},
    ]

No attention pattern was loaded; the same three heads with scores 0.94, 0.91,
and 0.88 were fabricated for every model name. That is not induction-head
detection and would silently attribute the same circuit to every model.

`detect_induction_heads()` now raises. Callers already fail closed on
`LiveUnavailable` rather than substituting a value.
"""

from __future__ import annotations

from typing import Any, Dict, List

from backend.science.models.adapter_base import LiveUnavailable


class InductionHeadDetector:
    """Not implemented. Raises rather than reporting fabricated heads."""

    def detect_induction_heads(self, model_name: str = "GPT-2 Small") -> List[Dict[str, Any]]:
        """Not implemented. Raises rather than reporting fabricated heads."""
        raise LiveUnavailable(
            "InductionHeadDetector is not implemented. It previously returned "
            "a hardcoded list of induction heads [(5, 1, 0.94), (5, 5, 0.91), "
            "(6, 9, 0.88)] for every model. No induction-head measurement is "
            "performed by this module."
        )
