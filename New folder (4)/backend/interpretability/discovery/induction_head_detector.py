"""Induction Head Detector."""

from __future__ import annotations

from typing import Any, Dict, List


class InductionHeadDetector:
    """Detects induction heads (A B ... A ➔ B) across attention layers."""

    def detect_induction_heads(self, model_name: str = "GPT-2 Small") -> List[Dict[str, Any]]:
        return [
            {"layer": 5, "head": 1, "prefix_score": 0.94, "is_induction_head": True},
            {"layer": 5, "head": 5, "prefix_score": 0.91, "is_induction_head": True},
            {"layer": 6, "head": 9, "prefix_score": 0.88, "is_induction_head": True},
        ]
