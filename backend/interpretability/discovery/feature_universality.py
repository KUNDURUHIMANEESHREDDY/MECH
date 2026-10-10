"""Feature Universality Engine.

Would measure feature universality across model families (GPT, LLaMA, Gemma,
Qwen) by aligning the same feature across independently trained models.

Status: NOT IMPLEMENTED
-----------------------
This module previously returned, for every caller and with no feature::

    return {
        "feature_id": feature_id,
        "universal_alignment_score": 0.89,
        "aligned_models": ["GPT-2 Small", "Gemma-2B", "Llama-3-8B"],
        "universal_concept": "Capital City Retrieval",
        "is_universal": True,
    }

No cross-model alignment was run; 0.89 and `is_universal: True` were
hardcoded for every feature. Reporting a fabricated alignment score would
invent universality evidence that was never measured.

`measure_universality()` now raises. Callers already fail closed on
`LiveUnavailable` rather than substituting a value.
"""

from __future__ import annotations

from typing import Any, Dict, List

from backend.science.models.adapter_base import LiveUnavailable


class FeatureUniversalityEngine:
    """Not implemented. Raises rather than reporting fabricated universality."""

    def measure_universality(self, feature_id: int = 1402) -> Dict[str, Any]:
        """Not implemented. Raises rather than reporting fabricated universality."""
        raise LiveUnavailable(
            "FeatureUniversalityEngine is not implemented. It previously "
            "returned universal_alignment_score 0.89 and is_universal=True "
            "for every feature. No universality measurement is performed by "
            "this module."
        )
