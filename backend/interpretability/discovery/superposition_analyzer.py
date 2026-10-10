"""Superposition Analyzer Engine.

Would analyze neural superposition, feature interference, and polysemanticity
metrics by measuring how feature directions share the residual stream and how
much they interfere with one another.

Status: NOT IMPLEMENTED
-----------------------
This module previously returned, for every caller and with no layer::

    return {
        "layer": layer,
        "feature_dim": feature_dim,
        "superposition_degree": 0.38,
        "interference_score": 0.12,
        "sparsity_level": 0.05,
        "interpretation": "High feature capacity stored in superposition via non-orthogonal directions.",
    }

No activation measurement was performed; 0.38, 0.12, and 0.05 were hardcoded
and identical for every layer and feature dimension. That is not a
superposition measurement and would mislead any interference or sparsity
conclusion drawn from it.

`analyze_superposition()` now raises. Callers already fail closed on
`LiveUnavailable` rather than substituting a value.
"""

from __future__ import annotations

from typing import Any, Dict

from backend.science.models.adapter_base import LiveUnavailable


class SuperpositionAnalyzerEngine:
    """Not implemented. Raises rather than reporting fabricated superposition."""

    def analyze_superposition(self, layer: int = 8, feature_dim: int = 768) -> Dict[str, Any]:
        """Not implemented. Raises rather than reporting fabricated metrics."""
        raise LiveUnavailable(
            "SuperpositionAnalyzerEngine is not implemented. It previously "
            "returned hardcoded values of 0.38 (superposition_degree), 0.12 "
            "(interference_score), and 0.05 (sparsity_level) for every layer "
            "and feature_dim. No superposition measurement is performed by "
            "this module."
        )
