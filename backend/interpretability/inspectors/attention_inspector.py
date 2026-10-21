"""Attention Inspector.

Analyzes multi-head attention patterns, query-key attention scores,
and attention head specialization (induction heads, previous-token heads).
"""

from __future__ import annotations

from typing import Any, Dict, List


class AttentionInspector:
    """Inspector for multi-head attention distributions."""

    def inspect(self, layer: int, head: int, tokens: List[str] | None = None) -> Dict[str, Any]:
        """Refuse: no model and no forward pass are bound to this inspector.

        Args:
            layer: Echoed back for correlation.
            head: Echoed back for correlation.
            tokens: Echoed back for correlation.

        Returns:
            An unavailable record carrying no attention matrix.

        This previously invented a causal matrix where row ``i`` held
        ``1.0 / (i + 1)`` for attended positions, a token list, and a
        ``head_type`` of "Induction Head" for any head in (8, 9) -- none of
        it read from a model, all of it shaped like attention. The
        implemented path is ``gpt2_engine.attention_head``.
        """
        return {
            "layer": layer,
            "head": head,
            "head_id": f"L{layer}_H{head}",
            "tokens": list(tokens or []),
            "head_type": None,
            "attention_matrix": [],
            "status": "unavailable",
            "provenance": "unavailable",
            "measured": False,
            "validation_eligible": False,
            "publication_eligible": False,
            "reason": (
                "Attention inspection is not implemented in this inspector: "
                "no forward pass was run, so there is no attention to "
                "report. See gpt2_engine.attention_head."
            ),
        }
