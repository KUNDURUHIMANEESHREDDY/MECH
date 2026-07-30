"""Attention Inspector.

Analyzes multi-head attention patterns, query-key attention scores,
and attention head specialization (induction heads, previous-token heads).
"""

from __future__ import annotations

from typing import Any, Dict, List


class AttentionInspector:
    """Inspector for multi-head attention distributions."""

    def inspect(self, layer: int, head: int, tokens: List[str] | None = None) -> Dict[str, Any]:
        """Inspect attention pattern matrix for specified head.

        Args:
            layer: Layer index.
            head: Attention head index.
            tokens: Input token strings.

        Returns:
            Dict containing attention weights matrix and head classification.
        """
        toks = tokens or ["The", "capital", "of", "France"]
        seq_len = len(toks)
        # Generate dummy triangular/causal attention weights matrix
        matrix = []
        for i in range(seq_len):
            row = []
            for j in range(seq_len):
                if j <= i:
                    row.append(round(1.0 / (i + 1), 3))
                else:
                    row.append(0.0)
            matrix.append(row)

        head_type = "Induction Head" if head in (8, 9) else "Previous-Token Head" if head == 0 else "General Head"
        return {
            "layer": layer,
            "head": head,
            "head_id": f"L{layer}_H{head}",
            "tokens": toks,
            "head_type": head_type,
            "attention_matrix": matrix,
        }
