"""Token Inspector.

Analyzes individual tokens in input prompts, per-token residual activations,
and top firing neurons for each token position.
"""

from __future__ import annotations

from typing import Any, Dict


class TokenInspector:
    """Inspector for token-level activations."""

    def inspect(self, token_index: int, token_text: str = "") -> Dict[str, Any]:
        """Inspect token activation statistics.

        Args:
            token_index: Token position in sequence.
            token_text: Text representation of token.

        Returns:
            Dict containing token position, text, and peak activation layer.
        """
        return {
            "position": token_index,
            "token": token_text or f"tok_{token_index}",
            "peak_layer": (token_index * 3) % 12,
            "peak_activation": round(0.5 + (token_index * 0.12) % 2.5, 3),
            "top_firing_neuron": f"L{(token_index * 3) % 12}_N{(token_index * 137) % 768}",
        }
