"""Token Inspector.

Analyzes individual tokens in input prompts, per-token residual activations,
and top firing neurons for each token position.
"""

from __future__ import annotations

from typing import Any, Dict


class TokenInspector:
    """Inspector for token-level activations."""

    def inspect(self, token_index: int, token_text: str = "") -> Dict[str, Any]:
        """Refuse: no model and no activation is bound to this inspector.

        Args:
            token_index: Echoed back for correlation.
            token_text: Echoed back for correlation.

        Returns:
            An unavailable record carrying no token statistics.

        This previously returned ``peak_layer = (token_index * 3) % 12`` and
        ``peak_activation = 0.5 + (token_index * 0.12) % 2.5``, and invented
        the token text when none was given. All three are arithmetic on the
        argument. The implemented path is ``gpt2_engine.layer_activations``.
        """
        return {
            "position": token_index,
            "token": token_text,
            "peak_layer": None,
            "peak_activation": None,
            "top_firing_neuron": None,
            "status": "unavailable",
            "provenance": "unavailable",
            "measured": False,
            "validation_eligible": False,
            "publication_eligible": False,
            "reason": (
                "Token inspection is not implemented in this inspector: no "
                "forward pass was run, so there is no per-token activation to "
                "report. See gpt2_engine.layer_activations."
            ),
        }
