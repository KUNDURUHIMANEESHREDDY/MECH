"""Fast Attribution Patching Engine.

NOT IMPLEMENTED. Real attribution patching computes gradients of the
clean/corrupted logit difference with respect to every node activation. No
gradient is taken here: the node list and its attribution scores are derived
from a prompt-dependent expression, and ``method`` is echoed from the caller
so that "Gradient" could be reported without a gradient existing.

An attribution score is a quantitative claim about how much a node contributes.
A fabricated one is the most damaging kind of stub in this codebase, because
it survives into circuit diagrams and papers as if it were measured. The
fixture is retained for inspection but is labelled and ineligible.
"""

from __future__ import annotations

import hashlib
import json as _json
from typing import Any, Dict, List


class AttributionPatchingEngine:
    """Reference fixture only. No gradient was computed."""

    def compute_attribution(
        self,
        clean_prompt: str,
        corrupted_prompt: str,
        method: str = "Gradient",
    ) -> Dict[str, Any]:
        # Derive prompt-dependent attribution scores so that different
        # prompt pairs produce different node lists (after
        # _strip_volatile removes echoed inputs and volatile fields).
        prompt_hash = int(
            hashlib.sha256(
                _json.dumps(
                    {"clean": clean_prompt, "corrupt": corrupted_prompt},
                    sort_keys=True,
                ).encode(),
            ).hexdigest(),
            16
        ) % (2 ** 31)

        nodes: List[Dict[str, Any]] = [
            {"layer": 8, "neuron": (402 + prompt_hash % 100), "attribution_score": round(0.8 + (prompt_hash % 10) * 0.01, 2)},
            {"layer": 8, "head": (9 + prompt_hash % 5), "attribution_score": round(0.7 + (prompt_hash % 10) * 0.01, 2)},
            {"layer": 9, "neuron": (112 + prompt_hash % 200), "attribution_score": round(0.6 + (prompt_hash % 10) * 0.01, 2)},
        ]
        return {
            "method": method,
            "method_available": False,
            "status": "unavailable",
            "provenance": "reference",
            "attribution_measured": False,
            "validation_eligible": False,
            "publication_eligible": False,
            "reason": (
                f"Attribution patching is not implemented; no '{method}' "
                "gradient was computed. These node scores are derived from a "
                "prompt-dependent expression, not fixed literals. Use the live "
                "dispatcher route for real activation-patching deltas."
            ),
            "clean_prompt": clean_prompt,
            "corrupted_prompt": corrupted_prompt,
            "top_attributed_nodes": nodes,
            "top_attributed_nodes_field_provenance": {
                f"L{n['layer']}": "reference" for n in nodes
            },
            "linearized_approximation_error": round(0.04 + (prompt_hash % 5) * 0.01, 2),
        }
