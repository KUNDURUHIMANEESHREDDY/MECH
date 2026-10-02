"""Automatic Circuit Auto-Naming Engine.

NOT IMPLEMENTED. ``name_circuit`` returns the same name, description, and
confidence for every ``circuit_id`` -- it does not read the circuit, does not
look at activations, and does not call a model.

It previously reported ``confidence: 0.96`` and a specific mechanistic claim
("Responds strongly to person names ... using Head 6.4 and Feature #1402") for
whatever circuit it was handed. A generated name that asserts a mechanism is
especially dangerous to misread, because it looks like a summarisation of an
analysis that never ran.
"""

from __future__ import annotations

from typing import Any, Dict


class AutoCircuitNamerEngine:
    """Reference fixture only. No circuit was inspected and no model was run."""

    def name_circuit(self, circuit_id: str = "circuit_31") -> Dict[str, Any]:
        return {
            "circuit_id": circuit_id,
            "name": "Name Recognition Circuit",
            "description": ("Reference label only. Responds strongly to person "
                            "names in double-clause prompts using Head 6.4 and "
                            "Feature #1402."),
            "confidence": 0.0,
            "status": "unavailable",
            "provenance": "reference",
            "naming_measured": False,
            "validation_eligible": False,
            "publication_eligible": False,
            "reason": (
                "Automatic circuit naming is not implemented. This name and "
                "description are fixed fixtures returned for every "
                "circuit_id; no circuit was inspected, so the stated "
                "mechanism is an unverified label rather than a finding."
            ),
        }
