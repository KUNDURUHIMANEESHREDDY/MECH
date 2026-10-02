"""Polysemanticity Detector Engine.

NOT IMPLEMENTED. `detect_polysemanticity` returns fixed fixtures: no neuron or
feature is ever inspected, and the "evidence" activations below are literals,
not measurements.

The values are retained for catalog/UI inspection, but every response declares
itself as synthetic and ineligible for validation or publication so it cannot
be mistaken for a measured finding. A real implementation must compute the
score from actual activations across concept-diverse prompts.
"""

from __future__ import annotations

from typing import Any, Dict, List


class PolysemanticityDetectorEngine:
    """Reference fixtures only. Not a measurement."""

    def detect_polysemanticity(self, target_type: str = "neuron", index: int = 402) -> Dict[str, Any]:
        return {
            "status": "unavailable",
            "provenance": "seeded",
            "validation_eligible": False,
            "publication_eligible": False,
            "reason": ("Polysemanticity detection is not implemented; these "
                       "are fixed fixtures and no activations were measured."),
            "target_type": target_type,
            "index": index,
            "polysemanticity_score": 0.28,  # Fixed fixture, not measured.
            "classification": "monosemantic",
            "primary_concept": "Indirect Object Identification",
            "secondary_concept": "Comma separator",
            "evidence_synthetic": True,
            "evidence": [
                {"context": "John gave a book to Mary", "concept": "IOI",
                 "activation": 4.12, "measured": False},
                {"context": "First, second, third", "concept": "List item",
                 "activation": 0.45, "measured": False},
            ],
        }
