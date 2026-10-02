"""Automated Feature Labeler Engine.

NOT IMPLEMENTED. ``label_feature`` returns the same label, the same description,
and the same ``confidence_score`` of 0.94 for every feature id. No model is
called -- there is no LLM interpretation, and no SAE activation is read, so the
"explanation_evidence" activations (+4.12, +3.85) are literals.

An auto-generated feature label is already a claim about what a feature does.
Claiming 0.94 confidence in a fixed string turns that into a fabricated
finding, so the confidence is removed and the response declares itself.
"""

from __future__ import annotations

from typing import Any, Dict, List


class AutomatedFeatureLabelerEngine:
    """Reference fixture only. No model was called and no activation read."""

    def label_feature(self, feature_id: int, provider: str = "LocalModel") -> Dict[str, Any]:
        return {
            "feature_id": feature_id,
            "provider": provider,
            "proposed_label": "Indirect Object Identifier",
            "proposed_description": ("Reference label only. Fires strongly on "
                                     "name tokens designated as indirect "
                                     "objects in multi-clause sentences."),
            "confidence_score": 0.0,
            "label_measured": False,
            "explanation_evidence": [],
            "status": "unavailable",
            "provenance": "reference",
            "validation_eligible": False,
            "publication_eligible": False,
            "reason": (
                "Automated feature labelling is not implemented. No model was "
                "called and no SAE activation was read; this label and its "
                "evidence are fixed fixtures, so no confidence is claimed."
            ),
        }
