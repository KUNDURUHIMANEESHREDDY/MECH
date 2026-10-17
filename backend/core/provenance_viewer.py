"""Experiment Provenance Viewer Engine.

NOT IMPLEMENTED. ``get_lineage`` returned a fixed six-stage lineage for any
``discovery_id`` -- dataset "IOI Prompts v1", model "GPT-2 Small", prompt
"When John and Mary...", runtime "Ray Execution Cluster", a discovery named
"L8_N402 IOI Circuit", and paper "ArXiv Package #9402" -- and asserted
``verified_lineage: True`` over it.

Provenance is the control that lets a reader tell a measured result from a
remembered one. A fabricated lineage is worse than a missing one: it has the
shape of a real trace and it claims to have been verified, so it defeats the
very check it appears to satisfy.
"""

from __future__ import annotations

from typing import Any, Dict


class ProvenanceViewerEngine:
    """Reference fixture only. No lineage was read for any discovery."""

    def get_lineage(self, discovery_id: str) -> Dict[str, Any]:
        """Refuse: no lineage is recorded for this discovery.

        ``verified_lineage`` is False because nothing was verified. Returning
        the empty lineage *with* the flag is what lets a caller tell "no lineage
        recorded" apart from "lineage checked and complete".
        """
        return {
            "discovery_id": discovery_id,
            "lineage": [],
            "verified_lineage": False,
            "status": "unavailable",
            "provenance": "unavailable",
            "measured": False,
            "validation_eligible": False,
            "publication_eligible": False,
            "reason": (
                "Lineage lookup is not implemented. No dataset, model, prompt, "
                "runtime, or publication record was read for this discovery. "
                "The lineage this method previously returned was a fixed "
                "fixture, and it reported itself as verified."
            ),
        }
