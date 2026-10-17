"""Automated Mechanistic Report Engine.

NOT IMPLEMENTED. ``generate_mechanistic_report`` returns a fixed four-step
causal narrative for any ``prompt`` and ``target_token``. It runs no forward
pass, reads no activation, and consults no circuit.

The narrative carried specific numbers -- "Layer 8 Neuron #402 fires
(activation +4.12)", "SAE Feature #1402", "Induction Head L8_H9", "the logit
for ' Paris' increases from 12.1 to 14.8" -- plus a ``causal_confidence`` of
0.95. A mechanistic explanation is the strongest claim this platform makes, so
invented numbers inside one are the worst case rather than the best: prose does
not look like a stub, and a reader has no way to tell it from a real analysis.
"""

from __future__ import annotations

import datetime as _dt
from typing import Any, Dict


class MechanisticReportEngine:
    """Reference fixture only. No model was run and no circuit was read."""

    def generate_mechanistic_report(
        self, prompt: str, target_token: str = " Paris"
    ) -> Dict[str, Any]:
        """Refuse: there is no report to generate.

        Returns the same keys a real report would carry, so consumers do not
        need to special-case it, but ``explanation_text`` is None and
        ``circuit_components`` is empty. ``causal_confidence`` is None rather
        than a number, because no causal claim was tested.
        """
        return {
            "prompt": prompt,
            "target_token": target_token,
            "explanation_text": None,
            "circuit_components": [],
            "causal_confidence": None,
            "status": "unavailable",
            "provenance": "unavailable",
            "measured": False,
            "validation_eligible": False,
            "publication_eligible": False,
            "timestamp": _dt.datetime.utcnow().isoformat() + "Z",
            "reason": (
                "Automated mechanistic report generation is not implemented. "
                "No forward pass was run, no activation was read, and no "
                "circuit was consulted, so there is no explanation to report. "
                "The narrative this method previously returned was a fixed "
                "fixture: its activation values, feature id, head label, and "
                "logit delta were all invented."
            ),
        }
