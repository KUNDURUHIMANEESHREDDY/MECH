"""Automated Peer Reviewer.

NOT IMPLEMENTED. ``review_manuscript`` returned a fixed decision of "Accept"
with a score of 9.2 and two canned strengths for any title. No manuscript was
read and no review criterion was applied.

An automated "Accept" is the most dangerous fabricated value in the validation
layer: downstream gates key off decisions, so an invented verdict does not
merely mislead a reader, it can carry a discovery past a check that exists to
stop it. The verdict is removed and the response declares itself instead.

``is_human_peer_review`` is False so a caller cannot mistake this for the human
review slot that the validation gates treat as authoritative.
"""

from __future__ import annotations

from typing import Any, Dict


class AutomatedPeerReviewer:
    """Reference fixture only. No manuscript was read and no review was run."""

    def review_manuscript(self, title: str) -> Dict[str, Any]:
        """Refuse: there is no review to report.

        ``decision`` and ``score`` are None rather than a verdict and a number.
        A caller testing ``if review["decision"]`` now gets a falsy value, which
        is the correct answer for a review that did not happen.
        """
        return {
            "title": title,
            "decision": None,
            "score": None,
            "strengths": [],
            "suggestions": [],
            "status": "unavailable",
            "provenance": "unavailable",
            "is_human_peer_review": False,
            "measured": False,
            "validation_eligible": False,
            "publication_eligible": False,
            "reason": (
                "Automated peer review is not implemented. No manuscript was "
                "read and no criterion was applied, so no decision and no score "
                "are reported. The verdict this method previously returned was "
                "a fixed 'Accept' with a score of 9.2 and canned strengths."
            ),
        }
