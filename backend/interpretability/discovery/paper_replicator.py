"""Automated Paper Replicator.

Status: NOT IMPLEMENTED
-----------------------
This module previously returned, for every caller and with no branch::

    return {
        "paper_id": paper_id,
        "title": "Interpretability of IOI Circuit in Transformers (Wang et al., 2022)",
        "reproduction_match_rate": 0.985,
        "replicated_at": _dt.datetime.utcnow().isoformat() + "Z",
        "status": "SuccessfullyReplicated",
    }

No paper was opened, no weights were loaded, and no replication run was
performed. 'paper_id' was echoed back, the title was constant, and 0.985 was
the same number for every id, so every caller received a fake successful
replication record.

replicate_paper() now raises. Callers that already treat a raise as "did not
run" remain fail-closed.
"""

from __future__ import annotations

from typing import Any, Dict

from backend.science.models.adapter_base import LiveUnavailable


class PublishedPaperReplicator:
    """Not implemented. Raises rather than reporting a fabricated replication match rate."""

    def replicate_paper(self, paper_id: str = "paper_ioi_2022") -> Dict[str, Any]:
        """Status: NOT IMPLEMENTED.

        This previously returned 'reproduction_match_rate: 0.985',
        'status: "SuccessfullyReplicated"', and a constant IOI title for any
        'paper_id', with 'replicated_at' generated at call time. That was wrong
        because the values were constant and not derived from running the paper,
        so the result could not be distinguished from a real replication.
        """
        raise LiveUnavailable(
            "PublishedPaperReplicator is not implemented. It previously returned "
            "reproduction_match_rate=0.985, status='SuccessfullyReplicated', and "
            "a constant IOI title for any paper_id, with replicated_at generated "
            "at call time. No paper was opened and no model measurement was made."
        )
