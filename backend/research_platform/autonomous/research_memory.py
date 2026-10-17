"""Research Memory Engine."""

from __future__ import annotations

import datetime as _dt
from typing import Any, Dict, List


class ResearchMemoryEngine:
    """Stores experiential research memory: successes, failed hypotheses, and prompt efficacy.

    Starts empty. This constructor used to seed two memories that read as
    recorded outcomes -- "Zeroing L8_N402 cleanly shifted logit preference from
    Mary to John" with `utility_score` 0.95, and a failed hypothesis at 0.12 --
    with no label marking them as illustration. A memory of a shift that was
    never measured is exactly the kind of record an agent later reasons from.
    """

    def __init__(self) -> None:
        self.memories: List[Dict[str, Any]] = []

    def record_memory(
        self,
        category: str,
        description: str,
        utility_score: float | None = None,
    ) -> Dict[str, Any]:
        """Record a memory. `utility_score` is None when the caller has none.

        It defaulted to 0.5, which put a quality number on a record whose
        caller had made no quality claim. `AutonomousResearchEngine` calls this
        with no score at all, and its own comment records that the score it
        used to pass (0.92) was invented. An absent measurement is None, which
        is distinguishable from a measured 0.5.
        """
        mem = {
            "memory_id": f"mem_{len(self.memories) + 1}",
            "category": category,
            "description": description,
            "utility_score": utility_score,
            "measured": utility_score is not None,
            "reason": (
                "" if utility_score is not None
                else "No utility score was supplied, so none is claimed."
            ),
            "recorded_at": _dt.datetime.utcnow().isoformat() + "Z",
        }
        self.memories.append(mem)
        return mem

    def list_memories(self, category: str | None = None) -> List[Dict[str, Any]]:
        if not category:
            return list(self.memories)
        return [m for m in self.memories if m["category"] == category]
