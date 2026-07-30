"""Research Memory Engine."""

from __future__ import annotations

import datetime as _dt
from typing import Any, Dict, List


class ResearchMemoryEngine:
    """Stores experiential research memory: successes, failed hypotheses, and prompt efficacy."""

    def __init__(self) -> None:
        self.memories: List[Dict[str, Any]] = [
            {
                "memory_id": "mem_1",
                "category": "successful_intervention",
                "description": "Zeroing L8_N402 cleanly shifted logit preference from Mary to John.",
                "utility_score": 0.95,
            },
            {
                "memory_id": "mem_2",
                "category": "failed_hypothesis",
                "description": "Attending to Layer 0 embedding tokens did not alter geographic capital output.",
                "utility_score": 0.12,
            },
        ]

    def record_memory(self, category: str, description: str, utility_score: float = 0.5) -> Dict[str, Any]:
        mem = {
            "memory_id": f"mem_{len(self.memories) + 1}",
            "category": category,
            "description": description,
            "utility_score": utility_score,
            "recorded_at": _dt.datetime.utcnow().isoformat() + "Z",
        }
        self.memories.append(mem)
        return mem

    def list_memories(self, category: str | None = None) -> List[Dict[str, Any]]:
        if not category:
            return list(self.memories)
        return [m for m in self.memories if m["category"] == category]
