"""Persistent Knowledge Base Engine."""

from __future__ import annotations

import datetime as _dt
from typing import Any, Dict, List


class KnowledgeBaseEngine:
    """Stores objective facts, discovered circuits, and mechanistic properties."""

    def __init__(self) -> None:
        self.facts: List[Dict[str, Any]] = [
            {
                "id": "fact_1",
                "entity": "Neuron L8_N402",
                "property": "Function",
                "value": "Indirect Object Identifier",
                "confidence": 0.94,
            },
            {
                "id": "fact_2",
                "entity": "SAE Feature #1402",
                "property": "Firing Pattern",
                "value": "Fires on name tokens in double-clause prompts",
                "confidence": 0.96,
            },
        ]

    def store_fact(self, entity: str, prop: str, value: Any, confidence: float = 0.9) -> Dict[str, Any]:
        fact = {
            "id": f"fact_{len(self.facts) + 1}",
            "entity": entity,
            "property": prop,
            "value": value,
            "confidence": confidence,
            "stored_at": _dt.datetime.utcnow().isoformat() + "Z",
        }
        self.facts.append(fact)
        return fact

    def query_knowledge(self, query: str = "") -> List[Dict[str, Any]]:
        if not query:
            return list(self.facts)
        q_lower = query.lower()
        return [
            f for f in self.facts
            if q_lower in f["entity"].lower() or q_lower in str(f["value"]).lower()
        ]
