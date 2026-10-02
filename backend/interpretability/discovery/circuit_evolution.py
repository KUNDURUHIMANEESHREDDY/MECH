"""Circuit Evolution Engine.

NOT IMPLEMENTED. The "evolution" returned here is a fixed three-step fixture
that does not depend on `circuit_id`, the prompt, or any measurement.

It previously reported ``confidence=0.92`` and a branching factor of 1.5 for
every circuit, which reads exactly like a measured growth curve. It now
declares itself and refuses to carry a confidence it did not compute.
"""

from __future__ import annotations

from typing import Any, Dict, List
from .discovery_result import DiscoveryResultDTO


class CircuitEvolutionEngine:
    """Reference fixture only. Not a measurement."""

    def track_evolution(self, circuit_id: str = "c_ioi") -> Dict[str, Any]:
        res = DiscoveryResultDTO(
            discovery_id=f"evo_{circuit_id}",
            discovery_type="CircuitEvolution",
            title=f"Evolution Graph for {circuit_id}",
            evidence=[],
            confidence=0.0,
        ).to_dict()
        res["evolution_steps"] = [
            {"step": 1, "context": "Single Name", "active_nodes_count": 4},
            {"step": 2, "context": "Double Name IOI", "active_nodes_count": 12},
            {"step": 3, "context": "Distractor Name Clause",
             "active_nodes_count": 18},
        ]
        res["status"] = "unavailable"
        res["provenance"] = "seeded"
        res["evolution_steps_synthetic"] = True
        res["validation_eligible"] = False
        res["publication_eligible"] = False
        res["reason"] = (
            "Circuit evolution is not implemented. These steps are fixed "
            "fixtures and are identical for every circuit_id; no trajectory "
            "was measured and no confidence was computed."
        )
        return res
