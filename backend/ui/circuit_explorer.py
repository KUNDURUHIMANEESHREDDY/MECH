from typing import Dict, Any, List

class CircuitExplorer:
    """
    Backend logic for the Interactive Circuit Explorer UI.
    Provides a trace from high-level circuits down to individual tokens and evidence.
    """

    @staticmethod
    def get_circuit_trace(circuit_id: str) -> Dict[str, Any]:
        """
        Returns a hierarchical trace of the circuit.
        """
        # Mocking the trace for 'IOI'
        if circuit_id.upper() == "IOI":
            return {
                "circuit": "Indirect Object Identification",
                "components": [
                    {
                        "type": "Attention Head",
                        "id": "L9H9",
                        "role": "Name Mover",
                        "impact_score": 0.94,
                        "downstream": [
                            {
                                "type": "SAE Feature",
                                "id": "SAE_8_4096",
                                "concept": "Recipient Name",
                                "tokens": ["Alice", "Bob", "Charlie"]
                            }
                        ]
                    }
                ],
                "causal_patching": {
                    "baseline": 3.2,
                    "patched": 0.2,
                    "recovery": 0.94
                },
                "evidence": {
                    "paper": "Wang et al., 2022",
                    "experiment_id": "exp_ioi_001",
                    "verification_report": "verified_001.json"
                }
            }
        return {"error": "Circuit not found"}

    @staticmethod
    def run_live_patch(component_id: str, value: float) -> float:
        """
        Simulates a live causal patch in the UI.
        """
        # In real use, this would trigger a HookedTransformer forward pass
        return value * 0.1 # Example attenuation
