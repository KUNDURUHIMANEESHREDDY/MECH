from typing import Dict, Any, List

class RepresentationAtlas:
    """
    Backend logic for the Representation Atlas UI.
    Organizes mechanistic interpretability findings into a semantic hierarchy.
    """
    
    HIERARCHY = {
        "Knowledge": {
            "Geography": {
                "Countries": {"concept_id": "concept_001", "features": ["SAE_318", "SAE_10291"]},
                "Cities": {
                    "Paris": {"neuron": "L8N402", "paper": "Bricken et al."},
                    "London": {"neuron": "L8N512"},
                    "Tokyo": {"neuron": "L9N12"}
                }
            },
            "Reasoning": {
                "Induction": {"circuit": "InductionHeads", "heads": ["L5H1", "L5H5"]}
            }
        }
    }

    @classmethod
    def get_node_details(cls, path: List[str]) -> Dict[str, Any]:
        """
        Retrieves detailed research artifacts for a specific node in the hierarchy.
        Example path: ["Knowledge", "Geography", "Cities", "Paris"]
        """
        current = cls.HIERARCHY
        try:
            for p in path:
                current = current[p]
        except KeyError:
            return {"error": "Node not found"}
            
        # In a real system, this would query the experiment database
        return {
            "node": path[-1],
            "data": current,
            "associated_experiments": ["exp_ioi_repro", "exp_sae_alignment"],
            "verification_status": "VERIFIED"
        }

    @classmethod
    def get_full_hierarchy(cls) -> Dict[str, Any]:
        return cls.HIERARCHY
