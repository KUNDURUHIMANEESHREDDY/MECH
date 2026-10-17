from typing import Dict, Any, List

class RepresentationAtlas:
    """
    Backend logic for the Representation Atlas UI.
    Organizes mechanistic interpretability findings into a semantic hierarchy.
    """
    
    # An illustrative taxonomy, not a set of findings.
    #
    # This previously attached specific neuron ids ("L8N402" to Paris, "L8N512"
    # to London, "L9N12" to Tokyo), SAE feature ids to Countries, a citation
    # ("Bricken et al.") to Paris, and a named circuit with two heads to
    # Induction. None of those associations was computed, and a neuron id or a
    # citation reads as a finding. The concept names are kept because callers
    # navigate by them; the claims are removed.
    HIERARCHY = {
        "Knowledge": {
            "Geography": {
                "Countries": {"concept_id": "concept_001", "features": []},
                "Cities": {
                    "Paris": {"neuron": None, "paper": None,
                              "association_measured": False},
                    "London": {"neuron": None,
                               "association_measured": False},
                    "Tokyo": {"neuron": None,
                              "association_measured": False}
                }
            },
            "Reasoning": {
                "Induction": {"circuit": None, "heads": [],
                              "association_measured": False}
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
            
        # No experiment database is queried here, so there are no associated
        # experiments and nothing is verified. This previously returned two
        # fixed experiment ids and asserted "VERIFIED" for any node reachable in
        # the hierarchy, which turned an illustrative taxonomy into a
        # verification claim.
        return {
            "node": path[-1],
            "data": current,
            "associated_experiments": [],
            "verification_status": "UNVERIFIED",
            "status": "unavailable",
            "provenance": "reference",
            "measured": False,
            "validation_eligible": False,
            "publication_eligible": False,
            "reason": (
                "This hierarchy is an illustrative taxonomy, not a finding. No "
                "experiment database was queried, so no experiment is "
                "associated with this node and its verification status is "
                "unknown."
            ),
        }

    @classmethod
    def get_full_hierarchy(cls) -> Dict[str, Any]:
        return cls.HIERARCHY
