"""Feature Genealogy Engine.

NOT IMPLEMENTED. The lineage returned here is a fixed fixture: the parent,
child, and depth values do not depend on ``feature_id`` or on any activation.
It previously reported a Lineage Overlap of 0.88 and confidence 0.95 for every
feature, which reads like a computed DAG.
"""

from __future__ import annotations

from typing import Any, Dict
from .discovery_result import DiscoveryResultDTO


class FeatureGenealogyEngine:
    """Reference fixture only. No lineage was inferred from activations."""

    def get_genealogy(self, feature_id: int = 1402) -> Dict[str, Any]:
        genealogy = {
            "feature_id": feature_id,
            "parents": [102, 304],
            "children": [2804, 3102],
            "merged_from": [102, 304],
            "lineage_depth": 3,
        }
        res = DiscoveryResultDTO(
            discovery_id=f"gen_{feature_id}",
            discovery_type="FeatureGenealogy",
            title=f"Lineage DAG for Feature #{feature_id}",
            evidence=[],
            confidence=0.0,
            related_features=[102, 304, feature_id, 2804, 3102],
        ).to_dict()
        res["genealogy"] = genealogy
        res["genealogy_field_provenance"] = {
            key: "reference" for key in genealogy
        }
        res["status"] = "unavailable"
        res["provenance"] = "reference"
        res["genealogy_measured"] = False
        res["validation_eligible"] = False
        res["publication_eligible"] = False
        res["reason"] = (
            "Feature genealogy is not implemented. These lineage values are "
            "fixed fixtures and are identical for every feature_id; no parent "
            "or child relationship was inferred from activations."
        )
        return res
