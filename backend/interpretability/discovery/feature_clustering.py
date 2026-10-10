"""SAE Feature Clustering Engine.

Status: NOT IMPLEMENTED
-----------------------
This module previously returned, for any method and with no features::

    return {
        "method": method,
        "num_clusters": num_clusters,
        "clusters": [
            {"cluster_id": "cluster_ioi", "label": "Indirect Object Identification", ...},
            {"cluster_id": "cluster_geo", "label": "Geographic Capital Cities", ...},
        ],
    }

No SAE features were loaded, no similarities were computed, and no clustering
method was run. The same 'cluster_ioi' and 'cluster_geo' groups were returned
for every call, so the output was a fixed fixture, not a clustering result.

cluster_features() now raises. Callers that need clustering should use an
implemented feature extractor rather than this stub.
"""

from __future__ import annotations

from typing import Any, Dict, List

from backend.science.models.adapter_base import LiveUnavailable


class FeatureClusteringEngine:
    """Not implemented. Raises rather than reporting fabricated feature clusters."""

    def cluster_features(self, method: str = "Cosine", num_clusters: int = 3) -> Dict[str, Any]:
        """Status: NOT IMPLEMENTED.

        This previously returned fixed clusters such as 'cluster_ioi' and
        'cluster_geo', with 'features: [1402, 1403, 1510]' and
        '[789, 790, 812]' and centroid similarities 0.88/0.91, for any
        'method' or 'num_clusters'. That was wrong because no clustering was
        computed from actual SAE feature activations.
        """
        raise LiveUnavailable(
            "FeatureClusteringEngine.cluster_features is not implemented. It "
            "previously returned fixed cluster_ioi and cluster_geo groups with "
            "features [1402, 1403, 1510] and [789, 790, 812] and centroid "
            "similarities 0.88/0.91 for any method or num_clusters. No SAE "
            "features were clustered."
        )
