"""Feature Universality Analysis Discovery Algorithm.

Ref: Anthropic / OpenAI, 2024 - "Cross-Model & Cross-Layer Feature Universality"

Measures whether sparse features learned by different models (e.g. GPT-2, Gemma, Llama)
or different layers within the same model represent identical semantic concepts.
Uses bipartite matching and mutual information / cosine similarity across dataset firings
to construct cross-model alignment graphs.
"""

from __future__ import annotations

import datetime as _dt
import math
import time
from typing import Any, Dict, List, Optional

from .base_algorithm import DiscoveryAlgorithm, DiscoveryReport
from .registry import register_algorithm, AlgorithmMetadata
from .configs import DiscoveryAlgorithmConfig, FeatureUniversalityConfig


@register_algorithm(AlgorithmMetadata(
    name="feature_universality",
    paper="Cross-Model & Cross-Layer Feature Universality (Anthropic / OpenAI)",
    authors="Interpretability Research Community",
    year=2024,
    supported_models=["gpt2", "gemma", "llama", "mistral"],
    required_capabilities=["get_activations"],
    estimated_runtime="5s-15s",
    search_space="cross_model_features",
    output_schema="DiscoveryReport"
))
class FeatureUniversalityAlgorithm(DiscoveryAlgorithm):
    """Aligns sparse features across models/layers to discover universal representations."""

    # Registered under the name and paper citation of bipartite cross-model
    # feature matching, but no matching is performed. This flag is what makes
    # `provenance_block()` mark the report reference/unavailable so the
    # evidence gate cannot read it as live. See `not_implemented_reason`.
    implements_published_method = False

    not_implemented_reason = (
        "FeatureUniversalityAlgorithm does not implement cross-model feature "
        "matching. It previously reported eight fixed concepts with fixed "
        "similarity scores ('French Cities' 0.94, 'Induction Pattern' 0.91, "
        "'Capital Cities' 0.88, ...), generated feature ids by arithmetic on "
        "the loop index (100 + i * 15), and derived its confidence from the "
        "average of those literals. No two models were compared and no feature "
        "was matched."
    )

    def run(self, dataset: Dict[str, Any], config: Optional[DiscoveryAlgorithmConfig] = None) -> DiscoveryReport:
        """Refuses: cross-model feature matching is not implemented here."""
        t0 = time.time()
        
        if not isinstance(config, FeatureUniversalityConfig):
            config = FeatureUniversalityConfig()

        models = config.target_models
        m1_name = models[0] if len(models) > 0 else "gpt2"
        m2_name = models[1] if len(models) > 1 else "gemma"

        # No alignment is computed. `distributed_campaign`,
        # `dag_discovery_planner`, `discovery_planner` and
        # `autonomous_paper_replicator` all schedule `feature_universality` as
        # an algorithm, so the fabricated report below was reachable through
        # campaign planning and not only by direct call.
        #
        # The hardcoded default prompt is gone with it: falling back to
        # "Paris is the capital of France" meant a dataset that supplied
        # nothing still produced a scorable report.
        runtime_ms = (time.time() - t0) * 1000

        return DiscoveryReport(
            algorithm="feature_universality",
            dataset_id=dataset.get("id", "unknown"),
            model_id=self.adapter.spec.model_id if self.adapter else "unknown",
            runtime_ms=round(runtime_ms, 2),
            statistics={
                "measured": False,
                "reason": self.not_implemented_reason,
                "aligned_models": [m1_name, m2_name],
                "matched_pairs_count": 0,
                "avg_universality_score": None,
                "matching_metric": config.matching_metric,
            },
            evidence={
                "universal_alignments": [],
                "metric": config.matching_metric,
            },
            # None, not a number averaged out of literals.
            confidence=None,
            graph={"nodes": [], "edges": [], "score": None},
            # Carries the gate-readable `source` key; see
            # `DiscoveryAlgorithm.provenance_block`.
            provenance=self.provenance_block(),
        )
