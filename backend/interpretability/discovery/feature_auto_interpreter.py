"""Feature Auto-Interpretation Engine.

Turns maximum-activating examples into a semantic description of a feature.

What is measured and what is not
--------------------------------
The text statistics here -- n-gram frequencies, histograms, entropy -- are real
computations over real strings. They are computed from caller-supplied prompts.

Interpreting an SAE feature is a different matter, and this module no longer
pretends to do it. What it used to do:

  * `_extract_sae_feature_activation` was a substring matcher keyed on two
    hardcoded feature indices::

        if feature_idx == 1042 and ("paris" in p or "france" in p or "baguette" in p):
            return 4.5 + (len(prompt) % 3)
        elif feature_idx == 2001 and ("def " in p or "return" in p or ...):
            return 5.2 + (len(prompt) % 2)
        return 0.0

    The activation is a function of string length, so two prompts of the same
    length over the same words get identical activations, and features other
    than 1042 and 2001 are always exactly zero. No SAE was involved.

  * `concept_family` was `"Geography" if "Paris" in name else "Syntax"` -- a
    substring test against the *generated label*. Every cluster whose label did
    not literally contain the word "Paris" was filed under Syntax, including
    clusters with no relationship to syntax.

  * `confidence` was the constant 0.85, applied to an automatic label that no
    check had been performed on.

  * `generate_discovery_report` returned `campaign_id: "REPR-772"`, a fixed
    `timestamp` of `2026-07-28T21:35:00Z`, `avg_polysemanticity: 0.12` and
    `overall_stability: 0.94`. None computed from its input.

It also held a 7-sentence "synthetic dataset for the MVP", which meant every
report described those sentences rather than anything the caller had.

What it does now
----------------
Prompts and feature activations come from the caller. A cluster can only be
described if per-feature activating examples are supplied; without them the
module says so. `concept_family` and `confidence` are never invented: the family
is either supplied or `None`, and an automatic label carries no confidence at
all, because nothing here can check one.

The label is explicitly `auto_suggested` and `verified: False`. A semantic name
derived from the top three n-grams of a few examples is a *suggestion for a human
to accept or reject*, and presenting it with a confidence implies otherwise.
"""

from __future__ import annotations

import collections
import datetime
import math
from typing import Any, Dict, List, Optional, Sequence

from backend.science.models.adapter_base import ModelAdapter


class FeatureAutoInterpreter:
    """Summarises a feature's activating examples into a reviewable description.

    Does not interpret features on its own. Given measured activations and the
    examples that produced them, it describes them; whether that description is
    *right* is a question for a human, and the output says so.
    """

    def __init__(
        self,
        adapter: Optional[ModelAdapter] = None,
        prompts: Optional[Sequence[str]] = None,
    ) -> None:
        self.adapter = adapter
        #: Text to analyse. Was a hardcoded 7-sentence "synthetic dataset for
        #: the MVP", so every report described those sentences instead of
        #: anything the caller had. Absent means no prompts were supplied, and
        #: any method needing them refuses.
        self._prompts: List[str] = [p for p in (prompts or []) if isinstance(p, str)]
        self._prompts_supplied = prompts is not None

    @property
    def prompts_available(self) -> bool:
        return bool(self._prompts)

    def _extract_sae_feature_activation(self, prompt: str, feature_idx: int) -> Optional[float]:
        """No longer invented.

        There is no SAE in this repository, so a feature activation cannot be
        obtained. Returns None, which callers must treat as "not measured" rather
        than as 0.0 -- the previous version returned a length-dependent formula
        for two hardcoded indices and exactly 0.0 for everything else, so a
        feature the simulator did not know about looked inert rather than absent.
        """
        return None

    def interpret_cluster(
        self,
        cluster_id: str,
        feature_ids: Sequence[int],
        activations: Optional[Dict[int, Sequence[Dict[str, Any]]]] = None,
        concept_family: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Describe a cluster from its measured activating examples.

        `activations` maps a feature id to that feature's activating examples,
        each ``{"prompt": str, "activation": float}``. Only examples whose
        activation is positive and whose provenance is a measurement contribute
        to the description.

        Without it there is nothing to describe: the function returns a record
        saying `measured: False` rather than a label built from nothing.
        """
        if not activations:
            return self._undescribed(
                cluster_id,
                reason=(
                    f"No measured activations were supplied for cluster "
                    f"{cluster_id}. A semantic description requires the examples "
                    f"that actually activate each feature; without them this "
                    f"previously produced a label from a hardcoded 7-sentence "
                    f"dataset, plus a constant confidence of 0.85 and a "
                    f"concept family chosen by testing whether the generated name "
                    f"contained the substring 'Paris'."
                ),
                feature_ids=list(feature_ids),
            )

        reports = [self.generate_feature_report(fid, activations.get(fid, []))
                   for fid in feature_ids
                   if activations.get(fid)]

        if not reports:
            return self._undescribed(
                cluster_id,
                reason=(
                    f"Activations were supplied for cluster {cluster_id}, but none "
                    f"of features {list(feature_ids)} had any recorded example."
                ),
                feature_ids=list(feature_ids),
            )

        # Aggregate n-grams across the features that actually had examples.
        all_words: collections.Counter = collections.Counter()
        for report in reports:
            for word, count in report["n_gram_frequencies"].items():
                all_words[word] += count

        top_words = [w for w, _ in all_words.most_common(3)]
        name = " / ".join(top_words) if top_words else None

        # Representative examples come from whichever feature had the largest
        # measured activation, not from `reports[0]` -- which was the first
        # feature id supplied and had no necessary relationship to the best one.
        best = max(
            (r for r in reports if r["max_activation"] is not None),
            key=lambda r: r["max_activation"],
            default=None,
        )
        examples = (best or reports[0])["top_positive_examples"]

        n_features_described = sum(
            1 for r in reports if r["max_activation"] is not None)

        return {
            "cluster_id": cluster_id,
            "measured": True,
            "semantic_name": name,
            "name_source": ("top n-grams across the features' activating examples"
                            if name else None),
            # Never inferred. The previous rule assigned "Geography" or "Syntax"
            # by testing whether the generated label contained "Paris".
            "concept_family": concept_family,
            "concept_family_source": "supplied by caller" if concept_family else None,
            # An automatic label has not been checked by anything here, so it
            # carries no confidence. A number here would be the 0.85 constant
            # wearing a different name.
            "confidence": None,
            "confidence_reason": (
                "An n-gram summary is a suggestion for human review. Nothing in "
                "this module can verify that the label describes the feature, so "
                "no confidence is reported."
            ),
            "auto_suggested": True,
            "verified": False,
            "n_features_described": n_features_described,
            "n_features_in_cluster": len(list(feature_ids)),
            "representative_examples": examples,
            "representative_source": (
                f"highest measured activation among {n_features_described} "
                f"described feature(s)"),
        }

    def _undescribed(
        self,
        cluster_id: str,
        reason: str,
        feature_ids: List[int],
    ) -> Dict[str, Any]:
        """The fail-closed shape. Same keys as a described cluster."""
        return {
            "cluster_id": cluster_id,
            "measured": False,
            "semantic_name": None,
            "name_source": None,
            "concept_family": None,
            "concept_family_source": None,
            "confidence": None,
            "confidence_reason": reason,
            "auto_suggested": False,
            "verified": False,
            "n_features_described": 0,
            "n_features_in_cluster": len(feature_ids),
            "feature_ids": feature_ids,
            "representative_examples": [],
            "representative_source": None,
            "reason": reason,
        }

    def generate_feature_report(
        self,
        feature_idx: int,
        examples: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """n-gram and histogram statistics for one feature's activating examples.

        `examples` are ``{"prompt": str, "activation": float}`` records. Only
        positive activations count as activating. With no examples every statistic
        is None or 0 with `measured: False`, rather than describing a built-in
        sentence list.
        """
        positive = [
            e for e in (examples or [])
            if isinstance(e, dict)
            and isinstance(e.get("prompt"), str)
            and isinstance(e.get("activation"), (int, float))
            and e["activation"] > 0
        ]

        if not positive:
            return {
                "feature_idx": feature_idx,
                "measured": False,
                "n_positive_examples": 0,
                "max_activation": None,
                "mean_activation": None,
                "total_activation": None,
                "ngram_entropy": None,
                "length_histogram": {},
                "n_gram_frequencies": {},
                "top_positive_examples": [],
                "reason": (
                    f"Feature {feature_idx} has no recorded activating example, "
                    f"so there is nothing to summarise."
                ),
            }

        prompts = [e["prompt"] for e in positive]
        activations = [float(e["activation"]) for e in positive]

        counter: collections.Counter = collections.Counter()
        for text in prompts:
            counter.update(text.lower().split())

        total = sum(counter.values())
        entropy = -sum(
            (c / total) * math.log2(c / total) for c in counter.values()
        ) if total else None

        histogram: Dict[str, int] = {}
        for text in prompts:
            key = f"{len(text.split()) // 10 * 10}-{len(text.split()) // 10 * 10 + 9}"
            histogram[key] = histogram.get(key, 0) + 1

        top = sorted(positive, key=lambda e: e["activation"], reverse=True)[:5]

        return {
            "feature_idx": feature_idx,
            "measured": True,
            "n_positive_examples": len(positive),
            "max_activation": round(max(activations), 6),
            "mean_activation": round(sum(activations) / len(activations), 6),
            "total_activation": round(sum(activations), 6),
            "ngram_entropy": round(entropy, 4) if entropy is not None else None,
            "length_histogram": histogram,
            "n_gram_frequencies": dict(counter.most_common(10)),
            "top_positive_examples": [
                {"prompt": e["prompt"], "activation": round(float(e["activation"]), 6)}
                for e in top
            ],
            "reason": None,
        }

    def generate_discovery_report(
        self,
        campaign_results: Dict[str, Any],
        clusters: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Assemble a campaign report from supplied cluster descriptions.

        Statistics are computed from `clusters`. With none supplied every
        statistic is None -- the previous version returned
        `avg_polysemanticity: 0.12` and `overall_stability: 0.94` regardless of
        input, under a fixed `campaign_id` and a fixed timestamp.
        """
        described = [c for c in (clusters or []) if c.get("measured")]

        polysemanticity = None
        stability = None
        if described:
            widths = [c.get("n_features_described", 0) for c in described]
            polysemanticity = round(sum(widths) / len(widths), 4)

            ratios = [
                c["n_features_described"] / c["n_features_in_cluster"]
                for c in described
                if c.get("n_features_in_cluster")
            ]
            stability = round(sum(ratios) / len(ratios), 4) if ratios else None

        return {
            "campaign_id": (campaign_results.get("campaign_id")
                            if isinstance(campaign_results, dict) else None),
            "timestamp": datetime.datetime.now(
                datetime.timezone.utc).isoformat(),
            "measured": bool(described),
            "discovered_concepts": list(clusters or []),
            "n_clusters_described": len(described),
            "n_clusters_requested": len(clusters or []),
            "global_statistics": {
                "avg_features_described_per_cluster": polysemanticity,
                "overall_coverage": stability,
                # Renamed from "avg_polysemanticity" and "overall_stability",
                # which named two quantities this module cannot compute.
                # Polysemanticity needs the features that co-fire on the same
                # tokens; stability needs the feature re-measured on held-out
                # text. Neither input exists here.
                "polysemanticity": None,
                "stability": None,
                "polysemanticity_reason": (
                    "Not computed: requires per-token co-firing across features, "
                    "which this module does not receive."
                ),
                "stability_reason": (
                    "Not computed: requires re-measuring each feature on held-out "
                    "text, which this module does not receive."
                ),
            },
            "reason": (None if described else
                       "No cluster descriptions with measured activations were "
                       "supplied, so no statistics can be reported."),
        }