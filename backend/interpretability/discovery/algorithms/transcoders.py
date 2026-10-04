"""Transcoder Dictionary Learning (Anthropic, 2024).

A transcoder decomposes an MLP layer's input-to-output transformation into a
sparse set of learned dictionary features. Measuring one requires a *trained*
encoder/decoder pair: the encoder maps activations to feature activations, the
decoder reconstructs the output from those features, and the fraction of variance
explained is how well that reconstruction works.

What this module used to do
---------------------------
It simulated all three.

  * Activations were fabricated on failure: `src_val = ... else 0.5` and
    `tgt_val = ... else 0.45`.
  * "Feature activations" were an arithmetic ramp over that fabricated value --
    ``feat_act = max(0.0, src_val * (0.8 + 0.1 * (f_idx % 3)) - 0.25)`` -- under
    a comment reading "Simulating L1 sparse encoder response". No encoder exists.
  * `encoder_weight = 0.5 + 0.05 * f_idx` and `decoder_weight = 0.4 + 0.04 *
    f_idx` were arithmetic ramps. These are the dictionary. Nothing was learned
    and nothing was loaded; a decoder weight that increases by 0.04 per feature
    index is not a property of any model.
  * Every edge carried `confidence: 0.95`, a constant applied to both the
    encode and decode direction of every feature.
  * ``fve = min(0.99, max(0.80, 0.92 + 0.05 * (len(active_features) / 10.0)))``
    was clamped into [0.80, 0.99], so it could never report a poor
    reconstruction. The only way to lower it was to have fewer than zero active
    features. This is the same never-failing formula already removed from ACDC.

So the algorithm reported a decomposition of an MLP layer, features with decoder
weights, and a variance-explained figure above 0.80, for any input at all.

What it does now
----------------
It refuses, and says what it would need. There is no trained dictionary anywhere
in this repository -- the only `decoder_weight` in the tree was the ramp above --
so there is nothing to measure and no partial result worth reporting.

`TranscoderConfig` and the algorithm registration are kept, so a research goal
naming transcoders still plans that stage and the report explains why it could not
run. That is the same arrangement as the unimplemented copy-task, arithmetic and
factual-recall pipelines.

To implement it
---------------
Supply `decoder` and `encoder` callables on the config, or a `dictionary_path`
pointing at trained weights. Then:

  * capture the true MLP input and output for the target layer
    (`capture_head_outputs` is the head analogue; an MLP equivalent is needed);
  * compute feature activations with the real encoder;
  * reconstruct with the real decoder;
  * report FVE as ``1 - Var(residual) / Var(target)`` -- unclamped, so a bad
    reconstruction can report a bad number.
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional

from .base_algorithm import DiscoveryAlgorithm, DiscoveryReport
from .registry import register_algorithm, AlgorithmMetadata
from .configs import DiscoveryAlgorithmConfig, TranscoderConfig


@register_algorithm(AlgorithmMetadata(
    name="transcoders",
    paper="Transcoders: Replacing MLP Layers with Dictionary Features (Anthropic)",
    authors="Anthropic Interpretability Team",
    year=2024,
    supported_models=["gpt2", "gemma", "llama", "mistral"],
    # Was ["get_activations", "get_logits"]. Those are sufficient for the
    # simulation that used to live here, and insufficient for the measurement
    # this algorithm is supposed to perform: a transcoder needs the MLP layer's
    # input and output vectors and a trained dictionary.
    required_capabilities=["capture_mlp_io", "transcoder_dictionary"],
    estimated_runtime="5s-20s",
    search_space="mlp_transcoders",
    output_schema="DiscoveryReport",
))
class TranscoderAlgorithm(DiscoveryAlgorithm):
    """Decomposes MLP transformations into sparse, interpretable features.

    Not implemented. Refuses rather than simulating -- see the module docstring
    for the four fabricated quantities this previously reported.
    """

    def run(
        self,
        dataset: Dict[str, Any],
        config: Optional[DiscoveryAlgorithmConfig] = None,
    ) -> DiscoveryReport:
        t0 = time.time()
        if not isinstance(config, TranscoderConfig):
            config = TranscoderConfig()

        num_layers = self.adapter.spec.num_layers if self.adapter else 0
        src_layer = min(config.source_layer, max(0, num_layers - 2))
        tgt_layer = min(config.target_layer, max(0, num_layers - 1))

        dictionary = getattr(config, "dictionary_path", None)
        encoder = getattr(config, "encoder", None)
        decoder = getattr(config, "decoder", None)

        missing = []
        if encoder is None:
            missing.append("encoder")
        if decoder is None:
            missing.append("decoder")
        if not missing and not dictionary:
            missing.append("dictionary_path (or inline weights)")

        if missing:
            reason = (
                f"No trained transcoder dictionary is available: "
                f"{', '.join(missing)} not supplied. A transcoder decomposition "
                f"requires a learned encoder and decoder; this algorithm "
                f"previously simulated both, deriving feature activations from an "
                f"arithmetic ramp over a fabricated input value, decoder weights "
                f"from `0.4 + 0.04 * feature_index`, and a fraction-of-variance-"
                f"explained clamped into [0.80, 0.99] so it could never report a "
                f"poor reconstruction. See the module docstring."
            )
            return DiscoveryReport(
                algorithm="transcoders",
                dataset_id=dataset.get("id", "unknown"),
                model_id=(self.adapter.spec.model_id
                          if self.adapter else "unavailable"),
                runtime_ms=(time.time() - t0) * 1000,
                statistics={
                    "measured": False,
                    "reason": reason,
                    "source_layer": src_layer,
                    "target_layer": tgt_layer,
                    "dict_size": config.dict_size,
                    # Present as None rather than as the clamped 0.80-0.99 the
                    # formula could produce.
                    "fve_variance_explained": None,
                    "l0_sparsity": None,
                    "active_features_count": None,
                },
                evidence={
                    "dictionary_path": dictionary,
                    "missing_components": missing,
                },
                confidence=None,
                graph={"nodes": [], "edges": [], "score": None, "measured": False},
                provenance={
                    "search_space": getattr(config, "search_space", "mlp_transcoders"),
                    "unavailable_reason": reason,
                    "previously_reported": {
                        "fve_range": "[0.80, 0.99] by construction",
                        "edge_confidence": 0.95,
                        "decoder_weight": "0.4 + 0.04 * feature_index",
                        "feature_activation": (
                            "max(0, src_val * (0.8 + 0.1 * (f_idx % 3)) - 0.25)"
                        ),
                    },
                },
            )

        # A dictionary was supplied. Running the real decomposition is not
        # implemented, and reporting a partial one would be the same defect under
        # a different name. Declining here is deliberate.
        raise NotImplementedError(
            "A transcoder dictionary was supplied, but the real decomposition is "
            "not implemented. It must: capture the MLP input and output for the "
            "target layer, encode to feature activations, decode to a "
            "reconstruction, and report FVE as 1 - Var(residual)/Var(target) "
            "unclamped. Returning anything else would reintroduce the "
            "fabrication this module previously performed."
        )