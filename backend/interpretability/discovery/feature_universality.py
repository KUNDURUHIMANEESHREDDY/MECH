"""Empirical Feature Universality & Cross-Layer Alignment Engine for MECH.

Measures geometric cosine similarity and semantic universality of feature directions
across transformer layers and dictionary representations using live weights.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional
import torch

logger = logging.getLogger("MECH.feature_universality")


class FeatureUniversalityEngine:
    """Computes empirical cross-layer and cross-dictionary feature universality."""

    def __init__(self, model: Any = None) -> None:
        self.model = model

    def _ensure_model(self) -> None:
        if self.model is None:
            import backend.services.gpt2_engine as gpt2_engine
            gpt2_engine.load()
            self.model = gpt2_engine._model

        if self.model is None:
            raise RuntimeError("Live model is uninitialized for feature universality analysis.")

    def measure_universality(
        self,
        source_layer: int = 8,
        source_neuron_idx: int = 412,
        target_layers: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        """Calculates exact cosine similarity alignments across comparison layers."""
        self._ensure_model()
        blocks = getattr(self.model, "transformer", getattr(self.model, "model", None))
        layers = getattr(blocks, "h", getattr(blocks, "layers", []))
        
        num_layers = len(layers)
        if source_layer >= num_layers:
            raise IndexError(f"Source layer {source_layer} out of range [0, {num_layers})")

        # Extract source feature vector [d_model]
        src_proj = layers[source_layer].mlp.c_proj.weight.data.float()
        if source_neuron_idx >= src_proj.shape[0]:
            source_neuron_idx = source_neuron_idx % src_proj.shape[0]

        v_source = src_proj[source_neuron_idx, :]  # [d_model]
        v_source_norm = v_source / (torch.norm(v_source) + 1e-8)

        comp_layers = target_layers if target_layers is not None else [l for l in range(num_layers) if l != source_layer]
        
        layer_alignments: List[Dict[str, Any]] = []
        max_overall_sim = -1.0
        best_match: Dict[str, Any] = {}

        for l in comp_layers:
            if l >= num_layers:
                continue
            tgt_proj = layers[l].mlp.c_proj.weight.data.float()  # [d_mlp, d_model]
            tgt_norms = tgt_proj / (torch.norm(tgt_proj, dim=-1, keepdim=True) + 1e-8)

            with torch.no_grad():
                cos_sims = torch.matmul(tgt_norms, v_source_norm)  # [d_mlp]

            max_sim_val, max_idx = torch.max(cos_sims, dim=0)
            sim_val = round(float(max_sim_val.item()), 4)
            best_idx = int(max_idx.item())

            layer_alignments.append({
                "layer": l,
                "best_matching_neuron": best_idx,
                "cosine_similarity": sim_val,
            })

            if sim_val > max_overall_sim:
                max_overall_sim = sim_val
                best_match = {
                    "layer": l,
                    "neuron_idx": best_idx,
                    "similarity": sim_val,
                }

        is_universal = max_overall_sim >= 0.65

        return {
            "source_feature": f"L{source_layer}_N{source_neuron_idx}",
            "source_layer": source_layer,
            "source_neuron": source_neuron_idx,
            "universal_alignment_score": max_overall_sim,
            "best_matching_feature": f"L{best_match.get('layer', 0)}_N{best_match.get('neuron_idx', 0)}",
            "is_universal": is_universal,
            "layer_alignments": layer_alignments,
            "provenance": "LIVE_PYTORCH",
        }
