"""
Attention experiment definitions.

Pre-built experiment workflows for attention head analysis using live GPT-2 weights.

All experiments require a live model connection. When no live model is
available, experiments fail closed with an explicit "unavailable" status
rather than returning synthetic data.
"""

from __future__ import annotations

from typing import Any, Dict, List

from .config import ExperimentConfig


class AttentionExperiment:
    """Pre-built attention analysis experiments using live GPT-2 weights.

    Parameters
    ----------
    config : ExperimentConfig
        Configuration for the experiment.
    """

    def __init__(self, config: ExperimentConfig = None):
        self.config = config or ExperimentConfig()

    def _require_live_engine(self):
        """Get the live GPT-2 engine or raise if unavailable."""
        from backend.services import gpt2_engine
        if not gpt2_engine.is_available():
            raise RuntimeError("torch/transformers not available — cannot run attention experiments without live weights")
        load_result = gpt2_engine.load()
        if load_result.get("status") != "loaded":
            raise RuntimeError(f"Model load failed: {load_result}")
        return gpt2_engine

    def run_single_head_analysis(
        self, layer_index: int = 0, head_index: int = 0
    ) -> Dict[str, Any]:
        """Analyze a single attention head using live weights.

        Parameters
        ----------
        layer_index : int
            Layer to analyze.
        head_index : int
            Head to analyze.

        Returns
        -------
        dict
            Analysis results including matrix, importance, and statistics.

        Raises
        ------
        RuntimeError
            If no live model is available.
        """
        engine = self._require_live_engine()
        return engine.head_detail(
            layer=layer_index,
            head=head_index,
        )

    def run_layer_attention_analysis(self, layer_index: int = 0) -> Dict[str, Any]:
        """Analyze all attention heads in a single layer using live weights.

        Parameters
        ----------
        layer_index : int
            Layer to analyze.

        Returns
        -------
        dict
            Analysis results including importance scores and summary.

        Raises
        ------
        RuntimeError
            If no live model is available.
        """
        engine = self._require_live_engine()
        layer_detail = engine.layer_detail(layer_index)
        if layer_detail.get("status") != "ok":
            return {"status": "unavailable", "layer_index": layer_index, "error": layer_detail.get("error", "Unknown error")}

        heads = layer_detail.get("attention_heads", [])
        importance_scores = [h.get("o_weight_l2", 0) for h in heads]

        return {
            "status": "ok",
            "layer": f"transformer.h.{layer_index}",
            "layer_index": layer_index,
            "token_index": self.config.token_index,
            "num_heads": len(heads),
            "importance_scores": importance_scores,
            "top_heads": [
                {
                    "head": h.get("head_index"),
                    "importance": h.get("o_weight_l2"),
                    "shape": [h.get("d_head"), h.get("d_head")],
                    "statistics": {
                        "q_weight_l2": h.get("q_weight_l2"),
                        "k_weight_l2": h.get("k_weight_l2"),
                        "v_weight_l2": h.get("v_weight_l2"),
                        "o_weight_l2": h.get("o_weight_l2"),
                    },
                }
                for h in heads[:5]
            ],
            "summary": {
                "max_importance": max(importance_scores) if importance_scores else None,
                "mean_importance": float(sum(importance_scores) / len(importance_scores)) if importance_scores else None,
                "min_importance": min(importance_scores) if importance_scores else None,
                "std_importance": None,  # Could compute if needed
            },
            "provenance": "live",
        }

    def run_all_layers_attention_analysis(self) -> List[Dict[str, Any]]:
        """Analyze attention heads across all configured layers using live weights.

        Returns
        -------
        list of dict
            Analysis results for each layer.

        Raises
        ------
        RuntimeError
            If no live model is available.
        """
        results = []
        for layer_idx in self.config.get_layers():
            try:
                results.append(self.run_layer_attention_analysis(layer_idx))
            except RuntimeError as e:
                results.append({
                    "status": "unavailable",
                    "layer_index": layer_idx,
                    "error": str(e),
                })
        return results

    def run_importance_analysis(self) -> Dict[str, Any]:
        """Analyze attention head importance across all layers using live weights.

        Returns
        -------
        dict
            Importance statistics per layer and overall trends.

        Raises
        ------
        RuntimeError
            If no live model is available.
        """
        engine = self._require_live_engine()
        import numpy as np

        layer_importance = []
        for layer_idx in self.config.get_layers():
            layer_detail = engine.layer_detail(layer_idx)
            if layer_detail.get("status") != "ok":
                layer_importance.append({
                    "layer": f"transformer.h.{layer_idx}",
                    "layer_index": layer_idx,
                    "importance_scores": [],
                    "max_importance": None,
                    "mean_importance": None,
                    "min_importance": None,
                    "top_head": None,
                    "error": layer_detail.get("error", "Unknown error"),
                })
                continue

            heads = layer_detail.get("attention_heads", [])
            scores = [h.get("o_weight_l2", 0) for h in heads]
            layer_importance.append({
                "layer": f"transformer.h.{layer_idx}",
                "layer_index": layer_idx,
                "importance_scores": scores,
                "max_importance": float(max(scores)) if scores else None,
                "mean_importance": float(np.mean(scores)) if scores else None,
                "min_importance": float(min(scores)) if scores else None,
                "top_head": int(np.argmax(scores)) if scores else None,
            })

        all_scores = [s for layer in layer_importance for s in layer.get("importance_scores", [])]
        return {
            "layer_importance": layer_importance,
            "overall_max_importance": float(max(all_scores)) if all_scores else None,
            "overall_mean_importance": float(np.mean(all_scores)) if all_scores else None,
            "overall_min_importance": float(min(all_scores)) if all_scores else None,
            "provenance": "live",
        }
