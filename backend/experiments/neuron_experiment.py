"""
Neuron experiment definitions.

Pre-built experiment workflows for neuron analysis using live GPT-2 weights.

All experiments require a live model connection. When no live model is
available, experiments fail closed with an explicit "unavailable" status
rather than returning synthetic data.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .config import ExperimentConfig


class NeuronExperiment:
    """Pre-built neuron analysis experiments using live GPT-2 weights.

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
            raise RuntimeError("torch/transformers not available — cannot run neuron experiments without live weights")
        load_result = gpt2_engine.load()
        if load_result.get("status") != "loaded":
            raise RuntimeError(f"Model load failed: {load_result}")
        return gpt2_engine

    def run_single_neuron_analysis(self, layer_index: int = 0, neuron_index: int = 0) -> Dict[str, Any]:
        """Analyze a single neuron across all tokens using live weights.

        Parameters
        ----------
        layer_index : int
            Layer to analyze.
        neuron_index : int
            Neuron to analyze.

        Returns
        -------
        dict
            Analysis results including activation, statistics, and
            top activating tokens.

        Raises
        ------
        RuntimeError
            If no live model is available.
        """
        engine = self._require_live_engine()
        return engine.neuron_detail(
            layer=layer_index,
            neuron_index=neuron_index,
            component="mlp",
            top_k_weights=16,
        )

    def run_layer_neuron_analysis(self, layer_index: int = 0) -> Dict[str, Any]:
        """Analyze top neurons in a single layer using live weights.

        Parameters
        ----------
        layer_index : int
            Layer to analyze.

        Returns
        -------
        dict
            Analysis results including top neurons and summary statistics.

        Raises
        ------
        RuntimeError
            If no live model is available.
        """
        engine = self._require_live_engine()
        return engine.list_neurons(
            layer=layer_index,
            component="mlp",
            page=0,
            page_size=self.config.top_k_neurons,
            sort_by="activation",
            order="desc",
        )

    def run_all_layers_neuron_analysis(self) -> List[Dict[str, Any]]:
        """Analyze neurons across all configured layers using live weights.

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
                results.append(self.run_layer_neuron_analysis(layer_idx))
            except RuntimeError as e:
                results.append({
                    "status": "unavailable",
                    "layer_index": layer_idx,
                    "error": str(e),
                })
        return results

    def run_sparsity_analysis(self) -> Dict[str, Any]:
        """Analyze neuron activation sparsity across all layers using live weights.

        Returns
        -------
        dict
            Sparsity statistics per layer and overall trends.

        Raises
        ------
        RuntimeError
            If no live model is available.
        """
        engine = self._require_live_engine()
        import numpy as np

        layer_sparsities = []
        for layer_idx in self.config.get_layers():
            layer_name = f"transformer.h.{layer_idx}"
            try:
                layer_detail = engine.layer_detail(layer_idx)
                if layer_detail.get("status") != "ok":
                    layer_sparsities.append({
                        "layer": layer_name,
                        "layer_index": layer_idx,
                        "sparsity": None,
                        "mean": None,
                        "max": None,
                        "variance": None,
                        "error": "Layer detail unavailable",
                    })
                    continue

                act_summary = layer_detail.get("activation_summary")
                if act_summary:
                    layer_sparsities.append({
                        "layer": layer_name,
                        "layer_index": layer_idx,
                        "sparsity": act_summary.get("fraction_active_last"),
                        "mean": act_summary.get("mean_abs_over_seq", {}).get("mean"),
                        "max": act_summary.get("last_token_stats", {}).get("max"),
                        "variance": act_summary.get("mean_abs_over_seq", {}).get("std", 0) ** 2,
                    })
                else:
                    layer_sparsities.append({
                        "layer": layer_name,
                        "layer_index": layer_idx,
                        "sparsity": None,
                        "mean": None,
                        "max": None,
                        "variance": None,
                        "note": "Run a prompt first to populate activations",
                    })
            except Exception as e:
                layer_sparsities.append({
                    "layer": layer_name,
                    "layer_index": layer_idx,
                    "sparsity": None,
                    "mean": None,
                    "max": None,
                    "variance": None,
                    "error": str(e),
                })

        valid_sparsities = [s["sparsity"] for s in layer_sparsities if s["sparsity"] is not None]
        return {
            "layer_sparsities": layer_sparsities,
            "overall_mean_sparsity": float(np.mean(valid_sparsities)) if valid_sparsities else None,
            "overall_max_sparsity": float(max(valid_sparsities)) if valid_sparsities else None,
            "overall_min_sparsity": float(min(valid_sparsities)) if valid_sparsities else None,
            "sparsity_trend": ("increasing" if len(valid_sparsities) >= 2 and np.polyfit(range(len(valid_sparsities)), valid_sparsities, 1)[0] > 0 else "decreasing") if valid_sparsities else "unknown",
        }
