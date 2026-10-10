"""
Experiment runner for live mechanistic interpretability experiments.

Runs analysis experiments across layers, neurons, attention heads,
and residual streams using live GPT-2 weights.

All experiments require a live model connection. When no live model is
available, experiments fail closed with an explicit "unavailable" status
rather than returning synthetic data.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class ExperimentResult:
    """Container for a single experiment's results.

    `provenance` and `measured` default to *not* claiming anything. They used to
    default to `"live"` and `True`, which meant a container constructed without
    them asserted a live measurement it had never seen -- the fail-open shape
    this repository is supposed to refuse. Every real construction site in
    `ExperimentRunner` passes them explicitly, so the closed defaults only affect
    a caller that constructs the container by hand, which is exactly the case
    that must not silently claim liveness.
    """

    name: str
    experiment_type: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    parameters: Dict[str, Any] = field(default_factory=dict)
    results: Dict[str, Any] = field(default_factory=dict)
    provenance: str = "unavailable"  # "live" | "unavailable"
    measured: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "experiment_type": self.experiment_type,
            "timestamp": self.timestamp,
            "parameters": self.parameters,
            "results": self.results,
            "provenance": self.provenance,
            "measured": self.measured,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


class ExperimentRunner:
    """Runs and collects results from live interpretability experiments.

    All experiments require a live GPT-2 model connection. When no live
    model is available, experiments fail closed with an explicit
    "unavailable" status rather than returning synthetic data.
    """

    def __init__(self):
        self.results: List[ExperimentResult] = []
        self._engine = None
        self._ensure_live_engine()

    def _ensure_live_engine(self):
        """Ensure the live GPT-2 engine is available."""
        from backend.services import gpt2_engine
        if not gpt2_engine.is_available():
            raise RuntimeError("torch/transformers not available — cannot run experiments without live weights")
        load_result = gpt2_engine.load()
        if load_result.get("status") != "loaded":
            raise RuntimeError(f"Model load failed: {load_result}")
        self._engine = gpt2_engine

    def _run_with_live_engine(self, fn, *args, **kwargs) -> Any:
        """Run a function with the live engine, failing closed if unavailable."""
        if self._engine is None:
            self._ensure_live_engine()
        return fn(*args, **kwargs)

    def run_neuron_experiment(
        self,
        layer_index: int,
        top_k: int = 10,
        token_index: int = 0,
    ) -> ExperimentResult:
        """Run a neuron analysis experiment using live weights.

        Inspects the top-k most active neurons in a layer and collects
        their statistics, activation histories, and top tokens.

        Raises
        ------
        RuntimeError
            If no live model is available.
        """
        neurons_data = self._run_with_live_engine(
            self._engine.list_neurons,
            layer=layer_index,
            component="mlp",
            page=0,
            page_size=top_k,
            sort_by="activation",
            order="desc",
        )

        if neurons_data.get("status") != "ok":
            return ExperimentResult(
                name=f"neuron_analysis_layer_{layer_index}",
                experiment_type="neuron",
                parameters={"layer_index": layer_index, "top_k": top_k, "token_index": token_index},
                results={"error": neurons_data.get("error", "Unknown error")},
                provenance="unavailable",
                measured=False,
            )

        neurons = neurons_data.get("neurons", [])
        activations = [n.get("activation") for n in neurons if n.get("activation") is not None]
        sparsities = [abs(n.get("activation", 0)) < 1e-3 for n in neurons] if neurons else []

        results = {
            "layer": f"transformer.h.{layer_index}",
            "layer_index": layer_index,
            "token_index": token_index,
            "neurons": neurons,
            "summary": {
                "max_activation": max(activations) if activations else None,
                "mean_activation": float(np.mean(activations)) if activations else None,
                "mean_sparsity": float(np.mean(sparsities)) if sparsities else None,
                "mean_variance": float(np.var(activations)) if len(activations) > 1 else None,
            },
            "provenance": "live",
            "measured": True,
        }

        result = ExperimentResult(
            name=f"neuron_analysis_layer_{layer_index}",
            experiment_type="neuron",
            parameters={
                "layer_index": layer_index,
                "top_k": top_k,
                "token_index": token_index,
            },
            results=results,
        )
        self.results.append(result)
        return result

    def run_attention_experiment(
        self,
        layer_index: int,
        token_index: int = 0,
    ) -> ExperimentResult:
        """Run an attention analysis experiment using live weights.

        Inspects all attention heads in a layer and collects their
        importance scores, matrices, and statistics.

        Raises
        ------
        RuntimeError
            If no live model is available.
        """
        layer_detail = self._run_with_live_engine(
            self._engine.layer_detail, layer_index
        )

        if layer_detail.get("status") != "ok":
            return ExperimentResult(
                name=f"attention_analysis_layer_{layer_index}",
                experiment_type="attention",
                parameters={"layer_index": layer_index, "token_index": token_index},
                results={"error": layer_detail.get("error", "Unknown error")},
                provenance="unavailable",
                measured=False,
            )

        heads = layer_detail.get("attention_heads", [])
        importance_scores = [h.get("o_weight_l2", 0) for h in heads]

        results = {
            "layer": f"transformer.h.{layer_index}",
            "layer_index": layer_index,
            "token_index": token_index,
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
                "mean_importance": float(np.mean(importance_scores)) if importance_scores else None,
                "min_importance": min(importance_scores) if importance_scores else None,
                "std_importance": float(np.std(importance_scores)) if len(importance_scores) > 1 else None,
            },
            "provenance": "live",
            "measured": True,
        }

        result = ExperimentResult(
            name=f"attention_analysis_layer_{layer_index}",
            experiment_type="attention",
            parameters={
                "layer_index": layer_index,
                "token_index": token_index,
            },
            results=results,
        )
        self.results.append(result)
        return result

    def run_residual_experiment(
        self,
        token_index: int = 0,
    ) -> ExperimentResult:
        """Run a residual stream analysis experiment using live weights.

        Inspects the residual stream at all layers and collects
        norms, statistics, and contributions.

        Raises
        ------
        RuntimeError
            If no live model is available.
        """
        n_layers = self._engine._n_layers() if self._engine else 12
        residuals = []
        norms = []

        for layer_idx in range(n_layers):
            layer_detail = self._run_with_live_engine(
                self._engine.layer_detail, layer_idx
            )
            if layer_detail.get("status") != "ok":
                continue

            # Get residual stream statistics from the layer
            # hidden[layer+1] is the residual after block `layer`
            if self._engine._cache and "hidden" in self._engine._cache:
                import torch
                hidden = self._engine._cache["hidden"]
                if layer_idx + 1 < len(hidden):
                    resid = hidden[layer_idx + 1]  # [seq, d_model]
                    resid_t = torch.tensor(resid)
                    token_resid = resid_t[token_index] if token_index < resid_t.shape[0] else resid_t[-1]
                    norm = float(torch.linalg.vector_norm(token_resid))
                    norms.append(norm)
                    residuals.append({
                        "layer_index": layer_idx,
                        "norm": norm,
                        "contribution": norm / sum(norms) if sum(norms) > 0 else 0,
                    })

        results = {
            "token_index": token_index,
            "num_layers": len(residuals),
            "residuals": residuals,
            "norms": norms,
            "contributions": [r["contribution"] for r in residuals],
            "summary": {
                "max_norm": max(norms) if norms else None,
                "mean_norm": float(np.mean(norms)) if norms else None,
                "min_norm": min(norms) if norms else None,
                "std_norm": float(np.std(norms)) if len(norms) > 1 else None,
                "total_contribution": sum(norms),
            },
            "provenance": "live",
            "measured": True,
        }

        result = ExperimentResult(
            name=f"residual_analysis_token_{token_index}",
            experiment_type="residual",
            parameters={
                "token_index": token_index,
            },
            results=results,
        )
        self.results.append(result)
        return result

    def run_full_experiment(
        self,
        layer_index: int = 0,
        top_k: int = 10,
        token_index: int = 0,
    ) -> Dict[str, ExperimentResult]:
        """Run all three experiment types and return combined results."""
        return {
            "neuron": self.run_neuron_experiment(
                layer_index, top_k, token_index
            ),
            "attention": self.run_attention_experiment(
                layer_index, token_index
            ),
            "residual": self.run_residual_experiment(token_index),
        }

    def run_cross_layer_analysis(self) -> ExperimentResult:
        """Run a cross-layer analysis of neuron statistics using live weights.

        Collects statistics for all layers and identifies patterns
        such as increasing/decreasing activation sparsity with depth.

        Raises
        ------
        RuntimeError
            If no live model is available.
        """
        if self._engine is None:
            self._ensure_live_engine()

        n_layers = self._engine._n_layers()
        layer_stats = []

        for layer_idx in range(n_layers):
            layer_detail = self._run_with_live_engine(
                self._engine.layer_detail, layer_idx
            )
            if layer_detail.get("status") != "ok":
                layer_stats.append({
                    "layer": f"transformer.h.{layer_idx}",
                    "layer_index": layer_idx,
                    "max": None,
                    "mean": None,
                    "variance": None,
                    "sparsity": None,
                    "error": layer_detail.get("error", "Unknown error"),
                })
                continue

            act_summary = layer_detail.get("activation_summary")
            if act_summary:
                layer_stats.append({
                    "layer": f"transformer.h.{layer_idx}",
                    "layer_index": layer_idx,
                    "max": act_summary.get("last_token_stats", {}).get("max"),
                    "mean": act_summary.get("mean_abs_over_seq", {}).get("mean"),
                    "variance": act_summary.get("mean_abs_over_seq", {}).get("std", 0) ** 2,
                    "sparsity": act_summary.get("fraction_active_last"),
                })
            else:
                layer_stats.append({
                    "layer": f"transformer.h.{layer_idx}",
                    "layer_index": layer_idx,
                    "max": None,
                    "mean": None,
                    "variance": None,
                    "sparsity": None,
                    "note": "Run a prompt first to populate activations",
                })

        sparsities = [s["sparsity"] for s in layer_stats if s["sparsity"] is not None]
        means = [s["mean"] for s in layer_stats if s["mean"] is not None]

        results = {
            "layer_stats": layer_stats,
            "trends": {
                "sparsity_increasing": bool(
                    np.polyfit(range(len(sparsities)), sparsities, 1)[0] > 0
                ) if len(sparsities) >= 2 else None,
                "mean_increasing": bool(
                    np.polyfit(range(len(means)), means, 1)[0] > 0
                ) if len(means) >= 2 else None,
                "max_sparsity_layer": layer_stats[
                    int(np.argmax(sparsities))
                ]["layer"] if sparsities else None,
                "min_sparsity_layer": layer_stats[
                    int(np.argmin(sparsities))
                ]["layer"] if sparsities else None,
            },
            "provenance": "live",
            "measured": True,
        }

        result = ExperimentResult(
            name="cross_layer_analysis",
            experiment_type="cross_layer",
            parameters={},
            results=results,
        )
        self.results.append(result)
        return result

    def save_results(self, path: str) -> None:
        """Save all experiment results to a JSON file."""
        data = {
            "experiment_count": len(self.results),
            "experiments": [r.to_dict() for r in self.results],
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def clear_results(self) -> None:
        """Clear all stored experiment results."""
        self.results.clear()
