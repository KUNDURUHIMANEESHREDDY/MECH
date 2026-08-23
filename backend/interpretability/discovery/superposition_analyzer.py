"""Empirical Superposition & Feature Interference Analyzer for MECH.

Measures dictionary coherence, off-diagonal Gram matrix interference,
and non-orthogonal superposition capacity on live model and SAE weights.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional
import torch

logger = logging.getLogger("MECH.superposition_analyzer")


class SuperpositionAnalyzerEngine:
    """Computes empirical superposition degree and feature interference from live model weights."""

    def __init__(self, model: Any = None) -> None:
        self.model = model

    def _ensure_model(self) -> None:
        if self.model is None:
            import backend.services.gpt2_engine as gpt2_engine
            gpt2_engine.load()
            self.model = gpt2_engine._model

        if self.model is None:
            raise RuntimeError("Live model is uninitialized for superposition analysis.")

    def analyze_superposition(
        self,
        layer: int = 8,
        sample_size: int = 128,
        custom_weights: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """Calculates exact Gram matrix off-diagonal coherence and interference metrics."""
        if custom_weights is not None:
            W = custom_weights.float()
        else:
            self._ensure_model()
            blocks = getattr(self.model, "transformer", getattr(self.model, "model", None))
            layers = getattr(blocks, "h", getattr(blocks, "layers", []))
            if layer >= len(layers):
                raise IndexError(f"Layer {layer} out of range for model with {len(layers)} layers.")

            target_block = layers[layer]
            # Extract MLP projection weights [d_mlp, d_model]
            if hasattr(target_block, "mlp") and hasattr(target_block.mlp, "c_proj"):
                W = target_block.mlp.c_proj.weight.data.float()  # [d_mlp, d_model]
            elif hasattr(target_block, "mlp") and hasattr(target_block.mlp, "c_fc"):
                W = target_block.mlp.c_fc.weight.data.float().T  # [d_mlp, d_model]
            else:
                raise RuntimeError(f"Could not locate MLP projection weights at layer {layer}.")

        # Take a subset of vectors if dimension is large
        n_vectors = min(sample_size, W.shape[0])
        sub_W = W[:n_vectors, :]  # [n_vectors, d_in]

        # Unit normalize rows
        norms = torch.norm(sub_W, dim=-1, keepdim=True) + 1e-8
        W_normed = sub_W / norms  # [n_vectors, d_in]

        # Compute cosine similarity Gram matrix: [n_vectors, n_vectors]
        with torch.no_grad():
            gram = torch.matmul(W_normed, W_normed.T)

        # Off-diagonal elements
        eye = torch.eye(n_vectors, device=gram.device)
        off_diag = gram * (1.0 - eye)

        max_coherence = float(torch.max(torch.abs(off_diag)).item())
        mean_interference = float(torch.sum(off_diag ** 2).item() / max(1, n_vectors * (n_vectors - 1)))
        
        # Superposition degree: fraction of pairs with non-negligible cross-talk (|cos| > 0.05)
        non_ortho_count = int(torch.sum(torch.abs(off_diag) > 0.05).item())
        total_pairs = n_vectors * (n_vectors - 1)
        superposition_degree = round(non_ortho_count / max(1, total_pairs), 4)

        return {
            "layer": layer,
            "feature_dim": int(W.shape[-1]),
            "num_features_evaluated": n_vectors,
            "max_dictionary_coherence": round(max_coherence, 4),
            "interference_score": round(mean_interference, 6),
            "superposition_degree": superposition_degree,
            "is_superposed": bool(superposition_degree > 0.1),
            "provenance": "LIVE_PYTORCH",
            "interpretation": (
                f"Evaluated {n_vectors} weight vectors at Layer {layer}. "
                f"Max dictionary coherence = {max_coherence:.4f}, mean interference = {mean_interference:.6f}. "
                f"{superposition_degree*100:.1f}% of feature pairs exhibit non-orthogonal geometric overlap."
            ),
        }
