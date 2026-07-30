"""Statistics Engine — Computes mathematical & statistical metrics on activations.

Separated from repository layer to keep data access clean.
"""

from __future__ import annotations

import torch
import math
from typing import Any


class StatsEngine:
    """Computes stats, norms, distributions, top-k probabilities, and GELU sparsity."""

    @staticmethod
    def compute_tensor_stats(tensor: torch.Tensor) -> dict[str, float]:
        if tensor is None or tensor.numel() == 0:
            return {"l2_norm": 0.0, "mean": 0.0, "max": 0.0, "min": 0.0, "std": 0.0, "sparsity": 0.0}

        t_float = tensor.float()
        l2_norm = float(torch.norm(t_float).item())
        mean_val = float(t_float.mean().item())
        max_val = float(t_float.max().item())
        min_val = float(t_float.min().item())
        std_val = float(t_float.std().item()) if t_float.numel() > 1 else 0.0

        zero_count = float((t_float == 0).sum().item())
        sparsity = zero_count / t_float.numel()

        return {
            "l2_norm": round(l2_norm, 4),
            "mean": round(mean_val, 4),
            "max": round(max_val, 4),
            "min": round(min_val, 4),
            "std": round(std_val, 4),
            "sparsity": round(sparsity, 4),
        }

    @staticmethod
    def compute_entropy(probs: torch.Tensor) -> float:
        """Calculate Shannon entropy in nats/bits."""
        if probs is None or probs.numel() == 0:
            return 0.0
        p = probs.float()
        p = p[p > 0]
        entropy = -torch.sum(p * torch.log2(p)).item()
        return round(float(entropy), 4)

    @staticmethod
    def compute_top_k(logits: torch.Tensor, tokenizer: Any, top_k: int = 5) -> list[dict[str, Any]]:
        """Compute top-K predicted tokens and probabilities from logits."""
        if logits is None or logits.numel() == 0:
            return []

        probs = torch.softmax(logits, dim=-1)
        top_probs, top_indices = torch.topk(probs, min(top_k, probs.size(-1)))

        results = []
        for prob, idx in zip(top_probs.tolist(), top_indices.tolist()):
            text = tokenizer.decode([idx], skip_special_tokens=True) if tokenizer else f"Token {idx}"
            results.append({
                "token_id": idx,
                "token_text": text if text else "<space>",
                "probability": round(float(prob), 4),
                "logit": round(float(logits[idx].item()), 4),
            })
        return results


stats_engine = StatsEngine()
