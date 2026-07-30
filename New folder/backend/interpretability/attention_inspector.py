"""
Attention Inspector.

Inspects attention heads in a transformer model's layers.
Returns head index, attention weight matrix, importance score,
matrix shape, and statistical summaries.

Also provides head ranking by entropy, importance, and sparsity.

Architecture:

    Runtime

    ↓

    ActivationRepository

    ↓

    AttentionInspector

    ↓

    AttentionInspection / HeadRanking
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from .repository import ActivationRepository
from .mock_runtime import get_default_runtime
from .models import AttentionInspection, HeadRanking, Statistics
from .statistics import StatisticsComputer


class AttentionInspector:
    """Inspects attention heads in a model's attention data.

    Parameters
    ----------
    repository : ActivationRepository, optional
        The activation repository to inspect.
    stats_computer : StatisticsComputer, optional
        The statistics computer to use.
    """

    def __init__(
        self,
        repository: Optional[ActivationRepository] = None,
        stats_computer: Optional[StatisticsComputer] = None,
    ):
        self.repository = repository if repository is not None else ActivationRepository(
            get_default_runtime()
        )
        self.stats_computer = (
            stats_computer if stats_computer is not None else StatisticsComputer()
        )

    def inspect(
        self,
        layer_index: int,
        head_index: int,
        token_index: Optional[int] = None,
    ) -> AttentionInspection:
        """Inspect a single attention head."""
        layer_name = self.repository.get_layer_name(layer_index)
        full_matrix = self.repository.get_attention_head(layer_name, head_index)

        if token_index is not None:
            matrix = full_matrix[token_index, :].reshape(1, -1)
            shape = [1, full_matrix.shape[1]]
        else:
            matrix = full_matrix
            shape = list(full_matrix.shape)

        stats = self.stats_computer.compute(matrix)
        importance = self._compute_importance(full_matrix)
        top_connections = self._find_top_connections(
            full_matrix, token_index=token_index, top_k=5
        )

        return AttentionInspection(
            head=head_index,
            num_heads=self.repository.num_heads,
            matrix=matrix.tolist(),
            importance=importance,
            shape=shape,
            layer=layer_name,
            layer_index=layer_index,
            statistics=stats,
            top_connections=top_connections,
        )

    def inspect_all_heads(
        self,
        layer_index: int,
        token_index: Optional[int] = None,
    ) -> List[AttentionInspection]:
        """Inspect all attention heads in a layer, sorted by importance."""
        results = [
            self.inspect(layer_index, h, token_index)
            for h in range(self.repository.num_heads)
        ]
        results.sort(key=lambda x: x.importance, reverse=True)
        return results

    def get_head_importance(
        self,
        layer_index: int,
        head_index: int,
    ) -> float:
        """Compute the importance score for a specific head."""
        layer_name = self.repository.get_layer_name(layer_index)
        matrix = self.repository.get_attention_head(layer_name, head_index)
        return self._compute_importance(matrix)

    def get_layer_importance(
        self,
        layer_index: int,
    ) -> List[float]:
        """Get importance scores for all heads in a layer."""
        return [
            self.get_head_importance(layer_index, h)
            for h in range(self.repository.num_heads)
        ]

    def rank_heads(
        self,
        layer_index: int,
        metric: str = "importance",
    ) -> HeadRanking:
        """Rank attention heads by a specific metric.

        Parameters
        ----------
        layer_index : int
            Layer to rank heads in.
        metric : str, default "importance"
            Metric to rank by: "entropy", "importance", or "sparsity".

        Returns
        -------
        HeadRanking
            Ranked list of heads with scores.
        """
        layer_name = self.repository.get_layer_name(layer_index)
        rankings = []

        for head_idx in range(self.repository.num_heads):
            matrix = self.repository.get_attention_head(layer_name, head_idx)

            if metric == "entropy":
                score = self._compute_entropy(matrix)
            elif metric == "importance":
                score = self._compute_importance(matrix)
            elif metric == "sparsity":
                score = self.stats_computer.sparsity_value(matrix)
            else:
                raise ValueError(f"Unknown metric: {metric}")

            rankings.append(
                {
                    "head": head_idx,
                    "score": float(score),
                }
            )

        # Sort by score (descending)
        rankings.sort(key=lambda x: x["score"], reverse=True)

        return HeadRanking(
            layer=layer_name,
            layer_index=layer_index,
            metric=metric,
            rankings=rankings,
        )

    # ------------------------------------------------------------------ #
    #  Internal helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _compute_importance(matrix: np.ndarray) -> float:
        """Compute an importance score for an attention matrix."""
        seq_len = matrix.shape[0]
        if seq_len == 0:
            return 0.0

        mean_mag = float(np.mean(matrix))

        eps = 1e-12
        entropies = []
        for row in matrix:
            p = row + eps
            p = p / np.sum(p)
            entropy = -np.sum(p * np.log(p))
            max_entropy = np.log(len(row)) if len(row) > 1 else 1.0
            if max_entropy > 0:
                entropies.append(entropy / max_entropy)
        avg_entropy = float(np.mean(entropies)) if entropies else 0.0

        importance = mean_mag * (1.0 - avg_entropy)
        return float(np.clip(importance, 0.0, 1.0))

    @staticmethod
    def _compute_entropy(matrix: np.ndarray) -> float:
        """Compute the average normalized entropy of an attention matrix."""
        eps = 1e-12
        entropies = []
        for row in matrix:
            p = row + eps
            p = p / np.sum(p)
            entropy = -np.sum(p * np.log(p))
            max_entropy = np.log(len(row)) if len(row) > 1 else 1.0
            if max_entropy > 0:
                entropies.append(entropy / max_entropy)
        return float(np.mean(entropies)) if entropies else 0.0

    @staticmethod
    def _find_top_connections(
        matrix: np.ndarray,
        token_index: Optional[int] = None,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """Find the top attention connections."""
        if token_index is not None:
            row = matrix[token_index, :]
            sorted_indices = np.argsort(row)[::-1][:top_k]
            return [
                {
                    "query": int(token_index),
                    "key": int(idx),
                    "weight": float(row[idx]),
                }
                for idx in sorted_indices
            ]

        flat = matrix.flatten()
        top_flat_indices = np.argsort(flat)[::-1][:top_k]
        results = []
        for idx in top_flat_indices:
            q, k = np.unravel_index(idx, matrix.shape)
            results.append(
                {
                    "query": int(q),
                    "key": int(k),
                    "weight": float(matrix[q, k]),
                }
            )
        return results
