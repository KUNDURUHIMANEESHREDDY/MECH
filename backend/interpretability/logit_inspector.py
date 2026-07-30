"""
Logit Inspector.

Inspects logits at each layer of a transformer model.
Provides layer logits and top predicted tokens.

Architecture:

    Runtime

    ↓

    ActivationRepository

    ↓

    LogitInspector

    ↓

    LogitInspection
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from .repository import ActivationRepository
from .mock_runtime import get_default_runtime
from .models import LogitInspection, Statistics
from .statistics import StatisticsComputer


class LogitInspector:
    """Inspects logits at each layer of a model.

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
        token_index: int = 0,
        top_k: int = 5,
    ) -> LogitInspection:
        """Inspect the logits at a specific layer and token.

        Parameters
        ----------
        layer_index : int
            Layer to inspect.
        token_index : int, default 0
            Token position to inspect.
        top_k : int, default 5
            Number of top predicted tokens to return.

        Returns
        -------
        LogitInspection
            Complete logit inspection data.
        """
        layer_name = self.repository.get_layer_name(layer_index)
        layer_logits = self.repository.get_logits(layer_index)

        # Get logits for the specified token
        token_logits = layer_logits[token_index, :]
        logits_list = token_logits.tolist()

        # Compute top-k predictions with softmax probabilities
        exp_logits = np.exp(token_logits - np.max(token_logits))
        probs = exp_logits / np.sum(exp_logits)

        top_indices = np.argsort(token_logits)[::-1][:top_k]
        top_tokens = [
            {
                "token_id": int(idx),
                "logit": float(token_logits[idx]),
                "probability": float(probs[idx]),
            }
            for idx in top_indices
        ]

        stats = self.stats_computer.compute(token_logits)

        return LogitInspection(
            layer=layer_name,
            layer_index=layer_index,
            logits=logits_list,
            shape=[int(token_logits.shape[0])],
            top_tokens=top_tokens,
            statistics=stats,
        )

    def inspect_all_layers(
        self,
        token_index: int = 0,
        top_k: int = 5,
    ) -> List[LogitInspection]:
        """Inspect logits at all layers.

        Parameters
        ----------
        token_index : int, default 0
            Token position to inspect.
        top_k : int, default 5
            Number of top predicted tokens.

        Returns
        -------
        list of LogitInspection
            Logit inspection data for each layer.
        """
        return [
            self.inspect(layer_idx, token_index, top_k)
            for layer_idx in range(self.repository.num_layers)
        ]

    def get_top_tokens(
        self,
        layer_index: int,
        token_index: int = 0,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """Get the top-k predicted tokens at a layer.

        Parameters
        ----------
        layer_index : int
            Layer to inspect.
        token_index : int, default 0
            Token position.
        top_k : int, default 5
            Number of top tokens.

        Returns
        -------
        list of dict
            Top predicted tokens with logits and probabilities.
        """
        layer_logits = self.repository.get_logits(layer_index)
        token_logits = layer_logits[token_index, :]

        exp_logits = np.exp(token_logits - np.max(token_logits))
        probs = exp_logits / np.sum(exp_logits)

        top_indices = np.argsort(token_logits)[::-1][:top_k]
        return [
            {
                "token_id": int(idx),
                "logit": float(token_logits[idx]),
                "probability": float(probs[idx]),
            }
            for idx in top_indices
        ]
