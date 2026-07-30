"""
Token Inspector.

Provides a token-centric view of model internals:
- Token embedding
- Attention from this token
- Residual stream at this token
- Logits and top predictions

Architecture:

    Runtime

    ↓

    ActivationRepository

    ↓

    TokenInspector

    ↓

    TokenInspection
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from .repository import ActivationRepository
from .mock_runtime import get_default_runtime
from .models import TokenInspection, Statistics
from .statistics import StatisticsComputer
from .attention_inspector import AttentionInspector
from .residual_inspector import ResidualInspector


class TokenInspector:
    """Inspects tokens in a model's sequence.

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
        self.attention_inspector = AttentionInspector(
            repository=self.repository, stats_computer=self.stats_computer
        )
        self.residual_inspector = ResidualInspector(
            repository=self.repository, stats_computer=self.stats_computer
        )

    def inspect(
        self,
        token_index: int,
        layer_index: int = 0,
        include_embedding: bool = True,
        include_attention: bool = True,
        include_residual: bool = True,
        include_logits: bool = True,
        top_k_predictions: int = 5,
    ) -> TokenInspection:
        """Inspect a single token.

        Parameters
        ----------
        token_index : int
            Index of the token in the sequence.
        layer_index : int, default 0
            Layer to use for attention and residual inspection.
        include_embedding : bool, default True
            Include token embedding.
        include_attention : bool, default True
            Include attention from this token.
        include_residual : bool, default True
            Include residual stream at this token.
        include_logits : bool, default True
            Include logits and top predictions.
        top_k_predictions : int, default 5
            Number of top predictions to return.

        Returns
        -------
        TokenInspection
            Complete token inspection data.
        """
        tokens = self.repository.get_tokens()
        token_id = int(tokens[token_index]) if token_index < len(tokens) else 0

        embedding = None
        embedding_shape = None
        if include_embedding:
            embeddings = self.repository.get_embeddings()
            embedding = embeddings[token_index, :].tolist()
            embedding_shape = [int(embeddings.shape[1])]

        attention = None
        if include_attention:
            attention = self.attention_inspector.inspect(
                layer_index, 0, token_index
            )

        residual = None
        if include_residual:
            residual = self.residual_inspector.inspect(
                layer_index, token_index
            )

        logits = None
        top_predictions = None
        if include_logits:
            layer_logits = self.repository.get_logits(layer_index)
            token_logits = layer_logits[token_index, :]
            logits = token_logits.tolist()

            # Get top-k predictions
            top_indices = np.argsort(token_logits)[::-1][:top_k_predictions]
            top_predictions = [
                {
                    "token_id": int(idx),
                    "logit": float(token_logits[idx]),
                    "probability": float(
                        np.exp(token_logits[idx]) / np.sum(np.exp(token_logits))
                    ),
                }
                for idx in top_indices
            ]

        stats = None
        if residual is not None:
            stats = residual.statistics

        return TokenInspection(
            token_index=token_index,
            token_id=token_id,
            embedding=embedding,
            embedding_shape=embedding_shape,
            attention=attention,
            residual=residual,
            logits=logits,
            top_predictions=top_predictions,
            statistics=stats,
        )

    def inspect_all_tokens(
        self,
        layer_index: int = 0,
        token_index: Optional[int] = None,
    ) -> List[TokenInspection]:
        """Inspect all tokens in the sequence.

        Parameters
        ----------
        layer_index : int, default 0
            Layer to use for inspection.
        token_index : int, optional
            If provided, only inspect this token.

        Returns
        -------
        list of TokenInspection
            Inspection data for each token.
        """
        if token_index is not None:
            return [self.inspect(token_index, layer_index)]

        return [
            self.inspect(i, layer_index)
            for i in range(self.repository.seq_len)
        ]
