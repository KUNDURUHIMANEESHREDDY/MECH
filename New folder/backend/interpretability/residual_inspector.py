"""
Residual Stream Inspector.

Inspects the residual stream vectors at each layer of a transformer
model. Returns the residual vector, layer information, statistics,
and the L2 norm of the vector.

Architecture:

    Runtime

    ↓

    ActivationRepository

    ↓

    ResidualInspector

    ↓

    ResidualInspection
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from .repository import ActivationRepository
from .mock_runtime import get_default_runtime
from .models import ResidualInspection, Statistics
from .statistics import StatisticsComputer


class ResidualInspector:
    """Inspects residual stream vectors in a model.

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
    ) -> ResidualInspection:
        """Inspect the residual stream at a specific layer and token."""
        layer_name = self.repository.get_layer_name(layer_index)
        residual_matrix = self.repository.get_residual(layer_name)
        residual_vector = residual_matrix[token_index, :]

        stats = self.stats_computer.compute(residual_vector)
        norm = float(np.linalg.norm(residual_vector))
        contribution = self._compute_contribution(layer_index, residual_vector)

        return ResidualInspection(
            layer=layer_name,
            layer_index=layer_index,
            residual_vector=residual_vector.tolist(),
            shape=[int(residual_vector.shape[0])],
            statistics=stats,
            norm=norm,
            contribution=contribution,
        )

    def inspect_all_layers(
        self,
        token_index: int = 0,
    ) -> List[ResidualInspection]:
        """Inspect the residual stream at all layers."""
        return [
            self.inspect(layer_idx, token_index)
            for layer_idx in range(self.repository.num_layers)
        ]

    def get_residual_norm(
        self,
        layer_index: int,
        token_index: int = 0,
    ) -> float:
        """Get the L2 norm of the residual vector at a layer."""
        layer_name = self.repository.get_layer_name(layer_index)
        residual_matrix = self.repository.get_residual(layer_name)
        return float(np.linalg.norm(residual_matrix[token_index, :]))

    def get_residual_norms(
        self,
        token_index: int = 0,
    ) -> List[float]:
        """Get L2 norms of residual vectors across all layers."""
        return [
            self.get_residual_norm(layer_idx, token_index)
            for layer_idx in range(self.repository.num_layers)
        ]

    def get_residual_vector(
        self,
        layer_index: int,
        token_index: int = 0,
    ) -> np.ndarray:
        """Get the raw residual vector at a layer."""
        layer_name = self.repository.get_layer_name(layer_index)
        residual_matrix = self.repository.get_residual(layer_name)
        return residual_matrix[token_index, :]

    # ------------------------------------------------------------------ #
    #  Internal helpers
    # ------------------------------------------------------------------ #

    def _compute_contribution(
        self,
        layer_index: int,
        residual_vector: np.ndarray,
    ) -> float:
        """Compute the relative contribution of this layer's residual."""
        all_norms = self.get_residual_norms()
        total_norm = sum(all_norms)
        if total_norm == 0:
            return 0.0
        return float(np.linalg.norm(residual_vector) / total_norm)
