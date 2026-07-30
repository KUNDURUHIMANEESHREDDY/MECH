"""
Statistics computation for neural network activations.

Computes max, mean, variance, sparsity, and additional statistics
from activation vectors and attention matrices.
"""

from __future__ import annotations

from typing import Optional, Union

import numpy as np

from .models import Statistics


class StatisticsComputer:
    """Computes statistical summaries of activation data.

    Supports both 1-D activation vectors and 2-D attention matrices.
    """

    # Threshold below which an element is considered "sparse" (near-zero).
    DEFAULT_SPARSE_THRESHOLD: float = 1e-4

    def __init__(self, sparse_threshold: float = DEFAULT_SPARSE_THRESHOLD):
        self.sparse_threshold = sparse_threshold

    # ------------------------------------------------------------------ #
    #  Public API
    # ------------------------------------------------------------------ #

    def compute(self, data: Union[np.ndarray, list, float]) -> Statistics:
        """Compute full statistics for a 1-D or 2-D array.

        Parameters
        ----------
        data : np.ndarray, list, or float
            The activation data to analyze.

        Returns
        -------
        Statistics
            A populated :class:`~backend.interpretability.models.Statistics`
            object.
        """
        arr = self._to_array(data)
        flat = arr.flatten()

        if flat.size == 0:
            return self._empty_statistics()

        max_val = float(np.max(flat))
        mean_val = float(np.mean(flat))
        var_val = float(np.var(flat))
        sparsity_val = self._compute_sparsity(flat)

        stats = Statistics(
            max=max_val,
            mean=mean_val,
            variance=var_val,
            sparsity=sparsity_val,
            min=float(np.min(flat)),
            std=float(np.std(flat)),
            median=float(np.median(flat)),
            l1_norm=float(np.sum(np.abs(flat))),
            l2_norm=float(np.sqrt(np.sum(flat ** 2))),
            num_elements=int(flat.size),
        )
        return stats

    def compute_scalar(self, value: float) -> Statistics:
        """Compute statistics for a single scalar activation value.

        Useful when inspecting a single neuron's activation.
        """
        arr = np.array([value], dtype=np.float64)
        return self.compute(arr)

    # ------------------------------------------------------------------ #
    #  Individual statistic helpers (exposed for granular use / testing)
    # ------------------------------------------------------------------ #

    @staticmethod
    def max_value(data: Union[np.ndarray, list]) -> float:
        """Return the maximum value in *data*."""
        arr = np.asarray(data, dtype=np.float64)
        return float(np.max(arr)) if arr.size else 0.0

    @staticmethod
    def mean_value(data: Union[np.ndarray, list]) -> float:
        """Return the arithmetic mean of *data*."""
        arr = np.asarray(data, dtype=np.float64)
        return float(np.mean(arr)) if arr.size else 0.0

    @staticmethod
    def variance_value(data: Union[np.ndarray, list]) -> float:
        """Return the population variance of *data*."""
        arr = np.asarray(data, dtype=np.float64)
        return float(np.var(arr)) if arr.size else 0.0

    def sparsity_value(
        self, data: Union[np.ndarray, list], threshold: Optional[float] = None
    ) -> float:
        """Return the sparsity of *data*.

        Sparsity is defined as the fraction of elements whose absolute
        value is below *threshold* (default ``self.sparse_threshold``).

        Returns
        -------
        float
            A value in ``[0.0, 1.0]`` where ``1.0`` means every element
            is near-zero.
        """
        arr = np.asarray(data, dtype=np.float64)
        if arr.size == 0:
            return 0.0
        thr = threshold if threshold is not None else self.sparse_threshold
        near_zero = np.sum(np.abs(arr) < thr)
        return float(near_zero / arr.size)

    # ------------------------------------------------------------------ #
    #  Internal helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _to_array(data: Union[np.ndarray, list, float]) -> np.ndarray:
        """Convert various input types to a float64 numpy array."""
        if isinstance(data, np.ndarray):
            return data.astype(np.float64)
        if isinstance(data, (list, tuple)):
            return np.array(data, dtype=np.float64)
        if isinstance(data, (int, float)):
            return np.array([float(data)], dtype=np.float64)
        raise TypeError(f"Unsupported data type for statistics: {type(data)}")

    def _compute_sparsity(self, flat: np.ndarray) -> float:
        """Compute sparsity on a flattened array."""
        if flat.size == 0:
            return 0.0
        near_zero = np.sum(np.abs(flat) < self.sparse_threshold)
        return float(near_zero / flat.size)

    @staticmethod
    def _empty_statistics() -> Statistics:
        """Return a Statistics object filled with zeros for empty input."""
        return Statistics(
            max=0.0,
            mean=0.0,
            variance=0.0,
            sparsity=0.0,
            min=0.0,
            std=0.0,
            median=0.0,
            l1_norm=0.0,
            l2_norm=0.0,
            num_elements=0,
        )
