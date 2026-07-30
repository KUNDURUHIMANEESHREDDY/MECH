"""Tests for the StatisticsComputer."""

import numpy as np
import pytest

from backend.interpretability.statistics import StatisticsComputer
from backend.interpretability.models import Statistics


class TestStatisticsComputer:
    """Tests for the StatisticsComputer class."""

    def test_compute_basic(self, stats_computer, sample_activation_vector):
        """Test basic statistics computation on a 1-D vector."""
        stats = stats_computer.compute(sample_activation_vector)

        assert isinstance(stats, Statistics)
        assert stats.max == pytest.approx(1.2)
        assert stats.mean == pytest.approx(np.mean(sample_activation_vector))
        assert stats.variance == pytest.approx(np.var(sample_activation_vector))
        assert stats.min == pytest.approx(-0.8)
        assert stats.std == pytest.approx(np.std(sample_activation_vector))
        assert stats.median == pytest.approx(np.median(sample_activation_vector))
        assert stats.num_elements == 8

    def test_compute_sparsity(self, stats_computer, sample_activation_vector):
        """Test sparsity computation.

        The sample vector has 4 zeros out of 8 elements, so sparsity
        should be 0.5.
        """
        stats = stats_computer.compute(sample_activation_vector)
        assert stats.sparsity == pytest.approx(0.5)

    def test_compute_l1_l2_norms(self, stats_computer, sample_activation_vector):
        """Test L1 and L2 norm computation."""
        stats = stats_computer.compute(sample_activation_vector)
        assert stats.l1_norm == pytest.approx(np.sum(np.abs(sample_activation_vector)))
        assert stats.l2_norm == pytest.approx(np.sqrt(np.sum(sample_activation_vector ** 2)))

    def test_compute_2d_array(self, stats_computer):
        """Test statistics computation on a 2-D array."""
        data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        stats = stats_computer.compute(data)

        assert stats.num_elements == 6
        assert stats.max == pytest.approx(6.0)
        assert stats.min == pytest.approx(1.0)
        assert stats.mean == pytest.approx(3.5)

    def test_compute_scalar(self, stats_computer):
        """Test statistics computation for a single scalar."""
        stats = stats_computer.compute_scalar(3.14)

        assert stats.max == pytest.approx(3.14)
        assert stats.mean == pytest.approx(3.14)
        assert stats.variance == pytest.approx(0.0)
        assert stats.num_elements == 1

    def test_compute_empty(self, stats_computer):
        """Test statistics computation for empty input."""
        stats = stats_computer.compute(np.array([]))

        assert stats.max == 0.0
        assert stats.mean == 0.0
        assert stats.variance == 0.0
        assert stats.sparsity == 0.0
        assert stats.num_elements == 0

    def test_max_value(self, stats_computer, sample_activation_vector):
        """Test the max_value static method."""
        assert stats_computer.max_value(sample_activation_vector) == pytest.approx(1.2)

    def test_mean_value(self, stats_computer, sample_activation_vector):
        """Test the mean_value static method."""
        expected = np.mean(sample_activation_vector)
        assert stats_computer.mean_value(sample_activation_vector) == pytest.approx(expected)

    def test_variance_value(self, stats_computer, sample_activation_vector):
        """Test the variance_value static method."""
        expected = np.var(sample_activation_vector)
        assert stats_computer.variance_value(sample_activation_vector) == pytest.approx(expected)

    def test_sparsity_value(self, stats_computer, sample_activation_vector):
        """Test the sparsity_value method."""
        # 4 zeros out of 8
        assert stats_computer.sparsity_value(sample_activation_vector) == pytest.approx(0.5)

    def test_sparsity_value_custom_threshold(self, stats_computer):
        """Test sparsity with a custom threshold."""
        data = np.array([0.0, 0.01, 0.02, 0.5, 1.0])
        # With threshold 0.05: 0.0, 0.01, 0.02 are near-zero -> 3/5 = 0.6
        assert stats_computer.sparsity_value(data, threshold=0.05) == pytest.approx(0.6)

    def test_sparsity_empty(self, stats_computer):
        """Test sparsity for empty input."""
        assert stats_computer.sparsity_value(np.array([])) == 0.0

    def test_to_array_from_list(self, stats_computer):
        """Test _to_array with a list input."""
        arr = stats_computer._to_array([1.0, 2.0, 3.0])
        assert isinstance(arr, np.ndarray)
        assert arr.dtype == np.float64
        assert len(arr) == 3

    def test_to_array_from_float(self, stats_computer):
        """Test _to_array with a scalar input."""
        arr = stats_computer._to_array(3.14)
        assert isinstance(arr, np.ndarray)
        assert len(arr) == 1

    def test_to_array_invalid_type(self, stats_computer):
        """Test _to_array with an invalid type."""
        with pytest.raises(TypeError):
            stats_computer._to_array("invalid")

    def test_custom_sparse_threshold(self):
        """Test StatisticsComputer with a custom sparse threshold."""
        computer = StatisticsComputer(sparse_threshold=0.5)
        data = np.array([0.1, 0.2, 0.6, 0.7])
        # With threshold 0.5: 0.1, 0.2 are near-zero -> 2/4 = 0.5
        assert computer.sparsity_value(data) == pytest.approx(0.5)

    def test_compute_with_list_input(self, stats_computer):
        """Test compute with a plain list."""
        stats = stats_computer.compute([1.0, 2.0, 3.0, 4.0])
        assert stats.max == pytest.approx(4.0)
        assert stats.mean == pytest.approx(2.5)
        assert stats.num_elements == 4
