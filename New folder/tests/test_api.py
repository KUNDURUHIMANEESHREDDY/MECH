"""Tests for the Activation Viewer API."""

import pytest
from fastapi.testclient import TestClient

from backend.interpretability.api import app


@pytest.fixture
def client():
    """Create a TestClient for the FastAPI app."""
    return TestClient(app)


class TestAPIRoot:
    """Tests for the root and health endpoints."""

    def test_root(self, client):
        """Test the root endpoint."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "Neuron Inspector API"
        assert "endpoints" in data

    def test_health(self, client):
        """Test the health check endpoint."""
        response = client.get("/api/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"


class TestGetNeuron:
    """Tests for the GetNeuron() API endpoint."""

    def test_get_neuron(self, client):
        """Test getting a single neuron."""
        response = client.get(
            "/api/neuron?layer_index=0&neuron_index=0&token_index=0"
        )
        assert response.status_code == 200
        data = response.json()
        assert data["neuron_id"] == "layer_0.neuron_0"
        assert data["layer"] == "layer_0"
        assert data["layer_index"] == 0
        assert data["neuron_index"] == 0
        assert "activation" in data
        assert "statistics" in data
        assert "activation_history" in data

    def test_get_neuron_different_layer(self, client):
        """Test getting a neuron from a different layer."""
        response = client.get(
            "/api/neuron?layer_index=3&neuron_index=5&token_index=0"
        )
        assert response.status_code == 200
        data = response.json()
        assert data["layer"] == "layer_3"
        assert data["neuron_index"] == 5

    def test_get_neuron_invalid_layer(self, client):
        """Test getting a neuron with an invalid layer index."""
        response = client.get(
            "/api/neuron?layer_index=999&neuron_index=0"
        )
        assert response.status_code == 404

    def test_get_neuron_batch(self, client):
        """Test batch neuron inspection."""
        response = client.get(
            "/api/neuron/batch?layer_index=0&neuron_indices=0,1,2&token_index=0"
        )
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 3
        assert data[0]["neuron_index"] == 0
        assert data[1]["neuron_index"] == 1
        assert data[2]["neuron_index"] == 2

    def test_get_top_neurons(self, client):
        """Test getting top neurons in a layer."""
        response = client.get(
            "/api/neuron/top?layer_index=0&top_k=5&token_index=0"
        )
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 5


class TestGetAttention:
    """Tests for the GetAttention() API endpoint."""

    def test_get_attention(self, client):
        """Test getting a single attention head."""
        response = client.get(
            "/api/attention?layer_index=0&head_index=0"
        )
        assert response.status_code == 200
        data = response.json()
        assert data["head"] == 0
        assert data["layer"] == "layer_0"
        assert "matrix" in data
        assert "importance" in data
        assert "shape" in data
        assert "statistics" in data

    def test_get_attention_with_token(self, client):
        """Test getting attention for a specific token."""
        response = client.get(
            "/api/attention?layer_index=0&head_index=0&token_index=3"
        )
        assert response.status_code == 200
        data = response.json()
        assert data["shape"] == [1, 16]  # seq_len

    def test_get_attention_all_heads(self, client):
        """Test getting all attention heads in a layer."""
        response = client.get(
            "/api/attention/all?layer_index=0"
        )
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 12  # default model has 12 heads

    def test_get_attention_importance(self, client):
        """Test getting attention importance scores."""
        response = client.get(
            "/api/attention/importance?layer_index=0"
        )
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 12
        assert all(0.0 <= s <= 1.0 for s in data)

    def test_get_attention_invalid_layer(self, client):
        """Test getting attention with an invalid layer index."""
        response = client.get(
            "/api/attention?layer_index=999&head_index=0"
        )
        assert response.status_code == 404


class TestGetResidual:
    """Tests for the GetResidual() API endpoint."""

    def test_get_residual(self, client):
        """Test getting a residual stream."""
        response = client.get(
            "/api/residual?layer_index=0&token_index=0"
        )
        assert response.status_code == 200
        data = response.json()
        assert data["layer"] == "layer_0"
        assert data["layer_index"] == 0
        assert "residual_vector" in data
        assert "statistics" in data
        assert "norm" in data
        assert "contribution" in data

    def test_get_residual_all_layers(self, client):
        """Test getting residuals at all layers."""
        response = client.get(
            "/api/residual/all?token_index=0"
        )
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 12  # default model has 12 layers

    def test_get_residual_norms(self, client):
        """Test getting residual norms."""
        response = client.get(
            "/api/residual/norms?token_index=0"
        )
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 12
        assert all(n > 0 for n in data)

    def test_get_residual_invalid_layer(self, client):
        """Test getting residual with an invalid layer index."""
        response = client.get(
            "/api/residual?layer_index=999&token_index=0"
        )
        assert response.status_code == 404


class TestGetLayer:
    """Tests for the GetLayer() API endpoint."""

    def test_get_layer_basic(self, client):
        """Test getting a layer with basic info."""
        response = client.get(
            "/api/layer?layer_index=0"
        )
        assert response.status_code == 200
        data = response.json()
        assert data["layer"] == "layer_0"
        assert data["layer_index"] == 0
        assert data["layer_type"] is not None
        assert data["statistics"] is not None
        assert data["neurons"] is None  # not included by default
        assert data["attention_heads"] is None  # not included by default
        assert data["residual"] is not None  # included by default

    def test_get_layer_with_neurons(self, client):
        """Test getting a layer with neuron data."""
        response = client.get(
            "/api/layer?layer_index=0&include_neurons=true&top_k_neurons=5"
        )
        assert response.status_code == 200
        data = response.json()
        assert data["neurons"] is not None
        assert len(data["neurons"]) == 5

    def test_get_layer_with_attention(self, client):
        """Test getting a layer with attention data."""
        response = client.get(
            "/api/layer?layer_index=0&include_attention=true"
        )
        assert response.status_code == 200
        data = response.json()
        assert data["attention_heads"] is not None
        assert len(data["attention_heads"]) == 12

    def test_get_layer_invalid(self, client):
        """Test getting a layer with an invalid index."""
        response = client.get(
            "/api/layer?layer_index=999"
        )
        assert response.status_code == 404


class TestHeatmapEndpoints:
    """Tests for the heatmap API endpoints."""

    def test_get_neuron_heatmap(self, client):
        """Test getting a neuron heatmap."""
        response = client.get(
            "/api/heatmap/neuron?layer_index=0&top_k=5"
        )
        assert response.status_code == 200
        data = response.json()
        assert data["heatmap_type"] == "neuron"
        assert len(data["data"]) == 5

    def test_get_attention_heatmap(self, client):
        """Test getting an attention heatmap."""
        response = client.get(
            "/api/heatmap/attention?layer_index=0&head_index=0"
        )
        assert response.status_code == 200
        data = response.json()
        assert data["heatmap_type"] == "attention"

    def test_get_residual_heatmap(self, client):
        """Test getting a residual heatmap."""
        response = client.get(
            "/api/heatmap/residual?token_index=0&num_dims=16"
        )
        assert response.status_code == 200
        data = response.json()
        assert data["heatmap_type"] == "residual"
        assert len(data["data"]) == 12  # num_layers

    def test_get_all_heatmaps(self, client):
        """Test getting all heatmaps."""
        response = client.get(
            "/api/heatmaps?layer_index=0&head_index=0&token_index=0"
        )
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 3


class TestUtilityEndpoints:
    """Tests for utility API endpoints."""

    def test_list_layers(self, client):
        """Test listing all layers."""
        response = client.get("/api/layers")
        assert response.status_code == 200
        data = response.json()
        assert "layers" in data
        assert len(data["layers"]) == 12
        assert data["num_layers"] == 12
        assert data["num_heads"] == 12
        assert data["hidden_dim"] == 768

    def test_get_stats(self, client):
        """Test getting statistics for a layer."""
        response = client.get(
            "/api/stats?layer_index=0"
        )
        assert response.status_code == 200
        data = response.json()
        assert "max" in data
        assert "mean" in data
        assert "variance" in data
        assert "sparsity" in data

    def test_get_stats_invalid_layer(self, client):
        """Test getting stats with an invalid layer index."""
        response = client.get(
            "/api/stats?layer_index=999"
        )
        assert response.status_code == 404
