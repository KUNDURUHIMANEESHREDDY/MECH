"""
Data models for the Neuron Inspector.

These Pydantic models define the structure of data returned by the
inspector components and the Activation Viewer API.

Naming convention: inspection results use the suffix "Inspection"
to describe what the API is delivering — an inspection result.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

import numpy as np
from pydantic import BaseModel, Field


class Statistics(BaseModel):
    """Statistical summary of an activation or attention matrix."""

    max: float = Field(..., description="Maximum value in the activation")
    mean: float = Field(..., description="Mean value of the activation")
    variance: float = Field(..., description="Variance of the activation")
    sparsity: float = Field(
        ..., description="Fraction of near-zero elements (0.0 to 1.0)"
    )
    min: Optional[float] = Field(None, description="Minimum value in the activation")
    std: Optional[float] = Field(None, description="Standard deviation")
    median: Optional[float] = Field(None, description="Median value")
    l1_norm: Optional[float] = Field(None, description="L1 norm of the vector")
    l2_norm: Optional[float] = Field(None, description="L2 (Euclidean) norm")
    num_elements: Optional[int] = Field(
        None, description="Total number of elements analyzed"
    )

    model_config = {"arbitrary_types_allowed": True}


class NeuronInspection(BaseModel):
    """Inspection result for a single neuron.

    Returned by the Neuron Inspector and the GetNeuron() API endpoint.
    """

    neuron_id: str = Field(..., description="Unique identifier for the neuron")
    layer: str = Field(..., description="Layer name where the neuron resides")
    layer_index: int = Field(..., description="Index of the layer in the model")
    neuron_index: int = Field(..., description="Index of the neuron within the layer")
    activation: float = Field(..., description="Raw activation value of the neuron")
    activation_history: Optional[List[float]] = Field(
        None, description="Activation values across multiple inputs/steps"
    )
    statistics: Statistics = Field(..., description="Statistical summary")
    top_tokens: Optional[List[Dict[str, Any]]] = Field(
        None,
        description="Top tokens that maximally activate this neuron",
    )
    description: Optional[str] = Field(
        None, description="Human-readable description of neuron function"
    )

    model_config = {"arbitrary_types_allowed": True}


class AttentionInspection(BaseModel):
    """Inspection result for a single attention head.

    Returned by the Attention Inspector and the GetAttention() API endpoint.
    """

    head: int = Field(..., description="Attention head index")
    num_heads: int = Field(..., description="Total number of attention heads")
    matrix: List[List[float]] = Field(
        ..., description="Attention weight matrix (query x key)"
    )
    importance: float = Field(
        ..., description="Importance score of this head (0.0 to 1.0)"
    )
    shape: List[int] = Field(
        ..., description="Shape of the attention matrix [seq_len, seq_len]"
    )
    layer: str = Field(..., description="Layer name where the head resides")
    layer_index: int = Field(..., description="Index of the layer")
    statistics: Statistics = Field(..., description="Statistical summary of the matrix")
    top_connections: Optional[List[Dict[str, Any]]] = Field(
        None,
        description="Top attention connections (query, key, weight)",
    )

    model_config = {"arbitrary_types_allowed": True}


class ResidualInspection(BaseModel):
    """Inspection result for a residual stream.

    Returned by the Residual Stream Inspector and the GetResidual() API endpoint.
    """

    layer: str = Field(..., description="Layer name where the residual is captured")
    layer_index: int = Field(..., description="Index of the layer")
    residual_vector: List[float] = Field(
        ..., description="The residual vector at this layer"
    )
    shape: List[int] = Field(..., description="Shape of the residual vector")
    statistics: Statistics = Field(..., description="Statistical summary")
    norm: Optional[float] = Field(None, description="L2 norm of the residual vector")
    contribution: Optional[float] = Field(
        None, description="Relative contribution to final output"
    )

    model_config = {"arbitrary_types_allowed": True}


class LayerInspection(BaseModel):
    """Inspection result for a full layer.

    Returned by the GetLayer() API endpoint.

    A layer contains:
    - Attention sub-layer
    - MLP sub-layer
    - Residual stream
    - Aggregate statistics
    - Metadata
    """

    layer: str = Field(..., description="Layer name")
    layer_index: int = Field(..., description="Index of the layer")
    layer_type: str = Field(..., description="Type of layer (e.g., attention, mlp)")
    attention: Optional[AttentionInspection] = Field(
        None, description="Attention sub-layer inspection"
    )
    mlp: Optional[Dict[str, Any]] = Field(
        None, description="MLP sub-layer inspection data"
    )
    residual: Optional[ResidualInspection] = Field(
        None, description="Residual stream inspection"
    )
    neurons: Optional[List[NeuronInspection]] = Field(
        None, description="Neuron data for this layer"
    )
    statistics: Optional[Statistics] = Field(
        None, description="Aggregate statistics for the layer"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        None, description="Additional layer metadata"
    )

    model_config = {"arbitrary_types_allowed": True}


class TokenInspection(BaseModel):
    """Inspection result for a single token.

    Returned by the Token Inspector. Provides a token-centric view:
    embedding, attention, residual, and prediction.
    """

    token_index: int = Field(..., description="Index of the token in the sequence")
    token_id: int = Field(..., description="Token ID in the vocabulary")
    embedding: Optional[List[float]] = Field(
        None, description="Token embedding vector"
    )
    embedding_shape: Optional[List[int]] = Field(
        None, description="Shape of the embedding"
    )
    attention: Optional[AttentionInspection] = Field(
        None, description="Attention from this token"
    )
    residual: Optional[ResidualInspection] = Field(
        None, description="Residual stream at this token"
    )
    logits: Optional[List[float]] = Field(
        None, description="Logits at this token"
    )
    top_predictions: Optional[List[Dict[str, Any]]] = Field(
        None, description="Top predicted tokens"
    )
    statistics: Optional[Statistics] = Field(
        None, description="Statistical summary"
    )

    model_config = {"arbitrary_types_allowed": True}


class LogitInspection(BaseModel):
    """Inspection result for logits at a layer.

    Returned by the Logit Inspector. Provides layer logits and
    top predicted tokens.
    """

    layer: str = Field(..., description="Layer name")
    layer_index: int = Field(..., description="Index of the layer")
    logits: List[float] = Field(
        ..., description="Logits at this layer"
    )
    shape: List[int] = Field(..., description="Shape of the logits")
    top_tokens: List[Dict[str, Any]] = Field(
        ..., description="Top predicted tokens with probabilities"
    )
    statistics: Statistics = Field(..., description="Statistical summary")

    model_config = {"arbitrary_types_allowed": True}


class HeadRanking(BaseModel):
    """Ranking of attention heads by a specific metric.

    Returned by the Head Ranking feature.
    """

    layer: str = Field(..., description="Layer name")
    layer_index: int = Field(..., description="Index of the layer")
    metric: str = Field(
        ..., description="Metric used for ranking (entropy, importance, sparsity)"
    )
    rankings: List[Dict[str, Any]] = Field(
        ..., description="Ranked list of heads with scores"
    )

    model_config = {"arbitrary_types_allowed": True}


class ActivationSearchResult(BaseModel):
    """Result of an activation search.

    Returned by the Activation Search feature.
    """

    query: str = Field(..., description="Search query description")
    matches: List[Dict[str, Any]] = Field(
        ..., description="Matching neurons with their activations"
    )
    total_matches: int = Field(..., description="Total number of matches")

    model_config = {"arbitrary_types_allowed": True}


class VisualizationDTO(BaseModel):
    """Visualization data prepared for frontend rendering.

    Contains all data needed to render visualizations (heatmaps,
    graphs, tables, 3D views) for neurons, attention, and residuals.

    The visualization type is determined by the frontend, not the
    interpretability engine.
    """

    visualization_type: str = Field(
        ..., description="Type: 'neuron', 'attention', or 'residual'"
    )
    title: str = Field(..., description="Title for the visualization")
    data: List[List[float]] = Field(
        ..., description="2D matrix of visualization values"
    )
    x_labels: Optional[List[str]] = Field(
        None, description="Labels for the x-axis"
    )
    y_labels: Optional[List[str]] = Field(
        None, description="Labels for the y-axis"
    )
    color_scale: Optional[str] = Field(
        None, description="Color scale name (e.g., 'viridis', 'plasma')"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        None, description="Additional metadata for the visualization"
    )

    model_config = {"arbitrary_types_allowed": True}


# Backward-compatible aliases
NeuronData = NeuronInspection
AttentionData = AttentionInspection
ResidualData = ResidualInspection
LayerData = LayerInspection
HeatmapData = VisualizationDTO
