"""
Activation Viewer API (v1).

Versioned REST API with session support for neural network
interpretability.

Architecture:

    Runtime

    ↓

    ActivationRepository

    ↓

    SessionManager

    ↓

    Inspectors

    ↓

    REST API (v1)

    ↓

    Frontend

API Structure:

    /api/v1/
        neurons/
        attention/
        residual/
        layers/
        tokens/
        logits/
        visualizations/
        sessions/
        experiments/

All inspection endpoints require a session_id, which is obtained
from the sessions/ endpoint.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Dict, List, Optional

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.routing import APIRouter

from .models import (
    ActivationSearchResult,
    AttentionInspection,
    HeadRanking,
    LayerInspection,
    LogitInspection,
    NeuronInspection,
    ResidualInspection,
    Statistics,
    TokenInspection,
    VisualizationDTO,
)
from .repository import ActivationRepository
from .mock_runtime import MockRuntime
from .statistics import StatisticsComputer
from .neuron_inspector import NeuronInspector
from .attention_inspector import AttentionInspector
from .residual_inspector import ResidualInspector
from .layer_inspector import LayerInspector
from .token_inspector import TokenInspector
from .logit_inspector import LogitInspector
from .heatmap import VisualizationGenerator


# ====================================================================== #
#  Session Management
# ====================================================================== #

class SessionInfo:
    """Information about an inspection session."""

    def __init__(self, session_id: str, runtime: MockRuntime):
        self.session_id = session_id
        self.created_at = datetime.now().isoformat()
        self.repository = ActivationRepository(runtime)
        self.stats_computer = StatisticsComputer()
        self.neuron_inspector = NeuronInspector(
            repository=self.repository, stats_computer=self.stats_computer
        )
        self.attention_inspector = AttentionInspector(
            repository=self.repository, stats_computer=self.stats_computer
        )
        self.residual_inspector = ResidualInspector(
            repository=self.repository, stats_computer=self.stats_computer
        )
        self.layer_inspector = LayerInspector(
            repository=self.repository, stats_computer=self.stats_computer
        )
        self.token_inspector = TokenInspector(
            repository=self.repository, stats_computer=self.stats_computer
        )
        self.logit_inspector = LogitInspector(
            repository=self.repository, stats_computer=self.stats_computer
        )
        self.visualization_generator = VisualizationGenerator(
            repository=self.repository
        )


class SessionManager:
    """Manages inspection sessions."""

    def __init__(self):
        self._sessions: Dict[str, SessionInfo] = {}

    def create_session(
        self,
        num_layers: int = 12,
        num_heads: int = 12,
        hidden_dim: int = 768,
        seq_len: int = 16,
        seed: int = 42,
    ) -> SessionInfo:
        """Create a new inspection session."""
        session_id = str(uuid.uuid4())
        runtime = MockRuntime(
            num_layers=num_layers,
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            seq_len=seq_len,
            seed=seed,
        )
        session = SessionInfo(session_id, runtime)
        self._sessions[session_id] = session
        return session

    def get_session(self, session_id: str) -> SessionInfo:
        """Get a session by ID."""
        if session_id not in self._sessions:
            raise HTTPException(
                status_code=404,
                detail=f"Session {session_id} not found",
            )
        return self._sessions[session_id]

    def delete_session(self, session_id: str) -> None:
        """Delete a session."""
        if session_id in self._sessions:
            del self._sessions[session_id]

    def list_sessions(self) -> List[Dict]:
        """List all sessions."""
        return [
            {
                "session_id": s.session_id,
                "created_at": s.created_at,
                "num_layers": s.repository.num_layers,
                "num_heads": s.repository.num_heads,
                "hidden_dim": s.repository.hidden_dim,
                "seq_len": s.repository.seq_len,
            }
            for s in self._sessions.values()
        ]


# Global session manager
session_manager = SessionManager()


def get_session(session_id: str = Query(..., description="Session ID")) -> SessionInfo:
    """Dependency to get a session."""
    return session_manager.get_session(session_id)


# ====================================================================== #
#  API v1 Router
# ====================================================================== #

v1_router = APIRouter(prefix="/api/v1")


# ------------------------------------------------------------------ #
#  Session endpoints
# ------------------------------------------------------------------ #

@v1_router.post("/sessions", response_model=Dict)
def create_session(
    num_layers: int = Query(12, ge=1, description="Number of layers"),
    num_heads: int = Query(12, ge=1, description="Number of attention heads"),
    hidden_dim: int = Query(768, ge=1, description="Hidden dimension"),
    seq_len: int = Query(16, ge=1, description="Sequence length"),
    seed: int = Query(42, description="Random seed"),
):
    """Create a new inspection session.

    Returns a session_id that must be used for all subsequent
    inspection endpoints.
    """
    session = session_manager.create_session(
        num_layers=num_layers,
        num_heads=num_heads,
        hidden_dim=hidden_dim,
        seq_len=seq_len,
        seed=seed,
    )
    return {
        "session_id": session.session_id,
        "created_at": session.created_at,
        "num_layers": session.repository.num_layers,
        "num_heads": session.repository.num_heads,
        "hidden_dim": session.repository.hidden_dim,
        "seq_len": session.repository.seq_len,
    }


@v1_router.get("/sessions", response_model=List[Dict])
def list_sessions():
    """List all active sessions."""
    return session_manager.list_sessions()


@v1_router.get("/sessions/{session_id}", response_model=Dict)
def get_session_info(session_id: str):
    """Get information about a session."""
    session = session_manager.get_session(session_id)
    return {
        "session_id": session.session_id,
        "created_at": session.created_at,
        "num_layers": session.repository.num_layers,
        "num_heads": session.repository.num_heads,
        "hidden_dim": session.repository.hidden_dim,
        "seq_len": session.repository.seq_len,
        "cache_info": session.repository.cache_info(),
    }


@v1_router.delete("/sessions/{session_id}")
def delete_session(session_id: str):
    """Delete a session."""
    session_manager.delete_session(session_id)
    return {"status": "deleted", "session_id": session_id}


# ------------------------------------------------------------------ #
#  Neuron endpoints
# ------------------------------------------------------------------ #

@v1_router.get("/neurons", response_model=NeuronInspection)
def get_neuron(
    session: SessionInfo = Depends(get_session),
    layer_index: int = Query(..., ge=0, description="Layer index"),
    neuron_index: int = Query(..., ge=0, description="Neuron index"),
    token_index: int = Query(0, ge=0, description="Token position"),
):
    """GetNeuron() - Inspect a single neuron."""
    try:
        return session.neuron_inspector.inspect(layer_index, neuron_index, token_index)
    except IndexError as e:
        raise HTTPException(status_code=404, detail=str(e))


@v1_router.get("/neurons/batch", response_model=List[NeuronInspection])
def get_neurons_batch(
    session: SessionInfo = Depends(get_session),
    layer_index: int = Query(..., ge=0, description="Layer index"),
    neuron_indices: str = Query(..., description="Comma-separated neuron indices"),
    token_index: int = Query(0, ge=0, description="Token position"),
):
    """GetNeuron() - Batch inspect multiple neurons."""
    indices = [int(x) for x in neuron_indices.split(",")]
    try:
        return session.neuron_inspector.inspect_batch(layer_index, indices, token_index)
    except IndexError as e:
        raise HTTPException(status_code=404, detail=str(e))


@v1_router.get("/neurons/top", response_model=List[NeuronInspection])
def get_top_neurons(
    session: SessionInfo = Depends(get_session),
    layer_index: int = Query(..., ge=0, description="Layer index"),
    top_k: int = Query(10, ge=1, le=100, description="Number of top neurons"),
    token_index: int = Query(0, ge=0, description="Token position"),
):
    """GetNeuron() - Inspect top-k most active neurons in a layer."""
    try:
        return session.neuron_inspector.inspect_layer(layer_index, top_k, token_index)
    except IndexError as e:
        raise HTTPException(status_code=404, detail=str(e))


@v1_router.get("/neurons/search", response_model=ActivationSearchResult)
def search_neurons(
    session: SessionInfo = Depends(get_session),
    layer_index: int = Query(..., ge=0, description="Layer index"),
    threshold: float = Query(5.0, description="Minimum absolute activation"),
    top_k: int = Query(20, ge=1, le=100, description="Max results"),
):
    """Search for neurons with activation above a threshold."""
    try:
        matches = session.neuron_inspector.search(layer_index, threshold, top_k)
        return ActivationSearchResult(
            query=f"activation > {threshold} in layer {layer_index}",
            matches=matches,
            total_matches=len(matches),
        )
    except IndexError as e:
        raise HTTPException(status_code=404, detail=str(e))


# ------------------------------------------------------------------ #
#  Attention endpoints
# ------------------------------------------------------------------ #

@v1_router.get("/attention", response_model=AttentionInspection)
def get_attention(
    session: SessionInfo = Depends(get_session),
    layer_index: int = Query(..., ge=0, description="Layer index"),
    head_index: int = Query(..., ge=0, description="Head index"),
    token_index: Optional[int] = Query(None, ge=0, description="Query token"),
):
    """GetAttention() - Inspect a single attention head."""
    try:
        return session.attention_inspector.inspect(layer_index, head_index, token_index)
    except IndexError as e:
        raise HTTPException(status_code=404, detail=str(e))


@v1_router.get("/attention/all", response_model=List[AttentionInspection])
def get_all_attention(
    session: SessionInfo = Depends(get_session),
    layer_index: int = Query(..., ge=0, description="Layer index"),
    token_index: Optional[int] = Query(None, ge=0, description="Query token"),
):
    """GetAttention() - Inspect all attention heads in a layer."""
    try:
        return session.attention_inspector.inspect_all_heads(layer_index, token_index)
    except IndexError as e:
        raise HTTPException(status_code=404, detail=str(e))


@v1_router.get("/attention/importance", response_model=List[float])
def get_attention_importance(
    session: SessionInfo = Depends(get_session),
    layer_index: int = Query(..., ge=0, description="Layer index"),
):
    """GetAttention() - Get importance scores for all heads."""
    try:
        return session.attention_inspector.get_layer_importance(layer_index)
    except IndexError as e:
        raise HTTPException(status_code=404, detail=str(e))


@v1_router.get("/attention/rank", response_model=HeadRanking)
def rank_heads(
    session: SessionInfo = Depends(get_session),
    layer_index: int = Query(..., ge=0, description="Layer index"),
    metric: str = Query("importance", description="Metric: entropy, importance, sparsity"),
):
    """Rank attention heads by a specific metric."""
    try:
        return session.attention_inspector.rank_heads(layer_index, metric)
    except (IndexError, ValueError) as e:
        raise HTTPException(status_code=404, detail=str(e))


# ------------------------------------------------------------------ #
#  Residual endpoints
# ------------------------------------------------------------------ #

@v1_router.get("/residual", response_model=ResidualInspection)
def get_residual(
    session: SessionInfo = Depends(get_session),
    layer_index: int = Query(..., ge=0, description="Layer index"),
    token_index: int = Query(0, ge=0, description="Token position"),
):
    """GetResidual() - Inspect the residual stream at a layer."""
    try:
        return session.residual_inspector.inspect(layer_index, token_index)
    except IndexError as e:
        raise HTTPException(status_code=404, detail=str(e))


@v1_router.get("/residual/all", response_model=List[ResidualInspection])
def get_all_residuals(
    session: SessionInfo = Depends(get_session),
    token_index: int = Query(0, ge=0, description="Token position"),
):
    """GetResidual() - Inspect residual streams at all layers."""
    return session.residual_inspector.inspect_all_layers(token_index)


@v1_router.get("/residual/norms", response_model=List[float])
def get_residual_norms(
    session: SessionInfo = Depends(get_session),
    token_index: int = Query(0, ge=0, description="Token position"),
):
    """GetResidual() - Get L2 norms of residual vectors across all layers."""
    return session.residual_inspector.get_residual_norms(token_index)


# ------------------------------------------------------------------ #
#  Layer endpoints
# ------------------------------------------------------------------ #

@v1_router.get("/layers", response_model=LayerInspection)
def get_layer(
    session: SessionInfo = Depends(get_session),
    layer_index: int = Query(..., ge=0, description="Layer index"),
    include_neurons: bool = Query(False, description="Include neuron data"),
    include_attention: bool = Query(False, description="Include attention data"),
    include_residual: bool = Query(True, description="Include residual data"),
    top_k_neurons: int = Query(10, ge=1, le=100, description="Top neurons"),
    token_index: int = Query(0, ge=0, description="Token position"),
):
    """GetLayer() - Inspect a full layer (Attention, MLP, Residual, Statistics)."""
    try:
        return session.layer_inspector.inspect(
            layer_index,
            include_neurons=include_neurons,
            include_attention=include_attention,
            include_residual=include_residual,
            top_k_neurons=top_k_neurons,
            token_index=token_index,
        )
    except IndexError as e:
        raise HTTPException(status_code=404, detail=str(e))


@v1_router.get("/layers/all", response_model=List[LayerInspection])
def get_all_layers(
    session: SessionInfo = Depends(get_session),
    include_neurons: bool = Query(False, description="Include neuron data"),
    include_attention: bool = Query(False, description="Include attention data"),
    include_residual: bool = Query(True, description="Include residual data"),
    top_k_neurons: int = Query(10, ge=1, le=100, description="Top neurons"),
    token_index: int = Query(0, ge=0, description="Token position"),
):
    """GetLayer() - Inspect all layers."""
    return session.layer_inspector.inspect_all_layers(
        include_neurons=include_neurons,
        include_attention=include_attention,
        include_residual=include_residual,
        top_k_neurons=top_k_neurons,
        token_index=token_index,
    )


# ------------------------------------------------------------------ #
#  Token endpoints
# ------------------------------------------------------------------ #

@v1_router.get("/tokens", response_model=TokenInspection)
def get_token(
    session: SessionInfo = Depends(get_session),
    token_index: int = Query(..., ge=0, description="Token index"),
    layer_index: int = Query(0, ge=0, description="Layer index"),
    include_embedding: bool = Query(True, description="Include embedding"),
    include_attention: bool = Query(True, description="Include attention"),
    include_residual: bool = Query(True, description="Include residual"),
    include_logits: bool = Query(True, description="Include logits"),
    top_k_predictions: int = Query(5, ge=1, le=20, description="Top predictions"),
):
    """Inspect a single token (embedding, attention, residual, logits)."""
    try:
        return session.token_inspector.inspect(
            token_index,
            layer_index=layer_index,
            include_embedding=include_embedding,
            include_attention=include_attention,
            include_residual=include_residual,
            include_logits=include_logits,
            top_k_predictions=top_k_predictions,
        )
    except IndexError as e:
        raise HTTPException(status_code=404, detail=str(e))


@v1_router.get("/tokens/all", response_model=List[TokenInspection])
def get_all_tokens(
    session: SessionInfo = Depends(get_session),
    layer_index: int = Query(0, ge=0, description="Layer index"),
):
    """Inspect all tokens in the sequence."""
    return session.token_inspector.inspect_all_tokens(layer_index)


# ------------------------------------------------------------------ #
#  Logit endpoints
# ------------------------------------------------------------------ #

@v1_router.get("/logits", response_model=LogitInspection)
def get_logits(
    session: SessionInfo = Depends(get_session),
    layer_index: int = Query(..., ge=0, description="Layer index"),
    token_index: int = Query(0, ge=0, description="Token position"),
    top_k: int = Query(5, ge=1, le=20, description="Top predictions"),
):
    """GetLogits() - Inspect logits at a layer."""
    try:
        return session.logit_inspector.inspect(layer_index, token_index, top_k)
    except IndexError as e:
        raise HTTPException(status_code=404, detail=str(e))


@v1_router.get("/logits/all", response_model=List[LogitInspection])
def get_all_logits(
    session: SessionInfo = Depends(get_session),
    token_index: int = Query(0, ge=0, description="Token position"),
    top_k: int = Query(5, ge=1, le=20, description="Top predictions"),
):
    """GetLogits() - Inspect logits at all layers."""
    return session.logit_inspector.inspect_all_layers(token_index, top_k)


# ------------------------------------------------------------------ #
#  Visualization endpoints
# ------------------------------------------------------------------ #

@v1_router.get("/visualizations/neuron", response_model=VisualizationDTO)
def get_neuron_visualization(
    session: SessionInfo = Depends(get_session),
    layer_index: int = Query(..., ge=0, description="Layer index"),
    top_k: int = Query(20, ge=1, le=200, description="Top neurons"),
    token_indices: Optional[str] = Query(None, description="Comma-separated token indices"),
):
    """Get visualization data for neuron activations."""
    tokens = None
    if token_indices:
        tokens = [int(x) for x in token_indices.split(",")]
    try:
        return session.visualization_generator.prepare_neuron_visualization(
            layer_index, top_k, tokens
        )
    except IndexError as e:
        raise HTTPException(status_code=404, detail=str(e))


@v1_router.get("/visualizations/attention", response_model=VisualizationDTO)
def get_attention_visualization(
    session: SessionInfo = Depends(get_session),
    layer_index: int = Query(..., ge=0, description="Layer index"),
    head_index: int = Query(..., ge=0, description="Head index"),
    token_index: Optional[int] = Query(None, ge=0, description="Query token"),
):
    """Get visualization data for an attention matrix."""
    try:
        return session.visualization_generator.prepare_attention_visualization(
            layer_index, head_index, token_index
        )
    except IndexError as e:
        raise HTTPException(status_code=404, detail=str(e))


@v1_router.get("/visualizations/residual", response_model=VisualizationDTO)
def get_residual_visualization(
    session: SessionInfo = Depends(get_session),
    token_index: int = Query(0, ge=0, description="Token position"),
    num_dims: int = Query(64, ge=1, le=1024, description="Dimensions to show"),
):
    """Get visualization data for the residual stream."""
    return session.visualization_generator.prepare_residual_visualization(
        token_index, num_dims
    )


@v1_router.get("/visualizations/all", response_model=List[VisualizationDTO])
def get_all_visualizations(
    session: SessionInfo = Depends(get_session),
    layer_index: int = Query(0, ge=0, description="Layer index"),
    head_index: int = Query(0, ge=0, description="Head index"),
    token_index: int = Query(0, ge=0, description="Token position"),
):
    """Get all visualization types."""
    heatmaps = session.visualization_generator.prepare_all_visualizations(
        layer_index, head_index, token_index
    )
    return list(heatmaps.values())


# ------------------------------------------------------------------ #
#  Utility endpoints
# ------------------------------------------------------------------ #

@v1_router.get("/layers/list", response_model=Dict)
def list_layers_v1(
    session: SessionInfo = Depends(get_session),
):
    """List all available layers with their indices."""
    return {
        "layers": [
            {"index": i, "name": name}
            for i, name in enumerate(session.repository.layer_names)
        ],
        "num_layers": session.repository.num_layers,
        "num_heads": session.repository.num_heads,
        "hidden_dim": session.repository.hidden_dim,
        "seq_len": session.repository.seq_len,
    }


@v1_router.get("/stats", response_model=Statistics)
def get_statistics(
    session: SessionInfo = Depends(get_session),
    layer_index: int = Query(..., ge=0, description="Layer index"),
):
    """Get aggregate statistics for a layer's activations."""
    try:
        layer_name = session.repository.get_layer_name(layer_index)
        activations = session.repository.get_activations(layer_name)
        return session.stats_computer.compute(activations)
    except IndexError as e:
        raise HTTPException(status_code=404, detail=str(e))


# ====================================================================== #
#  Main FastAPI App
# ====================================================================== #

app = FastAPI(
    title="Neuron Inspector API",
    description="Activation Viewer API for neural network interpretability (v1)",
    version="2.0.0",
)

# Include v1 router
app.include_router(v1_router)


# Root endpoint
@app.get("/")
def root():
    """Root endpoint with API documentation links."""
    return {
        "name": "Neuron Inspector API",
        "version": "2.0.0",
        "description": "Activation Viewer API for neural network interpretability",
        "api_version": "v1",
        "endpoints": {
            "Sessions": "/api/v1/sessions",
            "Neurons": "/api/v1/neurons",
            "Attention": "/api/v1/attention",
            "Residual": "/api/v1/residual",
            "Layers": "/api/v1/layers",
            "Tokens": "/api/v1/tokens",
            "Logits": "/api/v1/logits",
            "Visualizations": "/api/v1/visualizations",
            "Health": "/api/v1/health",
            "Docs": "/docs",
            "Redoc": "/redoc",
        },
    }


@app.get("/api/v1/health")
def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "version": "2.0.0",
        "api_version": "v1",
        "active_sessions": len(session_manager._sessions),
    }


# ====================================================================== #
#  Legacy /api router (session-less, stateless default model)
# ====================================================================== #

# Default session shared by all legacy endpoints
_default_session: SessionInfo = None


def _get_default_session() -> SessionInfo:
    """Return (and lazily create) the default stateless session."""
    global _default_session
    if _default_session is None:
        _default_session = session_manager.create_session()
    return _default_session


legacy_router = APIRouter(prefix="/api")


@legacy_router.get("/health")
def legacy_health():
    return {"status": "healthy"}


@legacy_router.get("/neuron", response_model=NeuronInspection)
def legacy_get_neuron(
    layer_index: int = Query(..., ge=0),
    neuron_index: int = Query(..., ge=0),
    token_index: int = Query(0, ge=0),
):
    session = _get_default_session()
    try:
        return session.neuron_inspector.inspect(layer_index, neuron_index, token_index)
    except IndexError as e:
        raise HTTPException(status_code=404, detail=str(e))


@legacy_router.get("/neuron/batch", response_model=List[NeuronInspection])
def legacy_get_neuron_batch(
    layer_index: int = Query(..., ge=0),
    neuron_indices: str = Query(...),
    token_index: int = Query(0, ge=0),
):
    session = _get_default_session()
    indices = [int(x) for x in neuron_indices.split(",")]
    try:
        return session.neuron_inspector.inspect_batch(layer_index, indices, token_index)
    except IndexError as e:
        raise HTTPException(status_code=404, detail=str(e))


@legacy_router.get("/neuron/top", response_model=List[NeuronInspection])
def legacy_get_top_neurons(
    layer_index: int = Query(..., ge=0),
    top_k: int = Query(10, ge=1, le=100),
    token_index: int = Query(0, ge=0),
):
    session = _get_default_session()
    try:
        return session.neuron_inspector.inspect_layer(layer_index, top_k, token_index)
    except IndexError as e:
        raise HTTPException(status_code=404, detail=str(e))


@legacy_router.get("/attention", response_model=AttentionInspection)
def legacy_get_attention(
    layer_index: int = Query(..., ge=0),
    head_index: int = Query(..., ge=0),
    token_index: Optional[int] = Query(None, ge=0),
):
    session = _get_default_session()
    try:
        return session.attention_inspector.inspect(layer_index, head_index, token_index)
    except IndexError as e:
        raise HTTPException(status_code=404, detail=str(e))


@legacy_router.get("/attention/all", response_model=List[AttentionInspection])
def legacy_get_all_attention(
    layer_index: int = Query(..., ge=0),
    token_index: Optional[int] = Query(None, ge=0),
):
    session = _get_default_session()
    try:
        return session.attention_inspector.inspect_all_heads(layer_index, token_index)
    except IndexError as e:
        raise HTTPException(status_code=404, detail=str(e))


@legacy_router.get("/attention/importance", response_model=List[float])
def legacy_get_attention_importance(
    layer_index: int = Query(..., ge=0),
):
    session = _get_default_session()
    try:
        return session.attention_inspector.get_layer_importance(layer_index)
    except IndexError as e:
        raise HTTPException(status_code=404, detail=str(e))


@legacy_router.get("/residual", response_model=ResidualInspection)
def legacy_get_residual(
    layer_index: int = Query(..., ge=0),
    token_index: int = Query(0, ge=0),
):
    session = _get_default_session()
    try:
        return session.residual_inspector.inspect(layer_index, token_index)
    except IndexError as e:
        raise HTTPException(status_code=404, detail=str(e))


@legacy_router.get("/residual/all", response_model=List[ResidualInspection])
def legacy_get_all_residuals(
    token_index: int = Query(0, ge=0),
):
    session = _get_default_session()
    return session.residual_inspector.inspect_all_layers(token_index)


@legacy_router.get("/residual/norms", response_model=List[float])
def legacy_get_residual_norms(
    token_index: int = Query(0, ge=0),
):
    session = _get_default_session()
    return session.residual_inspector.get_residual_norms(token_index)


@legacy_router.get("/layer")
def legacy_get_layer(
    layer_index: int = Query(..., ge=0),
    include_neurons: bool = Query(False),
    include_attention: bool = Query(False),
    include_residual: bool = Query(True),
    top_k_neurons: int = Query(10, ge=1, le=100),
    token_index: int = Query(0, ge=0),
):
    """Legacy layer endpoint — returns attention_heads list instead of single attention."""
    session = _get_default_session()
    try:
        result = session.layer_inspector.inspect(
            layer_index,
            include_neurons=include_neurons,
            include_attention=False,   # we handle attention ourselves below
            include_residual=include_residual,
            top_k_neurons=top_k_neurons,
            token_index=token_index,
        )
        attention_heads = None
        if include_attention:
            attention_heads = session.attention_inspector.inspect_all_heads(
                layer_index, token_index
            )
        data = result.model_dump()
        data.pop("attention", None)
        data["attention_heads"] = (
            [h.model_dump() for h in attention_heads] if attention_heads is not None else None
        )
        return data
    except IndexError as e:
        raise HTTPException(status_code=404, detail=str(e))


@legacy_router.get("/heatmap/neuron")
def legacy_get_neuron_heatmap(
    layer_index: int = Query(..., ge=0),
    top_k: int = Query(20, ge=1, le=200),
    token_indices: Optional[str] = Query(None),
):
    session = _get_default_session()
    tokens = [int(x) for x in token_indices.split(",")] if token_indices else None
    try:
        dto = session.visualization_generator.prepare_neuron_visualization(
            layer_index, top_k, tokens
        )
    except IndexError as e:
        raise HTTPException(status_code=404, detail=str(e))
    data = dto.model_dump()
    data["heatmap_type"] = data.pop("visualization_type")
    return data


@legacy_router.get("/heatmap/attention")
def legacy_get_attention_heatmap(
    layer_index: int = Query(..., ge=0),
    head_index: int = Query(..., ge=0),
    token_index: Optional[int] = Query(None, ge=0),
):
    session = _get_default_session()
    try:
        dto = session.visualization_generator.prepare_attention_visualization(
            layer_index, head_index, token_index
        )
    except IndexError as e:
        raise HTTPException(status_code=404, detail=str(e))
    data = dto.model_dump()
    data["heatmap_type"] = data.pop("visualization_type")
    return data


@legacy_router.get("/heatmap/residual")
def legacy_get_residual_heatmap(
    token_index: int = Query(0, ge=0),
    num_dims: int = Query(64, ge=1, le=1024),
):
    session = _get_default_session()
    dto = session.visualization_generator.prepare_residual_visualization(
        token_index, num_dims
    )
    data = dto.model_dump()
    data["heatmap_type"] = data.pop("visualization_type")
    return data


@legacy_router.get("/heatmaps")
def legacy_get_all_heatmaps(
    layer_index: int = Query(0, ge=0),
    head_index: int = Query(0, ge=0),
    token_index: int = Query(0, ge=0),
):
    session = _get_default_session()
    heatmaps = session.visualization_generator.prepare_all_visualizations(
        layer_index, head_index, token_index
    )
    result = []
    for dto in heatmaps.values():
        d = dto.model_dump()
        d["heatmap_type"] = d.pop("visualization_type")
        result.append(d)
    return result


@legacy_router.get("/layers")
def legacy_list_layers():
    session = _get_default_session()
    return {
        "layers": [
            {"index": i, "name": name}
            for i, name in enumerate(session.repository.layer_names)
        ],
        "num_layers": session.repository.num_layers,
        "num_heads": session.repository.num_heads,
        "hidden_dim": session.repository.hidden_dim,
        "seq_len": session.repository.seq_len,
    }


@legacy_router.get("/stats", response_model=Statistics)
def legacy_get_stats(
    layer_index: int = Query(..., ge=0),
):
    session = _get_default_session()
    try:
        layer_name = session.repository.get_layer_name(layer_index)
        activations = session.repository.get_activations(layer_name)
        return session.stats_computer.compute(activations)
    except IndexError as e:
        raise HTTPException(status_code=404, detail=str(e))


# Register legacy router
app.include_router(legacy_router)
