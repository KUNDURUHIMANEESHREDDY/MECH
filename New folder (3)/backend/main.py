"""Runtime API v2 — all endpoints under /api/v1/ with versioning.

API surface:
    GET    /api/v1/health
    GET    /api/v1/capabilities
    GET    /api/v1/models
    GET    /api/v1/models/{name}
    POST   /api/v1/models/load
    POST   /api/v1/models/unload
    POST   /api/v1/sessions
    GET    /api/v1/sessions
    GET    /api/v1/sessions/{id}
    POST   /api/v1/sessions/{id}/close
    POST   /api/v1/inference
    GET    /api/v1/layers
    GET    /api/v1/layers/{idx}/heads
    GET    /api/v1/hooks
    POST   /api/v1/hooks
    DELETE /api/v1/hooks/{id}
    GET    /api/v1/cache
    GET    /api/v1/cache/{id}
    GET    /api/v1/performance
    GET    /api/v1/inference/stream

Backward-compatible aliases at root level are provided.
"""

from __future__ import annotations

import time
from typing import Optional
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Query, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.routing import APIRoute

import torch
import psutil

from runtime.dto import (
    InferenceRequest, InferenceResponse, ModelLoadRequest, ModelLoadResponse,
    ModelInfoResponse, SessionResponse, SessionCreateResponse,
    HookRegisterRequest, HookHandleResponse,
    CacheQueryRequest, CacheEntryResponse,
    PerformanceReportResponse, ErrorResponse,
)
from runtime.model_descriptor import REGISTRY as MODEL_REGISTRY, list_all as list_descriptors
from runtime.model_manager import (
    list_models, model_info, load_model, unload_model, switch_model,
    get_model_and_tokenizer, is_loaded, _loaded_name,
)
from runtime.interpreter import run_inference
from runtime.session_manager import session_manager
from runtime.event_bus import bus
from runtime.hook_framework import HookManager, HOOK_TYPES
from runtime.activation_cache import cache
from runtime.profiler import Profiler
from runtime.health import get_health
from runtime.streaming import stream_inference
from runtime.errors import (
    RuntimeError, ModelLoadError, InferenceError, SessionNotFoundError,
    HookError, CacheError,
)
from repository.activation_repository import activation_repo
from repository.session_repository import session_repo
from repository.experiment_repository import experiment_repo
from interpretability.inspectors.token import token_inspector
from interpretability.inspectors.layer import layer_inspector
from interpretability.inspectors.prediction import prediction_inspector
from interpretability.inspectors.neuron import neuron_inspector
from interpretability.inspectors.attention import attention_inspector
from interpretability.inspectors.residual import residual_inspector

# ── Global state ─────────────────────────────────────────────────

hook_manager = HookManager()
profiler = Profiler()


# ── Client-ID auth (Epic 9) ──────────────────────────────────────

async def get_client_id(client_id: str = Header("", alias="Client-Id")):
    return client_id or "default"


# ── App setup ────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    yield


app = FastAPI(title="Model Explorer Runtime v2", version="2.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Exception handler ────────────────────────────────────────────

@app.exception_handler(RuntimeError)
async def runtime_error_handler(request, exc: RuntimeError):
    from fastapi.responses import JSONResponse
    return JSONResponse(
        status_code=400,
        content={"error": type(exc).__name__, "detail": str(exc)},
    )


# ── Health ───────────────────────────────────────────────────────

@app.get("/api/v1/health")
@app.get("/health")
def health():
    return get_health()


# ── Capabilities (Epic 7) ────────────────────────────────────────

@app.get("/api/v1/capabilities")
@app.get("/capabilities")
def capabilities():
    return {
        "version": "2.0.0",
        "models": list(MODEL_REGISTRY.keys()),
        "hooks": list(HOOK_TYPES.keys()),
        "supports_streaming": True,
        "supports_sessions": True,
        "supports_attention": True,
        "supports_residuals": True,
        "supports_mlp": True,
        "supports_embeddings": True,
        "supports_logits": True,
        "supports_patching": False,
        "supports_sae": False,
        "supports_circuits": False,
        "supports_gradients": False,
        "supports_client_auth": True,
        "supports_persistence": True,
        "supports_event_replay": True,
        "supports_hook_dependencies": True,
    }


# ── Models ───────────────────────────────────────────────────────

@app.get("/api/v1/models")
@app.get("/models")
def list_models_endpoint():
    return {"models": list_models()}


@app.get("/api/v1/models/{name}")
@app.get("/models/{name}")
def model_info_endpoint(name: str):
    desc = MODEL_REGISTRY.get(name)
    if desc is None:
        raise HTTPException(status_code=404, detail=f"Model '{name}' not found")
    return {
        "model_name": name,
        "hf_id": desc.hf_id,
        "family": desc.family,
        "architecture": desc.architecture,
        "hidden_size": desc.hidden_size,
        "num_layers": desc.num_layers,
        "num_heads": desc.num_heads,
        "vocab_size": desc.vocab_size,
        "max_position": desc.max_position,
        "activation_function": desc.activation_function,
        "description": desc.description,
        "supports_attention": desc.supports_attention,
        "supports_residuals": desc.supports_residuals,
        "supports_mlp": desc.supports_mlp,
        "loaded": name == _loaded_name,
    }


@app.post("/api/v1/models/load")
@app.post("/models/load", response_model=ModelLoadResponse)
def load_model_endpoint(req: ModelLoadRequest):
    with profiler.measure("load.model", model=req.model_name):
        info = load_model(req.model_name)
    hook_manager.bind(get_model_and_tokenizer()[0])
    return ModelLoadResponse(**info)


@app.post("/api/v1/models/unload")
@app.post("/models/unload")
def unload_model_endpoint():
    hook_manager.clear()
    return unload_model()


# ── Sessions ─────────────────────────────────────────────────────

@app.post("/api/v1/sessions")
@app.post("/sessions", response_model=SessionCreateResponse)
def create_session(
    model_name: str = Query("gpt2"),
    prompt: str = Query(""),
    client_id: str = Depends(get_client_id),
):
    s = session_manager.create_session(model_name=model_name, prompt=prompt, client_id=client_id)
    return SessionCreateResponse(session_id=s.session_id)


@app.get("/api/v1/sessions")
@app.get("/sessions", response_model=list[SessionResponse])
def list_sessions():
    return [
        SessionResponse(
            session_id=s.session_id, model_name=s.model_name, prompt=s.prompt,
            generated_text=s.generated_text, created_at=s.created_at,
            closed_at=s.closed_at, is_open=s.is_open,
            num_tokens=len(s.token_ids), num_layers=s.num_layers,
        )
        for s in session_manager.list_sessions()
    ]


@app.get("/api/v1/sessions/{session_id}")
@app.get("/sessions/{session_id}", response_model=SessionResponse)
def get_session(session_id: str):
    try:
        s = session_manager.get_session(session_id)
    except SessionNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return SessionResponse(
        session_id=s.session_id, model_name=s.model_name, prompt=s.prompt,
        generated_text=s.generated_text, created_at=s.created_at,
        closed_at=s.closed_at, is_open=s.is_open,
        num_tokens=len(s.token_ids), num_layers=s.num_layers,
    )


@app.post("/api/v1/sessions/{session_id}/close")
@app.post("/sessions/{session_id}/close")
def close_session(session_id: str):
    try:
        session_manager.close_session(session_id)
    except SessionNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return {"status": "closed", "session_id": session_id}


# ── Inference ────────────────────────────────────────────────────

@app.post("/api/v1/inference")
@app.post("/inference", response_model=InferenceResponse)
def inference_endpoint(
    req: InferenceRequest,
    client_id: str = Depends(get_client_id),
):
    if not is_loaded():
        raise HTTPException(status_code=400, detail="No model loaded. POST /api/v1/models/load first.")

    model, tokenizer = get_model_and_tokenizer()

    if req.session_id:
        try:
            session = session_manager.get_session(req.session_id)
        except SessionNotFoundError:
            raise HTTPException(status_code=404, detail=f"Session {req.session_id} not found")
    else:
        session = session_manager.create_session(
            model_name=_loaded_name or "unknown", prompt=req.prompt, client_id=client_id,
        )

    config = model.config
    session.num_layers = getattr(config, "n_layer", getattr(config, "num_hidden_layers", 12))
    session.num_heads = getattr(config, "n_head", getattr(config, "num_attention_heads", 12))

    with profiler.measure("inference.run", prompt=req.prompt, max_tokens=req.max_new_tokens, session=session.session_id):
        result = run_inference(
            prompt=req.prompt,
            max_new_tokens=req.max_new_tokens,
            model=model, tokenizer=tokenizer,
            hook_manager=hook_manager, session=session, cache=cache, profiler=profiler,
        )

    session.generated_text = result["generated_text"]
    session.token_texts = [t.text for t in result["tokens"]]
    session.token_ids = [t.id for t in result["tokens"]]
    session_manager.update_session(session)

    mem = psutil.Process().memory_percent()

    return InferenceResponse(
        session_id=session.session_id,
        model_name=_loaded_name or "unknown",
        tokens=result["tokens"],
        generated_text=result["generated_text"],
        attention_maps=result["attention_maps"],
        neuron_activations=result["neuron_activations"],
        profiling=profiler.report(),
        gpu_util=torch.cuda.is_available() and 1.0 or 0.0,
        memory_util=round(mem / 100.0, 2),
        cache_ids=result["cache_ids"],
    )


# ── Layers & Heads ───────────────────────────────────────────────

@app.get("/api/v1/layers")
@app.get("/layers")
def list_layers():
    if not is_loaded():
        raise HTTPException(status_code=400, detail="No model loaded")
    model, _ = get_model_and_tokenizer()
    config = model.config
    n = getattr(config, "n_layer", getattr(config, "num_hidden_layers", 0))
    return {"layers": list(range(n))}


@app.get("/api/v1/layers/{layer_idx}/heads")
@app.get("/layers/{layer_idx}/heads")
def list_heads(layer_idx: int):
    if not is_loaded():
        raise HTTPException(status_code=400, detail="No model loaded")
    model, _ = get_model_and_tokenizer()
    config = model.config
    num_layers = getattr(config, "n_layer", getattr(config, "num_hidden_layers", 0))
    if layer_idx < 0 or layer_idx >= num_layers:
        raise HTTPException(status_code=404, detail=f"Layer {layer_idx} out of range (0-{num_layers - 1})")
    num_heads = getattr(config, "n_head", getattr(config, "num_attention_heads", 0))
    return {"layer": layer_idx, "heads": list(range(num_heads))}


# ── Hooks ────────────────────────────────────────────────────────

@app.get("/api/v1/hooks")
@app.get("/hooks", response_model=list[HookHandleResponse])
def list_hooks():
    return [
        HookHandleResponse(id=h.id, name=h.name, enabled=h.enabled, layer_idx=h.layer_idx, hook_type=h.hook_type)
        for h in hook_manager.list_hooks()
    ]


@app.post("/api/v1/hooks")
@app.post("/hooks", response_model=HookHandleResponse)
def register_hook(req: HookRegisterRequest):
    if not is_loaded():
        raise HTTPException(status_code=400, detail="No model loaded")
    try:
        handle = hook_manager.register_hook(req.layer_idx, req.hook_type)
        return HookHandleResponse(id=handle.id, name=handle.name, enabled=handle.enabled, layer_idx=handle.layer_idx, hook_type=handle.hook_type)
    except HookError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.delete("/api/v1/hooks/{hook_id}")
@app.delete("/hooks/{hook_id}")
def remove_hook(hook_id: str):
    handles = hook_manager.list_hooks()
    for h in handles:
        if h.id == hook_id:
            hook_manager.remove_hook(h)
            return {"status": "removed", "hook_id": hook_id}
    raise HTTPException(status_code=404, detail=f"Hook {hook_id} not found")


# ── Cache ────────────────────────────────────────────────────────

@app.get("/api/v1/cache")
@app.get("/cache", response_model=list[CacheEntryResponse])
def search_cache(
    session_id: str | None = Query(None),
    layer: int | None = Query(None),
    component: str | None = Query(None),
    head: int | None = Query(None),
):
    results = cache.search(
        session_id=session_id, layer=layer, component=component, head=head,
    )
    return [
        CacheEntryResponse(
            activation_id=a.activation_id, session_id=a.session_id,
            prompt_id=a.prompt_id, layer=a.layer, head=a.head,
            component=a.component, shape=list(a.shape), timestamp=a.timestamp,
            metadata=a.metadata,
        )
        for a in results
    ]


@app.get("/api/v1/cache/{activation_id}")
@app.get("/cache/{activation_id}")
def get_cached_activation(activation_id: str):
    a = cache.get(activation_id)
    if a is None:
        raise HTTPException(status_code=404, detail=f"Activation {activation_id} not found")
    return CacheEntryResponse(
        activation_id=a.activation_id, session_id=a.session_id,
        prompt_id=a.prompt_id, layer=a.layer, head=a.head,
        component=a.component, shape=list(a.shape), timestamp=a.timestamp,
        metadata=a.metadata,
    )


# ── Event Log / Replay ───────────────────────────────────────────

@app.get("/api/v1/events")
@app.get("/events")
def list_events(event_type: str | None = Query(None), limit: int = Query(100)):
    return {
        "total": bus.log.count,
        "events_per_sec": round(bus.log.events_per_sec, 1),
        "events": [
            {"type": e.type, "timestamp": e.timestamp, "payload": e.payload}
            for e in bus.log.query(event_type=event_type, limit=limit)
        ],
    }


# ── Performance ──────────────────────────────────────────────────

@app.get("/api/v1/performance")
@app.get("/performance", response_model=PerformanceReportResponse)
def get_performance():
    return PerformanceReportResponse(**profiler.report())


# ── Streaming ────────────────────────────────────────────────────

@app.get("/api/v1/inference/stream")
@app.get("/inference/stream")
async def inference_stream(prompt: str = Query("Hello"), max_new_tokens: int = Query(10)):
    if not is_loaded():
        raise HTTPException(status_code=400, detail="No model loaded")
    model, tokenizer = get_model_and_tokenizer()
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
    return StreamingResponse(
        stream_inference(model, tokenizer, inputs["input_ids"], prompt),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )


# ── Capabilities & Search ────────────────────────────────────────

@app.get("/api/v1/capabilities")
@app.get("/capabilities")
def get_capabilities():
    return {
        "models": list_models(),
        "inspectors": ["token", "layer", "prediction", "neuron", "attention", "residual"],
        "panels": ["token", "layer", "prediction", "attention", "neuron"],
        "hooks": ["attention", "mlp", "residual"],
        "supports_streaming": True,
    }


@app.get("/api/v1/search")
@app.get("/search")
def search_endpoint(q: str = Query(...)):
    sessions = session_repo.list_sessions()
    matching_sessions = [s for s in sessions if q.lower() in s.prompt.lower() or q.lower() in s.model_name.lower()]
    matching_experiments = experiment_repo.search(q)
    return {
        "query": q,
        "results": {
            "sessions": [{"id": s.session_id, "prompt": s.prompt, "model": s.model_name} for s in matching_sessions],
            "experiments": [{"id": e.id, "name": e.name, "tags": e.tags} for e.repr() if False] if False else [{"id": e.id, "name": e.name, "tags": e.tags} for e in matching_experiments],
        }
    }


# ── Repository Endpoints ─────────────────────────────────────────

@app.get("/api/v1/repository/activations")
@app.get("/repository/activations")
def query_repository(
    session_id: str | None = Query(None),
    layer: int | None = Query(None),
    component: str | None = Query(None),
    head: int | None = Query(None),
    token_idx: int | None = Query(None),
):
    records = activation_repo.query(
        session_id=session_id, layer=layer, component=component, head=head, token_idx=token_idx
    )
    return [
        {
            "activation_id": r.activation_id,
            "session_id": r.session_id,
            "layer": r.layer,
            "head": r.head,
            "component": r.component,
            "shape": list(r.shape),
            "timestamp": r.timestamp,
        }
        for r in records
    ]


@app.get("/api/v1/workspaces")
@app.get("/workspaces")
def list_workspaces():
    return [
        {
            "id": w.id,
            "name": w.name,
            "models": w.models,
            "sessions": w.sessions,
            "experiments": w.experiments,
            "notes": [{"id": n.id, "title": n.title} for n in w.notes],
            "bookmarks": [{"id": b.id, "title": b.title} for b in w.bookmarks],
        }
        for w in session_repo.list_workspaces()
    ]


# ── Inspector Endpoints ──────────────────────────────────────────

@app.get("/api/v1/inspect/token")
@app.get("/inspect/token")
def inspect_token_endpoint(session_id: str = Query(...), token_idx: int = Query(0)):
    return token_inspector.inspect_token(session_id=session_id, token_idx=token_idx)


@app.get("/api/v1/inspect/layer")
@app.get("/inspect/layer")
def inspect_layer_endpoint(session_id: str = Query(...), layer_idx: int = Query(0)):
    return layer_inspector.inspect_layer(session_id=session_id, layer_idx=layer_idx)


@app.get("/api/v1/inspect/prediction")
@app.get("/inspect/prediction")
def inspect_prediction_endpoint(session_id: str = Query(...), top_k: int = Query(5)):
    model, tokenizer = None, None
    if is_loaded():
        model, tokenizer = get_model_and_tokenizer()
    return prediction_inspector.inspect_predictions(session_id=session_id, model=model, tokenizer=tokenizer, top_k=top_k)


# ── Legacy root redirect ─────────────────────────────────────────

from fastapi.responses import RedirectResponse

@app.get("/")
def root():
    return RedirectResponse(url="/api/v1/health")


# ── Main ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
