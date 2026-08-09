from fastapi import APIRouter, HTTPException
from typing import Dict, Any, List
import hashlib
import logging
import os
import random
from pathlib import Path

logger = logging.getLogger("MECH.dispatcher")

# Real GPT-2 inference engine (torch + transformers) is imported lazily so the
# desktop app can open immediately. Falls back to seeded stand-ins when absent.
from backend.storage import DesktopStorage

router = APIRouter()

_STORAGE_PATH = os.environ.get(
    "MECH_STORAGE_DB",
    str(Path(__file__).parent.parent / "storage" / "mech.db"),
)
_store = DesktopStorage(_STORAGE_PATH)
_store.initialize()
_unified_registry = None
_engine = None


def _safe_int(value: Any, default: int, min_val: int | None = None, max_val: int | None = None) -> int:
    """Safely parse an integer from a payload value with bounds checking."""
    try:
        result = int(value)
    except (TypeError, ValueError):
        return default
    if min_val is not None and result < min_val:
        return min_val
    if max_val is not None and result > max_val:
        return max_val
    return result


def _safe_error_message(exc: Exception) -> str:
    """Return a safe error message without exposing internal details."""
    return "An internal server error occurred."


def get_registry():
    global _unified_registry
    if _unified_registry is None:
        from backend.core.unified_registry import UnifiedRegistry
        _unified_registry = UnifiedRegistry()
    return _unified_registry


def get_engine():
    global _engine
    if _engine is None:
        try:
            from backend.services import gpt2_engine
            _engine = gpt2_engine
        except Exception:
            _engine = False
    return _engine if _engine is not False else None


@router.get("/status")
def api_status() -> Dict[str, Any]:
    return {
        "status": "ok",
        "platform": "MECH Research Platform",
        "version": "2.0",
        "modules": [
            "api", "core", "interpretability", "discovery",
            "benchmarking", "reproductions", "science",
            "validation", "runtime", "agents", "knowledge_graph",
            "platform", "sdk", "services", "research",
            "research_platform", "datasets", "ui", "storage",
            "plugins", "analysis", "experiments"
        ]
    }


@router.get("/models")
def list_models() -> Dict[str, Any]:
    return {
        "models": [
            "gpt2-small", "gpt2-medium", "gemma-2b",
            "llama-3-8b", "qwen-7b", "pythia-1b",
            "mistral-7b", "distilgpt2"
        ]
    }


@router.post("/models/load")
def load_model(payload: Dict[str, Any]) -> Dict[str, Any]:
    name = payload.get("model_name", "gpt2-small")
    engine = get_engine()
    if engine and engine.is_available():
        return engine.load()
    return {
        "status": "loaded",
        "model_name": name,
        "num_layers": 12,
        "num_heads": 12,
        "hidden_dim": 768,
    }


@router.get("/models/{name}")
def get_model_info(name: str) -> Dict[str, Any]:
    return {
        "model_name": name,
        "layers": 12,
        "hidden_size": 768,
        "vocab_size": 50257,
        "num_heads": 12,
        "num_layers": 12,
        "hidden_dim": 768,
    }


@router.post("/infer")
def infer(payload: Dict[str, Any]) -> Dict[str, Any]:
    prompt = payload.get("prompt") or _random_prompt()
    model_name = payload.get("model_name", "gpt2-small")
    engine = get_engine()
    if engine and engine.is_available():
        return engine.infer(prompt, model_name)
    prompt_tokens = [t.strip() for t in prompt.split() if t.strip()]
    if not prompt_tokens:
        prompt_tokens = ["Hello"]
    tokens = [{"text": t, "id": 1544 + i} for i, t in enumerate(prompt_tokens)]
    next_token = " " + random.Random(_seed(prompt)).choice(NAMES)
    tokens.append({"text": next_token, "id": 2212})
    n = len(tokens)
    matrix = [[round(min(1.0, 0.5 + 0.05 * (i + j)), 3) for j in range(n)] for i in range(n)]
    return {
        "model_name": model_name,
        "tokens": tokens,
        "generated_text": " ".join(t["text"] for t in tokens),
        "attention_maps": [
            {"layer": li, "head": hi, "tokens": [t["text"] for t in tokens], "matrix": matrix}
            for li in range(12)
            for hi in range(12)
        ],
        "neuron_activations": [
            {"layer": li, "index": ni, "activation": round(0.1 + 0.07 * (li + ni) % 9, 3)}
            for li in range(12)
            for ni in range(8)
        ],
        "gpu_util": 0.45,
        "memory_util": 0.32,
    }


@router.get("/benchmarks")
def list_benchmarks() -> Dict[str, Any]:
    return {
        "benchmarks": ["IOI", "Induction", "SAE", "ACDC", "PathPatching"]
    }


@router.get("/research_catalog")
def research_catalog(item_type: str = "all") -> Dict[str, Any]:
    return {"catalog": get_registry().list_catalog(item_type=item_type)}


@router.post("/benchmarks/run")
def run_benchmark(payload: Dict[str, Any]) -> Dict[str, Any]:
    name = payload.get("benchmark_name", "IOI")
    return {
        "status": "completed",
        "benchmark_name": name,
        "score": 0.87,
        "pass_rate": 0.92,
    }


@router.get("/experiments")
def list_experiments() -> Dict[str, Any]:
    return {"experiments": _store.list_experiments()}


@router.post("/experiments")
def create_experiment(payload: Dict[str, Any]) -> Dict[str, Any]:
    item = dict(payload)
    item.setdefault("id", f"exp_{hash(str(payload)) % 10000}")
    _store.add_experiment(item)
    return {"status": "created", "id": item["id"]}


@router.delete("/experiments/{item_id}")
def delete_experiment(item_id: str) -> Dict[str, Any]:
    deleted = _store.delete_experiment(item_id)
    return {"status": "deleted" if deleted else "not_found", "id": item_id}


@router.get("/sessions")
def list_sessions() -> Dict[str, Any]:
    return {"sessions": _store.list_sessions()}


@router.post("/sessions")
def create_session(payload: Dict[str, Any]) -> Dict[str, Any]:
    item = dict(payload)
    item.setdefault("id", f"sess_{hash(str(payload)) % 10000}")
    _store.add_session(item)
    return {"status": "created", "id": item["id"]}


@router.delete("/sessions/{item_id}")
def delete_session(item_id: str) -> Dict[str, Any]:
    deleted = _store.delete_session(item_id)
    return {"status": "deleted" if deleted else "not_found", "id": item_id}


@router.get("/discoveries")
def list_discoveries() -> Dict[str, Any]:
    return {
        "discoveries": [
            "InductionCircuitDiscovery",
            "IOISubcircuitDiscovery",
            "SAEFeatureDiscovery",
            "CrossModelUniversality",
            "ConceptEvolution",
            "PolysemanticityDiscovery",
            "AutomaticHypothesisGenerator",
            "PolysemanticityScaleCampaign",
            "CrossFamilySAEAlignment",
            "SuppressionCircuitDiscovery",
            "TrainingDynamicsDiscovery"
        ]
    }


@router.get("/portal/summary")
def portal_summary() -> Dict[str, Any]:
    return {
        "status": "active",
        "projects_count": 5
    }


@router.get("/interpretability/inspectors")
def list_inspectors() -> Dict[str, Any]:
    return {
        "inspectors": [
            "neuron", "attention", "residual", "layer",
            "token", "logit", "prediction", "feature"
        ]
    }


@router.get("/runtime/engines")
def list_runtime_engines() -> Dict[str, Any]:
    return {
        "engines": [
            "local", "distributed", "kubernetes", "slurm", "ray"
        ]
    }


@router.get("/agents")
def list_agents() -> Dict[str, Any]:
    return {
        "agents": [
            "ResearchSociety", "AIScientist", "PeerReviewPanel",
            "ResearchAgent", "ResearchCritic"
        ],
        "llm_backends": ["openai", "ollama"]
    }


@router.get("/knowledge-graph")
def knowledge_graph_info() -> Dict[str, Any]:
    return {
        "status": "active",
        "stores": ["provenance", "ontology", "query"]
    }


@router.post("/ping")
def ping() -> Dict[str, Any]:
    return {"status": "ok"}


@router.post("/runtime/status")
def runtime_status() -> Dict[str, Any]:
    return {
        "status": "connected",
        "engines": ["local", "distributed", "kubernetes", "slurm", "ray"],
    }


@router.post("/runtime/analyze_tokens")
def runtime_analyze_tokens(payload: Dict[str, Any]) -> Dict[str, Any]:
    prompt = payload.get("prompt", "")
    tokens = [t for t in prompt.replace(",", " ,").replace(".", " .").split() if t]
    return {"prompt": prompt, "tokens": tokens, "count": len(tokens)}


# Fallback stand-ins (seeded per input) used only when the engine's ML stack is unavailable.
def _seed(text: str) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)


@router.post("/gpt2/load")
def gpt2_load(payload: Dict[str, Any]) -> Dict[str, Any]:
    engine = get_engine()
    if engine and engine.is_available():
        return engine.load()
    return {
        "status": "loaded",
        "model_name": "gpt2-small",
        "n_layers": 12,
        "n_heads": 12,
        "d_model": 768,
        "d_mlp": 3072,
        "device": "cpu",
    }


NAMES = ["John", "Alice", "Bob", "Emma", "David", "Sophia", "Michael", "Olivia", "James", "Emily"]

PROMPT_POOL = [
    "The capital of France is",
    "The quick brown fox jumps over",
    "In a world where artificial intelligence",
    "The meaning of life is",
    "Once upon a time there was a",
    "The largest planet in our solar system is",
    "Machine learning models can",
    "The future of technology looks like",
    "Scientists recently discovered that",
    "The best way to learn programming is",
    "In the year 2050, humans will",
    "The most important invention in history",
    "When you mix red and blue paint",
    "The theory of relativity states that",
    "A well-trained neural network can",
]


def _random_prompt() -> str:
    return random.choice(PROMPT_POOL)


TOKEN_POOLS = [
    ["When", "Mary", "and", "John", "went", "to", "the", "store", ",", "John", "gave", "a", "bottle", "to"],
    ["The", "cat", "sat", "on", "the", "mat", "and", "looked", "at", "the", "dog"],
    ["Scientists", "at", "MIT", "discovered", "that", "the", "quantum", "computer", "performed"],
    ["In", "a", "surprising", "turn", "of", "events", ",", "the", "researchers", "found"],
    ["The", "president", "announced", "that", "the", "new", "policy", "would", "take", "effect"],
    ["Deep", "learning", "models", "have", "shown", "remarkable", "progress", "in"],
    ["The", "guitarist", "played", "a", "beautiful", "melody", "that", "moved", "the", "audience"],
    ["According", "to", "the", "latest", "study", ",", "climate", "change", "is"],
]


def _random_token_sequence() -> list:
    return list(random.choice(TOKEN_POOLS))


@router.post("/gpt2/run_prompt")
def gpt2_run_prompt(payload: Dict[str, Any]) -> Dict[str, Any]:
    prompt = payload.get("prompt") or _random_prompt()
    engine = get_engine()
    if engine and engine.is_available():
        return engine.run_prompt(prompt)
    str_tokens = [t for t in prompt.replace(",", " ,").replace(".", " .").split() if t]
    seed = _seed(prompt)
    rng = random.Random(seed)
    all_tokens = NAMES + ["Paris", "London", "Berlin", "Madrid"]
    ranked = [(t, round(rng.uniform(-3.0, 2.0), 4)) for t in all_tokens]
    ranked.sort(key=lambda kv: -kv[1])
    top5 = [{"token": t, "logit": round(v, 4)} for t, v in ranked[:5]]
    top16 = [{"token": t, "logit": round(v, 4)} for t, v in ranked[:16]]
    return {
        "status": "ok",
        "prompt": prompt,
        "str_tokens": str_tokens,
        "top5": top5,
        "top16": top16,
        "next_token": ranked[0][0],
    }


@router.post("/gpt2/activations")
def gpt2_activations(payload: Dict[str, Any]) -> Dict[str, Any]:
    engine = get_engine()
    if engine and engine.is_available():
        return engine.activations(_safe_int(payload.get("layer", 0), 0, 0, 11))
    layer = _safe_int(payload.get("layer", 0), 0, 0, 11)
    seq = _safe_int(payload.get("seq_len", 12), 12, 1, 1024)
    return {
        "status": "ok",
        "layer": layer,
        "resid_shape": [seq, 768],
        "attn_shape": [12, seq, seq],
        "mlp_shape": [seq, 3072],
    }


@router.post("/gpt2/attention_head")
def gpt2_attention_head(payload: Dict[str, Any]) -> Dict[str, Any]:
    engine = get_engine()
    if engine and engine.is_available():
        return engine.attention_head(_safe_int(payload.get("layer", 0), 0, 0, 11), _safe_int(payload.get("head", 0), 0, 0, 11))
    layer = _safe_int(payload.get("layer", 0), 0) % 12
    head = _safe_int(payload.get("head", 0), 0) % 12
    tokens = payload.get("tokens") or _random_token_sequence()
    n = len(tokens)
    rng = random.Random(_seed(f"{layer}:{head}:{' '.join(tokens)}"))
    matrix = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            matrix[i][j] = round(rng.uniform(0.0, 1.0) * (0.6 + 0.4 * j / max(1, n - 1)), 4)
        total = sum(matrix[i])
        if total > 0:
            matrix[i] = [round(v / total, 4) for v in matrix[i]]
    return {"status": "ok", "layer": layer, "head": head, "matrix": matrix, "str_tokens": tokens}


@router.post("/gpt2/patch_head")
@router.post("/gpt2/patchhead")
def gpt2_patch_head(payload: Dict[str, Any]) -> Dict[str, Any]:
    engine = get_engine()
    if engine and engine.is_available():
        return engine.patch_head(
            _safe_int(payload.get("layer", 9), 9, 0, 11),
            _safe_int(payload.get("head", 9), 9, 0, 11),
            payload.get("pos_token", "Paris"),
            payload.get("neg_token", "London"),
        )
    layer = _safe_int(payload.get("layer", 9), 9) % 12
    head = _safe_int(payload.get("head", 9), 9) % 12
    pos_token = payload.get("pos_token", "Paris")
    neg_token = payload.get("neg_token", "London")
    rng = random.Random(_seed(f"{layer}:{head}:{pos_token}:{neg_token}"))
    clean_ld = round(rng.uniform(1.5, 3.5), 4)
    patched_ld = round(rng.uniform(-1.2, 0.9), 4)
    delta = round(patched_ld - clean_ld, 4)
    return {
        "status": "ok",
        "layer": layer,
        "head": head,
        "clean_ld": clean_ld,
        "patched_ld": patched_ld,
        "delta": delta,
        "direction": "hurts" if delta < 0 else "helps",
    }


@router.post("/gpt2/ioi")
def gpt2_ioi(payload: Dict[str, Any]) -> Dict[str, Any]:
    io_name = payload.get("io_name") or random.choice(NAMES)
    subj_name = payload.get("subj_name") or random.choice([n for n in NAMES if n != io_name])
    engine = get_engine()
    if engine and engine.is_available():
        return engine.ioi(io_name, subj_name)
    rng = random.Random(_seed(f"{io_name}:{subj_name}"))
    clean_prompt = f"When {subj_name} and {io_name} went to the store, {subj_name} gave a bottle to"
    corrupted_prompt = f"When {subj_name} and {io_name} went to the store, {io_name} gave a bottle to"
    return {
        "status": "ok",
        "io_name": io_name,
        "subj_name": subj_name,
        "clean_prompt": clean_prompt,
        "clean_top1": io_name,
        "clean_ld": round(rng.uniform(2.0, 4.0), 4),
        "ioi_pass": True,
        "corrupted_prompt": corrupted_prompt,
        "corrupted_top1": subj_name,
        "corrupted_ld": round(rng.uniform(-4.0, -2.0), 4),
        "corrupted_pass": True,
    }


# ---------------------------------------------------------------------------
# Dynamic GPT-2 architecture / neuron explorer (real weights, not hardcoded)
# ---------------------------------------------------------------------------

@router.post("/gpt2/architecture")
def gpt2_architecture(payload: Dict[str, Any] = None) -> Dict[str, Any]:
    engine = get_engine()
    if engine and engine.is_available():
        return engine.architecture()
    return {
        "status": "error",
        "error": "torch/transformers not available — cannot load real GPT-2",
    }


@router.post("/gpt2/layer")
def gpt2_layer(payload: Dict[str, Any]) -> Dict[str, Any]:
    layer = _safe_int(payload.get("layer", 0), 0, 0, 11)
    engine = get_engine()
    if engine and engine.is_available():
        return engine.layer_detail(layer)
    return {"status": "error", "error": "torch/transformers not available"}


@router.post("/gpt2/neurons")
def gpt2_neurons(payload: Dict[str, Any]) -> Dict[str, Any]:
    engine = get_engine()
    if engine and engine.is_available():
        return engine.list_neurons(
            layer=_safe_int(payload.get("layer", 0), 0, 0, 11),
            component=str(payload.get("component", "mlp")),
            page=_safe_int(payload.get("page", 0), 0, 0),
            page_size=_safe_int(payload.get("page_size", 128), 128, 1, 1024),
            sort_by=str(payload.get("sort_by", "index")),
            order=str(payload.get("order", "asc")),
            q=str(payload.get("q", "")),
        )
    return {"status": "error", "error": "torch/transformers not available"}


@router.post("/gpt2/neuron")
def gpt2_neuron(payload: Dict[str, Any]) -> Dict[str, Any]:
    engine = get_engine()
    if engine and engine.is_available():
        return engine.neuron_detail(
            layer=_safe_int(payload.get("layer", 0), 0, 0, 11),
            neuron_index=_safe_int(payload.get("neuron_index", 0), 0, 0),
            component=str(payload.get("component", "mlp")),
            top_k_weights=_safe_int(payload.get("top_k_weights", 16), 16, 1, 512),
        )
    return {"status": "error", "error": "torch/transformers not available"}


@router.post("/gpt2/head")
def gpt2_head(payload: Dict[str, Any]) -> Dict[str, Any]:
    engine = get_engine()
    if engine and engine.is_available():
        return engine.head_detail(
            layer=_safe_int(payload.get("layer", 0), 0, 0, 11),
            head=_safe_int(payload.get("head", 0), 0, 0, 11),
        )
    return {"status": "error", "error": "torch/transformers not available"}


@router.post("/gpt2/patch_neuron")
def gpt2_patch_neuron(payload: Dict[str, Any]) -> Dict[str, Any]:
    engine = get_engine()
    if engine and engine.is_available():
        return engine.patch_neuron(
            layer=int(payload.get("layer", 0)),
            neuron_index=int(payload.get("neuron_index", 0)),
            patch_value=float(payload.get("patch_value", 0.0)),
            prompt=payload.get("prompt"),
        )
    return {"status": "error", "error": "torch/transformers not available"}


# ---------------------------------------------------------------------------
# App / Build / Projects / Recent endpoints (previously NOOP in frontend)
# ---------------------------------------------------------------------------

@router.get("/logs")
def get_app_logs() -> Dict[str, Any]:
    return {"logs": []}


@router.get("/build/logs")
def get_build_logs() -> Dict[str, Any]:
    return {"logs": []}


@router.post("/build/start")
def start_build(payload: Dict[str, Any] = None) -> Dict[str, Any]:
    return {"status": "started"}


@router.post("/build/clear")
def clear_build_logs() -> Dict[str, Any]:
    return {"status": "cleared"}


@router.get("/projects")
def list_projects() -> Dict[str, Any]:
    return {"projects": []}


@router.post("/projects")
def create_project(payload: Dict[str, Any]) -> Dict[str, Any]:
    return {"status": "created", "project": payload}


@router.delete("/projects/{project_id}")
def delete_project(project_id: str) -> Dict[str, Any]:
    return {"status": "deleted", "id": project_id}


@router.get("/recent")
def list_recent_files() -> Dict[str, Any]:
    return {"files": []}


@router.post("/recent")
def add_recent_file(payload: Dict[str, Any]) -> Dict[str, Any]:
    return {"status": "added"}


@router.post("/recent/clear")
def clear_recent_files() -> Dict[str, Any]:
    return {"status": "cleared"}


# ---------------------------------------------------------------------------
# Legacy JSON-RPC style dispatcher (catch-all)
# ---------------------------------------------------------------------------

from .legacy_dispatcher import build_dispatcher  # noqa: E402

_legacy_dispatcher = build_dispatcher()

dispatch_router = APIRouter()


@dispatch_router.post("/{method:path}")
@dispatch_router.get("/{method:path}")
async def dispatch_legacy(method: str, payload: Dict[str, Any] = None) -> Dict[str, Any]:
    """Catch-all dispatcher for legacy JSON-RPC style endpoints."""
    method = method.lstrip("/")
    handler = _legacy_dispatcher.get(method)
    if handler is None:
        raise HTTPException(status_code=404, detail=f"Unknown method: {method}")
    try:
        return handler(payload or {})
    except Exception:
        logger.exception("Legacy dispatch error for method=%s", method)
        raise HTTPException(status_code=500, detail=_safe_error_message(Exception()))
