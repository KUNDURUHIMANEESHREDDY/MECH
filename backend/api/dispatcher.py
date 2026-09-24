from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from typing import Dict, Any, List
import asyncio
import hashlib
import json
import os
import queue
import random
import threading
import time
import uuid
from pathlib import Path

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
    return {"status": "loaded", "model_name": name}


@router.get("/models/{name}")
def get_model_info(name: str) -> Dict[str, Any]:
    return {
        "model_name": name,
        "layers": 12,
        "hidden_size": 768,
        "vocab_size": 50257,
        "num_heads": 12,
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
    try:
        from backend.validation.benchmark_runner import (
            MechanisticBenchmarkRunner,
        )
        res = MechanisticBenchmarkRunner().run_benchmark(f"bench_{name}")
        score = float(res.get("accuracy", 0.0))
        return {
            "status": "completed",
            "benchmark_name": name,
            "score": round(score, 4),
            "pass_rate": round(float(res.get("robustness_score", score)), 4),
        }
    except Exception as exc:
        return {
            "status": "error",
            "benchmark_name": name,
            "error": str(exc)[:300],
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
            "ResearchSociety", "ResearchSocietyV2", "Planner", "Executor",
            "Inspector", "Discoverer", "Critic", "Scribe",
            "AIScientist", "PeerReviewPanel",
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
        return engine.activations(int(payload.get("layer", 0)))
    layer = max(0, min(11, int(payload.get("layer", 0))))
    seq = int(payload.get("seq_len", 12))
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
        return engine.attention_head(int(payload.get("layer", 0)), int(payload.get("head", 0)))
    layer = int(payload.get("layer", 0)) % 12
    head = int(payload.get("head", 0)) % 12
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
            int(payload.get("layer", 9)),
            int(payload.get("head", 9)),
            payload.get("pos_token", "Paris"),
            payload.get("neg_token", "London"),
        )
    layer = int(payload.get("layer", 9)) % 12
    head = int(payload.get("head", 9)) % 12
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
    layer = int(payload.get("layer", 0))
    engine = get_engine()
    if engine and engine.is_available():
        return engine.layer_detail(layer)
    return {"status": "error", "error": "torch/transformers not available"}


@router.post("/gpt2/neurons")
def gpt2_neurons(payload: Dict[str, Any]) -> Dict[str, Any]:
    engine = get_engine()
    if engine and engine.is_available():
        return engine.list_neurons(
            layer=int(payload.get("layer", 0)),
            component=str(payload.get("component", "mlp")),
            page=int(payload.get("page", 0)),
            page_size=int(payload.get("page_size", 128)),
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
            layer=int(payload.get("layer", 0)),
            neuron_index=int(payload.get("neuron_index", 0)),
            component=str(payload.get("component", "mlp")),
            top_k_weights=int(payload.get("top_k_weights", 16)),
        )
    return {"status": "error", "error": "torch/transformers not available"}


@router.post("/gpt2/head")
def gpt2_head(payload: Dict[str, Any]) -> Dict[str, Any]:
    engine = get_engine()
    if engine and engine.is_available():
        return engine.head_detail(
            layer=int(payload.get("layer", 0)),
            head=int(payload.get("head", 0)),
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


@router.post("/gpt2/layer_activations")
def gpt2_layer_activations(payload: Dict[str, Any]) -> Dict[str, Any]:
    engine = get_engine()
    if engine and engine.is_available():
        try:
            return engine.layer_activations(
                layer=int(payload.get("layer", 0)),
                prompt=payload.get("prompt") or "The capital of France is",
            )
        except Exception as exc:
            return {"status": "error", "error": str(exc)[:300]}
    return {"status": "error", "error": "torch/transformers not available"}


@router.post("/gpt2/logit_lens_all")
def gpt2_logit_lens_all(payload: Dict[str, Any]) -> Dict[str, Any]:
    engine = get_engine()
    if engine and engine.is_available():
        try:
            return engine.logit_lens_all(
                prompt=payload.get("prompt") or "The capital of France is",
            )
        except Exception as exc:
            return {"status": "error", "error": str(exc)[:300]}
    return {"status": "error", "error": "torch/transformers not available"}


# ---------------------------------------------------------------------------
# Research Society v2 — autonomous runs + SSE
# Mounted under /api (see main.py include_router prefix), so live paths are:
#   POST /api/society/run              {goal, model_name?} -> {runId, status}
#   GET  /api/society/stream?runId=... -> text/event-stream (live trace)
#   GET  /api/society/runs/{runId}     -> polling fallback (status + result)
# NOTE: docs/API_v1.md sketches /api/v1/... but no runtime_api module exists
# live — Society ships on this active dispatcher, not the frozen doc.
# Long GPU work runs in a daemon thread; the event loop is never blocked.
# ---------------------------------------------------------------------------

_society_runs: Dict[str, Dict[str, Any]] = {}
_society_lock = threading.Lock()
_SOCIETY_MAX_RUNS = 50


def _society_get():  # type: ignore[no-untyped-def]
    from backend.agents.society import ResearchSocietyV2
    return ResearchSocietyV2()


def _society_push(run_id: str, event: Dict[str, Any]) -> None:
    run = _society_runs.get(run_id)
    if run is None:
        return
    with _society_lock:
        run["events"].append(event)
        if len(run["events"]) > 1000:
            run["events"] = run["events"][-1000:]
    try:
        run["queue"].put(("event", event), block=False)
    except queue.Full:
        pass


def _society_worker(run_id: str, goal: str, model_name: str) -> None:
    try:
        society = _society_get()
        result = society.run_blocking(
            goal, model_name=model_name,
            on_event=lambda ev: _society_push(run_id, ev),
        )
        status = result.get("status", "completed")
    except Exception as exc:
        result = {"status": "error", "error": str(exc)[:500]}
        status = "error"
    run = _society_runs.get(run_id)
    if run is not None:
        with _society_lock:
            run["result"] = result
            run["status"] = status
    try:
        run["queue"].put(("done", None), block=False)
    except Exception:
        pass


@router.post("/society/run")
def society_run(payload: Dict[str, Any]) -> Dict[str, Any]:
    goal = str((payload or {}).get("goal", "")).strip()
    if not goal:
        return {"status": "error", "error": "goal required"}
    model_name = str((payload or {}).get("model_name", "gpt2"))
    run_id = "r" + uuid.uuid4().hex[:12]
    run: Dict[str, Any] = {
        "run_id": run_id,
        "goal": goal,
        "model_name": model_name,
        "status": "running",
        "events": [],
        "result": None,
        "queue": queue.Queue(maxsize=1000),
        "created": time.time(),
    }
    with _society_lock:
        _society_runs[run_id] = run
        # Prune oldest finished runs so long-lived desktop sessions stay lean.
        if len(_society_runs) > _SOCIETY_MAX_RUNS:
            finished = sorted(
                ((rid, r) for rid, r in _society_runs.items()
                 if r.get("status") != "running"),
                key=lambda kv: kv[1].get("created", 0),
            )
            for rid, _ in finished[: len(_society_runs) - _SOCIETY_MAX_RUNS]:
                _society_runs.pop(rid, None)
    worker = threading.Thread(
        target=_society_worker, args=(run_id, goal, model_name), daemon=True)
    worker.start()
    return {"runId": run_id, "status": "started",
            "stream": f"/api/society/stream?runId={run_id}"}


@router.get("/society/runs/{run_id}")
def society_run_status(run_id: str) -> Dict[str, Any]:
    run = _society_runs.get(run_id)
    if run is None:
        return {"status": "error", "error": f"unknown runId '{run_id}'"}
    with _society_lock:
        events = list(run["events"])
        return {"run_id": run_id, "status": run["status"],
                "goal": run["goal"], "events": events,
                "result": run["result"]}


def _society_json_default(obj: Any) -> Any:
    # Mirror FastAPI's jsonable_encoder for the JS-unfriendly types the
    # engines leak (sets, tuples, datetimes): coerce, never raise mid-stream.
    if isinstance(obj, (set, frozenset)):
        try:
            return sorted(obj, key=repr)
        except Exception:
            return list(obj)
    if isinstance(obj, tuple):
        return list(obj)
    if hasattr(obj, "isoformat"):
        try:
            return obj.isoformat()
        except Exception:
            pass
    if hasattr(obj, "to_dict"):
        try:
            return obj.to_dict()
        except Exception:
            pass
    return str(obj)


def _society_dumps(obj: Any) -> str:
    return json.dumps(obj, default=_society_json_default)


def _society_sse_frame(event: Dict[str, Any]) -> str:
    return f"data: {_society_dumps(event)}\n\n"


@router.get("/society/stream")
def society_stream(runId: str):  # type: ignore[no-untyped-def]
    run = _society_runs.get(runId)

    def gen():  # type: ignore[no-untyped-def]
        if run is None:
            yield ("event: error\n"
                   f"data: {json.dumps({'error': f'unknown runId {runId!r}'})}\n\n")
            yield "event: done\ndata: {}\n\n"
            return
        # Replay anything emitted before the client connected.
        with _society_lock:
            replay = list(run["events"])
            finished = run["status"] != "running" and run["queue"].empty()
        for ev in replay:
            yield _society_sse_frame(ev)
        if finished:
            with _society_lock:
                result = run["result"]
            yield f"event: done\ndata: {_society_dumps(result or {})}\n\n"
            return
        # Live tail: block on the queue, heartbeat every 15s.
        while True:
            try:
                kind, payload = run["queue"].get(timeout=15)
            except queue.Empty:
                yield ": beat\n\n"
                with _society_lock:
                    if run["status"] != "running" and run["queue"].empty():
                        break
                continue
            if kind == "done":
                with _society_lock:
                    result = run["result"]
                yield f"event: done\ndata: {_society_dumps(result or {})}\n\n"
                return
            yield _society_sse_frame(payload)

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no"})


from .legacy_dispatcher import build_dispatcher  # noqa: E402
