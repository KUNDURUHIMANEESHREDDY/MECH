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
    from backend.agents.evidence_policy import field_map

    return {
        "models": [
            "gpt2-small", "gpt2-medium", "gemma-2b",
            "llama-3-8b", "qwen-7b", "pythia-1b",
            "mistral-7b", "distilgpt2"
        ],
        "provenance": "reference",
        "field_provenance": field_map(("models",), "reference"),
    }


@router.post("/models/load")
def load_model(payload: Dict[str, Any]) -> Dict[str, Any]:
    name = payload.get("model_name", "gpt2-small")
    engine = get_engine()
    if engine and engine.is_available():
        return _mark(engine.load(), "live")
    return {
        "status": "loaded",
        "model_name": name,
        "provenance": "unavailable",
        "field_provenance": {
            "status": "unavailable",
            "model_name": "unavailable",
        },
        "reason": "The model registry is reference-only; no live model was loaded.",
    }


@router.get("/models/{name}")
def get_model_info(name: str) -> Dict[str, Any]:
    return {
        "model_name": name,
        "layers": 12,
        "hidden_size": 768,
        "vocab_size": 50257,
        "num_heads": 12,
        "provenance": "reference",
        "field_provenance": {
            "model_name": "reference",
            "layers": "reference",
            "hidden_size": "reference",
            "vocab_size": "reference",
            "num_heads": "reference",
        },
    }


@router.post("/infer")
def infer(payload: Dict[str, Any]) -> Dict[str, Any]:
    engine = get_engine()
    prompt = payload.get("prompt") or _default_prompt(engine)
    model_name = payload.get("model_name", "gpt2-small")
    if engine and engine.is_available():
        res = engine.infer(prompt, model_name)
        if isinstance(res, dict):
            # Provenance marker: live weights (never silently fake).
            res.setdefault("provenance", "live")
            res.setdefault("field_provenance", {
                "model_name": "live",
                "tokens": "live",
                "generated_text": "live",
                "attention_maps": "live",
                "neuron_activations": "live",
            })
        return res
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
        "provenance": "seeded",
        "field_provenance": {
            "model_name": "seeded",
            "tokens": "seeded",
            "generated_text": "seeded",
            "attention_maps": "seeded",
            "neuron_activations": "seeded",
            "gpu_util": "seeded",
            "memory_util": "seeded",
        },
        "provenance_note": "torch/transformers unavailable: deterministic "
                           "seeded stand-ins, not model measurements",
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
    from backend.agents.evidence_policy import field_map

    return {
        "benchmarks": ["IOI", "Induction", "SAE", "ACDC", "PathPatching"],
        "provenance": "reference",
        "field_provenance": field_map(("benchmarks",), "reference"),
    }


@router.get("/research_catalog")
def research_catalog(item_type: str = "all") -> Dict[str, Any]:
    return {"catalog": get_registry().list_catalog(item_type=item_type)}


@router.post("/benchmarks/run")
def run_benchmark(payload: Dict[str, Any]) -> Dict[str, Any]:
    from backend.agents.evidence_policy import field_map

    name = str(payload.get("benchmark_name", "IOI"))
    try:
        from backend.validation.benchmark_runner import (
            MechanisticBenchmarkRunner,
        )
        res = MechanisticBenchmarkRunner().run_benchmark(f"bench_{name}")
        status = str(res.get("status", "unavailable")).lower()
        provenance = str(res.get("provenance", "unavailable")).lower()
        if status not in {"completed", "passed"} or provenance != "live":
            return {
                "status": status if status in {"unavailable", "error"} else "unavailable",
                "benchmark_name": name,
                "provenance": provenance if provenance in {"unavailable", "seeded", "reference"} else "unavailable",
                "field_provenance": field_map(
                    ("status", "benchmark_name", "score", "pass_rate", "eval_samples"),
                    provenance if provenance in {"unavailable", "seeded", "reference"} else "unavailable",
                ),
                "reason": str(res.get("reason") or "Live benchmark execution is unavailable."),
            }

        try:
            score = float(res["accuracy"])
            pass_rate = float(res.get("robustness_score", score))
        except (KeyError, TypeError, ValueError):
            return {
                "status": "unavailable",
                "benchmark_name": name,
                "provenance": "unavailable",
                "field_provenance": field_map(
                    ("status", "benchmark_name", "score", "pass_rate", "eval_samples"),
                    "unavailable",
                ),
                "reason": "The live executor did not return numeric score fields.",
            }

        return {
            "status": "completed",
            "benchmark_name": name,
            "provenance": "live",
            "field_provenance": field_map(
                ("status", "benchmark_name", "score", "pass_rate", "eval_samples"),
                "live",
            ),
            "score": round(score, 4),
            "pass_rate": round(pass_rate, 4),
            "eval_samples": res.get("eval_samples"),
        }
    except Exception as exc:
        return {
            "status": "error",
            "benchmark_name": name,
            "provenance": "unavailable",
            "field_provenance": field_map(
                ("status", "benchmark_name", "score", "pass_rate", "eval_samples"),
                "unavailable",
            ),
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


@router.get("/settings")
def get_settings() -> Dict[str, Any]:
    from backend.agents.evidence_policy import field_map

    return {
        "status": "ok",
        "provenance": "live",
        "field_provenance": field_map(("status", "settings"), "live"),
        "settings": _store.get_settings(),
    }


@router.put("/settings")
def update_settings(payload: Dict[str, Any]) -> Dict[str, Any]:
    from backend.agents.evidence_policy import field_map
    from backend.storage.database import StorageError

    try:
        settings = _store.update_settings(dict(payload or {}))
    except StorageError as exc:
        return {
            "status": "error",
            "provenance": "unavailable",
            "field_provenance": field_map(("status", "error"), "unavailable"),
            "error": str(exc)[:300],
        }
    return {
        "status": "ok",
        "provenance": "live",
        "field_provenance": field_map(("status", "settings"), "live"),
        "settings": settings,
    }


@router.get("/projects")
def list_projects() -> Dict[str, Any]:
    from backend.agents.evidence_policy import field_map

    return {
        "status": "ok",
        "provenance": "live",
        "field_provenance": field_map(("status", "projects"), "live"),
        "projects": _store.list_recent_projects(),
    }


@router.post("/projects")
def add_project(payload: Dict[str, Any]) -> Dict[str, Any]:
    from backend.agents.evidence_policy import field_map
    from backend.storage.database import StorageError

    try:
        record = _store.add_recent_project(
            str((payload or {}).get("path", "")),
            str((payload or {}).get("name", "")) or None,
        )
    except StorageError as exc:
        return {
            "status": "error",
            "provenance": "unavailable",
            "field_provenance": field_map(("status", "error"), "unavailable"),
            "error": str(exc)[:300],
        }
    return {
        "status": "ok",
        "provenance": "live",
        "field_provenance": field_map(("status", "project"), "live"),
        "project": record,
    }


@router.get("/recent-files")
def list_recent_files() -> Dict[str, Any]:
    from backend.agents.evidence_policy import field_map

    return {
        "status": "ok",
        "provenance": "live",
        "field_provenance": field_map(("status", "files"), "live"),
        "files": _store.list_recent_files(),
    }


@router.post("/recent-files")
def add_recent_file(payload: Dict[str, Any]) -> Dict[str, Any]:
    from backend.agents.evidence_policy import field_map
    from backend.storage.database import StorageError

    try:
        record = _store.add_recent_file(
            str((payload or {}).get("path", "")),
            str((payload or {}).get("project_path", "")) or None,
        )
    except StorageError as exc:
        return {
            "status": "error",
            "provenance": "unavailable",
            "field_provenance": field_map(("status", "error"), "unavailable"),
            "error": str(exc)[:300],
        }
    return {
        "status": "ok",
        "provenance": "live",
        "field_provenance": field_map(("status", "file"), "live"),
        "file": record,
    }


@router.get("/logs")
def list_logs(limit: int = 100) -> Dict[str, Any]:
    from backend.agents.evidence_policy import field_map
    from backend.core.request_log import list_entries

    return {
        "status": "ok",
        "provenance": "live",
        "field_provenance": field_map(("status", "entries"), "live"),
        "source": "backend request log (metadata only, no bodies)",
        "entries": list_entries(limit),
    }


@router.get("/build")
def build_status() -> Dict[str, Any]:
    from backend.agents.evidence_policy import field_map
    from backend.runtime.build_runner import history, status

    return {
        "status": "ok",
        "provenance": "live",
        "field_provenance": field_map(("status", "build", "history"), "live"),
        "build": status(),
        "history": history(),
    }


@router.post("/build")
def start_build() -> Dict[str, Any]:
    from backend.agents.evidence_policy import field_map
    from backend.runtime.build_runner import start, status

    accepted = start()
    return {
        "status": "ok" if accepted.get("accepted") else "busy",
        "provenance": "live",
        "field_provenance": field_map(("status", "build"), "live"),
        "detail": accepted,
        "build": status(),
    }


def _plugin_service():  # type: ignore[no-untyped-def]
    from backend.plugins.service import get_service
    return get_service()


def _plugin_envelope(status: str, plugins: Any = None,
                     error: str = "") -> Dict[str, Any]:
    from backend.agents.evidence_policy import field_map

    body: Dict[str, Any] = {
        "status": status,
        "provenance": "live" if status == "ok" else "unavailable",
        "field_provenance": field_map(
            ("status", "plugins"), "live" if status == "ok" else "unavailable"),
    }
    if plugins is not None:
        body["plugins"] = plugins
    if error:
        body["error"] = error[:500]
    return body


@router.get("/plugins")
def list_plugins() -> Dict[str, Any]:
    try:
        return _plugin_envelope("ok", _plugin_service().list_all())
    except Exception as exc:
        return _plugin_envelope("error", error=str(exc))


@router.post("/plugins/install")
def install_plugin(payload: Dict[str, Any]) -> Dict[str, Any]:
    try:
        record = _plugin_service().install(str((payload or {}).get("path", "")))
        return _plugin_envelope("ok", [record])
    except (ValueError, RuntimeError) as exc:
        return _plugin_envelope("error", error=str(exc))


@router.post("/plugins/{name}/enable")
def enable_plugin(name: str) -> Dict[str, Any]:
    try:
        return _plugin_envelope("ok", [_plugin_service().enable(name)])
    except (ValueError, RuntimeError) as exc:
        return _plugin_envelope("error", error=str(exc))


@router.post("/plugins/{name}/disable")
def disable_plugin(name: str) -> Dict[str, Any]:
    try:
        return _plugin_envelope("ok", [_plugin_service().disable(name)])
    except (ValueError, RuntimeError) as exc:
        return _plugin_envelope("error", error=str(exc))


@router.delete("/plugins/{name}")
def uninstall_plugin(name: str) -> Dict[str, Any]:
    try:
        _plugin_service().uninstall(name)
        return _plugin_envelope("ok", [])
    except (ValueError, RuntimeError) as exc:
        return _plugin_envelope("error", error=str(exc))


@router.get("/discoveries")
def list_discoveries() -> Dict[str, Any]:
    from backend.agents.evidence_policy import field_map

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
        ],
        "provenance": "reference",
        "field_provenance": field_map(("discoveries",), "reference"),
    }


@router.get("/portal/summary")
def portal_summary() -> Dict[str, Any]:
    from backend.agents.evidence_policy import field_map

    return {
        "status": "active",
        "experiments_count": len(_store.list_experiments()),
        "sessions_count": len(_store.list_sessions()),
        "provenance": "live",
        "field_provenance": field_map(
            ("status", "experiments_count", "sessions_count"), "live"),
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


def _mark(res: Any, provenance: str) -> Any:
    """Stamp a response and each top-level returned field with provenance."""
    if isinstance(res, dict):
        res.setdefault("provenance", provenance)
        if not isinstance(res.get("field_provenance"), dict):
            res["field_provenance"] = {
                str(key): provenance
                for key in res
                if key not in {"provenance", "field_provenance"}
            }
    return res


@router.post("/gpt2/load")
def gpt2_load(payload: Dict[str, Any]) -> Dict[str, Any]:
    engine = get_engine()
    if engine and engine.is_available():
        return _mark(engine.load(), "live")
    return {
        "status": "loaded",
        "provenance": "seeded",
        "field_provenance": {
            "status": "seeded",
            "model_name": "seeded",
            "n_layers": "seeded",
            "n_heads": "seeded",
            "d_model": "seeded",
            "d_mlp": "seeded",
            "device": "seeded",
        },
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


# Short openers used only to seed live generation. The returned prompt text
# itself is always sampled from the model, never taken from a pool.
PRIMER_OPENERS = [
    "The", "When", "In", "Scientists", "Once", "Deep",
    "After", "A", "Researchers", "The future",
]


def _fresh_primer() -> str:
    return PRIMER_OPENERS[int(time.time() // 60) % len(PRIMER_OPENERS)]


def _generate_fresh_prompt(engine: Any, max_new_tokens: int = 12) -> str:
    """Ask the live model for a brand-new prompt; pool fallback on failure."""
    try:
        res = engine.generate_text(_fresh_primer(),
                                   max_new_tokens=max_new_tokens)
        text = str(res.get("text", "")).strip()
        if text:
            return text
    except Exception:
        pass
    return _random_prompt()


def _default_prompt(engine: Any) -> str:
    """Prompt default: live generation when the model is up, pool otherwise."""
    if engine is not None and engine.is_available():
        return _generate_fresh_prompt(engine, max_new_tokens=8)
    return _random_prompt()


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


@router.post("/gpt2/fresh_prompt")
def gpt2_fresh_prompt(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Generate a brand-new prompt from the live model (never from a pool).

    Samples a continuation from a rotating opener using the loaded weights.
    Fails closed when the engine is unavailable: no seeded or pooled text is
    ever substituted, since the whole point is model-created prompts.
    """
    engine = get_engine()
    if not (engine and engine.is_available()):
        return _mark({
            "status": "unavailable",
            "error": "torch/transformers not available — the model cannot create a prompt",
        }, "unavailable")
    primer = str(payload.get("primer") or _fresh_primer())
    try:
        max_new_tokens = max(1, min(48, int(payload.get("max_new_tokens", 12))))
    except (TypeError, ValueError):
        max_new_tokens = 12
    try:
        res = engine.generate_text(primer, max_new_tokens=max_new_tokens)
    except Exception as exc:
        return _mark({
            "status": "error",
            "primer": primer,
            "error": str(exc)[:300],
        }, "unavailable")
    if not isinstance(res, dict) or res.get("status") != "ok" or not str(res.get("text", "")).strip():
        return _mark({
            "status": "unavailable",
            "primer": primer,
            "error": "The model did not return usable prompt text.",
        }, "unavailable")
    return _mark({
        "status": "ok",
        "prompt": str(res["text"]).strip(),
        "primer": primer,
        "method": ("sampled continuation (temperature "
                   f"{res.get('temperature')}, top-k {res.get('top_k')}) "
                   "from live GPT-2 weights"),
        "tokens_added": res.get("tokens_added"),
    }, "live")


@router.post("/gpt2/run_prompt")
def gpt2_run_prompt(payload: Dict[str, Any]) -> Dict[str, Any]:
    engine = get_engine()
    prompt = payload.get("prompt") or _default_prompt(engine)
    if engine and engine.is_available():
        return _mark(engine.run_prompt(prompt), "live")
    str_tokens = [t for t in prompt.replace(",", " ,").replace(".", " .").split() if t]
    seed = _seed(prompt)
    rng = random.Random(seed)
    all_tokens = NAMES + ["Paris", "London", "Berlin", "Madrid"]
    ranked = [(t, round(rng.uniform(-3.0, 2.0), 4)) for t in all_tokens]
    ranked.sort(key=lambda kv: -kv[1])
    top5 = [{"token": t, "logit": round(v, 4)} for t, v in ranked[:5]]
    top16 = [{"token": t, "logit": round(v, 4)} for t, v in ranked[:16]]
    return _mark({
        "status": "ok",
        "prompt": prompt,
        "str_tokens": str_tokens,
        "top5": top5,
        "top16": top16,
        "next_token": ranked[0][0],
    }, "seeded")


@router.post("/gpt2/activations")
def gpt2_activations(payload: Dict[str, Any]) -> Dict[str, Any]:
    engine = get_engine()
    if engine and engine.is_available():
        return _mark(engine.activations(int(payload.get("layer", 0))), "live")
    layer = max(0, min(11, int(payload.get("layer", 0))))
    seq = int(payload.get("seq_len", 12))
    return _mark({
        "status": "ok",
        "layer": layer,
        "resid_shape": [seq, 768],
        "attn_shape": [12, seq, seq],
        "mlp_shape": [seq, 3072],
    }, "seeded")


@router.post("/gpt2/attention_head")
def gpt2_attention_head(payload: Dict[str, Any]) -> Dict[str, Any]:
    engine = get_engine()
    if engine and engine.is_available():
        return _mark(engine.attention_head(int(payload.get("layer", 0)), int(payload.get("head", 0))), "live")
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
    return _mark({"status": "ok", "layer": layer, "head": head,
                  "matrix": matrix, "str_tokens": tokens}, "seeded")


@router.post("/gpt2/patch_head")
@router.post("/gpt2/patchhead")
def gpt2_patch_head(payload: Dict[str, Any]) -> Dict[str, Any]:
    engine = get_engine()
    if engine and engine.is_available():
        return _mark(engine.patch_head(
            int(payload.get("layer", 9)),
            int(payload.get("head", 9)),
            payload.get("pos_token", "Paris"),
            payload.get("neg_token", "London"),
        ), "live")
    layer = int(payload.get("layer", 9)) % 12
    head = int(payload.get("head", 9)) % 12
    pos_token = payload.get("pos_token", "Paris")
    neg_token = payload.get("neg_token", "London")
    rng = random.Random(_seed(f"{layer}:{head}:{pos_token}:{neg_token}"))
    clean_ld = round(rng.uniform(1.5, 3.5), 4)
    patched_ld = round(rng.uniform(-1.2, 0.9), 4)
    delta = round(patched_ld - clean_ld, 4)
    return _mark({
        "status": "ok",
        "layer": layer,
        "head": head,
        "clean_ld": clean_ld,
        "patched_ld": patched_ld,
        "delta": delta,
        "direction": "hurts" if delta < 0 else "helps",
    }, "seeded")


@router.post("/gpt2/steer")
def gpt2_steer(payload: Dict[str, Any]) -> Dict[str, Any]:
    engine = get_engine()
    if engine and engine.is_available():
        return _mark(engine.steer(
            payload.get("prompt") or "The movie was",
            int(payload.get("layer", 8)),
            payload.get("pos_prompt") or "It was a fantastic wonderful amazing film. The movie was",
            payload.get("neg_prompt") or "It was a terrible awful horrible film. The movie was",
            float(payload.get("alpha", 25.0)),
        ), "live")
    return _mark({
        "status": "unavailable",
        "error": "torch/transformers not available — steering needs live weights",
    }, "unavailable")


@router.post("/gpt2/ioi")
def gpt2_ioi(payload: Dict[str, Any]) -> Dict[str, Any]:
    io_name = payload.get("io_name") or random.choice(NAMES)
    subj_name = payload.get("subj_name") or random.choice([n for n in NAMES if n != io_name])
    engine = get_engine()
    if engine and engine.is_available():
        return _mark(engine.ioi(io_name, subj_name), "live")
    rng = random.Random(_seed(f"{io_name}:{subj_name}"))
    clean_prompt = f"When {subj_name} and {io_name} went to the store, {subj_name} gave a bottle to"
    corrupted_prompt = f"When {subj_name} and {io_name} went to the store, {io_name} gave a bottle to"
    return _mark({
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
    }, "seeded")


# ---------------------------------------------------------------------------
# Dynamic GPT-2 architecture / neuron explorer (real weights, not hardcoded)
# ---------------------------------------------------------------------------

@router.post("/gpt2/architecture")
def gpt2_architecture(payload: Dict[str, Any] = None) -> Dict[str, Any]:
    engine = get_engine()
    if engine and engine.is_available():
        return _mark(engine.architecture(), "live")
    return _mark({
        "status": "unavailable",
        "error": "torch/transformers not available — cannot load real GPT-2",
    }, "unavailable")


@router.post("/gpt2/layer")
def gpt2_layer(payload: Dict[str, Any]) -> Dict[str, Any]:
    layer = int(payload.get("layer", 0))
    engine = get_engine()
    if engine and engine.is_available():
        return _mark(engine.layer_detail(layer), "live")
    return _mark({"status": "unavailable",
                  "error": "torch/transformers not available"}, "unavailable")


@router.post("/gpt2/neurons")
def gpt2_neurons(payload: Dict[str, Any]) -> Dict[str, Any]:
    engine = get_engine()
    if engine and engine.is_available():
        return _mark(engine.list_neurons(
            layer=int(payload.get("layer", 0)),
            component=str(payload.get("component", "mlp")),
            page=int(payload.get("page", 0)),
            page_size=int(payload.get("page_size", 128)),
            sort_by=str(payload.get("sort_by", "index")),
            order=str(payload.get("order", "asc")),
            q=str(payload.get("q", "")),
        ), "live")
    return _mark({"status": "unavailable",
                  "error": "torch/transformers not available"}, "unavailable")


@router.post("/gpt2/neuron")
def gpt2_neuron(payload: Dict[str, Any]) -> Dict[str, Any]:
    engine = get_engine()
    if engine and engine.is_available():
        return _mark(engine.neuron_detail(
            layer=int(payload.get("layer", 0)),
            neuron_index=int(payload.get("neuron_index", 0)),
            component=str(payload.get("component", "mlp")),
            top_k_weights=int(payload.get("top_k_weights", 16)),
        ), "live")
    return _mark({"status": "unavailable",
                  "error": "torch/transformers not available"}, "unavailable")


@router.post("/gpt2/head")
def gpt2_head(payload: Dict[str, Any]) -> Dict[str, Any]:
    engine = get_engine()
    if engine and engine.is_available():
        return _mark(engine.head_detail(
            layer=int(payload.get("layer", 0)),
            head=int(payload.get("head", 0)),
        ), "live")
    return _mark({"status": "unavailable",
                  "error": "torch/transformers not available"}, "unavailable")


@router.post("/gpt2/patch_neuron")
def gpt2_patch_neuron(payload: Dict[str, Any]) -> Dict[str, Any]:
    engine = get_engine()
    if engine and engine.is_available():
        return _mark(engine.patch_neuron(
            layer=int(payload.get("layer", 0)),
            neuron_index=int(payload.get("neuron_index", 0)),
            patch_value=float(payload.get("patch_value", 0.0)),
            prompt=payload.get("prompt"),
        ), "live")
    return _mark({"status": "unavailable",
                  "error": "torch/transformers not available"}, "unavailable")


@router.post("/gpt2/layer_activations")
def gpt2_layer_activations(payload: Dict[str, Any]) -> Dict[str, Any]:
    engine = get_engine()
    if engine and engine.is_available():
        try:
            return _mark(engine.layer_activations(
                layer=int(payload.get("layer", 0)),
                prompt=payload.get("prompt") or "The capital of France is",
            ), "live")
        except Exception as exc:
            return _mark({"status": "error", "error": str(exc)[:300]}, "unavailable")
    return _mark({"status": "unavailable",
                  "error": "torch/transformers not available"}, "unavailable")


@router.post("/gpt2/logit_lens_all")
def gpt2_logit_lens_all(payload: Dict[str, Any]) -> Dict[str, Any]:
    engine = get_engine()
    if engine and engine.is_available():
        try:
            return _mark(engine.logit_lens_all(
                prompt=payload.get("prompt") or "The capital of France is",
            ), "live")
        except Exception as exc:
            return _mark({"status": "error", "error": str(exc)[:300]}, "unavailable")
    return _mark({"status": "unavailable",
                  "error": "torch/transformers not available"}, "unavailable")


@router.get("/circuits")
def list_circuits() -> Dict[str, Any]:
    try:
        from backend.science.explorer.circuit_explorer import CircuitExplorer
        return {
            "circuits": CircuitExplorer().list_circuits(),
            "provenance": "reference",
            "field_provenance": {"circuits": "reference"},
        }
    except Exception as exc:
        return {
            "status": "error",
            "provenance": "unavailable",
            "field_provenance": {"circuits": "unavailable"},
            "error": str(exc)[:300],
        }


@router.get("/circuits/{circuit_id}")
def get_circuit(circuit_id: str) -> Dict[str, Any]:
    try:
        from backend.science.explorer.circuit_explorer import CircuitExplorer
        res = CircuitExplorer().get_circuit(circuit_id=circuit_id)
        if res is None:
            return {
                "status": "unavailable",
                "provenance": "unavailable",
                "field_provenance": {"circuit": "unavailable"},
                "error": f"unknown circuit_id '{circuit_id}'",
            }
        if isinstance(res, dict):
            return res
        return {
            "status": "unavailable",
            "provenance": "unavailable",
            "field_provenance": {"circuit": "unavailable"},
            "error": "Circuit registry returned an unreadable record.",
        }
    except Exception as exc:
        return {
            "status": "error",
            "provenance": "unavailable",
            "field_provenance": {"circuit": "unavailable"},
            "error": str(exc)[:300],
        }


@router.get("/figures/attention")
def figure_attention(prompt: str = "The capital of France is",
                     layer: int = 10, head: int = 7):
    """Publication-ready attention heatmap PNG from live weights."""
    from fastapi.responses import Response
    engine = get_engine()
    if not (engine and engine.is_available()):
        return {"status": "error",
                "error": "torch/transformers not available"}
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        # Prime the cache for the requested prompt, then read the head.
        engine.run_prompt(prompt)
        res = engine.attention_head(int(layer) % 12, int(head) % 12)
        matrix = res.get("matrix") or []
        tokens = res.get("str_tokens") or []
        if not matrix:
            return {"status": "error", "error": "empty attention matrix"}
        import io
        fig, ax = plt.subplots(
            figsize=(max(4.0, len(tokens) * 0.9), max(3.2, len(tokens) * 0.7)))
        im = ax.imshow(matrix, cmap="viridis", aspect="auto")
        ax.set_xticks(range(len(tokens)))
        ax.set_yticks(range(len(tokens)))
        ax.set_xticklabels(tokens, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(tokens, fontsize=8)
        ax.set_title(f"GPT-2 L{int(layer) % 12}H{int(head) % 12}: {prompt[:48]}",
                     fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=150)
        plt.close(fig)
        buf.seek(0)
        return Response(content=buf.read(), media_type="image/png",
                        headers={"Content-Disposition":
                                 f"inline; filename=attn_L{layer}_H{head}.png"})
    except Exception as exc:
        return {"status": "error", "error": str(exc)[:300]}


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
            run_id=run_id,
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
    # Durable per-run evidence record (survives restarts).
    try:
        from backend.core.evidence_graph import save_run_record
        save_run_record(run_id, {"run_id": run_id, "goal": goal,
                                 "model_name": model_name,
                                 "status": status, "result": result})
    except Exception:
        pass
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


@router.get("/society/runs")
def society_run_list() -> Dict[str, Any]:
    """Persisted per-run evidence records (durable across restarts)."""
    try:
        from backend.core.evidence_graph import list_run_records
        return {"runs": list_run_records()}
    except Exception as exc:
        return {"status": "error", "error": str(exc)[:300]}


@router.get("/society/runs/{run_id}")
def society_run_status(run_id: str) -> Dict[str, Any]:
    run = _society_runs.get(run_id)
    if run is not None:
        with _society_lock:
            events = list(run["events"])
            return {"run_id": run_id, "status": run["status"],
                    "goal": run["goal"], "events": events,
                    "result": run["result"]}
    # Fall back to the persisted record (post-restart reads).
    try:
        from backend.core.evidence_graph import load_run_record
        return load_run_record(run_id)
    except Exception:
        return {"status": "error", "error": f"unknown runId '{run_id}'"}


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
