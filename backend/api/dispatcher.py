from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from typing import Dict, Any, List, Optional
import asyncio
import importlib.util
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
from backend.core.identifiers import entity_id
from backend.core.provenance import pass_through

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
        "status": "unavailable",
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
    """Architecture of the model actually in memory, or one honest refusal.

    Fixed 12/12/768/50257 used to be returned for any name — including
    ``gemma-2b`` and ``llama-3-8b``, which have different shapes entirely and
    no loader in this engine. Now: a name matching the loaded weights gets
    real config-derived dimensions; anything else is reported as a mismatch
    with the loaded identity attached, so a response can never describe a
    model that did not produce it.
    """
    engine = get_engine()
    if engine and engine.is_available():
        res = engine.info()
        if isinstance(res, dict) and res.get("status") == "loaded":
            loaded = str(res.get("model_name", ""))
            if name == loaded:
                return _mark(res, "live")
            # Pass through the real description, then record the mismatch —
            # never plain dimensions under the requested name.
            out = dict(res)
            out["model_name"] = name
            out["model_requested"] = name
            out["model_loaded"] = loaded
            out["model_mismatch"] = True
            out["reason"] = (f"model mismatch: requested {name!r} but the engine "
                             f"has {loaded!r} weights in memory; the dimensions "
                             "below describe the loaded weights, not the "
                             "requested model")
            out["provenance"] = "live"
            out["attested"] = res.get("attested", False)
            out["field_provenance"] = {"model_name": "live", **{
                k: "live" for k in res if k not in
                {"status", "reason", "model_name"}
            }}
            return out
    return _mark({
        "status": "unavailable",
        "model_name": name,
        "error": "torch/transformers not available — no weights in memory",
    }, "unavailable")


@router.post("/infer")
def infer(payload: Dict[str, Any]) -> Dict[str, Any]:
    engine = get_engine()
    supplied = payload.get("prompt")
    if supplied is not None and not isinstance(supplied, str):
        raise HTTPException(400, detail="'prompt' must be a string")
    prompt = supplied or _default_prompt(engine)
    _check_prompt(prompt)
    model_name = _as_text(payload, "model_name", "gpt2-small")
    if engine and engine.is_available():
        res = engine.infer(prompt, model_name)
        # Provenance is passed through, never inferred.
        #
        # Was `res.setdefault("provenance", "live")` with a fixed
        # `field_provenance` naming five fields live. Two problems:
        #
        #  * `engine.is_available()` reports that torch and transformers
        #    imported. It does not report that a forward pass ran, so an engine
        #    returning `{"status": "unavailable"}` was relabelled live.
        #  * the fixed field list asserted `attention_maps` and
        #    `neuron_activations` were live even when the engine returned neither.
        #
        # The measurement layer now originates provenance and names the weights
        # it actually loaded -- see backend/core/provenance.py. If it attested
        # nothing, the response is withheld rather than upgraded here.
        return pass_through(res) if isinstance(res, dict) else res
    # Fail closed: no live model means no inference.
    # Seeded fallbacks produce convincing but fake results that can be
    # mistaken for real measurements. Return unavailable instead.
    return _mark({
        "status": "unavailable",
        "model_name": model_name,
        "error": "torch/transformers not available — inference requires live weights",
        "provenance_note": "No live model loaded. Seeded fallbacks were removed because they produce convincing but fake results.",
    }, "unavailable")


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
            # Propagated from the runner's own measurement, never asserted:
            # this route narrows the record to scores, it does not measure.
            "attested": res.get("attested", False),
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
    # Was `f"exp_{hash(str(payload)) % 10000}"`. See
    # backend.core.identifiers for why a salted 10,000-value namespace is the
    # wrong thing to hand a persisted row.
    item.setdefault("id", entity_id("exp_"))
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
    item.setdefault("id", entity_id("sess_"))
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


#: The four rungs every execution engine is measured on, weakest first.
#:
#: A single `reachable` boolean had to answer four unrelated questions, so the
#: cheapest one won: `importlib.util.find_spec("ray") is not None` reported a
#: Ray *cluster*. The ladder keeps them apart, and `_ladder` clamps it so a
#: probe cannot claim a rung it did not earn.
ENGINE_STATES = ("installed", "configured", "reachable", "execution_ready")

#: Wall-clock budget for any one network or subprocess handshake. A status
#: endpoint must not hang waiting on a cluster that is down.
_PROBE_TIMEOUT_SECONDS = 2.0


def _module_present(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError, AttributeError):
        return False


def _binary_on_path(name: str) -> Optional[str]:
    import shutil
    return shutil.which(name)


def _env_configured(*names: str) -> bool:
    return any(os.environ.get(n) for n in names)


def _tcp_reachable(host: str, port: int, timeout: float) -> bool:
    """A completed TCP handshake. Says the endpoint is there, nothing more."""
    import socket
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _run_probe(argv: List[str], timeout: Optional[float] = None):
    """Run a read-only cluster query under a hard timeout.

    Returns `(returncode, stdout)`. A failure is `(nonzero, "")` rather than
    an exception, so a dead controller reads as "did not answer" instead of
    crashing the endpoint.
    """
    import subprocess
    try:
        done = subprocess.run(
            argv, capture_output=True, text=True,
            timeout=timeout or _PROBE_TIMEOUT_SECONDS)
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        return -1, f"{type(exc).__name__}: {exc}"
    return done.returncode, (done.stdout or "")


def _ladder(installed: bool, configured: bool, reachable: bool,
            execution_ready: bool, detail: str) -> Dict[str, Any]:
    """Assemble the four rungs, enforcing that the ladder only goes up.

    Clamping here means no individual probe can accidentally over-claim: an
    engine with no client library cannot be reachable however willing the
    network is, and one that answered cannot be execution-ready unless it also
    has somewhere to execute.
    """
    held = [bool(installed), bool(configured),
            bool(reachable), bool(execution_ready)]
    for i in range(1, len(held)):
        held[i] = held[i] and held[i - 1]
    state = "absent"
    for name, ok in zip(ENGINE_STATES, held):
        if ok:
            state = name
    return {
        "installed": held[0],
        "configured": held[1],
        "reachable": held[2],
        "execution_ready": held[3],
        "state": state,
        "detail": detail,
    }


def _local_compute_path_works():
    """Does the compute backend actually run, not merely import?

    Importing torch proves a library is on disk. Allocating proves the
    runtime underneath it initialised -- which is the difference between
    "installed" and "reachable" for an in-process engine.
    """
    try:
        import torch
        torch.zeros(1)
        return True, "compute path verified by allocation"
    except Exception as exc:  # noqa: BLE001
        return False, f"compute path unusable: {type(exc).__name__}: {exc}"


def _local_model_resident(engine) -> tuple:
    """`execution_ready` for the local engine means a run would start now.

    `info()` returns an error dict when nothing is loaded, so this reads the
    engine's own state instead of assuming `installed` implies runnable.
    """
    try:
        described = engine.info()
    except Exception as exc:  # noqa: BLE001
        return False, f"model state unreadable: {type(exc).__name__}: {exc}"
    if not isinstance(described, dict) or described.get("status") == "error":
        return False, "executor reachable; no model resident yet"
    return True, "executor reachable with a resident model"


def _probe_local() -> Dict[str, Any]:
    try:
        from backend.services import gpt2_engine
    except Exception as exc:  # noqa: BLE001
        return _ladder(False, False, False, False,
                       f"executor import failed: {type(exc).__name__}: {exc}")
    if not gpt2_engine.is_available():
        return _ladder(False, False, False, False,
                       "torch/transformers not installed")
    # The local executor has no external endpoint, so configuration is
    # complete the moment the library is present.
    reachable, detail = _local_compute_path_works()
    if not reachable:
        return _ladder(True, True, False, False, detail)
    ready, ready_detail = _local_model_resident(gpt2_engine)
    return _ladder(True, True, True, ready, ready_detail)


def _probe_distributed() -> Dict[str, Any]:
    try:
        from backend.distributed import scheduler  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        return _ladder(False, False, False, False,
                       f"scheduler import failed: {type(exc).__name__}: {exc}")
    # Deliberately stops at `installed`. `ResourceManager` seeds two
    # "NVIDIA A100-80GB" profiles and a 64-core EPYC cluster as literals, and
    # `DistributedWorker` has no socket, HTTP or subprocess transport -- there
    # is no endpoint to reach. Reporting that registry as a live cluster is
    # how a laptop with no GPUs claims a datacentre, so the ladder refuses to
    # climb past a fact we can actually observe.
    return _ladder(True, False, False, False,
                   "scheduler imports, but no worker transport is configured; "
                   "its bundled worker registry is seeded with placeholder "
                   "profiles, so there is no endpoint to reach")


def _kubeconfig_endpoint():
    """Resolve `(host, port, context)` from the active kubeconfig."""
    try:
        from urllib.parse import urlparse
        from kubernetes import client as k8s_client, config as kube_config

        contexts, active = kube_config.list_kube_config_contexts()
        if not active:
            return None, None, None
        kube_config.load_kube_config(context=active["name"])
        host = k8s_client.CoreV1Api().api_client.configuration.host
        parsed = urlparse(host if "//" in host else f"//{host}")
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        return parsed.hostname, port, active["name"]
    except Exception:  # noqa: BLE001
        return None, None, None


def _kube_api_answers(context_name: str) -> tuple:
    try:
        from kubernetes import client as k8s_client, config as kube_config

        kube_config.load_kube_config(context=context_name)
        nodes = k8s_client.CoreV1Api().list_node(
            _request_timeout=(2, int(_PROBE_TIMEOUT_SECONDS)))
    except Exception as exc:  # noqa: BLE001
        return False, f"api server did not answer: {type(exc).__name__}: {exc}"
    count = len(getattr(nodes, "items", None) or [])
    if not count:
        return False, "api server answered, but the cluster has no nodes"
    return True, f"api server answered with {count} node(s)"


def _probe_kubernetes() -> Dict[str, Any]:
    if not _module_present("kubernetes"):
        return _ladder(False, False, False, False,
                       "kubernetes client not installed")
    host, port, context = _kubeconfig_endpoint()
    if not host or not context:
        return _ladder(True, False, False, False,
                       "client installed, but no kubeconfig context resolves")
    if not _tcp_reachable(host, port, _PROBE_TIMEOUT_SECONDS):
        return _ladder(True, True, False, False,
                       f"kubeconfig points at {host}:{port}, which did not answer")
    ready, detail = _kube_api_answers(context)
    return _ladder(True, True, True, ready, detail)


def _probe_slurm() -> Dict[str, Any]:
    if not _binary_on_path("sbatch"):
        return _ladder(False, False, False, False, "sbatch not on PATH")
    if not _env_configured("SLURM_CLUSTER_NAME", "SLURM_CONF", "SCRATCH"):
        return _ladder(True, False, False, False,
                       "sbatch on PATH, but no cluster named by the environment")
    code, out = _run_probe(["sinfo", "--noheader", "--format=%D %t"])
    if code != 0:
        return _ladder(True, True, False, False,
                       f"controller did not answer sinfo (exit {code})")
    partitions = [line for line in out.splitlines() if line.strip()]
    if not partitions:
        return _ladder(True, True, True, False,
                       "controller answered, but reports no partitions")
    return _ladder(True, True, True, True,
                   f"controller reports {len(partitions)} partition(s)")


def _ray_endpoint(address: str) -> tuple:
    stripped = address.split("://", 1)[-1].split("/", 1)[0]
    host, _, port = stripped.partition(":")
    try:
        return host, int(port) if port else 6379
    except ValueError:
        return host, 6379


def _ray_cluster_answers(address: str) -> tuple:
    try:
        import ray

        ray.init(address=address, ignore_reinit_error=True,
                 connect_timeout_ms=int(_PROBE_TIMEOUT_SECONDS * 1000),
                 logging_level="ERROR")
        try:
            resources = ray.cluster_resources() or {}
        finally:
            ray.shutdown()
    except Exception as exc:  # noqa: BLE001
        return False, f"ray did not answer: {type(exc).__name__}: {exc}"
    if not resources.get("CPU"):
        return False, "ray answered, but reports no CPU capacity"
    return True, f"ray answered with {resources.get('CPU')} CPU(s)"


def _probe_ray() -> Dict[str, Any]:
    if not _module_present("ray"):
        return _ladder(False, False, False, False, "ray not installed")
    address = os.environ.get("RAY_ADDRESS")
    if not address:
        return _ladder(True, False, False, False,
                       "ray installed, but RAY_ADDRESS names no cluster")
    host, port = _ray_endpoint(address)
    if not host or not _tcp_reachable(host, port, _PROBE_TIMEOUT_SECONDS):
        return _ladder(True, True, False, False,
                       f"RAY_ADDRESS {address} did not answer")
    ready, detail = _ray_cluster_answers(address)
    return _ladder(True, True, True, ready, detail)


#: Engine name -> probe. Kept as a table so a probe can be substituted in a
#: test without reaching into the dispatch logic.
_ENGINE_PROBES = {
    "local": _probe_local,
    "distributed": _probe_distributed,
    "kubernetes": _probe_kubernetes,
    "slurm": _probe_slurm,
    "ray": _probe_ray,
}


def _probe_engine(name: str) -> Dict[str, Any]:
    """Measure one engine on the capability ladder. Fails closed on error."""
    probe = _ENGINE_PROBES.get(name)
    if probe is None:
        return _ladder(False, False, False, False,
                       f"no probe is defined for engine {name!r}")
    try:
        return probe()
    except Exception as exc:  # noqa: BLE001
        return _ladder(False, False, False, False,
                       f"probe failed: {type(exc).__name__}: {exc}")


@router.post("/runtime/status")
def runtime_status() -> Dict[str, Any]:
    """Report what is actually reachable, not a fixed engine list.

    This previously returned `{"status": "connected", "engines": [local,
    distributed, kubernetes, slurm, ray]}` unconditionally -- claiming a Slurm
    and Ray cluster connection that was never probed, so a UI reading this had
    no way to tell an idle laptop from a cluster.

    Each engine is now measured on the four-rung ladder in `ENGINE_STATES`,
    and `engines` lists only the ones that were actually reached.
    `execution_ready_engines` is the narrower list that could accept a job
    right now -- an engine that answers but has no capacity belongs in the
    first list and not the second.
    """
    engines = {name: _probe_engine(name) for name in _ENGINE_PROBES}

    reachable = [n for n, v in engines.items() if v["reachable"]]
    ready = [n for n, v in engines.items() if v["execution_ready"]]
    return {
        # "connected" is only correct when at least one engine was verified.
        "status": "connected" if reachable else "unavailable",
        "engines": reachable,
        "execution_ready_engines": ready,
        "engine_detail": engines,
        "reason": (None if reachable else
                   "No execution engine was reachable; the platform is "
                   "serving but cannot execute a run."),
        "provenance": "live",
        "validation_eligible": False,
        "publication_eligible": False,
    }


@router.post("/runtime/analyze_tokens")
def runtime_analyze_tokens(payload: Dict[str, Any]) -> Dict[str, Any]:
    prompt = payload.get("prompt", "")
    if not isinstance(prompt, str):
        raise HTTPException(400, detail="'prompt' must be a string")
    _check_prompt(prompt)
    tokens = [t for t in prompt.replace(",", " ,").replace(".", " .").split() if t]
    return {"prompt": prompt, "tokens": tokens, "count": len(tokens)}


# NOTE: the seeded per-input stand-ins (`_seed`, `_seeded_tokens`,
# `TOKEN_POOLS`, `_random_token_sequence`) were deleted. They produced
# deterministic fake measurements; every route now fails closed instead.
# `PROMPT_POOL`/`_random_prompt` survive only as *input* defaults for live
# runs (a prompt to measure, never a measurement).


# ── Payload coercion ────────────────────────────────────────────────────────
# Every route below takes an untyped JSON body. Coercing raw user input with a
# bare int()/float()/str() turns a malformed request into an unhandled
# ValueError and a 500. These helpers validate first and raise HTTPException
# (400) so a bad payload is a client error, never a server crash.

def _as_int(payload: Dict[str, Any], key: str, default: int) -> int:
    value = payload.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise HTTPException(400, detail=f"'{key}' must be an integer")
    try:
        return int(value)
    except (TypeError, ValueError):
        raise HTTPException(400, detail=f"'{key}' must be an integer")


def _as_float(payload: Dict[str, Any], key: str, default: float) -> float:
    value = payload.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise HTTPException(400, detail=f"'{key}' must be a number")
    try:
        return float(value)
    except (TypeError, ValueError):
        raise HTTPException(400, detail=f"'{key}' must be a number")


def _as_text(payload: Dict[str, Any], key: str, default: str = "") -> str:
    value = payload.get(key, default)
    if value is None:
        return default
    if not isinstance(value, str):
        raise HTTPException(400, detail=f"'{key}' must be a string")
    return value


def _opt_text(payload: Dict[str, Any], key: str) -> Optional[str]:
    """Optional string; a supplied non-string is still a client error."""
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise HTTPException(400, detail=f"'{key}' must be a string")
    return value


_MAX_PROMPT_CHARS = 2048


def _check_prompt(prompt: str) -> None:
    """Reject prompt-bearing API inputs above a fixed size bound.

    The live engine path is bounded by GPT-2's 1024-token context, but an
    unbounded prompt can still demand excessive compute, so bound at the
    boundary.
    """
    if isinstance(prompt, str) and len(prompt) > _MAX_PROMPT_CHARS:
        raise HTTPException(
            400, detail=f"prompt exceeds {_MAX_PROMPT_CHARS} characters")


# NOTE: `_prime_engine_cache` was removed. Split prime-then-read had a race
# window: thread A primes prompt A, thread B primes prompt B, then A reads
# B's activations. Readers now take an explicit `prompt=` and do
# ensure-then-read atomically under MODEL_LOCK inside the engine.


def _mark(res: Any, provenance: str) -> Any:
    """Ensure a response carries a provenance label, never inventing `live`.

    The `provenance` argument is legacy: availability (`engine.is_available()`)
    says the ML stack imports, not that a forward pass ran, so a wrapper's
    claim can never be the source of scientific `live`. An engine label —
    attested or not — is preserved exactly as the measurement layer set it;
    an unlabeled record is withheld as unmeasured rather than upgraded.
    """
    if isinstance(res, dict):
        from backend.core.provenance import pass_through
        return pass_through(res)
    return res


@router.post("/gpt2/load")
def gpt2_load(payload: Dict[str, Any]) -> Dict[str, Any]:
    engine = get_engine()
    if engine and engine.is_available():
        return _mark(engine.load(), "live")
    # Fail closed: hardcoded dims (12/12/768/3072) are indistinguishable
    # from a real measurement. No live weights => unavailable.
    return _mark({
        "status": "unavailable",
        "error": "torch/transformers not available — cannot load real GPT-2",
    }, "unavailable")


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
        generated = _generate_fresh_prompt(engine, max_new_tokens=8)
        _check_prompt(generated)
        return generated[:_MAX_PROMPT_CHARS]
    return _random_prompt()


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
    primer = _as_text(payload, "primer", "") or _fresh_primer()
    max_new_tokens = max(1, min(48, _as_int(payload, "max_new_tokens", 12)))
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
    _check_prompt(prompt)
    if engine and engine.is_available():
        return _mark(engine.run_prompt(prompt), "live")
    # Fail closed: no live model means no prompt execution.
    # Seeded fallbacks produce convincing but fake results.
    return _mark({
        "status": "unavailable",
        "prompt": prompt,
        "error": "torch/transformers not available — run_prompt requires live weights",
    }, "unavailable")


@router.post("/gpt2/activations")
def gpt2_activations(payload: Dict[str, Any]) -> Dict[str, Any]:
    engine = get_engine()
    layer = _as_int(payload, "layer", 0)
    seq = _as_int(payload, "seq_len", 12)
    prompt = _opt_text(payload, "prompt") or ""
    _check_prompt(prompt)
    if engine and engine.is_available():
        # Atomic ensure-then-read inside the engine (under MODEL_LOCK):
        # the shapes always describe the requested prompt, never whatever
        # another thread ran last. The old split prime-then-read had a race
        # window between the two calls.
        return _mark(engine.activations(layer, prompt=prompt or None), "live")
    # Fail closed: no live model means no activation data.
    # Seeded fallbacks produce convincing but fake shape data.
    return _mark({
        "status": "unavailable",
        "layer": layer,
        "error": "torch/transformers not available — activations require live weights",
    }, "unavailable")


@router.post("/gpt2/attention_head")
def gpt2_attention_head(payload: Dict[str, Any]) -> Dict[str, Any]:
    engine = get_engine()
    raw_layer = _as_int(payload, "layer", 0)
    raw_head = _as_int(payload, "head", 0)
    prompt = _opt_text(payload, "prompt") or ""
    _check_prompt(prompt)
    if engine and engine.is_available():
        # Atomic ensure-then-read inside the engine (under MODEL_LOCK).
        return _mark(engine.attention_head(raw_layer, raw_head, prompt=prompt or None), "live")
    layer = raw_layer % 12
    head = raw_head % 12
    # Fail closed: no live model means no attention patterns.
    # Seeded fallbacks produce convincing but fake attention matrices.
    return _mark({
        "status": "unavailable",
        "layer": layer,
        "head": head,
        "error": "torch/transformers not available — attention_head requires live weights",
    }, "unavailable")


@router.post("/gpt2/patch_head")
@router.post("/gpt2/patchhead")
def gpt2_patch_head(payload: Dict[str, Any]) -> Dict[str, Any]:
    engine = get_engine()
    raw_layer = _as_int(payload, "layer", 9)
    raw_head = _as_int(payload, "head", 9)
    pos_token = _as_text(payload, "pos_token", "Paris")
    neg_token = _as_text(payload, "neg_token", "London")
    _check_prompt(pos_token)
    _check_prompt(neg_token)
    if engine and engine.is_available():
        return _mark(engine.patch_head(
            raw_layer, raw_head, pos_token, neg_token,
        ), "live")
    layer = raw_layer % 12
    head = raw_head % 12
    # Fail closed: no live model means no causal intervention.
    # Seeded fallbacks produce convincing but fake causal effects.
    return _mark({
        "status": "unavailable",
        "layer": layer,
        "head": head,
        "error": "torch/transformers not available — patch_head requires live weights",
    }, "unavailable")


@router.post("/gpt2/steer")
def gpt2_steer(payload: Dict[str, Any]) -> Dict[str, Any]:
    engine = get_engine()
    prompt = _as_text(payload, "prompt", "") or "The movie was"
    layer = _as_int(payload, "layer", 8)
    pos_prompt = _as_text(payload, "pos_prompt", "") or (
        "It was a fantastic wonderful amazing film. The movie was")
    neg_prompt = _as_text(payload, "neg_prompt", "") or (
        "It was a terrible awful horrible film. The movie was")
    alpha = _as_float(payload, "alpha", 25.0)
    if engine and engine.is_available():
        return _mark(engine.steer(
            prompt, layer, pos_prompt, neg_prompt, alpha,
        ), "live")
    return _mark({
        "status": "unavailable",
        "error": "torch/transformers not available — steering needs live weights",
    }, "unavailable")


@router.post("/gpt2/ioi")
def gpt2_ioi(payload: Dict[str, Any]) -> Dict[str, Any]:
    raw_io = payload.get("io_name")
    raw_subj = payload.get("subj_name")
    if raw_io is not None and not isinstance(raw_io, str):
        raise HTTPException(400, detail="'io_name' must be a string")
    if raw_subj is not None and not isinstance(raw_subj, str):
        raise HTTPException(400, detail="'subj_name' must be a string")
    # Deterministic defaults: the previous `random.choice` meant two
    # identical requests ran different experiments, and only a careful
    # caller reading the echoed names could reproduce them. The fixed pair
    # (NAMES[0], NAMES[1]) is arbitrary but stable, and `io_name_defaulted`
    # records that the names were chosen here rather than by the caller.
    defaulted = False
    io_name = (raw_io or "").strip()
    subj_name = (raw_subj or "").strip()
    if not io_name:
        io_name, defaulted = NAMES[0], True
    if not subj_name or subj_name == io_name:
        subj_name = next(n for n in NAMES if n != io_name)
        defaulted = True
    _check_prompt(io_name)
    _check_prompt(subj_name)
    engine = get_engine()
    if engine and engine.is_available():
        res = engine.ioi(io_name, subj_name)
        if isinstance(res, dict):
            res["io_name_defaulted"] = defaulted
        return _mark(res, "live")
    # Fail closed: no live model means no IOI circuit analysis.
    # Seeded fallbacks produce convincing but fake circuit measurements.
    return _mark({
        "status": "unavailable",
        "io_name": io_name,
        "subj_name": subj_name,
        "io_name_defaulted": defaulted,
        "error": "torch/transformers not available — IOI analysis requires live weights",
    }, "unavailable")


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
    layer = _as_int(payload, "layer", 0)
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
            layer=_as_int(payload, "layer", 0),
            component=_as_text(payload, "component", "mlp"),
            page=_as_int(payload, "page", 0),
            page_size=_as_int(payload, "page_size", 128),
            sort_by=_as_text(payload, "sort_by", "index"),
            order=_as_text(payload, "order", "asc"),
            q=_as_text(payload, "q", ""),
        ), "live")
    return _mark({"status": "unavailable",
                  "error": "torch/transformers not available"}, "unavailable")


@router.post("/gpt2/neuron")
def gpt2_neuron(payload: Dict[str, Any]) -> Dict[str, Any]:
    engine = get_engine()
    if engine and engine.is_available():
        return _mark(engine.neuron_detail(
            layer=_as_int(payload, "layer", 0),
            neuron_index=_as_int(payload, "neuron_index", 0),
            component=_as_text(payload, "component", "mlp"),
            top_k_weights=_as_int(payload, "top_k_weights", 16),
        ), "live")
    return _mark({"status": "unavailable",
                  "error": "torch/transformers not available"}, "unavailable")


@router.post("/gpt2/head")
def gpt2_head(payload: Dict[str, Any]) -> Dict[str, Any]:
    engine = get_engine()
    if engine and engine.is_available():
        return _mark(engine.head_detail(
            layer=_as_int(payload, "layer", 0),
            head=_as_int(payload, "head", 0),
        ), "live")
    return _mark({"status": "unavailable",
                  "error": "torch/transformers not available"}, "unavailable")


@router.post("/gpt2/patch_neuron")
def gpt2_patch_neuron(payload: Dict[str, Any]) -> Dict[str, Any]:
    engine = get_engine()
    if engine and engine.is_available():
        return _mark(engine.patch_neuron(
            layer=_as_int(payload, "layer", 0),
            neuron_index=_as_int(payload, "neuron_index", 0),
            patch_value=_as_float(payload, "patch_value", 0.0),
            prompt=_opt_text(payload, "prompt"),
        ), "live")
    return _mark({"status": "unavailable",
                  "error": "torch/transformers not available"}, "unavailable")


@router.post("/gpt2/layer_activations")
def gpt2_layer_activations(payload: Dict[str, Any]) -> Dict[str, Any]:
    engine = get_engine()
    if engine and engine.is_available():
        try:
            return _mark(engine.layer_activations(
                layer=_as_int(payload, "layer", 0),
                prompt=_as_text(payload, "prompt", "") or "The capital of France is",
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
                prompt=_as_text(payload, "prompt", "") or "The capital of France is",
            ), "live")
        except Exception as exc:
            return _mark({"status": "error", "error": str(exc)[:300]}, "unavailable")
    return _mark({"status": "unavailable",
                  "error": "torch/transformers not available"}, "unavailable")


# ---------------------------------------------------------------------------
# SAE — live surface over the existing pipeline + checkpoint loader.
#
# The trainer (SAEReproductionPipeline), the checkpoint providers
# (local/HF/SAELens) and the fail-closed inspector all existed, but no API
# route reached them — so every UI SAE panel could only show invented
# features. These routes expose exactly what exists:
#   GET  /api/sae/status   capability probe (never features)
#   POST /api/sae/inspect  encode one live hidden state with a real checkpoint
#   POST /api/sae/train    background top-k training job + poll
# Anything unmeasurable returns unavailable with the reason, never zeros.
# ---------------------------------------------------------------------------

_SAE_CHECKPOINT_ENV = "MECH_SAE_CHECKPOINT"
_SAE_SOURCE_ENV = "MECH_SAE_SOURCE"
_SAE_DIR_ENV = "MECH_SAE_DIR"
_SAE_DEFAULT_DIR = Path(__file__).parent.parent / "storage" / "sae_checkpoints"
#: Checkpoints larger than this are refused before deserialisation.
_SAE_MAX_CHECKPOINT_BYTES = 512 * 1024 * 1024
#: Caller corpora larger than this are rejected before tokenisation.
_SAE_MAX_CORPUS_CHARS = 200_000

_sae_runs: Dict[str, Dict[str, Any]] = {}
_sae_lock = threading.Lock()
_sae_train_slot = threading.Lock()
_SAE_MAX_RUNS = 20


def _sae_checkpoint_ref(payload: Dict[str, Any]) -> tuple[str, str, str]:
    """Resolve (source, identifier, version) for a checkpoint.

    Explicit payload wins; otherwise the deployment's env pointer; otherwise
    empty — which every caller treats as unavailable, not as a default.
    """
    source = _as_text(payload, "source", "") or os.environ.get(_SAE_SOURCE_ENV, "")
    identifier = _as_text(payload, "identifier", "") or os.environ.get(_SAE_CHECKPOINT_ENV, "")
    version = _as_text(payload, "version", "") or "latest"
    return source, identifier, version


def _sae_checkpoint_dir() -> Path:
    configured = os.environ.get(_SAE_DIR_ENV, "").strip()
    target = Path(configured) if configured else _SAE_DEFAULT_DIR
    target.mkdir(parents=True, exist_ok=True)
    return target.resolve()


def _resolve_local_checkpoint(identifier: str, checkpoint_dir: Path) -> Optional[str]:
    """Confine a user-supplied local identifier to the checkpoint directory.

    Bare filenames only: absolute paths, parent traversal (`..`), hidden
    names, UNC shares and separators are malformed requests (400), because
    the resolved path must stay inside `checkpoint_dir` after symlink
    resolution. A well-formed name with no file behind it is *not* malformed
    -- it returns None so the caller reports unavailable. The `env:`
    indirection is operator configuration, honoured only from the
    environment itself -- never from a request payload, which would let
    a caller aim the loader at an arbitrary server-side path.
    """
    name = (identifier or "").strip()
    if name.startswith("env:"):
        raise HTTPException(
            400, detail="The 'env:' checkpoint indirection is server "
                       "configuration and is not accepted in requests.")
    if (not name or name in (".", "..") or os.path.basename(name) != name
            or name.startswith(".")):
        raise HTTPException(
            400, detail="Local checkpoint identifiers must be bare filenames "
                       "inside the configured checkpoint directory.")
    root = checkpoint_dir.resolve()
    candidate = (root / name).resolve()
    try:
        inside = candidate.is_relative_to(root)
    except Exception:
        inside = False
    if not inside:
        raise HTTPException(
            400, detail="Checkpoint path escapes the configured directory.")
    if not candidate.is_file():
        return None
    if candidate.stat().st_size > _SAE_MAX_CHECKPOINT_BYTES:
        raise HTTPException(
            400, detail="Checkpoint exceeds the maximum loadable size.")
    return str(candidate)


def _sae_probe() -> Dict[str, bool]:
    try:
        import torch  # noqa: F401
        torch_ok = True
    except Exception:
        torch_ok = False
    try:
        from backend.science.reproducibility.sae_pipeline import (  # noqa: F401
            SAEReproductionPipeline,
        )
        pipeline_ok = True
    except Exception:
        pipeline_ok = False
    engine = get_engine()
    return {
        "torch_available": torch_ok,
        "gpt2_available": bool(engine and engine.is_available()),
        "pipeline_importable": pipeline_ok,
        "checkpoint_configured": bool(os.environ.get(_SAE_CHECKPOINT_ENV, "").strip()),
    }


@router.get("/sae/status")
def sae_status() -> Dict[str, Any]:
    from backend.agents.evidence_policy import field_map

    probe = _sae_probe()
    with _sae_lock:
        training_running = any(r.get("phase") == "running" for r in _sae_runs.values())
    return {
        "status": "ok",
        "torch_available": probe["torch_available"],
        "gpt2_available": probe["gpt2_available"],
        "pipeline_importable": probe["pipeline_importable"],
        "checkpoint_configured": probe["checkpoint_configured"],
        "checkpoint_source": os.environ.get(_SAE_SOURCE_ENV, "") or None,
        "training_running": training_running,
        "provenance": "live",
        "field_provenance": field_map(
            ("status", "torch_available", "gpt2_available",
             "pipeline_importable", "checkpoint_configured",
             "training_running"),
            "live",
        ),
    }


@router.post("/sae/inspect")
def sae_inspect(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Encode one live hidden state with a real SAE checkpoint.

    Hidden state comes from a live GPT-2 forward over the caller-supplied
    prompt; encoding is the checkpoint's own matrix. Dimension mismatch
    between checkpoint and model state fails closed — a 768-wide encoder
    applied to a 3072-wide MLP state would be a shape error reported as a
    finding.
    """
    from backend.agents.evidence_policy import field_map

    prompt = _as_text(payload, "prompt", "") or "The capital of France is"
    _check_prompt(prompt)
    layer = _as_int(payload, "layer", 8)
    source, identifier, version = _sae_checkpoint_ref(payload)
    from_payload = bool((_as_text(payload, "identifier", "") or "").strip())
    if not source or not identifier:
        return {
            "status": "unavailable",
            "provenance": "unavailable",
            "field_provenance": field_map(("status",), "unavailable"),
            "reason": (
                "No SAE checkpoint is configured. Set "
                f"{_SAE_SOURCE_ENV}/{_SAE_CHECKPOINT_ENV} or pass "
                "source+identifier explicitly. No encoder weights are loaded, "
                "so no feature decomposition is reported."
            ),
        }
    # Local identifiers are request-supplied paths: confine them to the
    # checkpoint directory before the loader ever sees them. The `env:`
    # indirection stays available only via server environment (operator
    # configuration), never via request payload.
    if (not source or source.strip().lower() == "local") and from_payload:
        source = "local"
        identifier = _resolve_local_checkpoint(identifier, _sae_checkpoint_dir())
        if identifier is None:
            return {
                "status": "unavailable",
                "provenance": "unavailable",
                "field_provenance": field_map(("status",), "unavailable"),
                "reason": ("No checkpoint file with that name exists in the "
                           "configured checkpoint directory."),
            }
    try:
        from backend.interpretability.sae.loader import SAELoader, SAELoadError
    except Exception as exc:
        return {
            "status": "error",
            "provenance": "unavailable",
            "field_provenance": field_map(("status",), "unavailable"),
            "error": f"SAE loader unavailable: {exc}"[:300],
        }
    try:
        sae = SAELoader().load_sae(source, identifier, version)
    except Exception as exc:
        return {
            "status": "unavailable",
            "provenance": "unavailable",
            "field_provenance": field_map(("status",), "unavailable"),
            "reason": f"SAE checkpoint could not be loaded: {exc}"[:500],
        }
    engine = get_engine()
    if not (engine and engine.is_available()):
        return {
            "status": "unavailable",
            "provenance": "unavailable",
            "field_provenance": field_map(("status",), "unavailable"),
            "reason": "torch/transformers not available — no live hidden state to encode.",
        }
    try:
        import torch
    except Exception:
        return {
            "status": "unavailable",
            "provenance": "unavailable",
            "field_provenance": field_map(("status",), "unavailable"),
            "reason": "torch is not importable, so the hidden state cannot be tensorised.",
        }
    try:
        detail = engine.layer_activations(layer, prompt)
    except Exception as exc:
        return {
            "status": "error",
            "provenance": "unavailable",
            "field_provenance": field_map(("status",), "unavailable"),
            "error": str(exc)[:300],
        }
    if not isinstance(detail, dict) or detail.get("status") != "ok":
        return {
            "status": "unavailable",
            "provenance": "unavailable",
            "field_provenance": field_map(("status",), "unavailable"),
            "reason": f"Live activations unavailable: {detail.get('error', 'unknown')}"[:300],
        }
    d_in = int(sae.config.d_in)
    resid = detail.get("resid_post") or []
    mlp = detail.get("mlp_post") or []
    if d_in == int(detail.get("d_model", -1)) and resid:
        state = torch.tensor(resid[-1], dtype=torch.float32)
        tensor_source = f"residual stream, last token, layer {detail.get('layer')}"
    elif d_in == int(detail.get("d_mlp", -1)) and mlp:
        state = torch.tensor(mlp[-1], dtype=torch.float32)
        tensor_source = f"MLP post-activation, last token, layer {detail.get('layer')}"
    else:
        return {
            "status": "unavailable",
            "provenance": "unavailable",
            "field_provenance": field_map(("status",), "unavailable"),
            "reason": (
                f"Checkpoint expects d_in={d_in} but the live state offers "
                f"residual {detail.get('d_model')} / MLP {detail.get('d_mlp')}. "
                "Encoding across mismatched dimensions would be a shape "
                "error reported as features."
            ),
        }
    try:
        encoded = sae.activate(state)
    except Exception as exc:
        return {
            "status": "error",
            "provenance": "unavailable",
            "field_provenance": field_map(("status",), "unavailable"),
            "error": str(exc)[:300],
        }
    if not isinstance(encoded, dict) or encoded.get("status") != "completed":
        return encoded if isinstance(encoded, dict) else {
            "status": "unavailable",
            "provenance": "unavailable",
            "field_provenance": field_map(("status",), "unavailable"),
            "reason": "SAE encoding did not complete.",
        }
    encoded["prompt"] = prompt
    encoded["layer"] = detail.get("layer")
    encoded["tensor_source"] = tensor_source
    encoded["d_in"] = d_in
    encoded["d_sae"] = sae.config.d_sae
    encoded["weights_sha256"] = sae.weights_sha256()
    # The loader ships encoder weights only: encoding is measured,
    # reconstruction fidelity of this checkpoint is not.
    encoded["reconstruction_measured"] = False
    encoded.setdefault("evidence_level", "OBSERVATIONAL")
    encoded["field_provenance"] = field_map(
        ("status", "feature_indices", "activations", "prompt", "layer",
         "tensor_source", "d_in", "d_sae", "weights_sha256"),
        "live",
    )
    return encoded


def _sae_train_worker(run_id: str, run: Dict[str, Any]) -> None:
    from backend.services.model_lock import MODEL_LOCK

    if run is None:
        return
    params = run.get("params", {})
    # Training forwards interleave with interactive forwards on the shared
    # model; hold the lock for the whole fit (see model_lock.py).
    with MODEL_LOCK:
        try:
            from backend.science.reproducibility.sae_pipeline import (
                SAEReproductionPipeline,
            )
            from backend.science.models.adapter_base import LiveUnavailable
        except Exception as exc:
            _sae_finish(run_id, {
                "status": "error", "provenance": "unavailable",
                "error": f"SAE pipeline unavailable: {exc}"[:300],
            })
            return
        try:
            result = SAEReproductionPipeline().run(
                n_features=params["n_features"], seed=params["seed"],
                corpus=run.get("corpus"), n_tokens=params["n_tokens"],
                k=params.get("k"), steps=params["steps"],
                checkpoint_dir=params.get("checkpoint_dir"),
            )
        except LiveUnavailable as exc:
            result = {
                "status": "unavailable", "provenance": "unavailable",
                "reason": str(exc)[:500],
            }
        except Exception as exc:
            result = {
                "status": "error", "provenance": "unavailable",
                "error": f"{type(exc).__name__}: {exc}"[:300],
            }
    _sae_finish(run_id, result if isinstance(result, dict) else {
        "status": "error", "provenance": "unavailable",
        "error": "SAE training returned an unreadable result.",
    })


def _sae_finish(run_id: str, result: Dict[str, Any]) -> None:
    run = _sae_runs.get(run_id)
    if run is None:
        return
    with _sae_lock:
        run["result"] = result
        run["phase"] = result.get("status", "error")
        run["finished"] = time.time()


def _sae_clamp_params(payload: Dict[str, Any]) -> Dict[str, int]:
    """Clamp training resource bounds before any allocation.

    A pure function of the payload so the caps are unit-testable without
    starting a job. Values above the caps are pulled down to them; the
    capped values are what the run record stores, so a request for a
    billion features is visibly a 512-feature run, not a silent DoS.
    """
    return {
        "n_features": max(2, min(512, _as_int(payload, "n_features", 64))),
        "n_tokens": max(64, min(16384, _as_int(payload, "n_tokens", 1024))),
        "steps": max(1, min(2000, _as_int(payload, "steps", 100))),
    }


@router.post("/sae/train")
def sae_train(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Start a top-k SAE training job in a background thread.

    One training slot: a second start while one runs gets `status: busy`,
    not a second concurrent fit on the shared model.
    """
    from backend.agents.evidence_policy import field_map

    bounds = _sae_clamp_params(payload)
    n_features, n_tokens, steps = (
        bounds["n_features"], bounds["n_tokens"], bounds["steps"])
    seed = _as_int(payload, "seed", 42)
    k_raw = payload.get("k", None)
    k = None if k_raw is None else max(1, min(512, _as_int(payload, "k", 8)))
    corpus = _opt_text(payload, "corpus")
    if corpus is not None and not corpus.strip():
        return {
            "status": "unavailable",
            "provenance": "unavailable",
            "field_provenance": field_map(("status",), "unavailable"),
            "reason": "The supplied corpus is empty; nothing to train on.",
        }
    # Bounds are enforced before allocation: an unbounded corpus or
    # feature count turns this endpoint into a local DoS. Clamping
    # silently would change the experiment, so oversize is refused.
    if corpus is not None and len(corpus) > _SAE_MAX_CORPUS_CHARS:
        raise HTTPException(
            400, detail=f"corpus exceeds {_SAE_MAX_CORPUS_CHARS} characters")
    try:
        checkpoint_dir = _sae_checkpoint_dir()
    except Exception as exc:
        return {
            "status": "error",
            "provenance": "unavailable",
            "field_provenance": field_map(("status",), "unavailable"),
            "error": f"Checkpoint directory unavailable: {exc}"[:300],
        }
    if not _sae_train_slot.acquire(blocking=False):
        return {
            "status": "busy",
            "provenance": "unavailable",
            "field_provenance": field_map(("status",), "unavailable"),
            "reason": "An SAE training job is already running; one fit at a time on the shared model.",
        }
    run_id = "sae_" + uuid.uuid4().hex[:12]
    run: Dict[str, Any] = {
        "run_id": run_id,
        "phase": "running",
        "result": None,
        "created": time.time(),
        "params": {"n_features": n_features, "seed": seed,
                   "n_tokens": n_tokens, "steps": steps, "k": k,
                   "corpus_chars": len(corpus) if corpus else None,
                   "checkpoint_dir": str(checkpoint_dir)},
        # The corpus text travels with the run record (outside `params`, so
        # status polls stay small) so the worker trains on what was
        # requested. Previously only `corpus_chars` was stored and the worker
        # silently fell back to the default corpus.
        "corpus": corpus,
    }
    with _sae_lock:
        _sae_runs[run_id] = run
        if len(_sae_runs) > _SAE_MAX_RUNS:
            finished = sorted(
                ((rid, r) for rid, r in _sae_runs.items()
                 if r.get("phase") != "running"),
                key=lambda kv: kv[1].get("created", 0),
            )
            for rid, _ in finished[: len(_sae_runs) - _SAE_MAX_RUNS]:
                _sae_runs.pop(rid, None)

    def _guarded(run_id: str = run_id, record: Dict[str, Any] = run) -> None:
        try:
            _sae_train_worker(run_id, record)
        finally:
            try:
                _sae_train_slot.release()
            except RuntimeError:
                pass

    worker = threading.Thread(target=_guarded, daemon=True)
    worker.start()
    return {
        "status": "started",
        "run_id": run_id,
        "poll": f"/api/sae/runs/{run_id}",
        "provenance": "unavailable",
        "field_provenance": field_map(("status", "run_id"), "unavailable"),
    }


@router.get("/sae/runs/{run_id}")
def sae_run_status(run_id: str) -> Dict[str, Any]:
    from backend.agents.evidence_policy import field_map

    run = _sae_runs.get(run_id)
    if run is None:
        return {
            "status": "unavailable",
            "provenance": "unavailable",
            "field_provenance": field_map(("status",), "unavailable"),
            "error": f"unknown run_id '{run_id}'",
        }
    with _sae_lock:
        body: Dict[str, Any] = {
            "status": "ok" if run.get("phase") == "running" else str(run.get("phase", "error")),
            "run_id": run_id,
            "phase": run.get("phase"),
            "params": run.get("params"),
            "result": run.get("result"),
            "provenance": "live",
            "field_provenance": field_map(("status", "run_id", "phase"), "live"),
        }
    return body


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
    _check_prompt(prompt)
    if not (engine and engine.is_available()):
        return {"status": "error",
                "error": "torch/transformers not available"}
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        # Atomic ensure-then-read: the matrix always describes `prompt`,
        # even with concurrent requests on other prompts.
        res = engine.attention_head(int(layer) % 12, int(head) % 12, prompt=prompt)
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
#: Retention, not admission: finished records kept for the runs view.
_SOCIETY_MAX_RUNS = 50
#: Admission control: at most this many runs *alive* at once, and this many
#: queued. The SAE surface already uses a one-slot lock; a Society run is a
#: daemon thread plus a 1000-event queue per request, so an unauthenticated
#: caller (or a runaway UI poll) could otherwise spend the process's threads
#: and memory without ever touching the model. Requests above the cap are
#: refused with 503 rather than queued forever.
_SOCIETY_MAX_ACTIVE_RUNS = 2
_SOCIETY_MAX_QUEUED_RUNS = 5
_SOCIETY_MAX_GOAL_CHARS = 4096
_SOCIETY_MAX_MODEL_NAME_CHARS = 128


def _society_admission_reason() -> str:
    with _society_lock:
        active = sum(1 for r in _society_runs.values()
                     if r.get("status") == "running")
        queued = sum(1 for r in _society_runs.values()
                     if r.get("status") == "queued")
    if active >= _SOCIETY_MAX_ACTIVE_RUNS:
        return (f"active run limit reached ({active}/"
                f"{_SOCIETY_MAX_ACTIVE_RUNS})")
    # Queue capacity is a cap on queued work in its own right: admitting
    # another run when MAX_QUEUED are already waiting would exceed it
    # regardless of how much room the active slot has.
    if queued >= _SOCIETY_MAX_QUEUED_RUNS:
        return (f"queue capacity reached ({queued}/"
                f"{_SOCIETY_MAX_QUEUED_RUNS} queued)")
    return ""


def _society_get():  # type: ignore[no-untyped-def]
    from backend.agents.society import ResearchSocietyV2
    return ResearchSocietyV2()


def _society_push(run_id: str, event: Dict[str, Any]) -> None:
    run = _society_runs.get(run_id)
    if run is None:
        return
    if run.get("cancel_requested"):
        return  # cooperative cancel: drop post-cancel events (see loop_stop)
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
        # Cooperative cancel (requested via mech_loop_stop): the in-flight
        # step ran to completion, but the run is reported cancelled rather
        # than completed. The worker thread itself cannot be killed.
        if run.get("cancel_requested"):
            status = "cancelled"
            result = {"status": "cancelled", "run_id": run_id, "goal": goal,
                      "reason": "cancel requested via control plane; "
                                "in-flight step ran to completion."}
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
    if len(goal) > _SOCIETY_MAX_GOAL_CHARS:
        return {"status": "error",
                "error": f"goal exceeds {_SOCIETY_MAX_GOAL_CHARS} characters"}
    model_name = str((payload or {}).get("model_name", "gpt2"))
    if len(model_name) > _SOCIETY_MAX_MODEL_NAME_CHARS:
        return {"status": "error", "error": "model_name is too long"}
    # Admission control before any state is allocated.
    reason = _society_admission_reason()
    if reason:
        return {"status": "busy", "error": reason,
                "queue": {"active": _SOCIETY_MAX_ACTIVE_RUNS,
                          "max_queued": _SOCIETY_MAX_QUEUED_RUNS}}
    run_id = "r" + uuid.uuid4().hex[:12]
    run: Dict[str, Any] = {
        "run_id": run_id,
        "goal": goal,
        "model_name": model_name,
        "status": "running",
        "cancel_requested": False,
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
        from backend.core.evidence_graph import envelope_status, load_run_record
        body = load_run_record(run_id)
        if isinstance(body, dict):
            try:
                body["envelope_status"] = envelope_status(run_id)
            except Exception:
                body["envelope_status"] = {"envelope": "unknown",
                                           "attested": False,
                                           "signature": "unverifiable",
                                           "reason": "status check failed"}
        return body
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
