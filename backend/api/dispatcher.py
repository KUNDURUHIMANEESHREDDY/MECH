from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from typing import Dict, Any, List, Optional
import asyncio
import importlib.util
import json
import logging
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

#: Logger for this module. A failure that must not be swallowed has to have
#: somewhere to go: an exception that is caught and dropped is a run that is
#: reported as successful when it was not.
logger = logging.getLogger("MECH.dispatcher")

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

    # The small-model catalogue is reported separately and also marked
    # `reference`: it describes published checkpoints and no model was executed
    # to produce it. `load_state`/`loadable_now` say whether the weights are
    # cached locally, which is not the same claim as "this name loads".
    try:
        from backend.science.models.model_load_service import catalog_entries

        small_models = catalog_entries()
    except Exception as exc:  # pragma: no cover - reported, never hidden
        small_models = []
        small_models_error = str(exc)[:200]
    else:
        small_models_error = None

    payload: Dict[str, Any] = {
        "models": [
            "gpt2-small", "gpt2-medium", "gemma-2b",
            "llama-3-8b", "qwen-7b", "pythia-1b",
            "mistral-7b", "distilgpt2"
        ],
        "small_models": small_models,
        "provenance": "reference",
        "field_provenance": field_map(
            ("models", "small_models"), "reference"),
    }
    if small_models_error is not None:
        payload["small_models_error"] = small_models_error
    return payload


@router.post("/models/load")
def load_model(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Load a named model: a registered small model, or the GPT-2 engine.

    A name in the small-model registry is served by
    ``backend.science.models.model_load_service``, which returns the loaded
    checkpoint's own architecture numbers or an honest refusal from the loader.
    Any other name falls through to the GPT-2 engine path unchanged, and a name
    that is neither is refused -- never substituted by something that looks
    like an answer.
    """
    name = payload.get("model_name", "gpt2-small")
    try:
        from backend.science.models.model_load_service import (
            is_small_model,
            load_model_request,
        )

        if is_small_model(name):
            return load_model_request(payload)
    except ImportError:
        # The small-model seam is optional for a runtime that does not ship it;
        # GPT-2 loading must not become unavailable because of it.
        pass

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


# --- P5: caller-sized benchmark panels + aggregate read-models -------------
#
# `n_prompts` defaults to 100 (the panel size the IOI pipelines were built
# around) and is validated into a 400 rather than a 500. The response always
# echoes the panel actually used plus the pipeline's own adequacy ladder.

_BENCHMARK_MIN_PROMPTS = 1
_BENCHMARK_MAX_PROMPTS = 1000
_BENCHMARK_DEFAULT_PROMPTS = 100

#: Panel/adequacy fields echoed on every benchmarks/run response.
_PANEL_FIELDS = ("n_prompts", "requested_n_prompts", "n_prompts_honoured",
                 "validation_eligible", "publication_eligible",
                 "ineligible_because", "adequacy_reported_by_executor")


def _benchmark_n_prompts(payload: Dict[str, Any]) -> int:
    """Validate the requested panel size, or raise HTTPException(400)."""
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400,
                            detail="request body must be an object")
    raw = payload.get("n_prompts")
    if raw is None:
        return _BENCHMARK_DEFAULT_PROMPTS
    if isinstance(raw, bool) or not isinstance(raw, int):
        raise HTTPException(
            status_code=400,
            detail="n_prompts must be an integer between "
                   f"{_BENCHMARK_MIN_PROMPTS} and {_BENCHMARK_MAX_PROMPTS}")
    if raw < _BENCHMARK_MIN_PROMPTS or raw > _BENCHMARK_MAX_PROMPTS:
        raise HTTPException(
            status_code=400,
            detail=f"n_prompts must be between {_BENCHMARK_MIN_PROMPTS} and "
                   f"{_BENCHMARK_MAX_PROMPTS}, got {raw}")
    return raw


def _panel_fields(res: Dict[str, Any], requested: int) -> Dict[str, Any]:
    """Echo the panel actually used, plus the executor's adequacy ladder.

    Nothing is asserted: `validation_eligible`, `publication_eligible` and
    `ineligible_because` are propagated when the executor reports them and left
    `None` -- with `adequacy_reported_by_executor: False` -- when it does not.
    Defaulting them to True would claim the run was adequate to publish.
    """
    used = res.get("n_prompts", res.get("eval_samples")) if isinstance(res, dict) else None
    honest = requested
    if isinstance(used, int) and not isinstance(used, bool) and used > 0:
        honest = used
    reported = bool(isinstance(res, dict) and (
        "validation_eligible" in res or "publication_eligible" in res))
    return {
        "n_prompts": honest,
        "requested_n_prompts": requested,
        "n_prompts_honoured": honest == requested,
        "validation_eligible": res.get("validation_eligible") if isinstance(res, dict) else None,
        "publication_eligible": res.get("publication_eligible") if isinstance(res, dict) else None,
        "ineligible_because": res.get("ineligible_because") if isinstance(res, dict) else None,
        "adequacy_reported_by_executor": reported,
    }


def _measured_weights_fingerprint(result: Dict[str, Any]) -> str:
    """The weight fingerprint the measurement bound itself to, if authentic.

    The measurement record carries ``model_identity.content_hash`` together
    with ``content_hash_source``. Only ``model_weights`` and ``explicit``
    are authentic -- they hash the loaded weight bytes. ``model_name_fallback``
    hashes the model NAME, which proves nothing about which weights ran, so
    it is refused rather than stored as if it were a fingerprint.
    """
    record = result.get("measurement_record")
    if not isinstance(record, dict):
        return ""
    identity = record.get("model_identity")
    if not isinstance(identity, dict):
        return ""
    source = str(identity.get("content_hash_source", "")).strip().lower()
    if source not in {"model_weights", "explicit"}:
        return ""
    digest = identity.get("content_hash")
    if not isinstance(digest, str) or not digest.strip():
        return ""
    return digest.strip()


def _repository_code_revision() -> str:
    """The commit the working tree is on, or empty when git cannot say.

    A revision is only useful if it is the real one. An unavailable
    repository yields no revision at all: a placeholder object id would
    bind a stored run to nothing while looking exactly like a real one.
    """
    try:
        from backend.science.reproducibility.run_attestation import code_revision
    except Exception:
        return ""
    try:
        revision = code_revision()
    except Exception:
        return ""
    value = getattr(revision, "revision", "")
    return value.strip() if isinstance(value, str) else ""


@router.post("/benchmarks/run")
def run_benchmark(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Run a benchmark over a caller-sized prompt panel."""
    from backend.agents.evidence_policy import field_map

    n_prompts = _benchmark_n_prompts(payload)
    name = str(payload.get("benchmark_name", "IOI"))
    base_fields = ("status", "benchmark_name", "score", "pass_rate",
                   "eval_samples")
    echo_fields = base_fields + _PANEL_FIELDS
    try:
        from backend.validation.benchmark_runner import (
            MechanisticBenchmarkRunner,
        )
        runner = MechanisticBenchmarkRunner()
        try:
            res = runner.run_benchmark(f"bench_{name}", n_prompts=n_prompts)
        except TypeError:
            # This runner fixes its own panel size. Run it and let the payload
            # say the requested size was not honoured -- never imply it was.
            res = runner.run_benchmark(f"bench_{name}")
        if not isinstance(res, dict):
            res = {"status": "error",
                   "error": "the benchmark runner returned a non-dict result"}
        panel = _panel_fields(res, n_prompts)
        status = str(res.get("status", "unavailable")).lower()
        provenance = str(res.get("provenance", "unavailable")).lower()
        if status not in {"completed", "passed"} or provenance != "live":
            label = (provenance if provenance in {"unavailable", "seeded",
                                                  "reference"}
                     else "unavailable")
            return {
                "status": status if status in {"unavailable", "error"} else "unavailable",
                "benchmark_name": name,
                "provenance": label,
                "field_provenance": field_map(echo_fields, label),
                **panel,
                "reason": str(res.get("reason") or "Live benchmark execution is unavailable."),
                # A refusal is not a run. The success branch reports these two
                # keys, so a consumer reading `body["recorded"]` must not have
                # to know which branch produced the response: the absence of a
                # run is a fact, not a missing key.
                "run_id": None,
                "recorded": False,
            }

        try:
            score = float(res["accuracy"])
            pass_rate = float(res.get("robustness_score", score))
        except (KeyError, TypeError, ValueError):
            return {
                "status": "unavailable",
                "benchmark_name": name,
                "provenance": "unavailable",
                "field_provenance": field_map(echo_fields, "unavailable"),
                **panel,
                "reason": "The live executor did not return numeric score fields.",
                # Same contract as the refusal above: no run, and the keys
                # that say so are present rather than absent.
                "run_id": None,
                "recorded": False,
            }

        body = {
            "status": "completed",
            "benchmark_name": name,
            "provenance": "live",
            # Propagated from the runner's own measurement, never asserted:
            # this route narrows the record to scores, it does not measure.
            "attested": res.get("attested", False),
            "field_provenance": field_map(echo_fields, "live"),
            "score": round(score, 4),
            "pass_rate": round(pass_rate, 4),
            **panel,
        }
        # Which model, which weights and which code produced this run. The
        # history store keeps these three columns, and they were arriving
        # NULL because the route narrowed the executor's record to scores
        # and never carried them across. A stored row that cannot say which
        # weights produced it is not evidence of anything repeatable, so
        # the provenance is taken from the measurement itself -- never
        # invented here, and never a placeholder.
        #
        # `model_id` is the executor's own claim about the model it ran.
        model_id = res.get("model_id")
        if isinstance(model_id, str) and model_id.strip():
            body["model_id"] = model_id
        # The weights fingerprint is only recorded when the measurement
        # bound it to real weight bytes. A name-derived fallback hashes the
        # model NAME, which proves nothing about which weights ran, so it is
        # refused rather than stored as if it were a fingerprint.
        weights_fingerprint = _measured_weights_fingerprint(res)
        if weights_fingerprint:
            body["weights_fingerprint"] = weights_fingerprint
        # The commit the working tree is on, as git reports it. When git
        # cannot be asked, the column stays empty: an invented object id
        # would bind the row to nothing and would be indistinguishable from
        # a real one.
        revision = _repository_code_revision()
        if revision:
            body["git_commit"] = revision
        body["field_provenance"] = {
            **body["field_provenance"],
            "model_id": "live" if body.get("model_id") else "unavailable",
            "weights_fingerprint": (
                "live" if body.get("weights_fingerprint") else "unavailable"),
            "git_commit": "live" if body.get("git_commit") else "unavailable",
        }
        # Persist the completed live run and hand back the stored id.
        # `record_benchmark_run` returns None for anything that is not a
        # completed live run carrying a real finite score, so a run storage
        # declined to keep leaves `run_id` None rather than inventing one.
        #
        # The call is isolated deliberately. This handler's outer
        # ``except Exception`` would otherwise convert a storage failure into a
        # 200 ``status: "error"``, discarding a measurement that did in fact
        # happen and relabelling it as a failed run. Isolating it keeps the
        # measurement, reports the storage failure in ``recording_error``, and
        # never claims the run was stored.
        try:
            from backend.benchmarking.run_history import record_benchmark_run

            stored = record_benchmark_run(benchmark_name=name, result=body)
        except Exception as recording_exc:
            stored = None
            body["recording_error"] = str(recording_exc)[:300]
        if stored is None:
            body.setdefault(
                "recording_error",
                "run history did not record this run: it is not a completed "
                "live run carrying a finite score",
            )
        body["run_id"] = stored["run_id"] if stored else None
        body["recorded"] = stored is not None
        body["field_provenance"] = {
            **body["field_provenance"],
            # A stored id is a real fact about the history table when it
            # exists, and absent when it does not -- never a placeholder.
            "run_id": "live" if stored else "unavailable",
            "recorded": "live",
        }
        return body
    except Exception as exc:
        return {
            "status": "error",
            "benchmark_name": name,
            "provenance": "unavailable",
            "field_provenance": field_map(echo_fields, "unavailable"),
            **_panel_fields({}, n_prompts),
            "error": str(exc)[:300],
            # The handler failed before a measurement existed, so no run was
            # recorded. The keys are present so a consumer does not have to
            # treat a missing key as "maybe recorded".
            "run_id": None,
            "recorded": False,
        }


@router.get("/benchmarks/history")
def benchmark_history(
    benchmark: Optional[str] = None, limit: int = 20
) -> Dict[str, Any]:
    """Persisted benchmark runs, newest first: the rows storage returned.

    ``limit`` is clamped to 1..500 by the store. A non-integer limit is a
    client error, not an empty history, so it is refused rather than answered.
    """
    from backend.benchmarking.run_history import RunHistoryError, history_payload

    try:
        return history_payload(benchmark=benchmark, limit=limit)
    except RunHistoryError as exc:
        raise HTTPException(
            status_code=getattr(exc, "status_code", 400) or 400,
            detail=str(exc),
        ) from exc


@router.get("/benchmarks/compare")
def benchmark_compare(run_a: str, run_b: str) -> Dict[str, Any]:
    """Compare two stored runs.

    An unknown run id is a 404, never a 500 and never a delta. When the two
    runs are not comparable the reasons are carried through verbatim and the
    delta stays None -- a zero would read as "no change", which is a claim the
    stored rows do not support.
    """
    from backend.benchmarking.run_history import RunHistoryError, compare_payload

    try:
        return compare_payload(run_a, run_b)
    except RunHistoryError as exc:
        raise HTTPException(
            status_code=getattr(exc, "status_code", 400) or 400,
            detail=str(exc),
        ) from exc


@router.get("/scientific_health")
def scientific_health() -> Dict[str, Any]:
    """One payload: subsystem probes, run ledger, attestation, validation.

    Deliberately carries no health score -- see `backend/core/aggregates.py`.
    Sources that could not be read are named in `unavailable`.
    """
    from backend.core import aggregates

    return aggregates.scientific_health()


@router.get("/analytics")
def analytics() -> Dict[str, Any]:
    """Descriptive inventory: run/experiment/session counts and durations.

    No quality or productivity metric is computed, because nothing here
    measures one.
    """
    from backend.core import aggregates

    return aggregates.aggregate_analytics()


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
# Prompts that aren't supplied by a caller are sampled from live GPT-2
# weights; there is no fixed prompt pool or canned input fallback.


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

@router.post("/runtime/compare")
def compare_model_checkpoints(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Compare two independently loaded checkpoints and persist live evidence.

    The route accepts caller-provided model IDs/revisions and a prompt; it
    invents neither a checkpoint, an input, nor a numeric result. Only the
    trusted built-in Hugging Face backend can return an attested measurement.
    """
    from datetime import datetime, timezone
    from backend.runtime.comparison import (
        HuggingFaceComparisonBackend,
        ModelComparisonEngine,
    )

    if not isinstance(payload, dict):
        raise HTTPException(status_code=422, detail="Request body must be an object.")
    prompt = _as_text(payload, "prompt", "")
    if not prompt.strip():
        raise HTTPException(status_code=422, detail="Provide a non-empty input prompt.")
    _check_prompt(prompt)
    model_a = _as_text(payload, "model_a", "").strip()
    model_b = _as_text(payload, "model_b", "").strip()
    revision_a = _as_text(payload, "revision_a", "main").strip() or "main"
    revision_b = _as_text(payload, "revision_b", "main").strip() or "main"
    if not model_a or not model_b:
        raise HTTPException(status_code=422, detail="Provide independent model_a and model_b Hugging Face checkpoint IDs.")

    result = ModelComparisonEngine(backend=HuggingFaceComparisonBackend()).compare(
        prompt=prompt, model_a=model_a, model_b=model_b,
        revision_a=revision_a, revision_b=revision_b,
    )
    if result.get("status") != "completed" or result.get("provenance") != "live" or result.get("measured") is not True:
        return result

    experiment_id = "cross-model-" + uuid.uuid4().hex
    experiment = {
        "id": experiment_id,
        "name": f"Cross-model comparison: {model_a} vs {model_b}",
        "experiment_type": "cross_model_comparison",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "provenance": "live",
        "measured": True,
        "parameters": {
            "prompt": prompt,
            "model_a": model_a,
            "model_b": model_b,
            "revision_a": revision_a,
            "revision_b": revision_b,
        },
        "results": result,
    }
    try:
        stored = _store.add_experiment(experiment)
        result["experiment_id"] = stored.get("id")
        result["stored"] = bool(stored.get("id"))
    except Exception as exc:
        result["stored"] = False
        result["recording_error"] = f"{type(exc).__name__}: {exc}"[:300]
    return result


NAMES = ["John", "Alice", "Bob", "Emma", "David", "Sophia", "Michael", "Olivia", "James", "Emily"]


def _generate_fresh_prompt(engine: Any, max_new_tokens: int = 12) -> str:
    """Return only model-sampled text; failure produces no substitute input."""
    if not (engine is not None and engine.is_available()):
        return ""
    try:
        result = engine.generate_text("", max_new_tokens=max_new_tokens)
        text = (
            str(result.get("prompt", "")).strip()
            if isinstance(result, dict)
            and result.get("status") == "ok"
            and result.get("provenance") == "live"
            and result.get("attested") is True
            else ""
        )
        return text[:_MAX_PROMPT_CHARS] if text else ""
    except Exception as exc:
        logger.warning("GPT-2 prompt generation failed: %s", exc)
        return ""


def _default_prompt(engine: Any) -> str:
    """Generate an input from live GPT-2 weights, or return no input."""
    generated = _generate_fresh_prompt(engine, max_new_tokens=12)
    if generated:
        _check_prompt(generated)
    return generated


@router.post("/gpt2/fresh_prompt")
def gpt2_fresh_prompt(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Return a prompt generated and attested by the live GPT-2 measurement layer.

    There is no text primer, prompt pool, or wrapper-created measurement.
    Fail closed if the model fails to attest the generated prompt.
    """
    engine = get_engine()
    if not (engine and engine.is_available()):
        return _mark({
            "status": "unavailable",
            "error": "torch/transformers not available — the model cannot create a prompt",
        }, "unavailable")
    # Fresh prompts are generated from GPT-2's own special-token context.
    # Ignore caller text primers: the final prompt must contain model-sampled
    # tokens only, not an arbitrary or hardcoded text prefix.
    primer = ""
    max_new_tokens = max(1, min(48, _as_int(payload, "max_new_tokens", 12)))
    try:
        res = engine.generate_text("", max_new_tokens=max_new_tokens)
    except Exception as exc:
        return _mark({
            "status": "error",
            "primer": primer,
            "error": str(exc)[:300],
        }, "unavailable")
    if (
        not isinstance(res, dict)
        or res.get("status") != "ok"
        or res.get("provenance") != "live"
        or res.get("attested") is not True
        or not str(res.get("prompt", "")).strip()
    ):
        return _mark({
            "status": "unavailable",
            "primer": primer,
            "error": "The model did not return an attested, usable GPT-2 prompt.",
        }, "unavailable")
    # The measurement layer has already attested the exact prompt field.
    # Return that record unchanged; wrapping it in a new dict would lose scope.
    return pass_through(res)


@router.post("/gpt2/run_prompt")
def gpt2_run_prompt(payload: Dict[str, Any]) -> Dict[str, Any]:
    engine = get_engine()
    prompt = payload.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        prompt = _default_prompt(engine)
    if not prompt:
        return _mark({
            "status": "unavailable",
            "error": "No prompt was supplied and live GPT-2 could not generate one. No fallback input was substituted.",
        }, "unavailable")
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
            prompt = _as_text(payload, "prompt", "").strip()
            if not prompt:
                prompt = _default_prompt(engine)
            if not prompt:
                return _mark({
                    "status": "unavailable",
                    "error": "No prompt supplied and GPT-2 could not generate a live prompt. No fallback input was substituted.",
                }, "unavailable")
            _check_prompt(prompt)
            return pass_through(engine.layer_activations(
                layer=_as_int(payload, "layer", 0),
                prompt=prompt,
            ))
        except Exception as exc:
            return _mark({"status": "error", "error": str(exc)[:300]}, "unavailable")
    return _mark({"status": "unavailable",
                  "error": "torch/transformers not available"}, "unavailable")


@router.post("/gpt2/logit_lens_all")
def gpt2_logit_lens_all(payload: Dict[str, Any]) -> Dict[str, Any]:
    engine = get_engine()
    if engine and engine.is_available():
        try:
            prompt = _as_text(payload, "prompt", "").strip()
            if not prompt:
                prompt = _default_prompt(engine)
            if not prompt:
                return _mark({
                    "status": "unavailable",
                    "error": "No prompt supplied and GPT-2 could not generate a live prompt. No fallback input was substituted.",
                }, "unavailable")
            _check_prompt(prompt)
            return pass_through(engine.logit_lens_all(prompt=prompt))
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


def _is_foreign_path(name: str) -> bool:
    r"""True for a path that is absolute on a platform other than this one.

    `os.path.basename` splits on the *host's* separator, so on POSIX a
    Windows path is indistinguishable from a bare filename:
    `C:\Windows\sae.pt` and `\\server\share\sae.pt` both pass the basename
    check and are then resolved as relative names inside the checkpoint
    directory. The same request is refused on Windows and accepted on
    POSIX, so one identifier behaves differently depending on where the
    server runs -- and a Windows path is not a valid identifier on any
    host.
    """
    if "\\" in name:
        return True
    if len(name) >= 2 and name[0].isalpha() and name[1] == ":":
        return len(name) == 2 or name[2] in "\\/"
    return False


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
            or name.startswith(".") or _is_foreign_path(name)):
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
    try:
        from backend.interpretability.tl_compat import check_compatibility
        tl_ok = bool(check_compatibility().get("compatible"))
    except Exception:
        tl_ok = False
    engine = get_engine()
    return {
        "torch_available": torch_ok,
        "gpt2_available": bool(engine and engine.is_available()),
        "transformer_lens_available": tl_ok,
        "pipeline_importable": pipeline_ok,
        "checkpoint_configured": bool(os.environ.get(_SAE_CHECKPOINT_ENV, "").strip()),
    }


@router.get("/sae/status")
def sae_status() -> Dict[str, Any]:
    from backend.agents.evidence_policy import field_map

    probe = _sae_probe()
    with _sae_lock:
        training_running = any(r.get("phase") == "running" for r in _sae_runs.values())
    from backend.interpretability.sae.local_checkpoints import (
        checkpoint_dir,
        checkpoint_names,
    )
    return {
        "status": "ok",
        "torch_available": probe["torch_available"],
        "gpt2_available": probe["gpt2_available"],
        "transformer_lens_available": probe["transformer_lens_available"],
        "pipeline_importable": probe["pipeline_importable"],
        "checkpoint_configured": probe["checkpoint_configured"],
        "checkpoint_source": os.environ.get(_SAE_SOURCE_ENV, "") or None,
        "checkpoint_identifier": os.environ.get(_SAE_CHECKPOINT_ENV, "").strip() or None,
        "checkpoint_dir": str(checkpoint_dir()),
        "available_checkpoints": checkpoint_names(),
        "training_running": training_running,
        "provenance": "live",
        "field_provenance": field_map(
            ("status", "torch_available", "gpt2_available",
             "transformer_lens_available", "pipeline_importable",
             "checkpoint_configured", "checkpoint_identifier", "checkpoint_dir",
             "available_checkpoints", "training_running"),
            "live",
        ),
    }


def _sae_activation(engine, tensor_source: str, prompt: str):
    """The activation a checkpoint names, or ``(None, reason)``.

    The tensor source is the fact that binds a dictionary to an activation, so
    it is resolved rather than guessed at: ``transformer_lens:<hook>`` is read
    from TransformerLens' own cache, and ``transformers:blocks.<L>.<name>``
    from this engine. A source this platform cannot read is a refusal, never
    a fallback to some other tensor of the right width.
    """
    source = str(tensor_source or "").strip()
    library, _, name = source.partition(":")
    library = library.strip().lower()
    name = name.strip()
    if not name:
        return None, (
            f"Tensor source {source!r} names no activation, so there is "
            "nothing to encode.")
    if library == "transformer_lens":
        result = engine.hook_activation(name, prompt)
        if not isinstance(result, dict) or result.get("status") != "ok":
            reason = (result or {}).get("error") or "unknown error"
            return None, (
                f"The activation {name!r} could not be read from "
                f"TransformerLens: {reason}")
        return result.get("tensor"), (
            f"TransformerLens hook {name}, last token")
    if library == "transformers":
        parts = name.split(".")
        if len(parts) >= 3 and parts[0] == "blocks":
            try:
                layer = int(parts[1])
            except ValueError:
                layer = -1
            kind = ".".join(parts[2:])
            if kind == "mlp_write" and layer >= 0:
                result = engine.mlp_write(layer, prompt)
                if not isinstance(result, dict) or result.get("status") != "ok":
                    reason = (result or {}).get("error") or "unknown error"
                    return None, (
                        f"The activation {name!r} could not be read from the "
                        f"live model: {reason}")
                tensor = result.get("tensor")
                if tensor is not None and getattr(tensor, "ndim", 1) > 1:
                    tensor = tensor[-1]
                return tensor, (
                    "the MLP's write into the residual stream at block "
                    f"{layer} (c_proj(act(c_fc(ln_2(resid_post))))), last token")
        return None, (
            f"Tensor source {source!r} is not an activation this platform can "
            "reproduce, so nothing was encoded.")
    return None, (
        f"Tensor source {source!r} names an unknown library, so nothing was "
        "encoded.")


@router.post("/sae/inspect")
def sae_inspect(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Encode one live hidden state with a real SAE checkpoint.

    The activation is the one the checkpoint describes. A dictionary is
    trained on one activation at one hook point on one model, so the
    checkpoint says which -- its ``hook_point`` and ``tensor_source`` -- and
    the route reads exactly that tensor: from TransformerLens when the hook is
    a TransformerLens hook, from this engine when it is not. The `transformers`
    residual stream is a *different* tensor from the TransformerLens one, and
    the same dictionary reconstructs the TransformerLens tensor at relative L2
    0.243 against 3.43 for the `transformers` one at layer 8, so encoding
    whichever tensor was convenient would report a measurement of the wrong
    quantity.

    A layer the dictionary does not describe is refused rather than encoded,
    and a checkpoint that records no tensor source is refused rather than
    applied to an arbitrary tensor. Dimension mismatch between checkpoint and
    model state also fails closed: a 768-wide encoder applied to a 3072-wide
    MLP state would be a shape error reported as a finding.
    """
    from backend.agents.evidence_policy import field_map

    prompt = _as_text(payload, "prompt", "").strip()
    if not prompt:
        prompt = _default_prompt(get_engine())
    if not prompt:
        return {
            "status": "unavailable",
            "provenance": "unavailable",
            "field_provenance": field_map(("status", "prompt"), "unavailable"),
            "reason": "No prompt supplied and live GPT-2 could not generate one. No fallback input was substituted.",
        }
    _check_prompt(prompt)
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
    elif (not source or source.strip().lower() == "local"):
        # Operator configuration (the environment pointer). An absolute path is
        # honoured as given; a bare file name is resolved inside the
        # checkpoint directory, so `MECH_SAE_CHECKPOINT=<file name>` means
        # what it says instead of being resolved against the server's cwd.
        raw = identifier.strip()
        if raw and os.path.basename(raw) == raw and not raw.startswith("."):
            resolved = _resolve_local_checkpoint(raw, _sae_checkpoint_dir())
            if resolved is None:
                return {
                    "status": "unavailable",
                    "provenance": "unavailable",
                    "field_provenance": field_map(("status",), "unavailable"),
                    "reason": (
                        f"No checkpoint file named {raw!r} exists in the "
                        "configured checkpoint directory "
                        f"({_sae_checkpoint_dir()})."
                    ),
                }
            identifier = resolved
    try:
        from backend.interpretability.sae.loader import (
            SAELoadError,
            SAELoader,
            hook_at_layer,
        )
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

    # Which activation does this dictionary describe? The checkpoint says, and
    # the route encodes that one. A checkpoint that says nothing describes no
    # activation, so there is nothing honest to encode it with.
    config = sae.config
    declared_hook = str(getattr(config, "hook_point", "") or "")
    tensor_source = str(getattr(config, "tensor_source", "") or "")
    try:
        declared_layer = int(getattr(config, "layer", -1))
    except (TypeError, ValueError):
        declared_layer = -1
    if not tensor_source:
        return {
            "status": "unavailable",
            "provenance": "unavailable",
            "field_provenance": field_map(("status",), "unavailable"),
            "reason": (
                "This checkpoint records no hook point and no tensor source, "
                "so the activation it was trained on cannot be reproduced "
                "from these weights. Encoding an arbitrary tensor instead "
                "would be a silent mismatch reported as a decomposition, so "
                "no feature indices are reported."),
        }
    if declared_layer < 0:
        return {
            "status": "unavailable",
            "provenance": "unavailable",
            "field_provenance": field_map(("status",), "unavailable"),
            "reason": (
                "This checkpoint does not record which layer its dictionary "
                "describes, so a layer mismatch cannot be refused. Nothing "
                "was encoded."),
        }

    # The layer asked for, defaulting to the dictionary's own layer. A layer
    # the dictionary does not describe is refused below, before any encoding.
    if payload.get("layer") is None:
        layer = declared_layer
    else:
        layer = _as_int(payload, "layer", declared_layer)
    try:
        sae.assert_applies_to(layer, hook_at_layer(declared_hook, layer))
    except SAELoadError as exc:
        return {
            "status": "unavailable",
            "provenance": "unavailable",
            "field_provenance": field_map(("status",), "unavailable"),
            "reason": str(exc)[:500],
        }

    engine = get_engine()
    if not (engine and engine.is_available()):
        return {
            "status": "unavailable",
            "provenance": "unavailable",
            "field_provenance": field_map(("status",), "unavailable"),
            "reason": "torch/transformers not available — no live hidden state to encode.",
        }

    state, tensor_detail = _sae_activation(engine, tensor_source, prompt)
    if state is None:
        return {
            "status": "unavailable",
            "provenance": "unavailable",
            "field_provenance": field_map(("status",), "unavailable"),
            "reason": tensor_detail[:500],
        }
    d_in = int(sae.config.d_in)
    width = int(getattr(state, "shape", [-1])[-1])
    if d_in != width:
        return {
            "status": "unavailable",
            "provenance": "unavailable",
            "field_provenance": field_map(("status",), "unavailable"),
            "reason": (
                f"Checkpoint expects d_in={d_in} but the declared tensor "
                f"{tensor_source} is {width} wide. Encoding across "
                "mismatched dimensions would be a shape error reported as "
                "features."),
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
    encoded["layer"] = layer
    encoded["hook_point"] = declared_hook
    encoded["tensor_source"] = tensor_source
    encoded["tensor_source_detail"] = tensor_detail
    encoded["d_in"] = d_in
    encoded["d_sae"] = sae.config.d_sae
    encoded["weights_sha256"] = sae.weights_sha256()
    encoded["model_id"] = sae.config.model_id
    # Reconstruction fidelity is a measurement, not a constant: it is computed
    # from the checkpoint's own decoder when it has one, and refused with a
    # reason when it does not. It used to be hardcoded to False here, which
    # reported a measurable quantity as absent.
    try:
        reconstruction = sae.reconstruction_error(
            state, list(encoded.get("feature_indices") or []))
    except Exception as exc:  # pragma: no cover - reported, never hidden
        reconstruction = {
            "status": "unavailable",
            "measured": False,
            "reason": f"reconstruction error could not be measured: {type(exc).__name__}: {exc}"[:300],
        }
    if isinstance(reconstruction, dict) and reconstruction.get("measured"):
        encoded["reconstruction_measured"] = True
        error = reconstruction.get("relative_l2_error")
        encoded["relative_l2_error"] = error
        encoded["n_active_features"] = reconstruction.get("n_active_features")
        encoded["reconstruction_hook_point"] = reconstruction.get("hook_point")
        # A relative L2 of 1.0 is the predict-the-mean baseline: at or above
        # it the reconstruction explains no more of the activation than
        # predicting the mean would, so it is not a reconstruction at all.
        # The number stays as measured -- it is a measurement, not a score --
        # and the flag is the qualification a reader needs, the same one the
        # training pipeline applies to its own fit.
        try:
            useful = error is not None and float(error) < 1.0
        except (TypeError, ValueError):
            useful = False
        encoded["reconstruction_useful"] = bool(useful)
        if not useful:
            encoded["reconstruction_note"] = (
                f"Relative L2 {error} is at or above 1.0, which is the "
                "predict-the-mean baseline: this dictionary explains no more "
                "of the activation than predicting the mean would. The "
                "figure is the measurement, not a score.")
    else:
        encoded["reconstruction_measured"] = False
        encoded["reconstruction_useful"] = False
        encoded["reconstruction_reason"] = (
            (reconstruction or {}).get("reason")
            or "the checkpoint carries no decoder weights, so the "
               "reconstruction cannot be measured"
        )
    # The activation encoded here is the one the checkpoint declares, and the
    # requested layer was checked against the checkpoint's own hook point
    # before anything was encoded. The note says so, because the reader of a
    # reconstruction number needs to know what it is a reconstruction of.
    encoded["tensor_mismatch_note"] = (
        f"No mismatch: the activation encoded is {tensor_source!r}, the "
        "tensor this dictionary was trained on, and the requested layer was "
        "checked against the checkpoint's own hook point before encoding.")
    encoded.setdefault("evidence_level", "OBSERVATIONAL")
    encoded["attested"] = False
    encoded["field_provenance"] = field_map(
        ("status", "feature_indices", "activations", "prompt", "layer",
         "tensor_source", "tensor_source_detail", "hook_point", "d_in", "d_sae",
         "weights_sha256", "reconstruction_measured", "relative_l2_error",
         "reconstruction_useful", "reconstruction_note",
         "n_active_features", "tensor_mismatch_note", "model_id"),
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
                layer=int(params.get("layer", 0)),
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
    # Which block's MLP write is fitted. The checkpoint records it, so the
    # route can reproduce the training tensor instead of guessing at it.
    layer = max(0, _as_int(payload, "layer", 0))
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
                   "layer": layer,
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
def figure_attention(prompt: str = "",
                     layer: int = 10, head: int = 7):
    """Publication-ready attention heatmap PNG from live weights."""
    from fastapi.responses import Response
    engine = get_engine()
    prompt = (prompt or "").strip()
    if not prompt:
        prompt = _default_prompt(engine)
    if not prompt:
        return {
            "status": "unavailable",
            "provenance": "unavailable",
            "error": "No prompt supplied and live GPT-2 could not generate one. No fallback input was substituted.",
        }
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
#   POST /api/society/run                  {goal, model_name?} -> {runId, status}
#   GET  /api/society/stream?runId=...    -> text/event-stream (live trace)
#   GET  /api/society/runs/{runId}        -> polling fallback (status + result)
#   POST /api/society/runs/{runId}/cancel -> cooperative cancel, mirroring
#                                            /api/v1/jobs/{jobId}/cancel
# NOTE: docs/API_v1.md sketches /api/v1/... and backend/api/runtime_api.py now
# exists, but Society still ships on this active dispatcher: the v2 module
# exposes the job/cancel surface, not these routes.
# Long GPU work runs in a daemon thread; the event loop is never blocked.
#
# A run must not be able to hold an admission slot forever. Every admitted
# run carries a deadline and a worker thread: a run that outlives its
# deadline is finalized as a failure whose reason names the timeout, and a
# run whose worker thread is gone is finalized as interrupted. Without both,
# one hung pipeline kept status "running" for the lifetime of the process and
# every later submission was refused ("active run limit reached (2/2)").
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
#: Bounded lifetime of one run, in seconds. A normal Society run takes
#: 75-160s on a CPU-only host, so the default leaves room for the slow end
#: while still bounding how long a hung pipeline may hold its slot.
#: Override with MECH_SOCIETY_RUN_TIMEOUT_S.
_SOCIETY_DEFAULT_RUN_TIMEOUT_S = 600.0
#: Grace period for attaching a worker thread to an admitted run. Past it the
#: run is finalized as interrupted: nothing will ever start it.
_SOCIETY_STARTUP_GRACE_S = 30.0
#: Statuses that mean "this run is over". Anything else still holds an
#: admission slot.
_SOCIETY_TERMINAL_STATUSES = frozenset({
    "completed", "completed_unrecorded", "blocked", "error", "failed",
    "cancelled", "interrupted",
})
#: Statuses that mean "this run finished successfully".
_SOCIETY_SUCCESS_STATUSES = frozenset({"completed"})
#: The status a successful run carries when its evidence record could not be
#: written. The work happened; the proof of it did not. Reporting a plain
#: "completed" would claim a durable record that does not exist, so the run is
#: reported as finished-but-unrecorded instead -- still terminal, so it still
#: releases its admission slot, but not a success anyone can cite.
_SOCIETY_UNRECORDED_STATUS = "completed_unrecorded"
#: Statuses that hold an admission slot.
_SOCIETY_LIVE_STATUSES = frozenset({"running"})


#: How much a Society run may cost, by name.
#:
#: A live discovery sweeps every attention head of GPT-2 on a prompt and
#: verifies the candidates on every panel prompt, so a run's cost is set by
#: its sample: the "standard" preset is the sample the platform considers
#: adequate for a validation claim (10 prompts, 5 interaction prompts), and it
#: takes minutes on a CPU. "quick" is a smaller sample -- it cannot support a
#: validation claim, and the run says so -- for callers that need a run to
#: finish now. The effort is reported with the result, so a quick run is
#: never mistaken for a thorough one.
_SOCIETY_EFFORTS: Dict[str, Dict[str, int]] = {
    "quick": {"n_prompts": 2, "n_interaction_prompts": 1, "max_heads": 4},
    "standard": {"n_prompts": 10, "n_interaction_prompts": 5, "max_heads": 10},
    "full": {"n_prompts": 25, "n_interaction_prompts": 5, "max_heads": 10},
}

_SOCIETY_DEFAULT_EFFORT = "standard"


def _society_effort(payload: Dict[str, Any]) -> Dict[str, int]:
    """The effort a run may spend, and an error when the name is unknown."""
    name = str((payload or {}).get("effort", "") or _SOCIETY_DEFAULT_EFFORT).strip()
    effort = _SOCIETY_EFFORTS.get(name)
    if effort is None:
        raise HTTPException(
            status_code=400,
            detail=(f"unknown effort {name!r}; one of "
                    f"{sorted(_SOCIETY_EFFORTS)}"))
    return dict(effort)


def _society_run_timeout_s() -> float:
    """Bounded lifetime of one Society run, in seconds.

    MECH_SOCIETY_RUN_TIMEOUT_S overrides the default (600s). A malformed or
    non-positive value falls back to the default instead of removing the
    bound: an unbounded run is exactly the leak this guards against.
    """
    raw = os.environ.get("MECH_SOCIETY_RUN_TIMEOUT_S", "")
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return _SOCIETY_DEFAULT_RUN_TIMEOUT_S
    if value <= 0:
        return _SOCIETY_DEFAULT_RUN_TIMEOUT_S
    return value


def _society_is_terminal(status: Any) -> bool:
    """True when the run is over and no longer holds an admission slot."""
    return str(status) in _SOCIETY_TERMINAL_STATUSES


def _society_finalize_locked(run: Dict[str, Any], status: str,
                             result: Dict[str, Any], reason: str = "") -> None:
    """Write the terminal state of a run. The caller holds the lock."""
    run["status"] = status
    run["result"] = result
    run["reason"] = reason
    run["finished_at"] = time.time()
    try:
        run["queue"].put(("done", None), block=False)
    except Exception:
        pass


def _society_persist_record(run_id: str, goal: str, model_name: str,
                             status: str, result: Any) -> bool:
    """Durable per-run evidence record (survives restarts).

    Returns True when the record was written. A failed write is reported,
    not swallowed: a run whose evidence was not recorded must not be
    reported as a plain success (see ``_society_mark_unrecorded``).
    """
    try:
        from backend.core.evidence_graph import save_run_record
        save_run_record(run_id, {"run_id": run_id, "goal": goal,
                                 "model_name": model_name,
                                 "status": status, "result": result})
        return True
    except Exception as exc:
        logger.warning("Society run %s: evidence record not written (%s: %s)",
                       run_id, type(exc).__name__, exc)
        return False


def _society_mark_unrecorded(run_id: str, status: str) -> None:
    """Downgrade a successful run whose evidence could not be recorded.

    The work ran; the durable record of it did not. Reporting plain
    "completed" would claim an evidence record that does not exist, so the
    run carries ``completed_unrecorded`` and a reason that says why. The
    status is still terminal -- the run is over and releases its admission
    slot -- it simply is not a plain success.
    """
    if status not in _SOCIETY_SUCCESS_STATUSES:
        return
    reason = ("the run finished but its evidence record could not be "
              "written, so the result is not durable")
    with _society_lock:
        run = _society_runs.get(run_id)
        if run is None:
            return
        if run.get("status") not in _SOCIETY_SUCCESS_STATUSES:
            return
        run["status"] = _SOCIETY_UNRECORDED_STATUS
        run["reason"] = reason
        result = run.get("result")
        if isinstance(result, dict):
            result["status"] = _SOCIETY_UNRECORDED_STATUS
            result["reason"] = reason


def _society_reap_locked(now: float) -> list:
    """Finalize runs that can no longer make progress. Caller holds the lock.

    Two cases end a run that would otherwise hold its admission slot for the
    lifetime of the process:

    * the worker thread is gone and the run never produced a result -- the
      thread died without finalizing, so nothing can complete it; and
    * the run outlived its deadline (MECH_SOCIETY_RUN_TIMEOUT_S).

    A thread that is alive but stuck cannot be killed in CPython, so the
    timeout bounds the *slot*, not the thread: the run is reported as failed
    and the worker is asked to stop cooperatively.

    Returns the finalized runs so the caller can persist them after
    releasing the lock.
    """
    pending = []
    for run_id, run in list(_society_runs.items()):
        if _society_is_terminal(run.get("status")):
            continue
        deadline = run.get("deadline")
        if deadline is not None and now >= float(deadline):
            timeout = float(run.get("timeout_s")
                            or _SOCIETY_DEFAULT_RUN_TIMEOUT_S)
            reason = ("run exceeded the %.0fs society run timeout "
                      "(MECH_SOCIETY_RUN_TIMEOUT_S=%.0f); it was finalized as "
                      "a failure" % (timeout, timeout))
            run["cancel_requested"] = True  # ask the stuck worker to stop
            stop = run.get("stop")
            if stop is not None:
                # The worker observes this between stages and between
                # measurements, so the sweep unwinds instead of running to
                # the end of the sample.
                stop.set()
            result = {"status": "failed", "error": reason, "reason": reason,
                      "timeout_s": timeout, "run_id": run_id,
                      "goal": run.get("goal", "")}
            _society_finalize_locked(run, "failed", result, reason)
            pending.append((run_id, run.get("goal", ""),
                            run.get("model_name", ""), "failed", result))
            continue
        worker = run.get("worker")
        if worker is None:
            # Admitted, but no worker was ever attached: it can never start.
            created = float(run.get("created") or now)
            if now - created <= _SOCIETY_STARTUP_GRACE_S:
                continue
            reason = ("the run was admitted but its worker thread was never "
                      "started, so it can no longer progress")
            result = {"status": "interrupted", "error": reason,
                      "reason": reason, "run_id": run_id,
                      "goal": run.get("goal", "")}
            _society_finalize_locked(run, "interrupted", result, reason)
            pending.append((run_id, run.get("goal", ""),
                            run.get("model_name", ""), "interrupted", result))
            continue
        if not worker.is_alive() and run.get("result") is None:
            reason = ("the worker thread exited without finalizing the run, "
                      "so the run was interrupted")
            result = {"status": "interrupted", "error": reason,
                      "reason": reason, "run_id": run_id,
                      "goal": run.get("goal", "")}
            _society_finalize_locked(run, "interrupted", result, reason)
            pending.append((run_id, run.get("goal", ""),
                            run.get("model_name", ""), "interrupted", result))
    return pending


def _society_reap() -> list:
    """Reap dead/expired runs, then persist what changed."""
    with _society_lock:
        pending = _society_reap_locked(time.time())
    for run_id, goal, model_name, status, result in pending:
        if not _society_persist_record(run_id, goal, model_name, status, result):
            _society_mark_unrecorded(run_id, status)
    return [item[0] for item in pending]


def _society_finalize(run_id: str, status: str, result: Dict[str, Any],
                      reason: str = "") -> bool:
    """Finalize a run by id. False when it was already finalized."""
    goal = ""
    model_name = ""
    with _society_lock:
        run = _society_runs.get(run_id)
        if run is None or _society_is_terminal(run.get("status")):
            return False
        goal = run.get("goal", "")
        model_name = run.get("model_name", "")
        _society_finalize_locked(run, status, result, reason)
    if not _society_persist_record(run_id, goal, model_name, status, result):
        _society_mark_unrecorded(run_id, status)
    return True


def _society_admit_and_reserve() -> tuple[str, dict | None]:
    """Check admission and reserve a slot atomically under one lock.

    Returns (reason, run_dict_or_None). If reason is empty string, admission
    was granted and run dict (with run_id, etc.) is returned. If reason is
    non-empty, admission was denied and None is returned.
    """
    reason = ""
    run = None
    pending = []
    with _society_lock:
        # Runs that can no longer finish release their slot before the count,
        # so the counter only ever counts runs that can still finish.
        pending = _society_reap_locked(time.time())
        active = sum(1 for r in _society_runs.values()
                     if r.get("status") in _SOCIETY_LIVE_STATUSES)
        queued = sum(1 for r in _society_runs.values()
                     if r.get("status") == "queued")
        if active >= _SOCIETY_MAX_ACTIVE_RUNS:
            reason = (f"active run limit reached ({active}/"
                      f"{_SOCIETY_MAX_ACTIVE_RUNS})")
        elif queued >= _SOCIETY_MAX_QUEUED_RUNS:
            reason = (f"queue capacity reached ({queued}/"
                      f"{_SOCIETY_MAX_QUEUED_RUNS} queued)")
        else:
            # Admission granted — create and reserve the run slot atomically
            run_id = "r" + uuid.uuid4().hex[:12]
            timeout = _society_run_timeout_s()
            created = time.time()
            run = {
                "run_id": run_id,
                "goal": "",  # will be set by caller
                "model_name": "",  # will be set by caller
                "status": "running",
                "cancel_requested": False,
                "events": [],
                "result": None,
                "queue": queue.Queue(maxsize=1000),
                "created": created,
                "started_at": None,
                "finished_at": None,
                "deadline": created + timeout,
                "timeout_s": timeout,
                "worker": None,        # the worker thread, once attached
                "pipeline_started": False,
                "reason": "",
                # Set when the run is cancelled or times out. The worker
                # checks it between stages and between measurements, so a
                # stopped run stops instead of sweeping every remaining
                # candidate on a CPU nobody is waiting for.
                "stop": threading.Event(),
                "effort": {},
            }
            _society_runs[run_id] = run

            # Prune oldest finished runs
            if len(_society_runs) > _SOCIETY_MAX_RUNS:
                finished = sorted(
                    ((rid, r) for rid, r in _society_runs.items()
                     if _society_is_terminal(r.get("status"))),
                    key=lambda kv: kv[1].get("created", 0),
                )
                for rid, _ in finished[: len(_society_runs) - _SOCIETY_MAX_RUNS]:
                    _society_runs.pop(rid, None)
    for item in pending:
        if not _society_persist_record(*item):
            _society_mark_unrecorded(item[0], item[3])
    return (reason, run)


def _society_admission_reason() -> str:
    """Check admission without reserving a slot.

    Returns reason string if denied, empty string if admitted.
    This is kept for backward compatibility with other callers.
    """
    _society_reap()
    with _society_lock:
        active = sum(1 for r in _society_runs.values()
                     if r.get("status") in _SOCIETY_LIVE_STATUSES)
        queued = sum(1 for r in _society_runs.values()
                     if r.get("status") == "queued")
    if active >= _SOCIETY_MAX_ACTIVE_RUNS:
        return (f"active run limit reached ({active}/"
                f"{_SOCIETY_MAX_ACTIVE_RUNS})")
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


def _society_begin_pipeline(run_id: str) -> bool:
    """Mark that the run's pipeline has started.

    Returns False when the pipeline must not run: the record is gone, the run
    was already finalized (timeout, interrupt), or a cancel arrived before
    the pipeline started -- nothing is in flight then, so the terminal state
    is certain.
    """
    pending = None
    with _society_lock:
        run = _society_runs.get(run_id)
        if run is None:
            # The in-memory record is gone (pruned, or the process restarted).
            # The run still happens and the durable record keeps the evidence.
            return True
        if _society_is_terminal(run.get("status")):
            return False
        if run.get("cancel_requested"):
            reason = "cancelled before the pipeline started"
            result = {"status": "cancelled", "run_id": run_id,
                      "goal": run.get("goal", ""), "reason": reason}
            _society_finalize_locked(run, "cancelled", result, reason)
            pending = (run_id, run.get("goal", ""), run.get("model_name", ""),
                       "cancelled", result)
        else:
            run["pipeline_started"] = True
            run["started_at"] = time.time()
    if pending is not None:
        if not _society_persist_record(*pending):
            _society_mark_unrecorded(pending[0], pending[3])
        return False
    return True


def _society_worker(run_id: str, goal: str, model_name: str) -> None:
    if not _society_begin_pipeline(run_id):
        # Cancelled before it started, already finalized, or the record is
        # gone: the pipeline must not run.
        return
    try:
        society = _society_get()
        with _society_lock:
            stop = _society_runs.get(run_id, {}).get("stop")
            effort = dict(_society_runs.get(run_id, {}).get("effort") or {})
        # The stop check and the effort are optional: a society implementation
        # that does not accept them still runs, it just cannot be stopped
        # between measurements. The signature is checked rather than the call
        # wrapped in a try, so a real error inside the run is not mistaken for
        # an unsupported argument.
        import inspect as _inspect
        try:
            accepted = _inspect.signature(society.run_blocking).parameters
        except (TypeError, ValueError):
            accepted = {}
        extra = {}
        if stop is not None and "should_stop" in accepted:
            extra["should_stop"] = stop.is_set
        if effort and "effort" in accepted:
            extra["effort"] = effort
        result = society.run_blocking(
            goal, model_name=model_name,
            on_event=lambda ev: _society_push(run_id, ev),
            run_id=run_id,
            **extra,
        )
        status = result.get("status", "completed")
    except Exception as exc:
        result = {"status": "error", "error": str(exc)[:500]}
        status = "error"
    persist = False
    with _society_lock:
        run = _society_runs.get(run_id)
        if run is None:
            persist = True  # the record was dropped; keep the evidence
        elif not _society_is_terminal(run.get("status")):
            # Cooperative cancel (requested via mech_loop_stop or the cancel
            # route): the in-flight step ran to completion, but the run is
            # reported cancelled rather than completed. The worker thread
            # itself cannot be killed.
            if run.get("cancel_requested"):
                status = "cancelled"
                result = {"status": "cancelled", "run_id": run_id, "goal": goal,
                          "reason": "cancel requested via control plane; "
                                    "in-flight step ran to completion."}
            reason = str((result or {}).get("reason")
                         or (result or {}).get("error") or "")
            _society_finalize_locked(run, status, result, reason)
            persist = True
        # Otherwise a timeout or an interrupt already finalized this run: the
        # first finalization wins, a late result never overwrites it.
    if persist:
        if not _society_persist_record(run_id, goal, model_name, status, result):
            _society_mark_unrecorded(run_id, status)


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
        return {"status": "error", "error": f"model_name is too long"}
    effort = _society_effort(payload)

    # Atomic admission check + slot reservation under one lock.
    # This eliminates the check-then-act race where concurrent requests
    # could both pass the admission check and exceed limits.
    reason, run = _society_admit_and_reserve()
    if reason:
        return {"status": "busy", "error": reason,
                "queue": {"active": _SOCIETY_MAX_ACTIVE_RUNS,
                          "max_queued": _SOCIETY_MAX_QUEUED_RUNS}}
    if run is None:
        return {"status": "error", "error": "admission failed"}

    worker = threading.Thread(
        target=_society_worker, args=(run["run_id"], goal, model_name),
        daemon=True)
    try:
        # Attach and start the worker under the lock: a reaper running
        # between the two would see a thread that is not alive yet and could
        # declare the run interrupted.
        with _society_lock:
            run["goal"] = goal
            run["model_name"] = model_name
            run["effort"] = effort
            run["worker"] = worker
            worker.start()
    except Exception as exc:
        message = f"could not start the run worker: {exc}"
        with _society_lock:
            run["goal"] = goal
            run["model_name"] = model_name
        _society_finalize(run["run_id"], "failed",
                          {"status": "failed", "error": message,
                           "reason": message, "run_id": run["run_id"]},
                          reason=message)
        return {"status": "error", "error": message}
    return {"runId": run["run_id"], "status": "started",
            "effort": effort,
            "stream": f"/api/society/stream?runId={run['run_id']}"}


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
    pending = []
    body = None
    with _society_lock:
        # A run that can no longer progress must never be reported as
        # running: reap it before reading the record.
        pending = _society_reap_locked(time.time())
        run = _society_runs.get(run_id)
        if run is not None:
            body = {"run_id": run_id, "status": run["status"],
                    "goal": run["goal"], "events": list(run["events"]),
                    "result": run["result"], "reason": run.get("reason", ""),
                    "cancel_requested": bool(run.get("cancel_requested")),
                    "started_at": run.get("started_at"),
                    "finished_at": run.get("finished_at")}
    for item in pending:
        if not _society_persist_record(*item):
            _society_mark_unrecorded(item[0], item[3])
    if body is not None:
        return body
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


@router.post("/society/runs/{run_id}/cancel")
def society_run_cancel(run_id: str) -> Dict[str, Any]:
    """Request cancellation of a Society run (cooperative, honest).

    Mirrors the job queue's cancel (/api/v1/jobs/{jobId}/cancel): a finished
    run is a 409, an unknown run is a 404, and the answer says what actually
    happened. A run whose pipeline never started has nothing in flight, so it
    is cancelled outright. A running run is flagged and keeps reporting
    "running" until its worker observes the flag: the worker thread cannot be
    killed, and this route does not pretend otherwise.
    """
    pending = []
    outcome = ""
    detail = ""
    body: Dict[str, Any] = {}
    with _society_lock:
        pending = _society_reap_locked(time.time())
        run = _society_runs.get(run_id)
        if run is None:
            record = None
            try:
                from backend.core.evidence_graph import load_run_record
                record = load_run_record(run_id)
            except Exception:
                record = None
            if isinstance(record, dict):
                outcome = "finished"
                detail = (f"run {run_id!r} already finished with status "
                          f"{record.get('status', 'stored')!r}; it cannot be "
                          f"cancelled")
            else:
                outcome = "unknown"
                detail = f"unknown runId {run_id!r}"
        elif _society_is_terminal(run.get("status")):
            outcome = "finished"
            detail = (f"run {run_id!r} already finished with status "
                      f"{run.get('status')!r}; it cannot be cancelled")
        else:
            run["cancel_requested"] = True
            stop = run.get("stop")
            if stop is not None:
                stop.set()
            if not run.get("pipeline_started"):
                # Nothing is in flight: the terminal state is certain.
                reason = "cancelled before the pipeline started"
                result = {"status": "cancelled", "run_id": run_id,
                          "goal": run.get("goal", ""), "reason": reason}
                _society_finalize_locked(run, "cancelled", result, reason)
                pending.append((run_id, run.get("goal", ""),
                                run.get("model_name", ""), "cancelled", result))
                outcome = "cancelled"
                detail = reason
            else:
                outcome = "cancel_requested"
                detail = ("cancel requested: the run is still running until "
                          "its worker observes the flag; an in-flight step "
                          "runs to completion")
            body = {"run_id": run_id, "status": run["status"],
                    "goal": run.get("goal", ""), "result": run.get("result"),
                    "reason": run.get("reason", ""),
                    "cancel_requested": True,
                    "outcome": outcome, "note": detail,
                    "provenance": "live"}
    for item in pending:
        if not _society_persist_record(*item):
            _society_mark_unrecorded(item[0], item[3])
    if outcome == "unknown":
        raise HTTPException(status_code=404, detail=detail)
    if outcome == "finished":
        raise HTTPException(status_code=409, detail=detail)
    return body


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
    # A run that timed out must not keep streaming forever.
    _society_reap()
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
            finished = (_society_is_terminal(run["status"])
                        and run["queue"].empty())
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
                    if (_society_is_terminal(run["status"])
                            and run["queue"].empty()):
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
