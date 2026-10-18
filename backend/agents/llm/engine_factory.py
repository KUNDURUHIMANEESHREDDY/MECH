"""Deterministic selection of the LLM engine the Research Society may use.

Why this module exists
----------------------
`backend/agents/society.py` had no way to reach a language model at all, so
"autonomous hypothesis generation" was a label on a workflow that generated
none. The engines in this package already fail closed -- a missing key or an
unreachable backend produces an `unavailable` envelope with a reason rather
than canned text -- but nothing was constructing one, so the safe behavior and
the reachable behavior were the same: nothing happened.

This factory is the single, documented place where an engine is selected. It
is the only switch between "no model" and "a model", and it is deliberately
default-off.

Selection (environment only, never inferred)
--------------------------------------------
`MECH_LLM_ENGINE` selects the engine:

* unset, empty, or ``"none"`` -> **no engine**. This is the default on purpose.
  Every pre-existing behaviour and every existing test is unchanged, and the
  Society's hypothesis node reports NOT_EXECUTED rather than inventing a
  hypothesis. *Unset must never mean "pick something for me".*
* ``"ollama"`` -> `OllamaEngine`, with the endpoint from `MECH_OLLAMA_ENDPOINT`
  (default ``http://127.0.0.1:11434/v1/chat/completions``) and the model from
  `MECH_OLLAMA_MODEL` (default ``llama3``).
* ``"openai"`` -> `OpenAIEngine`, which keeps its own existing convention
  (`OPENAI_API_KEY`), plus `MECH_OPENAI_MODEL` overriding its default model.
* anything else -> no engine, and `engine_description()` says the value was
  unrecognised. An unknown name is a configuration error, not a licence to
  guess an engine the operator did not ask for.

Availability
------------
`build_engine()` only constructs; it performs no I/O. `engine_description()`
performs a real reachability probe (`_probe`) and reports `available: bool`
plus a `reason` when it is not. The probe is bounded by
`MECH_LLM_PROBE_TIMEOUT` seconds (default 2.0). Nothing here ever raises: an
exception is a failed probe reported as `available: False`, never a fabricated
success. A caller that wants to know whether generation can happen reads
`engine_description()["available"]`; it does not assume.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

from .engine_base import LLMEngine

#: Accepted values of `MECH_LLM_ENGINE`. `None` means "no engine".
SUPPORTED_ENGINES = ("none", "ollama", "openai")

#: Ollama's OpenAI-compatible chat endpoint for the default host.
DEFAULT_OLLAMA_ENDPOINT = "http://127.0.0.1:11434/v1/chat/completions"
DEFAULT_OLLAMA_MODEL = "llama3"

#: OpenAI's chat endpoint and the model used when `MECH_OPENAI_MODEL` is unset.
DEFAULT_OPENAI_ENDPOINT = "https://api.openai.com/v1/chat/completions"
DEFAULT_OPENAI_MODEL = "gpt-4o"

#: Seconds to wait for a reachability probe. Short on purpose: availability is
#: asked on a request path, and a probe that can hang is worse than one that
#: reports unavailable.
DEFAULT_PROBE_TIMEOUT = 2.0


def _env(name: str) -> str:
    return str(os.environ.get(name, "") or "").strip()


def _timeout(name: str, default: float) -> float:
    raw = _env(name)
    if not raw:
        return default
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return default
    return value if value > 0 else default


def selected_engine_name() -> Optional[str]:
    """The engine `MECH_LLM_ENGINE` asks for, or None for "no engine".

    Returning None for unset/empty/``"none"``/unknown is the whole point: the
    default must be the inert one.
    """
    requested = _env("MECH_LLM_ENGINE").lower()
    if requested in ("", "none"):
        return None
    if requested in SUPPORTED_ENGINES:
        return requested
    return None


@dataclass(frozen=True)
class EngineSelection:
    """What `MECH_LLM_ENGINE` resolved to, before any engine is constructed."""

    #: "none" | "ollama" | "openai" | "unknown"
    name: str
    #: The raw environment value, kept so a report can quote it back.
    requested: str
    endpoint: str = ""
    model: str = ""

    @property
    def is_none(self) -> bool:
        return self.name == "none"


def select_engine() -> EngineSelection:
    """Resolve the environment into a named, documented selection."""
    requested = _env("MECH_LLM_ENGINE")
    lowered = requested.lower()

    if lowered in ("", "none"):
        return EngineSelection(name="none", requested=requested)

    if lowered == "ollama":
        # `MECH_OLLAMA_ENDPOINT` may name either Ollama's native base URL or a
        # full chat path; normalising here means a caller cannot accidentally
        # POST to the base and get a 404 that reads like an outage.
        endpoint = _env("MECH_OLLAMA_ENDPOINT") or DEFAULT_OLLAMA_ENDPOINT
        return EngineSelection(
            name="ollama", requested=requested,
            endpoint=_ollama_chat_url(endpoint),
            model=_env("MECH_OLLAMA_MODEL") or DEFAULT_OLLAMA_MODEL,
        )

    if lowered == "openai":
        return EngineSelection(
            name="openai", requested=requested,
            endpoint=DEFAULT_OPENAI_ENDPOINT,
            model=_env("MECH_OPENAI_MODEL") or DEFAULT_OPENAI_MODEL,
        )

    return EngineSelection(name="unknown", requested=requested)


def _ollama_chat_url(endpoint: str) -> str:
    """Accept a base URL or a full chat path; return a full chat path."""
    url = endpoint.rstrip("/")
    parsed = urllib.parse.urlsplit(url)
    if not parsed.scheme or not parsed.netloc:
        # No scheme/host to reason about -- treat the value as a bare host:port.
        url = "http://" + url
        parsed = urllib.parse.urlsplit(url)
    path = parsed.path.rstrip("/")
    if path.endswith("/v1/chat/completions"):
        return url
    if path.endswith("/v1"):
        return urllib.parse.urlunsplit(
            (parsed.scheme, parsed.netloc, path + "/chat/completions", "", ""))
    return urllib.parse.urlunsplit(
        (parsed.scheme, parsed.netloc, path + "/v1/chat/completions", "", ""))


def build_engine() -> Optional[LLMEngine]:
    """Construct the engine the environment selects, or None for "no engine".

    Deterministic and side-effect free with respect to the network: this does
    not check reachability. A configuration error (an unrecognised
    `MECH_LLM_ENGINE`) yields None, exactly like unset -- guessing an engine is
    the failure mode this function exists to prevent.

    Never raises. A construction failure is reported by `engine_description()`
    as an unavailable engine with a reason; it is not converted into an engine
    that appears to work.
    """
    selection = select_engine()

    if selection.name == "ollama":
        from .ollama_engine import OllamaEngine
        return OllamaEngine(
            model=selection.model,
            endpoint=selection.endpoint,
            timeout=_timeout("MECH_OLLAMA_TIMEOUT", 10.0),
        )

    if selection.name == "openai":
        from .openai_engine import OpenAIEngine
        return OpenAIEngine(model=selection.model)

    # "none" (unset / empty / explicit) and "unknown" both mean no engine.
    return None


# -- reachability ---------------------------------------------------------
# The probe is injectable so a test can prove the unavailable path without a
# live server, and never has to mock a *successful generation* to do it.

Probe = Callable[[str, float], Dict[str, Any]]


def _http_get(url: str, timeout: float,
              headers: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    request = urllib.request.Request(url, headers=headers or {}, method="GET")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = response.read(2048)
    return {"status": int(getattr(response, "status", 200) or 200),
            "body": body}


def _probe_ollama(endpoint: str, timeout: float) -> Dict[str, Any]:
    """Ask Ollama's native tags endpoint whether the server answers.

    Deliberately narrow: answering HTTP is availability of the *server*. It is
    not a claim that the configured model is pulled, because that would need a
    generation call and this must stay bounded and side-effect free.
    """
    base = endpoint
    for suffix in ("/v1/chat/completions", "/v1", "/chat/completions"):
        if base.endswith(suffix):
            base = base[: -len(suffix)]
            break
    tags_url = base.rstrip("/") + "/api/tags"
    try:
        response = _http_get(tags_url, timeout)
    except urllib.error.HTTPError as exc:
        if int(getattr(exc, "code", 0) or 0) < 500:
            # The server answered -- it just disliked this route.
            return {"available": True, "detail": f"HTTP {exc.code} from {tags_url}"}
        return {"available": False,
                "reason": f"Ollama at {tags_url} answered HTTP {exc.code}."}
    except Exception as exc:  # noqa: BLE001 - availability is a report, not a raise
        return {"available": False,
                "reason": (f"Ollama is not reachable at {tags_url}: "
                           f"{type(exc).__name__}: {exc}")}

    models = []
    try:
        parsed = json.loads(response.get("body") or b"{}")
        models = [str(m.get("name", "")) for m in parsed.get("models", [])
                  if isinstance(m, dict)]
    except Exception:
        models = []
    return {"available": True, "models": models}


def _probe_openai(endpoint: str, timeout: float) -> Dict[str, Any]:
    key = _env("OPENAI_API_KEY")
    if not key:
        return {"available": False,
                "reason": "OPENAI_API_KEY environment variable is not set."}
    models_url = endpoint.split("/v1/")[0].rstrip("/") + "/v1/models"
    try:
        _http_get(models_url, timeout, {"Authorization": f"Bearer {key}"})
    except urllib.error.HTTPError as exc:
        if int(getattr(exc, "code", 0) or 0) in (401, 403):
            return {"available": False,
                    "reason": f"OpenAI rejected the API key (HTTP {exc.code})."}
        if int(getattr(exc, "code", 0) or 0) < 500:
            return {"available": True, "detail": f"HTTP {exc.code} from {models_url}"}
        return {"available": False,
                "reason": f"OpenAI answered HTTP {exc.code} at {models_url}."}
    except Exception as exc:  # noqa: BLE001
        return {"available": False,
                "reason": (f"OpenAI is not reachable at {models_url}: "
                           f"{type(exc).__name__}: {exc}")}
    return {"available": True}


def _probe(selection: EngineSelection, timeout: float) -> Dict[str, Any]:
    if selection.name == "ollama":
        return _probe_ollama(selection.endpoint, timeout)
    if selection.name == "openai":
        return _probe_openai(selection.endpoint, timeout)
    return {"available": False, "reason": "No LLM engine is configured."}


def engine_description(probe: Optional[Probe] = None) -> Dict[str, Any]:
    """Which engine is selected, and whether it actually answers right now.

    Returns a payload that can travel with a Society run so the claim "an LLM
    produced this" is traceable to an engine identity:

        {"engine": "ollama"|"openai"|"none", "selected": bool,
         "endpoint": str, "model": str, "available": bool, "reason": str,
         "detail": str, "requested": str}

    `selected` is a statement about configuration; `available` is a statement
    about this moment, established by a real probe. They are separate fields
    because they fail for different reasons and a reader needs to tell them
    apart. Nothing raises out of here: an exception inside the probe is a
    failed probe.
    """
    selection = select_engine()
    timeout = _timeout("MECH_LLM_PROBE_TIMEOUT", DEFAULT_PROBE_TIMEOUT)

    if selection.is_none:
        return {"engine": "none", "selected": False,
                "endpoint": "", "model": "", "available": False,
                "reason": ("No LLM engine is configured; set MECH_LLM_ENGINE "
                           "to 'ollama' or 'openai' to enable live hypothesis "
                           "generation."),
                "detail": "", "requested": selection.requested}

    if selection.name == "unknown":
        return {"engine": "unknown", "selected": False,
                "endpoint": "", "model": "", "available": False,
                "reason": (f"MECH_LLM_ENGINE={selection.requested!r} is not a "
                           f"supported engine; expected one of "
                           f"{list(SUPPORTED_ENGINES)}."),
                "detail": "", "requested": selection.requested}

    probe_fn = probe or _probe
    try:
        result = probe_fn(selection, timeout)
    except Exception as exc:  # noqa: BLE001 - never a fabricated success
        result = {"available": False,
                  "reason": f"{type(exc).__name__}: {exc}"}
    if not isinstance(result, dict):
        result = {"available": False,
                  "reason": f"probe returned {type(result).__name__}"}

    description = {
        "engine": selection.name,
        "selected": True,
        "endpoint": selection.endpoint,
        "model": selection.model,
        "available": bool(result.get("available", False)),
        "reason": str(result.get("reason", "") or ""),
        "detail": str(result.get("detail", "") or ""),
        "requested": selection.requested,
    }
    if isinstance(result.get("models"), list):
        description["models"] = list(result["models"])
    return description
