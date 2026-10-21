"""NNsight adapter (spec §9.2, priority 3).

Trace-based activation access and interventions, reported honestly when the
dependency is missing
-----------------------------------------------------------------------
``nnsight`` is not installed here, and it is deliberately not in
``requirements.txt``. That absence is a first-class state rather than an error
to paper over:

* ``describe()`` reports ``available: False``, ``version: None``, every
  operation ``False``, and a reason naming the package;
* ``build_trace_request()`` still works -- it is data construction, and a
  caller needs to see the request it *would* send before a GPU is touched;
* ``run_trace()`` returns ``NOT_EXECUTED`` with that reason and no data.

``NOT_EXECUTED`` here means "registered, nothing ran" (AGENTS.md: no fabricated
results), not "the run failed". A failed run is ``FAILED``, and it carries no
data either -- see ``_refusal`` and ``_failed``.

Request construction is separate from execution on purpose, and nothing is
imported at module import time, so ``test_all_backend_modules_import.py`` stays
green on a machine without the package.

The execution path follows NNsight's documented trace API
(``model.trace(...)``, ``proxy.save()``, in-place proxy assignment). It has
*not* been exercised against an installed nnsight in this environment and the
tests do not fake an execution. If it is wrong it fails loudly rather than
returning plausible activations: every declared intervention must resolve and
be applied, and a trace that raises is reported ``FAILED``.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import itertools
import re
from contextlib import contextmanager
from typing import Any, Iterable, Optional

from backend.core.execution_status import (
    FAILED,
    IMPLEMENTED,
    NOT_EXECUTED,
    not_executed_payload,
)

ADAPTER_NAME = "nnsight"
PACKAGE = "nnsight"

#: The trace operations this adapter expresses. All come from one entry point
#: (``nnsight.LanguageModel``), so they rise and fall together.
OPERATIONS = ("trace_activation_capture", "trace_intervention", "trace_logits")

#: Intervention kinds a request may declare, and what each means on a site.
INTERVENTION_OPS = ("set", "add", "scale", "zero")

_INSTALL_HINT = "pip install nnsight"

_ABSENT_REASON = (
    f"the '{PACKAGE}' package is not installed, so no NNsight trace can be "
    f"built or run; this adapter reports {NOT_EXECUTED} rather than falling "
    f"back to a different engine"
)


class TraceRequestError(ValueError):
    """A trace request is malformed, or a site does not resolve on the model."""


# ── lazy load ────────────────────────────────────────────────────────

#: Success is cached; absence is not. Installing the package -- or a test
#: injecting it into ``sys.modules`` -- is picked up on the next call without a
#: process restart, while a cached import keeps repeated diagnostics cheap.
_CACHE: dict[str, Any] = {}


def _version_of(module: Any) -> Optional[str]:
    """Installed distribution version, else the module's own ``__version__``.

    A module injected for a test has no distribution metadata, and reporting
    ``version: None`` for a package that is plainly importable would be lying
    about what the adapter can see.
    """
    try:
        return importlib.metadata.version(PACKAGE)
    except importlib.metadata.PackageNotFoundError:
        pass
    value = getattr(module, "__version__", None)
    return str(value) if value else None


def _load_nnsight() -> tuple[Any, Optional[str], Optional[str]]:
    """Return ``(module, version, reason)``. Absence is reported, never raised."""
    cached = _CACHE.get("module")
    if cached is not None:
        return cached, _CACHE.get("version"), None
    try:
        module = importlib.import_module(PACKAGE)
    except ImportError as exc:
        return None, None, (
            f"{_ABSENT_REASON} ({type(exc).__name__}: {exc}); install it with "
            f"`{_INSTALL_HINT}`"
        )
    version = _version_of(module)
    _CACHE["module"], _CACHE["version"] = module, version
    return module, version, None


def _trace_entry_point(module: Any) -> Optional[str]:
    """The class that opens a trace, or None when the module offers none.

    Both spellings are checked: 0.2/0.3 exported ``NNsight``, 0.4+ exports
    ``LanguageModel``. A module with neither cannot trace whatever its version
    string says, so it is not "available" in any sense a caller cares about.
    """
    if module is None:
        return None
    for name in ("LanguageModel", "NNsight"):
        if hasattr(module, name):
            return name
    return None


def describe() -> dict[str, Any]:
    """Machine-readable adapter report for diagnostics.

    Name, package, version, availability, the operations actually supported,
    and -- when nothing is -- the reason. ``supported_operations`` is empty
    whenever ``available`` is False: there is no state in which a capability is
    claimed without the package behind it.
    """
    module, version, reason = _load_nnsight()
    entry_point = _trace_entry_point(module)
    available = entry_point is not None
    if not available and reason is None:
        reason = (
            f"the '{PACKAGE}' module does not expose a trace entry point "
            f"(expected LanguageModel or NNsight); no NNsight operation is "
            f"supported"
        )
    operations = {name: available for name in OPERATIONS}
    return {
        "adapter": ADAPTER_NAME,
        "package": PACKAGE,
        "version": version,
        "available": available,
        "status": IMPLEMENTED if available else NOT_EXECUTED,
        "reason": None if available else reason,
        "operations": operations,
        "supported_operations": [n for n, ok in operations.items() if ok],
        "entry_point": entry_point,
        "install_hint": None if available else _INSTALL_HINT,
    }


#: ``capabilities()`` is the name the diagnostics surfaces use for this
#: mapping. It is the same report, not a second one.
capabilities = describe


# ── activation sites (§9.3) ──────────────────────────────────────────

_STEP = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)((?:\[\s*-?\d+\s*\])*)$")


def _parse_site_path(path: str) -> list[Any]:
    """``transformer.h[0].output`` -> ``["transformer", "h", 0, "output"]``.

    Attribute and index steps are both needed because NNsight addresses sites
    that way, and an unparseable path must be rejected before a run starts
    rather than at the first ``getattr`` halfway through one.
    """
    steps: list[Any] = []
    for part in str(path).split("."):
        match = _STEP.match(part)
        if match is None:
            raise TraceRequestError(
                f"unparseable site path {path!r}: segment {part!r} is not an "
                f"attribute name with optional integer indices"
            )
        steps.append(match.group(1))
        steps.extend(
            int(i) for i in re.findall(r"\[\s*(-?\d+)\s*\]", match.group(2))
        )
    return steps


def _resolve_site(root: Any, path: str) -> Any:
    """Walk ``path`` from ``root``, failing loudly when the site is absent."""
    current = root
    for step in _parse_site_path(path):
        try:
            current = (
                current[step] if isinstance(step, int) else getattr(current, step)
            )
        except (AttributeError, IndexError, KeyError, TypeError) as exc:
            raise TraceRequestError(
                f"site path {path!r} does not resolve on this model at step "
                f"{step!r}: {type(exc).__name__}: {exc}"
            ) from exc
    return current


def _site_entries(sites: Iterable[Any]) -> list[dict[str, Any]]:
    """Normalise sites to ``{site_id, canonical_path, ...}`` rows (§9.3).

    A bare string is its own site id, which is what a caller passing a hook
    path means. Ids and paths must be unique: two captures under one name is a
    silent overwrite, and the second one wins.
    """
    entries: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    seen_paths: set[str] = set()
    for raw in sites:
        if isinstance(raw, str):
            raw = {"canonical_path": raw}
        if not isinstance(raw, dict):
            raise TraceRequestError(
                f"a site must be a str or a dict, got {type(raw).__name__}"
            )
        path = str(raw.get("canonical_path") or raw.get("site_id") or "")
        _parse_site_path(path)
        site_id = str(raw.get("site_id") or path)
        if site_id in seen_ids:
            raise TraceRequestError(f"duplicate site_id {site_id!r}")
        if path in seen_paths:
            raise TraceRequestError(f"duplicate site path {path!r}")
        seen_ids.add(site_id)
        seen_paths.add(path)
        entry: dict[str, Any] = {
            "site_id": site_id,
            "canonical_path": path,
            "axis_names": list(raw.get("axis_names") or []),
        }
        for key in ("display_name", "component", "layer"):
            if raw.get(key) is not None:
                entry[key] = raw[key]
        entries.append(entry)
    return entries


def _intervention_specs(interventions: Iterable[Any],
                        paths: dict[str, str]) -> list[dict[str, Any]]:
    """Validate intervention specs; every one names a concrete site path."""
    specs: list[dict[str, Any]] = []
    for raw in interventions:
        if not isinstance(raw, dict):
            raise TraceRequestError(
                f"an intervention must be a dict, got {type(raw).__name__}"
            )
        site_id = raw.get("site_id")
        path = raw.get("canonical_path")
        if site_id is not None:
            path = paths.get(str(site_id)) or path
        if not path:
            raise TraceRequestError(
                f"an intervention must name a declared site_id or supply a "
                f"canonical_path; got {dict(raw)!r}"
            )
        _parse_site_path(path)
        op = str(raw.get("op") or "set").lower()
        if op not in INTERVENTION_OPS:
            raise TraceRequestError(
                f"unknown intervention op {op!r}; supported: "
                f"{list(INTERVENTION_OPS)}"
            )
        if op != "zero" and "value" not in raw:
            raise TraceRequestError(f"intervention op {op!r} needs a 'value'")
        specs.append({
            "site_id": str(site_id) if site_id is not None else str(path),
            "canonical_path": str(path),
            "op": op,
            "value": raw.get("value"),
        })
    return specs


_JOB_IDS = itertools.count(1)


def build_trace_request(prompt: str, sites: Iterable[Any], *,
                        interventions: Iterable[Any] = (),
                        job_id: Optional[str] = None) -> dict[str, Any]:
    """Build a trace request. Nothing is imported, resolved, or run.

    The result describes the work: the prompt, one row per site, the save name
    each capture lands in (the site id), and the intervention specs that must be
    applied. It carries ``executed: False`` and ``NOT_EXECUTED`` -- including
    when nnsight *is* installed, where a built request is still not a
    measurement.
    """
    if not isinstance(prompt, str) or not prompt.strip():
        raise TraceRequestError("a trace request needs a non-empty prompt")
    entries = _site_entries(sites)
    if not entries:
        raise TraceRequestError("a trace request needs at least one site to capture")
    paths = {e["site_id"]: e["canonical_path"] for e in entries}
    report = describe()
    return {
        "adapter": ADAPTER_NAME,
        "package": PACKAGE,
        "version": report["version"],
        "available": report["available"],
        "job_id": str(job_id or f"{ADAPTER_NAME}-{next(_JOB_IDS)}"),
        "prompt": prompt,
        "sites": entries,
        #: site_id -> native site path; the site id is also the save name.
        "saves": dict(paths),
        "interventions": _intervention_specs(interventions, paths),
        "executed": False,
        "status": NOT_EXECUTED,
        "reason": "request built; nothing has been executed",
    }


# ── job-scoped hook lifecycle (§9.4) ─────────────────────────────────

#: job_id -> record of what that job registered. Never survives the job.
_ACTIVE: dict[str, dict[str, Any]] = {}


@contextmanager
def trace_scope(job_id: str):
    """Scope one job's saves/interventions, dropping the record in ``finally``.

    NNsight owns the hooks themselves -- they live and die with the
    ``model.trace(...)`` block. What this registry owns is MECH's record of
    what the job registered, and it must not survive an exception: a stale
    entry here is how a later job inherits an intervention it never asked for.
    """
    if job_id in _ACTIVE:
        raise TraceRequestError(f"a trace is already active for job {job_id!r}")
    job: dict[str, Any] = {"job_id": job_id, "saves": {}, "interventions": []}
    _ACTIVE[job_id] = job
    try:
        yield job
    finally:
        _ACTIVE.pop(job_id, None)


def active_traces() -> list[str]:
    """Job ids with a trace in flight. Empty after any run, successful or not."""
    return sorted(_ACTIVE)


# ── execution ────────────────────────────────────────────────────────

def _refusal(reason: str, version: Optional[str]) -> dict[str, Any]:
    """``NOT_EXECUTED`` envelope: the reason, and no measurement of any kind."""
    payload = not_executed_payload(reason, engine=ADAPTER_NAME, capability=PACKAGE)
    payload.update({"adapter": ADAPTER_NAME, "package": PACKAGE, "version": version})
    return payload


def _failed(reason: str, version: Optional[str]) -> dict[str, Any]:
    """``FAILED`` envelope -- distinct from NOT_EXECUTED, and equally data-free."""
    return {
        "status": FAILED,
        "engine": ADAPTER_NAME,
        "adapter": ADAPTER_NAME,
        "package": PACKAGE,
        "version": version,
        "reason": reason,
        "measured": False,
    }


def _apply_intervention(proxy: Any, spec: dict[str, Any]) -> None:
    """Write the intervention onto the resolved site, inside the trace.

    In-place proxy assignment is how NNsight registers an edit on the graph.
    Raising here is the point: a run that cannot apply what it declared must not
    continue as though it had.
    """
    op = spec["op"]
    if op == "set":
        proxy[:] = spec["value"]
    elif op == "zero":
        proxy[:] = proxy * 0
    elif op == "add":
        proxy[:] = proxy + spec["value"]
    else:  # scale
        proxy[:] = proxy * spec["value"]


def _saved_value(saved: Any) -> Any:
    """The value behind a ``.save()`` result, when the proxy exposes one."""
    return getattr(saved, "value", saved)


def run_trace(request: dict[str, Any], model: Any = None, *,
              job_id: Optional[str] = None) -> dict[str, Any]:
    """Execute a request built by :func:`build_trace_request`.

    Returns ``NOT_EXECUTED`` and no data when nnsight is absent. That check
    happens *before* the model is touched, so a caller learns the dependency is
    missing rather than something about its model.

    When nnsight is present each declared intervention must resolve and be
    applied. A capture without the intervention it was asked for is not a
    slightly worse result; it is a different experiment, so it is ``FAILED``.
    """
    report = describe()
    if not report["available"]:
        return _refusal(report["reason"], report["version"])
    if model is None:
        return _failed(
            "no model was supplied; an NNsight trace needs an already-constructed "
            "LanguageModel",
            report["version"],
        )

    scope = str(job_id or request.get("job_id") or f"{ADAPTER_NAME}-{next(_JOB_IDS)}")
    saves = dict(request.get("saves") or {})
    specs = [dict(spec) for spec in (request.get("interventions") or [])]
    saved_vars: dict[str, Any] = {}
    applied: list[dict[str, Any]] = []

    try:
        with trace_scope(scope) as job:
            with model.trace(request["prompt"]):
                for site_id, path in saves.items():
                    saved_vars[site_id] = _resolve_site(model, path).save()
                    job["saves"][site_id] = path
                for spec in specs:
                    resolved = _resolve_site(model, spec["canonical_path"])
                    _apply_intervention(resolved, spec)
                    applied.append({"site_id": spec["site_id"], "op": spec["op"]})
                    job["interventions"].append(dict(spec))
        if len(applied) != len(specs):
            return _failed(
                f"{len(specs) - len(applied)} declared intervention(s) were not "
                f"applied: {[s['site_id'] for s in specs if s not in applied]}",
                report["version"],
            )
    except TraceRequestError as exc:
        return _failed(f"the trace request is invalid: {exc}", report["version"])
    except Exception as exc:  # noqa: BLE001 - a failed hook invalidates the run
        return _failed(
            f"the NNsight trace did not complete: {type(exc).__name__}: {exc}",
            report["version"],
        )

    # Eligibility is the validation layer's call; this only states what ran.
    return {
        "status": IMPLEMENTED,
        "engine": ADAPTER_NAME,
        "adapter": ADAPTER_NAME,
        "package": PACKAGE,
        "version": report["version"],
        "provenance": "live",
        "job_id": scope,
        "prompt": request["prompt"],
        "saves": {name: _saved_value(var) for name, var in saved_vars.items()},
        "interventions": applied,
    }
