"""Subsystem health probes.

`GET /health` answers one question only: is this process alive and serving? It
deliberately returns a fixed body, because a liveness probe that runs real
dependency checks turns a slow torch import into an orchestrator restart.

That leaves a gap the old hardcoded `{"status": "healthy"}` papered over: a
process can be perfectly alive while GPT-2 is missing, CUDA is absent, the
database is read-only, or no executor can run. This module answers the second
question -- *can this deployment actually do science?* -- by probing each
subsystem and reporting what it found, including the reason for every failure.

A subsystem is only `ok` if the probe actually exercised it. Nothing is
reported healthy by default.
"""

from __future__ import annotations

import importlib
import importlib.util
import os
import sqlite3
import tempfile
from typing import Any, Callable, Dict, Optional

# Status vocabulary, deliberately narrower than "healthy".
OK = "ok"
DEGRADED = "degraded"
UNAVAILABLE = "unavailable"
UNKNOWN = "unknown"


def _probe_process() -> Dict[str, Any]:
    """The one thing /health already covers, stated explicitly."""
    return {"status": OK, "detail": "process is serving requests"}


def _probe_dependencies() -> Dict[str, Any]:
    """Check the ML stack actually imports, at the pinned versions."""
    missing, incompatible, versions = [], [], {}
    for package in ("torch", "transformers", "transformer_lens"):
        try:
            module = importlib.import_module(package)
        except Exception as exc:
            missing.append(f"{package} ({type(exc).__name__}: {exc})")
            continue
        versions[package] = getattr(module, "__version__", "unknown")

    try:
        from backend.interpretability.tl_compat import check_compatibility
        report = check_compatibility()
        if not report.get("compatible"):
            incompatible.append(report.get("reason") or "version mismatch")
    except Exception as exc:
        incompatible.append(f"compat probe failed: {exc}")

    if missing:
        status = UNAVAILABLE
    elif incompatible:
        status = DEGRADED
    else:
        status = OK
    return {
        "status": status,
        "versions": versions,
        "missing": missing,
        "incompatible": incompatible,
        "detail": "ML stack import and version check",
    }


def _probe_model() -> Dict[str, Any]:
    """Report whether GPT-2 is loaded, without loading it as a side effect."""
    try:
        from backend.services import gpt2_engine
    except Exception as exc:
        return {"status": UNAVAILABLE, "reason": f"gpt2_engine import failed: {exc}"}

    available = False
    try:
        available = bool(gpt2_engine.is_available())
    except Exception as exc:
        return {"status": UNAVAILABLE, "reason": f"is_available() raised: {exc}"}

    loaded = None
    device = None
    try:
        state = gpt2_engine.status() if hasattr(gpt2_engine, "status") else {}
        loaded = state.get("loaded")
        device = state.get("device")
    except Exception:
        pass

    return {
        "status": OK if (available and loaded) else DEGRADED,
        "stack_available": available,
        "weights_loaded": bool(loaded),
        "device": device,
        "detail": ("weights loaded" if loaded else
                   "ML stack present but weights are not loaded"),
    }


def _probe_executor() -> Dict[str, Any]:
    """Can a causal executor actually run? Check, don't execute a sweep."""
    try:
        spec = importlib.util.find_spec(
            "backend.interpretability.discovery.live_discovery")
        if spec is None:
            return {"status": UNAVAILABLE,
                    "reason": "live_discovery module is absent"}
        from backend.interpretability.discovery.live_discovery import LiveIOIDiscovery
        available = bool(LiveIOIDiscovery.available())
    except Exception as exc:
        return {"status": UNAVAILABLE, "reason": f"executor probe failed: {exc}"}

    return {
        "status": OK if available else UNAVAILABLE,
        "live_discovery_available": available,
        "detail": ("live executor connected" if available else
                   "no live executor; discovery cannot produce live evidence"),
    }


def _probe_storage() -> Dict[str, Any]:
    """The evidence database must be writable, not merely present."""
    try:
        from backend.storage.database import get_default_db_path
        path = get_default_db_path()
    except Exception as exc:
        return {"status": UNAVAILABLE, "reason": f"db path resolution failed: {exc}"}

    parent = path.parent
    if not parent.exists():
        return {"status": UNAVAILABLE,
                "reason": f"storage directory does not exist: {parent}"}

    try:
        with tempfile.NamedTemporaryFile(dir=parent, suffix=".probe") as handle:
            handle.write(b"probe")
            handle.flush()
    except Exception as exc:
        return {"status": UNAVAILABLE,
                "reason": f"storage is not writable: {exc}"}

    try:
        connection = sqlite3.connect(str(path))
        try:
            mode = connection.execute("PRAGMA journal_mode").fetchone()
            journal = mode[0] if mode else "unknown"
        finally:
            connection.close()
    except Exception as exc:
        journal = f"unreadable: {exc}"

    return {
        "status": OK,
        "path": str(path),
        "exists": path.exists(),
        "journal_mode": journal,
        "detail": "storage directory is writable",
    }


def _probe_evidence() -> Dict[str, Any]:
    """The evidence pipeline must be importable and its policy present."""
    checks = {}
    for label, module in (
        ("boundary", "backend.core.evidence_boundary"),
        ("evidence_graph", "backend.core.evidence_graph"),
        ("evidence_policy", "backend.agents.evidence_policy"),
    ):
        try:
            importlib.import_module(module)
            checks[label] = OK
        except Exception as exc:
            checks[label] = f"{UNAVAILABLE}: {exc}"

    failed = [k for k, v in checks.items() if v != OK]
    return {
        "status": OK if not failed else UNAVAILABLE,
        "components": checks,
        "detail": ("evidence boundary, graph, and policy are importable"
                   if not failed else f"missing: {', '.join(failed)}"),
    }


PROBES: Dict[str, Callable[[], Dict[str, Any]]] = {
    "process": _probe_process,
    "dependency": _probe_dependencies,
    "model": _probe_model,
    "executor": _probe_executor,
    "storage": _probe_storage,
    "evidence": _probe_evidence,
}


def run_probes(
        which: Optional[list] = None) -> Dict[str, Dict[str, Any]]:
    """Run the named probes. A probe that raises is reported, never propagated."""
    results: Dict[str, Dict[str, Any]] = {}
    for name, probe in PROBES.items():
        if which and name not in which:
            continue
        try:
            results[name] = probe()
        except Exception as exc:
            results[name] = {
                "status": UNAVAILABLE,
                "reason": f"probe raised {type(exc).__name__}: {exc}",
            }
    return results


def overall(results: Dict[str, Dict[str, Any]]) -> str:
    """Worst-status-wins. Never 'healthy': the vocabulary has no such word."""
    statuses = {r.get("status", UNKNOWN) for r in results.values()}
    for worst in (UNAVAILABLE, UNKNOWN, DEGRADED):
        if worst in statuses:
            return worst
    return OK


def health_snapshot() -> Dict[str, Any]:
    """A full report: every subsystem, its status, and why."""
    results = run_probes()
    overall_status = overall(results)
    return {
        # "healthy" is deliberately absent: nothing here proves health.
        "status": overall_status,
        "capable_of_live_evidence": overall_status == OK,
        "subsystems": results,
        "evidence_dir": os.environ.get("MECH_EVIDENCE_DIR",
                                       "backend/storage/evidence"),
    }
