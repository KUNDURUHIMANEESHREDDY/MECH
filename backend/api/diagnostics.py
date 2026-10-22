"""The diagnostics surface: one place a support engineer can read.

``GET /health`` stays liveness only -- it answers "is this process serving?"
and nothing else, because a liveness probe that loads torch turns a slow
dependency into an orchestrator restart.

This module answers the other question: *what is actually happening here?*
It reports

* the per-subsystem status **probed** by ``backend/core/health.py`` -- the
  status vocabulary has no "healthy" member, because nothing here proves
  health;
* engine and model availability, with the real reason when unavailable
  (never a synthesised success);
* storage and disk state for every directory the control plane writes to;
* the versions of the dependencies that were actually imported;
* the last error the process recorded;
* the configuration in use, and every value that fell back to its default.

Nothing here asserts health. Every status is the output of a probe that ran.
"""

from __future__ import annotations

import importlib
import os
import platform
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import APIRouter

from backend.core import health as health_mod
from backend.core.observability import (
    APP_VERSION,
    Settings,
    get_settings,
    last_error,
    recent_errors,
    utcnow_iso,
)

router = APIRouter()


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def _engine_report() -> Dict[str, Any]:
    """Model/engine availability, with the real reason when unavailable.

    The reason comes from the engine itself (``gpt2_engine.info()`` reports
    why there is no model) or from the import error that made the ML stack
    unavailable. It is never replaced with a guess.
    """
    try:
        from backend.services import gpt2_engine
    except Exception as exc:
        return {
            "status": health_mod.UNAVAILABLE,
            "stack_available": False,
            "weights_loaded": False,
            "reason": "gpt2_engine import failed: %s: %s" % (type(exc).__name__, exc),
        }

    try:
        stack_available = bool(gpt2_engine.is_available())
    except Exception as exc:
        return {
            "status": health_mod.UNAVAILABLE,
            "stack_available": False,
            "weights_loaded": False,
            "reason": "is_available() raised: %s: %s" % (type(exc).__name__, exc),
        }

    if not stack_available:
        # The engine swallowed the import error, so it is re-derived here:
        # the reason the stack is unavailable is the exception that made it
        # unavailable, not a guess.
        reason = None
        try:
            importlib.import_module("numpy")
            importlib.import_module("torch")
            importlib.import_module("transformers")
        except Exception as exc:
            reason = "%s: %s" % (type(exc).__name__, exc)
        return {
            "status": health_mod.UNAVAILABLE,
            "stack_available": False,
            "weights_loaded": False,
            "reason": reason or "the engine reports the ML stack is unavailable",
        }

    try:
        info = gpt2_engine.info()
    except Exception as exc:
        info = {"status": "error", "error": "%s: %s" % (type(exc).__name__, exc)}

    if not isinstance(info, dict):
        info = {"status": "error", "error": "the engine returned a non-dict status"}

    loaded = info.get("status") == "loaded"
    return {
        "status": health_mod.OK if loaded else health_mod.DEGRADED,
        "stack_available": True,
        "weights_loaded": loaded,
        "model_name": info.get("model_name"),
        "device": info.get("device"),
        "n_params": info.get("n_params"),
        "reason": None if loaded else (info.get("error") or
                                       "the ML stack is present but no weights are loaded"),
    }


def _directory_state(path_value: str) -> Dict[str, Any]:
    """One directory: does it exist, can it be written to."""
    path = Path(path_value)
    state: Dict[str, Any] = {"path": str(path), "exists": path.exists()}
    try:
        state["writable"] = bool(path.exists() and os.access(str(path), os.W_OK))
    except OSError as exc:  # pragma: no cover - defensive
        state["writable"] = False
        state["error"] = str(exc)
    return state


def _storage_report(settings: Settings) -> Dict[str, Any]:
    """Storage and disk state for every directory the control plane writes to."""
    root = Path(settings.storage_root)
    report: Dict[str, Any] = {
        "root": str(root),
        "exists": root.exists(),
        "writable": False,
        "directories": {
            "storage_root": _directory_state(settings.storage_root),
            "export_root": _directory_state(settings.export_root),
            "artifact_dir": _directory_state(settings.artifact_dir),
            "model_cache_dir": _directory_state(settings.model_cache_dir),
        },
    }
    try:
        report["writable"] = bool(root.exists() and os.access(str(root), os.W_OK))
    except OSError as exc:  # pragma: no cover - defensive
        report["error"] = str(exc)

    probe_root = root if root.exists() else root.parent
    try:
        usage = shutil.disk_usage(str(probe_root))
        report["disk"] = {
            "path": str(probe_root),
            "total_bytes": usage.total,
            "used_bytes": usage.used,
            "free_bytes": usage.free,
            "percent_used": round(usage.used / usage.total * 100.0, 2) if usage.total else None,
        }
    except OSError as exc:
        report["disk"] = {"error": str(exc)}
    return report


def _dependency_report(probes: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """The versions of the dependencies that were actually imported."""
    dependency_probe = probes.get("dependency") or {}
    versions = dependency_probe.get("versions")
    report: Dict[str, Any] = {
        "python": sys.version.split()[0],
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "packages": dict(versions) if isinstance(versions, dict) else {},
        "missing": list(dependency_probe.get("missing") or []),
        "incompatible": list(dependency_probe.get("incompatible") or []),
    }
    return report


def _log_report(settings: Settings) -> Dict[str, Any]:
    """Where the log file is, and whether it exists."""
    path = Path(settings.log_path)
    return {
        "path": str(path),
        "exists": path.is_file(),
        "level": settings.log_level,
        "max_bytes": settings.log_file_max_bytes,
        "backup_count": settings.log_file_backup_count,
    }


# --------------------------------------------------------------------------- #
# routes
# --------------------------------------------------------------------------- #


@router.get("/diagnostics")
def diagnostics() -> Dict[str, Any]:
    """Everything a support engineer needs, in one response.

    Every subsystem status here is the output of a probe that ran during
    this request; nothing is carried over from a previous run and nothing
    is asserted.
    """
    settings = get_settings()
    probes = health_mod.run_probes()
    status = health_mod.overall(probes)

    return {
        "app_version": APP_VERSION,
        "generated_at": utcnow_iso(),
        "status": status,
        "capable_of_live_evidence": status == health_mod.OK,
        "subsystems": probes,
        "engines": {
            "model": _engine_report(),
            "live_executor": probes.get("executor") or {
                "status": health_mod.UNKNOWN,
                "reason": "the executor probe did not run",
            },
        },
        "storage": _storage_report(settings),
        "dependencies": _dependency_report(probes),
        "last_error": last_error(),
        "recent_errors": recent_errors(5),
        "configuration": settings.as_dict(),
        "configuration_problems": list(settings.problems),
        "log": _log_report(settings),
    }


@router.get("/config")
def read_config() -> Dict[str, Any]:
    """The configuration in use, and every value that fell back to a default.

    Read-only on purpose: a configuration that can be changed at runtime
    would have to be re-read by every consumer, and a value that changes
    under a running request is a value no log line can explain. Change the
    environment and restart.
    """
    settings = get_settings()
    return {
        "app_version": APP_VERSION,
        "configuration": settings.as_dict(),
        "problems": list(settings.problems),
    }
