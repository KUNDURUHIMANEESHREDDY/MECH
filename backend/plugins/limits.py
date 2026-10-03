"""OS-level resource limits for out-of-process plugin workers.

This module is the actual containment boundary for untrusted plugin code. The
AST scan and restricted builtins in :mod:`plugin_sandbox` are defence in depth;
a worker running in a separate, capability-restricted process is what keeps a
hostile plugin from owning the backend.

Two mechanisms, chosen per platform:

* **POSIX** — ``resource.setrlimit`` for CPU seconds, address space, file size,
  open descriptors, and process count, plus ``setsid`` so the worker cannot
  signal the backend's process group. Applied by the child itself before it
  loads any plugin code, so there is no window between exec and enforcement.

* **Windows** — a Job Object bounding active processes, per-process and total
  job memory, and CPU time, with ``KILL_ON_JOB_CLOSE`` so the worker cannot
  outlive the backend. Assigned by the parent before the worker is told to
  load its plugin (see :mod:`.runner`), which closes the create/assign race
  without needing ``CREATE_SUSPENDED``.

Everything here degrades to a no-op with a logged warning when the platform
refuses a limit, so a missing capability is visible rather than silent.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Conservative defaults for a research plugin: enough to load a transformer
# and run a forward pass, not enough to fork-bomb or fill the disk.
DEFAULT_LIMITS: Dict[str, int] = {
    "cpu_seconds": 120,
    "address_space_mb": 4096,
    "file_size_mb": 64,
    "open_files": 256,
    "processes": 64,
}


def apply_posix_limits(
    cpu_seconds: int = DEFAULT_LIMITS["cpu_seconds"],
    address_space_mb: int = DEFAULT_LIMITS["address_space_mb"],
    file_size_mb: int = DEFAULT_LIMITS["file_size_mb"],
    open_files: int = DEFAULT_LIMITS["open_files"],
    processes: int = DEFAULT_LIMITS["processes"],
) -> Dict[str, Any]:
    """Apply rlimits in the current (child) process. Returns what was applied.

    Safe to call on any platform; non-POSIX platforms are a no-op.
    """
    applied: Dict[str, Any] = {"platform": sys.platform}
    if sys.platform == "win32":
        return applied

    try:
        import resource
    except ImportError:  # pragma: no cover - non-POSIX without resource
        logger.warning("resource module unavailable; no rlimits applied")
        return applied

    def _set(name: str, res: int, soft: int, hard: Optional[int] = None) -> None:
        try:
            cur_soft, cur_hard = resource.getrlimit(res)
            # Never try to raise an inherited hard limit.
            if cur_hard != resource.RLIM_INFINITY:
                if hard is None:
                    hard = cur_hard
                soft = min(soft, cur_hard)
                if hard != resource.RLIM_INFINITY:
                    hard = min(hard, cur_hard)
            resource.setrlimit(res, (soft, hard if hard is not None else cur_hard))
            applied[name] = soft
        except (ValueError, OSError) as exc:
            # A limit the current hard limit forbids is not fatal; record it.
            applied[name] = f"unavailable: {exc}"
            logger.warning("Could not set rlimit %s: %s", name, exc)

    _set("cpu_seconds", resource.RLIMIT_CPU, cpu_seconds)
    _set("address_space_mb", resource.RLIMIT_AS, address_space_mb * 1024 * 1024)
    _set("file_size_mb", resource.RLIMIT_FSIZE, file_size_mb * 1024 * 1024)
    _set("open_files", resource.RLIMIT_NOFILE, open_files)
    _set("processes", resource.RLIMIT_NPROC, processes)

    try:
        os.setsid()
        applied["setsid"] = True
    except OSError as exc:
        applied["setsid"] = f"unavailable: {exc}"

    return applied


# --------------------------------------------------------------------------- #
# Windows Job Object
# --------------------------------------------------------------------------- #

_JOB_HANDLE = None


def _win_job_types():  # type: ignore[no-untyped-def]
    """Define the Job Object structures. Returns None off Windows."""
    import ctypes
    from ctypes import wintypes

    class IO_COUNTERS(ctypes.Structure):
        _fields_ = [
            ("ReadOperationCount", ctypes.c_ulonglong),
            ("WriteOperationCount", ctypes.c_ulonglong),
            ("OtherOperationCount", ctypes.c_ulonglong),
            ("ReadTransferCount", ctypes.c_ulonglong),
            ("WriteTransferCount", ctypes.c_ulonglong),
            ("OtherTransferCount", ctypes.c_ulonglong),
        ]

    class JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
        _fields_ = [
            ("PerProcessUserTimeLimit", ctypes.c_int64),
            ("PerJobUserTimeLimit", ctypes.c_int64),
            ("LimitFlags", wintypes.DWORD),
            ("MinimumWorkingSetSize", ctypes.c_size_t),
            ("MaximumWorkingSetSize", ctypes.c_size_t),
            ("ActiveProcessLimit", wintypes.DWORD),
            ("Affinity", ctypes.c_size_t),
            ("PriorityClass", wintypes.DWORD),
            ("SchedulingClass", wintypes.DWORD),
        ]

    class JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
        _fields_ = [
            ("BasicLimitInformation", JOBOBJECT_BASIC_LIMIT_INFORMATION),
            ("IoInfo", IO_COUNTERS),
            ("ProcessMemoryLimit", ctypes.c_size_t),
            ("JobMemoryLimit", ctypes.c_size_t),
            ("PeakProcessMemoryUsed", ctypes.c_size_t),
            ("PeakJobMemoryUsed", ctypes.c_size_t),
        ]

    return ctypes, JOBOBJECT_EXTENDED_LIMIT_INFORMATION


def assign_windows_job(
    pid: int,
    cpu_seconds: int = DEFAULT_LIMITS["cpu_seconds"],
    address_space_mb: int = DEFAULT_LIMITS["address_space_mb"],
    processes: int = DEFAULT_LIMITS["processes"],
) -> Dict[str, Any]:
    """Assign ``pid`` to a bounded Job Object. No-op off Windows.

    Call this immediately after spawning the worker and before sending it the
    load command, so the worker cannot run plugin code while unbounded.
    """
    result: Dict[str, Any] = {"platform": sys.platform}
    if sys.platform != "win32":
        return result

    global _JOB_HANDLE
    try:
        types = _win_job_types()
        if types is None:
            return result
        ctypes, JOBOBJECT_EXTENDED_LIMIT_INFORMATION = types

        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]

        JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
        JOB_OBJECT_LIMIT_ACTIVE_PROCESS = 0x00000008
        JOB_OBJECT_LIMIT_PROCESS_MEMORY = 0x00000100
        JOB_OBJECT_LIMIT_JOB_MEMORY = 0x00000200
        JOB_OBJECT_LIMIT_PROCESS_TIME = 0x00000002

        if _JOB_HANDLE is None:
            handle = kernel32.CreateJobObjectW(None, None)
            if not handle:
                result["error"] = "CreateJobObjectW failed"
                return result
            _JOB_HANDLE = handle

        info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
        info.BasicLimitInformation.LimitFlags = (
            JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
            | JOB_OBJECT_LIMIT_ACTIVE_PROCESS
            | JOB_OBJECT_LIMIT_PROCESS_MEMORY
            | JOB_OBJECT_LIMIT_JOB_MEMORY
            | JOB_OBJECT_LIMIT_PROCESS_TIME
        )
        info.BasicLimitInformation.ActiveProcessLimit = processes
        info.BasicLimitInformation.PerProcessUserTimeLimit = cpu_seconds * 10_000_000
        info.BasicLimitInformation.PerJobUserTimeLimit = cpu_seconds * 10_000_000
        info.ProcessMemoryLimit = address_space_mb * 1024 * 1024
        info.JobMemoryLimit = address_space_mb * 1024 * 1024

        ok = kernel32.SetInformationJobObject(
            _JOB_HANDLE, 9, ctypes.byref(info), ctypes.sizeof(info)
        )
        if not ok:
            result["error"] = "SetInformationJobObject failed"
            return result

        process = kernel32.OpenProcess(0x1F0FFF, False, pid)  # PROCESS_SET_QUOTA|TERMINATE|...
        if not process:
            result["error"] = "OpenProcess failed"
            return result
        try:
            assigned = kernel32.AssignProcessToJobObject(_JOB_HANDLE, process)
        finally:
            kernel32.CloseHandle(process)

        result["assigned"] = bool(assigned)
        return result
    except Exception as exc:  # noqa: BLE001
        # A missing capability must be loud, not silent.
        result["error"] = str(exc)
        logger.warning("Windows Job Object assignment failed: %s", exc)
        return result


def describe_limits() -> Dict[str, Any]:
    """Report which containment mechanisms are available on this host."""
    return {
        "platform": sys.platform,
        "rlimits": sys.platform != "win32",
        "job_object": sys.platform == "win32",
        "defaults": dict(DEFAULT_LIMITS),
    }
