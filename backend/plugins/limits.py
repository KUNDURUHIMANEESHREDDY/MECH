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


def resolve_limits(requested: Optional[Dict[str, Any]] = None) -> Dict[str, int]:
    """Merge a caller's requested limits over the defaults, clamped to them.

    Clamped, not merely defaulted. A caller that may raise its own containment
    cap may as well remove it, so a request above the default is reduced to the
    default rather than honoured. Unknown keys are dropped, and non-numeric or
    non-positive values fall back, so a malformed request cannot produce an
    unbounded worker.

    This exists because the two platforms were reading the same request
    differently. `start_remote_plugin(..., limits={"cpu_seconds": 2})` reached
    `apply_posix_limits` in the child, so the cap applied on POSIX -- while the
    parent's `assign_windows_job(proc.pid)` call took no arguments and silently
    used `DEFAULT_LIMITS`, so on Windows the same worker was allowed 120 seconds.
    The containment boundary enforced different limits depending on the host, and
    the test that would have shown it was skipped on Windows.
    """
    resolved = dict(DEFAULT_LIMITS)
    if not isinstance(requested, dict):
        return resolved

    for key, value in requested.items():
        if key not in DEFAULT_LIMITS:
            logger.warning("ignoring unknown plugin limit %r", key)
            continue
        try:
            number = int(value)
        except (TypeError, ValueError, OverflowError):
            # OverflowError matters: `int(float("inf"))` raises it, and an
            # unbounded-looking value that raises here would otherwise escape as
            # an exception from inside the containment boundary.
            logger.warning("ignoring non-numeric plugin limit %s=%r", key, value)
            continue
        if number <= 0:
            logger.warning("ignoring non-positive plugin limit %s=%r", key, value)
            continue
        resolved[key] = min(number, DEFAULT_LIMITS[key])
    return resolved


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
    limits: Optional[Dict[str, int]] = None,
) -> Dict[str, Any]:
    """Assign ``pid`` to a bounded Job Object. No-op off Windows.

    Call this immediately after spawning the worker and before sending it the
    load command, so the worker cannot run plugin code while unbounded.

    `limits` is the dict from :func:`resolve_limits`. A Job Object cannot enforce
    every key -- there is no equivalent of `RLIMIT_FSIZE` or `RLIMIT_NOFILE` -- so
    the unenforceable ones are reported under ``result["not_enforceable"]`` rather
    than dropped in silence. That mirrors what `apply_posix_limits` records as
    ``"unavailable: ..."``, and it matters because the platform's containment is
    then visibly partial instead of looking complete.

    **One job per worker.** This used to cache the handle in a module-level
    `_JOB_HANDLE` and reuse it, so every worker the backend ever started landed
    in the *same* job. Three limits are job-wide rather than per-process:
    `PerJobUserTimeLimit`, `JobMemoryLimit` and (in aggregate) `ActiveProcessLimit`.
    With one shared job those budgets were shared across every concurrent plugin,
    so one misbehaving plugin burning its CPU allocation could terminate its
    siblings, and `ActiveProcessLimit` capped the backend's plugin count in total
    rather than any one plugin's children.

    Verified before the fix, on this host:

        assign(worker A) -> {'platform': 'win32', 'assigned': True}
        assign(worker B) -> {'platform': 'win32', 'assigned': True}
        worker A in the module-level job : True
        worker B in the module-level job : True

    The handle is returned as `result["handle"]` so the caller can close it. That
    matters twice over: closing is what triggers `KILL_ON_JOB_CLOSE` for any
    surviving descendants, and the handle is a real resource that would otherwise
    leak for the life of the backend.
    """
    result: Dict[str, Any] = {"platform": sys.platform}
    if sys.platform != "win32":
        return result

    resolved = resolve_limits(limits)
    cpu_seconds = resolved["cpu_seconds"]
    address_space_mb = resolved["address_space_mb"]
    processes = resolved["processes"]

    # RLIMIT_FSIZE and RLIMIT_NOFILE have no Job Object equivalent. Reported, not
    # silently ignored: a caller reading `not_enforceable` can see that Windows
    # containment is partial rather than assume it matches POSIX.
    unenforceable = {
        key: "no Windows Job Object equivalent"
        for key in ("file_size_mb", "open_files")
    }
    if unenforceable:
        result["not_enforceable"] = unenforceable
    try:
        types = _win_job_types()
        if types is None:
            # Previously this returned without setting `assigned`, and the
            # runner's check was `job.get("assigned") is False` -- so a result
            # with no `assigned` key at all satisfied it and the worker went on
            # to load plugin code with no containment on Windows. Said plainly
            # instead: the job object could not be built here.
            result["assigned"] = False
            result["error"] = ("Windows Job Object types are unavailable, so "
                               "no containment could be applied")
            return result
        ctypes, JOBOBJECT_EXTENDED_LIMIT_INFORMATION = types

        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]

        # HANDLE is pointer-sized. Without this, ctypes defaults the return type
        # to c_int and a handle whose value exceeds 32 bits is sign-extended into
        # a different -- and possibly valid -- handle. That was harmless while the
        # handle was never closed; now that `close_windows_job` closes it, the
        # truncation would close an unrelated handle.
        kernel32.CreateJobObjectW.restype = ctypes.c_void_p
        kernel32.CreateJobObjectW.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p]
        kernel32.SetInformationJobObject.restype = ctypes.c_int
        kernel32.OpenProcess.restype = ctypes.c_void_p
        kernel32.AssignProcessToJobObject.restype = ctypes.c_int
        kernel32.CloseHandle.restype = ctypes.c_int
        kernel32.TerminateJobObject.restype = ctypes.c_int

        JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
        JOB_OBJECT_LIMIT_ACTIVE_PROCESS = 0x00000008
        JOB_OBJECT_LIMIT_PROCESS_MEMORY = 0x00000100
        JOB_OBJECT_LIMIT_JOB_MEMORY = 0x00000200
        JOB_OBJECT_LIMIT_PROCESS_TIME = 0x00000002

        # Fresh job per worker. See the docstring.
        handle = kernel32.CreateJobObjectW(None, None)
        if not handle:
            result["error"] = "CreateJobObjectW failed"
            return result

        info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
        info.BasicLimitInformation.LimitFlags = (
            JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
            | JOB_OBJECT_LIMIT_ACTIVE_PROCESS
            | JOB_OBJECT_LIMIT_PROCESS_MEMORY
            | JOB_OBJECT_LIMIT_JOB_MEMORY
            | JOB_OBJECT_LIMIT_PROCESS_TIME
        )
        info.BasicLimitInformation.ActiveProcessLimit = processes
        # 100-nanosecond intervals, per the Win32 documentation. Both limits are
        # set to the same value; now that the job holds exactly one worker tree,
        # per-process and per-job mean the same thing and neither is shared.
        info.BasicLimitInformation.PerProcessUserTimeLimit = cpu_seconds * 10_000_000
        info.BasicLimitInformation.PerJobUserTimeLimit = cpu_seconds * 10_000_000
        info.ProcessMemoryLimit = address_space_mb * 1024 * 1024
        info.JobMemoryLimit = address_space_mb * 1024 * 1024

        ok = kernel32.SetInformationJobObject(
            handle, 9, ctypes.byref(info), ctypes.sizeof(info)
        )
        if not ok:
            kernel32.CloseHandle(handle)
            result["error"] = "SetInformationJobObject failed"
            return result

        process = kernel32.OpenProcess(0x1F0FFF, False, pid)  # PROCESS_SET_QUOTA|TERMINATE|...
        if not process:
            kernel32.CloseHandle(handle)
            result["error"] = "OpenProcess failed"
            return result
        try:
            assigned = kernel32.AssignProcessToJobObject(handle, process)
        finally:
            kernel32.CloseHandle(process)

        if not assigned:
            kernel32.CloseHandle(handle)
            result["assigned"] = False
            result["error"] = "AssignProcessToJobObject failed"
            return result

        result["assigned"] = True
        result["handle"] = handle
        return result
    except Exception as exc:  # noqa: BLE001
        # A missing capability must be loud, not silent.
        result["error"] = str(exc)
        logger.warning("Windows Job Object assignment failed: %s", exc)
        return result


def close_windows_job(handle: Any, terminate: bool = False) -> bool:
    """Release a job handle returned by :func:`assign_windows_job`.

    Closing a job handle flagged `KILL_ON_JOB_CLOSE` terminates every process
    still assigned to it, which is the intended backstop when a worker
    outlives its usefulness. Pass ``terminate=True`` to kill the job explicitly
    before closing.

    No-op off Windows. Returns whether the handle was closed.
    """
    if sys.platform != "win32" or not handle:
        return False
    try:
        import ctypes

        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        kernel32.CloseHandle.restype = ctypes.c_int
        kernel32.TerminateJobObject.restype = ctypes.c_int
        kernel32.TerminateJobObject.argtypes = [ctypes.c_void_p, ctypes.c_uint]
        if terminate:
            kernel32.TerminateJobObject(handle, 1)
        return bool(kernel32.CloseHandle(handle))
    except Exception as exc:  # noqa: BLE001
        logger.warning("could not close Windows Job Object handle: %s", exc)
        return False


def describe_limits() -> Dict[str, Any]:
    """Report which containment mechanisms are available on this host."""
    return {
        "platform": sys.platform,
        "rlimits": sys.platform != "win32",
        "job_object": sys.platform == "win32",
        "defaults": dict(DEFAULT_LIMITS),
    }
