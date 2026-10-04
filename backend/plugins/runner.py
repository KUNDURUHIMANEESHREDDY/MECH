"""Spawns and supervises plugin worker processes.

:func:`start_remote_plugin` launches ``python -m backend.plugins.worker``,
waits for its ``ready`` handshake, applies the Windows Job Object bound (before
the worker is allowed to load any plugin code), then sends the ``load``
command and waits for the plugin manifest. The returned
:class:`~backend.plugins.proxy.RemotePluginProxy` behaves like a
:class:`~backend.plugins.plugin_base.MechPlugin` but holds no plugin code in
this process.

This is the containment boundary. The AST scan and restricted builtins in
:mod:`plugin_sandbox` run inside the worker as defence in depth; the process
split and OS limits are what actually bound a hostile plugin.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

from .limits import assign_windows_job, close_windows_job, resolve_limits
from .protocol import MAX_MESSAGE_BYTES, ProtocolError, decode, encode, ready
from .proxy import RemotePluginError, RemotePluginProxy, _manifest_from_dict

logger = logging.getLogger(__name__)

_WORKER_BOOT_TIMEOUT = 60.0
_LOAD_TIMEOUT = 120.0


class WorkerBootError(Exception):
    """The worker failed to start, load the plugin, or stay within limits."""


def _child_env() -> Dict[str, str]:
    """A scrubbed environment for the worker.

    The backend may hold API keys and database credentials in its environment.
    A plugin worker has no business seeing them, so pass only what a Python
    process needs to boot plus the HF cache location for model weights.
    """
    keep = {
        "PATH", "SYSTEMROOT", "WINDIR", "COMSPEC", "PATHEXT", "TEMP", "TMP",
        "TMPDIR", "HOME", "USERPROFILE", "APPDATA", "LOCALAPPDATA",
        "PYTHONPATH", "PYTHONHASHSEED", "PYTHONIOENCODING", "PYTHONUTF8",
        "SYSTEMDRIVE", "HOMEDRIVE", "HOMEPATH", "NUMBER_OF_PROCESSORS",
        "HF_HOME", "HF_HUB_CACHE", "TRANSFORMERS_CACHE", "LANG", "LC_ALL",
    }
    env = {k: v for k, v in os.environ.items() if k in keep}
    env.setdefault("PYTHONUNBUFFERED", "1")
    env["MECH_PLUGIN_WORKER"] = "1"
    return env


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def _set_nonblocking(stream) -> None:
    """Best-effort switch of a pipe to non-blocking mode.

    ``os.set_blocking`` only exists on Unix. On Windows, anonymous pipes created
    by ``subprocess`` do not honour ``ioctl``/``fcntl`` toggles either, so the
    boot handshake relies on the worker honouring its own timeout there. Failing
    to switch modes is not fatal: the blocking read still returns on a clean
    worker exit, and the boot timeout below is a backstop.
    """
    try:
        os.set_blocking(stream.fileno(), False)
    except (AttributeError, OSError) as exc:
        logger.debug("pipe left blocking (%s): %s", stream, exc)


def _drain_stderr(proc: subprocess.Popen) -> str:
    """Best-effort read of the worker's stderr for diagnostics."""
    if proc.stderr is None:
        return "<no stderr pipe>"
    try:
        _set_nonblocking(proc.stderr)
        data = proc.stderr.read() or b""
    except Exception:  # noqa: BLE001
        return "<stderr unavailable>"
    return (data.decode("utf-8", errors="replace") or "<empty>")[-2000:]


def _abort(proc: subprocess.Popen, reason: str) -> None:
    """Terminate a worker that failed to come up, logging why."""
    logger.warning("aborting plugin worker (pid=%s): %s", proc.pid, reason)
    try:
        proc.kill()
        proc.wait(timeout=5)
    except Exception:  # noqa: BLE001
        pass





def start_remote_plugin(
    plugin_path: str | Path,
    timeout: float = 60.0,
    limits: Optional[Dict[str, int]] = None,
) -> RemotePluginProxy:
    """Start a worker, load ``plugin_path`` in it, and return a proxy.

    Args:
        plugin_path: Absolute path to the plugin's single .py source file.
        timeout: Per-call timeout for the returned proxy, in seconds.
        limits: Optional override of the OS resource limits, e.g.
            ``{"cpu_seconds": 3}``. Tests use a small CPU ceiling rather than
            waiting out the 120s production default.

    Raises:
        WorkerBootError: The worker failed to boot, was refused by its own
            sandbox, or exceeded its limits.
    """
    plugin_path = Path(plugin_path)
    if not plugin_path.is_file():
        raise WorkerBootError(f"plugin file not found: {plugin_path}")

    root = _repo_root()
    env = _child_env()
    # -E/-s (implied by -I) strip PYTHONPATH and ignore user site-packages, both
    # of which we want here. They are enforced by the OS limits, the scrubbed
    # environment, and the AST gate -- not by interpreter flags, which a plugin
    # cannot influence anyway since it never runs in this interpreter.
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{root}{os.pathsep}{existing}" if existing else str(root)
    # -E keeps the interpreter from picking up a user-level sitecustomize or
    # .pth that could inject code into the worker before the limits apply.
    proc = subprocess.Popen(
        [sys.executable, "-E", "-s", "-m", "backend.plugins.worker"],
        cwd=str(root),
        env=env,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        close_fds=(os.name != "nt"),
    )

    if proc.stdout is None or proc.stdin is None:  # pragma: no cover
        proc.kill()
        raise WorkerBootError("plugin worker pipes unavailable")

    # A crash, memory bomb, or hang must not wedge the backend on a blocking
    # read of the worker's stdout.
    deadline = time.monotonic() + _WORKER_BOOT_TIMEOUT
    for stream in (proc.stdout, proc.stdin):
        _set_nonblocking(stream)

    def _await(what: str) -> Dict[str, Any]:
        """Read one frame, giving up at the boot deadline."""
        while True:
            line = proc.stdout.readline()
            if line:
                if len(line) > MAX_MESSAGE_BYTES:
                    raise WorkerBootError(
                        f"plugin worker {what} message exceeded the size cap")
                try:
                    return decode(line)
                except ProtocolError as exc:
                    raise WorkerBootError(
                        f"malformed {what} message: {exc}") from exc
            if proc.poll() is not None:
                raise WorkerBootError(
                    f"plugin worker exited (code {proc.returncode}) during {what}; "
                    f"stderr: {_drain_stderr(proc)}")
            if time.monotonic() > deadline:
                _abort(proc, f"boot timeout during {what}")
                raise WorkerBootError(
                    f"plugin worker timed out after {_WORKER_BOOT_TIMEOUT:.0f}s "
                    f"during {what}; stderr: {_drain_stderr(proc)}")
            time.sleep(0.02)

    try:
        boot = _await("ready")
        if boot.get("event") != "ready":
            raise WorkerBootError(f"unexpected handshake: {boot.get('event')!r}")
        if boot.get("event") == "ready" and isinstance(boot.get("limits"), dict):
            unavailable = [
                k for k, v in boot["limits"].items()
                if isinstance(v, str) and v.startswith("unavailable")
            ]
            if unavailable:
                logger.warning(
                    "plugin worker could not apply limits: %s", ", ".join(unavailable))

        # Bound the worker before it is permitted to load any plugin code.
        #
        # The requested limits are resolved and passed here. This call used to
        # take no arguments, so `limits={"cpu_seconds": 2}` was honoured by the
        # child on POSIX and ignored here on Windows -- the same request produced
        # a 2-second cap on Linux and the 120-second default on this host.
        job = assign_windows_job(proc.pid, resolve_limits(limits))
        if job.get("error") or job.get("assigned") is False:
            proc.kill()
            raise WorkerBootError(
                f"could not apply containment to worker: "
                f"{job.get('error', 'job object assignment refused')}")
        if job.get("not_enforceable"):
            logger.warning(
                "Windows containment cannot enforce %s; the worker is bounded on "
                "CPU, memory and process count only",
                ", ".join(sorted(job["not_enforceable"])),
            )
        # Per-worker job handle. `limits.assign_windows_job` used to cache one
        # module-level handle for every worker, so the job-wide CPU, memory and
        # process budgets were shared across all of them. It is returned now so
        # `stop_remote_plugin` can close it, which is also what makes
        # KILL_ON_JOB_CLOSE fire as the intended backstop.
        job_handle = job.get("handle")

        proc.stdin.write(encode({"op": "load", "path": str(plugin_path),
                                 "limits": limits or {}}))
        proc.stdin.flush()

        loaded_msg = _await("loaded")
        if loaded_msg.get("event") == "failed":
            proc.kill()
            raise WorkerBootError(
                f"plugin rejected in worker ({loaded_msg.get('stage')}): "
                f"{loaded_msg.get('error')}")
        if loaded_msg.get("event") != "loaded":
            proc.kill()
            raise WorkerBootError(
                f"unexpected load response: {loaded_msg.get('event')!r}")

        manifest = _manifest_from_dict(loaded_msg.get("manifest") or {})
        if not manifest.plugin_id:
            proc.kill()
            raise WorkerBootError("plugin returned an empty plugin_id")
    except Exception:
        try:
            proc.kill()
        except Exception:  # noqa: BLE001
            pass
        # Close the job on the boot-failure paths too, so a rejected plugin does
        # not leave a handle -- and a KILL_ON_JOB_CLOSE group -- behind.
        try:
            close_windows_job(locals().get("job_handle"), terminate=True)
        except Exception:  # noqa: BLE001
            pass
        raise

    # Restore blocking mode. The non-blocking flags set above live on the open
    # file description, which the child inherited across fork/exec -- so the
    # worker's stdin became non-blocking too and its readline() returned EOF
    # immediately, killing the worker on the first hook call. This only showed
    # up on Linux: os.set_blocking does not exist on Windows, so the call
    # silently no-opped there and the bug stayed hidden.
    for stream in (proc.stdin, proc.stdout):
        try:
            os.set_blocking(stream.fileno(), True)
        except (AttributeError, OSError):
            pass  # Windows: pipes are always blocking here.

    proxy = RemotePluginProxy(proc.stdin, proc.stdout, manifest, timeout=timeout)
    # Keep the handle so the caller can terminate the worker. The Windows Job
    # Object's KILL_ON_JOB_CLOSE is the backstop if they forget -- and it can only
    # fire if the handle is eventually closed, so `stop_remote_plugin` closes it.
    proxy._proc = proc  # type: ignore[attr-defined]
    proxy._job_handle = job_handle  # type: ignore[attr-defined]
    return proxy


def stop_remote_plugin(proxy: RemotePluginProxy, proc: Optional[Any] = None) -> None:
    """Ask a worker's plugin to unload, then terminate the worker."""
    try:
        proxy._roundtrip({"op": "shutdown"})  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001
        pass
    if proc is not None:
        try:
            proc.terminate()
            proc.wait(timeout=5)
        except Exception:  # noqa: BLE001
            try:
                proc.kill()
            except Exception:  # noqa: BLE001
                pass
    # Close this worker's Job Object, terminating whatever is still in it.
    #
    # This is what makes `KILL_ON_JOB_CLOSE` do its job. With one shared,
    # never-closed handle the flag never fired, so a plugin that outlived its
    # proxy stayed alive until the whole backend exited. With a handle per worker
    # the backstop is scoped to that worker's own process tree.
    try:
        close_windows_job(getattr(proxy, "_job_handle", None), terminate=True)
    except Exception:  # noqa: BLE001
        pass
