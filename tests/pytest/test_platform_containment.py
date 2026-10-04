"""Each plugin worker must be contained on its own.

The rule under test
-------------------
**A containment boundary that is shared between mutually distrusting workers is
not a boundary.** If every plugin worker lands in one Job Object, then the
job-wide CPU, memory and process budgets are shared across all of them, and one
plugin exhausting its allowance terminates its siblings.

What was wrong
--------------
`backend/plugins/limits.py` cached the Job Object handle in a module-level
`_JOB_HANDLE`:

    global _JOB_HANDLE
    if _JOB_HANDLE is None:
        handle = kernel32.CreateJobObjectW(None, None)
        ...
        _JOB_HANDLE = handle

`runner.start_remote_plugin` calls `assign_windows_job(proc.pid)` once per worker,
so every worker the backend ever started was assigned to that one job. Measured
on this host before the fix:

    assign(worker A) -> {'platform': 'win32', 'assigned': True}
    assign(worker B) -> {'platform': 'win32', 'assigned': True}
    worker A in the module-level job : True
    worker B in the module-level job : True

Three of the limits applied are job-wide rather than per-process --
`PerJobUserTimeLimit`, `JobMemoryLimit`, and `ActiveProcessLimit` in aggregate --
so with one shared job:

  * a plugin burning 120 CPU-seconds exhausted the allowance for every other
    concurrently running plugin, and the job's CPU-time termination took them
    with it;
  * `ActiveProcessLimit = 64` capped the backend's *total* plugin count rather
    than any one plugin's children;
  * `JobMemoryLimit` was the combined memory of every worker, so three plugins
    each within budget could jointly exceed it.

`KILL_ON_JOB_CLOSE` was also inert. `runner.py` documented it as the backstop for
a worker that outlives its proxy, but the handle was never closed, so the flag
never fired and a stray worker survived until the whole backend exited.

The handle type was a latent hazard too: `CreateJobObjectW` was called without
setting `restype`, so ctypes defaulted the return to `c_int` and a handle above
32 bits would be truncated. Harmless while the handle was never closed; once
`close_windows_job` closes it, truncation would close an unrelated handle.

Everything here is Windows-specific and skips elsewhere. That is the honest
shape: POSIX containment is `resource.setrlimit` in the child and is covered by
`test_plugin_isolation.py`.
"""

from __future__ import annotations

import subprocess
import sys
import time

import pytest

pytestmark = pytest.mark.skipif(
    sys.platform != "win32",
    reason="Windows Job Objects; POSIX uses resource.setrlimit in the child",
)

pytest.importorskip("ctypes")

#: Give a child something to do so it stays alive while we inspect it.
SLEEPY = "import time; time.sleep(30)"

PROCESS_QUERY_LIMITED_INFORMATION = 0x1000


@pytest.fixture
def sleepy_process():
    """A live child process standing in for a plugin worker."""
    proc = subprocess.Popen([sys.executable, "-c", SLEEPY])
    try:
        yield proc
    finally:
        try:
            proc.kill()
            proc.wait(timeout=5)
        except Exception:  # noqa: BLE001
            pass


def _in_job(pid: int, job) -> bool:
    """Whether `pid` is assigned to `job`."""
    import ctypes

    k32 = ctypes.windll.kernel32
    k32.OpenProcess.restype = ctypes.c_void_p
    k32.IsProcessInJob.restype = ctypes.c_int
    handle = k32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    try:
        out = ctypes.c_int(0)
        if not k32.IsProcessInJob(handle, job, ctypes.byref(out)):
            return False
        return bool(out.value)
    finally:
        k32.CloseHandle(handle)


def test_each_worker_gets_its_own_job(sleepy_process):
    """The core fix. Both workers were in one job before."""
    from backend.plugins.limits import assign_windows_job, close_windows_job

    first = assign_windows_job(sleepy_process.pid)
    assert first.get("assigned") is True, first
    handle = first.get("handle")
    assert handle, f"no handle returned; KILL_ON_JOB_CLOSE cannot fire: {first}"

    second = subprocess.Popen([sys.executable, "-c", SLEEPY])
    other_handle = None
    try:
        other = assign_windows_job(second.pid)
        assert other.get("assigned") is True, other
        other_handle = other.get("handle")

        assert handle != other_handle, (
            "two workers received the same Job Object handle, so their CPU, "
            "memory and process budgets are shared")
        assert _in_job(sleepy_process.pid, handle)
        assert _in_job(second.pid, other_handle)
        assert not _in_job(sleepy_process.pid, other_handle), (
            "the first worker is in the second worker's job")
        assert not _in_job(second.pid, handle), (
            "the second worker is in the first worker's job")
    finally:
        try:
            second.kill()
        except Exception:  # noqa: BLE001
            pass
        close_windows_job(other_handle)


def test_closing_one_job_leaves_the_other_worker_running(sleepy_process):
    """KILL_ON_JOB_CLOSE must be scoped to the worker that owns the handle."""
    from backend.plugins.limits import assign_windows_job, close_windows_job

    keep = assign_windows_job(sleepy_process.pid)
    assert keep.get("assigned") is True, keep

    doomed = subprocess.Popen([sys.executable, "-c", SLEEPY])
    doomed_job = assign_windows_job(doomed.pid)
    try:
        assert doomed_job.get("assigned") is True, doomed_job
        close_windows_job(doomed_job.get("handle"), terminate=True)

        deadline = time.monotonic() + 10
        while doomed.poll() is None and time.monotonic() < deadline:
            time.sleep(0.05)

        assert doomed.poll() is not None, (
            "closing a job flagged KILL_ON_JOB_CLOSE must terminate what is "
            "still assigned to it")
        time.sleep(0.3)
        assert sleepy_process.poll() is None, (
            "closing one worker's job killed an unrelated worker -- the jobs are "
            "not independent")
    finally:
        try:
            doomed.kill()
        except Exception:  # noqa: BLE001
            pass


def test_the_job_cpu_limit_terminates_a_spinning_worker():
    """The property `test_infinite_loop_is_bounded` checks on POSIX.

    It was skipped here because it was written against `RLIMIT_CPU`, and the
    premise of the skip -- that a hung plugin cannot be bounded on Windows -- is
    false. A Job Object's `JOB_OBJECT_LIMIT_PROCESS_TIME` does it.
    """
    from backend.plugins.limits import assign_windows_job, close_windows_job

    proc = subprocess.Popen([sys.executable, "-c", "while True: pass"])
    handle = None
    try:
        assigned = assign_windows_job(proc.pid, {"cpu_seconds": 2})
        assert assigned.get("assigned") is True, assigned
        handle = assigned.get("handle")

        deadline = time.monotonic() + 60
        while proc.poll() is None and time.monotonic() < deadline:
            time.sleep(0.1)

        assert proc.poll() is not None, (
            "a spinning worker was not terminated by the job's CPU limit within "
            "60s; a 2-second allowance should have ended it in single digits")
    finally:
        try:
            proc.kill()
        except Exception:  # noqa: BLE001
            pass
        close_windows_job(handle)


def test_an_unbounded_control_keeps_spinning():
    """Negative control: without the job, nothing stops the spin.

    Without this, the test above could pass for the wrong reason -- a typo in the
    payload that made the process exit immediately would look like a successful
    kill.
    """
    proc = subprocess.Popen([sys.executable, "-c", "while True: pass"])
    try:
        time.sleep(3)
        assert proc.poll() is None, (
            "the control process exited on its own, so the CPU-limit test is "
            "not measuring what it claims")
    finally:
        proc.kill()
        proc.wait(timeout=5)


def test_close_windows_job_is_a_noop_without_a_handle():
    from backend.plugins.limits import close_windows_job

    assert close_windows_job(None) is False
    assert close_windows_job(0) is False


def test_the_module_no_longer_caches_a_job_handle():
    """The global was the bug. It must not come back."""
    import backend.plugins.limits as limits

    assert not hasattr(limits, "_JOB_HANDLE"), (
        "a module-level job handle means every worker shares one job again")


def test_unenforceable_limits_are_reported_not_dropped():
    """Windows containment is partial, and says so.

    `RLIMIT_FSIZE` and `RLIMIT_NOFILE` have no Job Object equivalent. Silently
    ignoring them would make the platform's containment look identical to POSIX
    when it is not.
    """
    from backend.plugins.limits import assign_windows_job

    proc = subprocess.Popen([sys.executable, "-c", SLEEPY])
    from backend.plugins.limits import close_windows_job

    try:
        result = assign_windows_job(proc.pid)
        assert "file_size_mb" in result.get("not_enforceable", {})
        assert "open_files" in result.get("not_enforceable", {})
        close_windows_job(result.get("handle"))
    finally:
        try:
            proc.kill()
        except Exception:  # noqa: BLE001
            pass


def test_describe_limits_reports_the_platform_mechanism():
    from backend.plugins.limits import describe_limits

    described = describe_limits()
    assert described["platform"] == "win32"
    assert described["job_object"] is True
    assert described["rlimits"] is False
