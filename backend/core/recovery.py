"""Failure recovery for the control plane: timeouts, bounded retries, cleanup.

The rules this module exists to keep
------------------------------------

* **A call that does not finish in time is a failure.** It is reported as
  one, with the budget it exceeded. It is never a slow success, and it is
  never swallowed into an empty result.
* **A retry happens only where retrying is safe.** The caller asks for one
  by calling :func:`retry_call`; there is no global "retry everything"
  setting, because retrying a mutation can duplicate it.
* **A failed run leaves nothing half-written behind.** Partial files are
  removed by :func:`cleanup_partial_files`, which never raises: cleanup runs
  in the failure path, and a cleanup error would replace the real error with
  a less useful one.

Everything here is standard library.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Type


class EngineTimeout(TimeoutError):
    """A call that did not finish inside its budget."""

    def __init__(self, label: str, timeout_s: float) -> None:
        super().__init__("%s did not finish within %.3fs" % (label, timeout_s))
        self.label = label
        self.timeout_s = timeout_s


class EngineFailure(RuntimeError):
    """The engine reported a failure instead of raising.

    Some engines answer with a status rather than an exception -- a load
    that returns ``{"status": "error"}`` is a failure, not a result. This
    wraps that shape so it can be retried and reported like any other
    failure.
    """

    def __init__(self, reason: Any) -> None:
        super().__init__(str(reason))
        self.reason = reason


def call_with_timeout(func: Callable[..., Any], timeout_s: Optional[float],
                      label: str = "call", *args: Any, **kwargs: Any) -> Any:
    """Run a blocking callable with a time budget.

    The callable runs on its own thread, so a hung call cannot wedge the
    caller. Python cannot kill a thread, so a thread that overruns is left
    to finish on its own; what matters is that the caller is told the truth:
    :class:`EngineTimeout` is raised, and a timeout is a failure.

    ``timeout_s`` of ``None`` or a non-positive value means "no budget",
    which is the previous behaviour.
    """
    if timeout_s is None or float(timeout_s) <= 0:
        return func(*args, **kwargs)

    box: Dict[str, Any] = {}

    def _runner() -> None:
        try:
            box["result"] = func(*args, **kwargs)
        except BaseException as exc:  # noqa: BLE001 - re-raised in the caller
            box["error"] = exc

    thread = threading.Thread(target=_runner, name="mech-timeout-%s" % label,
                              daemon=True)
    thread.start()
    thread.join(float(timeout_s))
    if thread.is_alive():
        raise EngineTimeout(label, float(timeout_s))
    if "error" in box:
        raise box["error"]
    return box.get("result")


def retry_call(func: Callable[[], Any], attempts: int = 2,
               backoff_s: float = 0.5,
               retry_on: Sequence[Type[BaseException]] = (Exception,),
               label: str = "call",
               sleep: Callable[[float], None] = time.sleep) -> Tuple[Any, int]:
    """Call ``func``, retrying a bounded number of times with backoff.

    Only use this where retrying is safe: a read, an idempotent load, a
    probe. Retrying a mutation can duplicate it.

    A timeout is never retried: the thread that overran may still be
    running, and a second attempt would race it.

    Returns the result and the number of attempts it took. The last
    exception is raised when every attempt fails.
    """
    total = max(1, int(attempts))
    allowed = tuple(retry_on)
    for attempt in range(1, total + 1):
        try:
            return func(), attempt
        except EngineTimeout:
            raise
        except allowed as exc:
            if attempt >= total:
                raise
            sleep(max(0.0, float(backoff_s)) * (2 ** (attempt - 1)))
    raise AssertionError("unreachable")  # pragma: no cover


def cleanup_partial_files(paths: Iterable[Any]) -> List[str]:
    """Delete the partial files a failed run left behind.

    Returns the paths that were actually removed. Never raises: this runs in
    the failure path, and a cleanup error would replace the real error with
    a less useful one.
    """
    removed: List[str] = []
    for raw in paths or ():
        try:
            path = Path(raw)
        except TypeError:
            continue
        try:
            if path.is_file():
                path.unlink()
                removed.append(str(path))
        except OSError:
            continue
    return removed


def failure_payload(reason: Any, kind: str = "error",
                    **extra: Any) -> Dict[str, Any]:
    """A failure payload that cannot be mistaken for a success.

    ``status`` is never ``ok`` or ``completed``, ``measured`` is ``False``
    and ``provenance`` is ``unavailable``: nothing was measured, so nothing
    may claim otherwise.
    """
    payload: Dict[str, Any] = {
        "status": "failed" if kind == "error" else str(kind),
        "provenance": "unavailable",
        "measured": False,
        "reason": str(reason),
    }
    payload.update(extra)
    return payload
