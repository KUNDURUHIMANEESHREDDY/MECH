"""Record benchmark runs and read them back as history / comparisons.

This module is the only place a benchmark result is turned into a persisted
row, and the only place the two read models are shaped for the control plane.

What it refuses to do
---------------------
A run is recorded **only** when the result itself says a live measurement
happened:

* ``status == "completed"`` -- ``"passed"`` is deliberately *not* accepted.
  ``POST /api/benchmarks/run`` promotes a ``passed`` executor response to
  ``completed`` itself, so the only way a ``passed`` status arrives here is as
  an un-promoted raw executor payload, which is a different record from the one
  the route reports. Accepting it would store a status the API never shows.
* ``provenance == "live"`` -- an unavailable, seeded or reference result is a
  refusal, and **a refusal is never persisted as a run**. There is no "recorded
  with score 0" path.
* a **real, finite** primary score. ``bool`` is rejected explicitly (it is an
  ``int`` subclass, so ``True`` would otherwise be stored as the score 1.0),
  as are NaN, +-inf, and anything non-numeric. A missing measurement stays
  missing: it is never coerced to ``0.0``.

Recording is idempotent on ``run_id`` because the store uses
``INSERT OR REPLACE``: re-recording the same id overwrites the earlier attempt
rather than failing or appending a duplicate.

Value provenance of the read models
-----------------------------------
``history_payload`` reports ``provenance: "live"`` because the returned rows
were read from the persistent store in this call -- each row carries its own
``provenance`` field, and that is the one that says how the measurement was
made. ``field_provenance`` describes *this read* the same way: a field is
``"live"`` when the rows actually carry it and ``"unavailable"`` when they do
not, so the map can never claim a field the rows do not have. Fields present in
the rows are never renamed, reordered, filled in or synthesised.
"""
from __future__ import annotations

import math
from typing import Any

#: `StorageError` is the store's own refusal (a bad limit, an unknown run). It
#: is imported at module scope rather than inside a handler so every function
#: in this module can name the same class -- a local import in one function
#: left the other referencing an undefined name.
from backend.storage import StorageError

__all__ = [
    "RunHistoryError",
    "compare_payload",
    "history_payload",
    "record_benchmark_run",
]

#: Statuses that mean a live measurement happened. `"passed"` is absent on
#: purpose -- see the module docstring.
_RECORDABLE_STATUSES = frozenset({"completed"})

#: The provenance label that means the numbers came from a real measurement.
_LIVE = "live"

#: The field a score is read from. The route's own payload calls it `score` and
#: that is what is written as `primary_metric` so `compare` (which requires both
#: runs to name the same metric) can subtract two of these.
_PRIMARY_METRIC = "score"

#: Depth bound for the JSON sanitisation of the stored `detail`. A panel payload
#: is shallow; a self-referential structure would otherwise recurse forever.
_MAX_JSON_DEPTH = 12

#: Fields the method name implies are carried through to `detail` when present.
#: Everything the route returned is kept as well -- these are named so the
#: common fields are greppable, not so the rest are dropped.
_DETAIL_FIELDS = (
    "pass_rate",
    "eval_samples",
    "n_prompts",
    "requested_n_prompts",
    "n_prompts_honoured",
    "validation_eligible",
    "publication_eligible",
    "ineligible_because",
    "adequacy_reported_by_executor",
    "attested",
)

#: Candidate columns for `field_provenance`, mapped to their storage column name
#: in the returned row.
_PROVENANCE_FIELDS = (
    "run_id",
    "benchmark",
    "primary_metric",
    "primary_score",
    "status",
    "provenance",
    "model_id",
    "weights_fingerprint",
    "git_commit",
    "created_at",
    "traces",
    "detail",
)


class RunHistoryError(Exception):
    """A history/compare request that cannot be answered from the store.

    ``run_id`` names the requested run when the failure is about a specific
    one, and ``status_code`` is the HTTP status the control plane should
    answer with -- 404 for an unknown run, 400 for a malformed argument. The
    route maps the error; it does not have to guess from the message.
    """

    def __init__(
        self,
        message: str,
        run_id: str | None = None,
        status_code: int = 400,
    ) -> None:
        super().__init__(message)
        self.run_id = run_id
        self.status_code = status_code


def _store() -> Any:
    """The storage authority. Imported lazily so this module stays cheap."""
    from backend.storage import DesktopStorage, get_default_db_path

    store = DesktopStorage(get_default_db_path())
    store.initialize()
    return store


def _is_real_number(value: Any) -> bool:
    """A finite int/float that is not a bool.

    ``bool`` is excluded because it is an ``int`` subclass, so ``True`` would
    otherwise be accepted as the score 1.0 -- a fabricated measurement.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError, OverflowError):
        return False


def _real_number(record: Any, key: str) -> float | None:
    """``record[key]`` as a float when it is a real finite number, else None."""
    if not isinstance(record, dict):
        return None
    value = record.get(key)
    if not _is_real_number(value):
        return None
    return float(value)


def _text(value: Any) -> str | None:
    """A non-empty string, or None. Never stringifies a non-string."""
    if isinstance(value, str) and value.strip():
        return value
    return None


def _require_run_id(run_id: Any) -> str:
    if not isinstance(run_id, str) or not run_id.strip():
        raise RunHistoryError("run_a and run_b must be non-empty run ids")
    return run_id


def _jsonable(value: Any, _depth: int = 0) -> Any:
    """A JSON-serialisable copy of ``value``.

    The payload arrives from a runner, so an unexpected object is a real
    possibility, and ``json.dumps`` raising inside the recording path would
    turn a stored run into a 500 on a benchmark that actually succeeded.
    Unencodable values become their ``repr`` rather than being dropped, so the
    detail stays honest about what it could not represent.
    """
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, str):
        # NUL cannot be stored in a SQLite TEXT column; an embedded one would
        # truncate the detail rather than fail, so it is replaced here.
        return value.replace("\x00", "\ufffd")
    if isinstance(value, float):
        # `json.dumps` would write a bare NaN/Infinity, which is not valid JSON
        # and is not a measurement; both are recorded as absent instead.
        return value if math.isfinite(value) else None
    if _depth >= _MAX_JSON_DEPTH:
        return repr(value)
    if isinstance(value, dict):
        return {
            str(key): _jsonable(item, _depth + 1)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_jsonable(item, _depth + 1) for item in value]
    try:
        if hasattr(value, "item") and hasattr(value, "dtype"):
            # numpy scalar: recover the Python number rather than a repr.
            return _jsonable(value.item(), _depth + 1)
    except Exception:  # noqa: BLE001 - fall through to repr
        pass
    return repr(value)


def _detail_for(result: dict[str, Any]) -> dict[str, Any] | None:
    """The stored ``detail`` blob: the full result plus the named panel fields.

    The whole payload is kept because the panel/adequacy fields are exactly the
    ones the route spread into the response and the UI shows, and a projection
    list would silently drop whichever one the route adds next. Naming
    ``_DETAIL_FIELDS`` makes the common ones stable.
    """
    detail = {
        key: value
        for key, value in result.items()
        if _jsonable(value) is not None
    }
    for key in _DETAIL_FIELDS:
        if key in result:
            detail[key] = result[key]
    detail = _jsonable(detail)
    return detail or None


def record_benchmark_run(
    *, benchmark_name: str, result: dict
) -> dict[str, Any] | None:
    """Persist one completed, live benchmark run; ``None`` when refused.

    See the module docstring for the exact refusal conditions. A refusal
    writes nothing at all -- it is not a row with a zeroed score.
    """
    if not isinstance(benchmark_name, str) or not benchmark_name.strip():
        raise RunHistoryError("benchmark_name must be a non-empty string")
    if not isinstance(result, dict):
        return None
    if str(result.get("status", "")).strip().lower() not in _RECORDABLE_STATUSES:
        return None
    if str(result.get("provenance", "")).strip().lower() != _LIVE:
        return None

    score = _real_number(result, _PRIMARY_METRIC)
    if score is None:
        # No real measurement: nothing is stored, and no 0.0 is invented.
        return None

    store = _store()
    return store.record_run(
        run_id=result.get("run_id") or None,
        benchmark=benchmark_name,
        primary_metric=result.get("primary_metric") or _PRIMARY_METRIC,
        primary_score=score,
        status="completed",
        provenance=_LIVE,
        # Copied only when the result carries them, never defaulted.
        model_id=_text(result.get("model_id")),
        weights_fingerprint=_text(result.get("weights_fingerprint")),
        git_commit=_text(result.get("git_commit")),
        traces=result.get("traces") if isinstance(result.get("traces"), list) else None,
        detail=_detail_for(result),
    )


def _field_provenance(runs: list[dict[str, Any]]) -> dict[str, str]:
    """Per-field label for this read: ``live`` when the rows carry it.

    A field is ``live`` when at least one returned row holds a value for it,
    and ``unavailable`` when the rows are missing it. This describes what was
    read, and never asserts a field the rows do not have.
    """
    return {
        field: ("live" if any(row.get(field) is not None for row in runs)
                else "unavailable")
        for field in _PROVENANCE_FIELDS
    }


def history_payload(benchmark: str | None = None, limit: int = 20) -> dict[str, Any]:
    """The persisted run history, exactly as storage returned it.

    Rows are passed through untouched: nothing is sorted again, filled in or
    synthesised here. ``count`` is ``len(runs)`` -- the number of rows actually
    returned, not the number in the table.
    """
    if benchmark is not None and not isinstance(benchmark, str):
        raise RunHistoryError("benchmark must be a string or None")
    try:
        runs = _store().history(benchmark=benchmark, limit=limit)
    except StorageError as exc:
        # The store refuses a non-integer limit; that is a client error (400),
        # not a missing resource, and it is not swallowed into an empty list.
        raise RunHistoryError(str(exc), status_code=400) from exc
    return {
        "runs": runs,
        "count": len(runs),
        "benchmark": benchmark,
        "provenance": _LIVE,
        "field_provenance": _field_provenance(runs),
    }


def compare_payload(run_a: str, run_b: str) -> dict[str, Any]:
    """Two stored runs compared, with the reasons a delta is absent.

    Delegates to :meth:`DesktopStorage.compare`, which returns ``score_delta``
    as ``None`` -- never ``0.0`` -- whenever the two runs do not measure the
    same metric on the same benchmark, or when either score is NULL. Those
    reasons travel through ``not_comparable`` unchanged; this wrapper adds
    ``comparable`` as a summary flag and never computes a delta of its own.

    Raises :class:`RunHistoryError` when a run id is not stored, so the route
    can answer 404 rather than 500.
    """
    first = _require_run_id(run_a)
    second = _require_run_id(run_b)
    try:
        return _compare(first, second)
    except RunHistoryError:
        raise
    except Exception as exc:  # noqa: BLE001 - storage errors become 404s
        from backend.storage import StorageError

        if isinstance(exc, StorageError):
            missing = _missing_run_id(str(exc), first, second)
            raise RunHistoryError(
                str(exc), run_id=missing, status_code=404 if missing else 400
            ) from exc
        raise RunHistoryError(
            f"benchmark comparison failed: {type(exc).__name__}: {exc}"
        ) from exc


def _compare(run_a: str, run_b: str) -> dict[str, Any]:
    from backend.storage import compare

    comparison = compare(run_a, run_b)
    if not isinstance(comparison, dict):
        raise RunHistoryError("the store returned an unreadable comparison")
    reasons = comparison.get("not_comparable")
    if not isinstance(reasons, list):
        reasons = []
    delta = comparison.get("score_delta")
    # `score_delta` is None exactly when the comparison is not comparable;
    # a delta that is present but not a real number is a store defect and is
    # reported as not comparable rather than shown as a change.
    if delta is not None and not _is_real_number(delta):
        reasons = [*reasons, "score_delta was not a real number"]
        delta = None
    result = dict(comparison)
    result["score_delta"] = delta
    result["not_comparable"] = reasons
    result["comparable"] = bool(
        (comparison.get("same_benchmark") or delta is not None)
        and delta is not None
    )
    return result


def _missing_run_id(message: str, run_a: str, run_b: str) -> str | None:
    """Which requested id the storage error is about, when it names one."""
    for run_id in (run_a, run_b):
        if run_id and run_id in message:
            return run_id
    return None
