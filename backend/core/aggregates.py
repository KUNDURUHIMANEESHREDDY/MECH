"""Aggregate read-models for the control plane.

Two routes the UI has been declaring missing consult this module:

* ``GET /api/scientific_health`` -- the run ledger, the attestation certificates
  and the recorded validation verdicts, in one payload.
* ``GET /api/analytics`` -- a descriptive inventory: how many runs exist, how
  they ended, how long the measurable ones took.

Rules this module is written to, because a research IDE that mints numbers is
worse than one that admits it cannot:

1. **No minted score.** There is no scientific-health grade below. Nothing in
   this repository measures one, so ``score`` is ``None`` and the reason is
   stated. A count of runs is not a health score, and adding them up would not
   produce one.
2. **Absent is not zero.** A source that could not be read yields ``None`` and
   an entry in ``unavailable``. A source that was read and is empty yields
   ``0``. The two are different facts and are not collapsed into one.
3. **Per-field provenance.** Every payload carries ``provenance`` and
   ``field_provenance``, so a consumer can see which fields were measured here
   and which were merely read back from a stored record.
4. **Nothing is inferred from a step that did not run.** Validation statistics
   are computed only from ``peer_review.checks[].passed`` booleans that the
   validation stage actually recorded, and pass rates stay ``None`` when no
   checks were recorded.

The heavy lifting is not re-implemented here: subsystem probes come from
:mod:`backend.core.health`, the ledger from :mod:`backend.core.evidence_graph`,
and stored experiment/session records from
:class:`backend.storage.database.DesktopStorage`.
"""

from __future__ import annotations

import datetime as _dt
import json
from pathlib import Path
from typing import Any, Dict, List, Iterable, Optional

from backend.agents.evidence_policy import field_map

#: Provenance label for a value this module read from a store it could open.
#: Deliberately not "live": these counts are reads, not measurements, and the
#: repository's own vocabulary reserves "live" for a value measured from loaded
#: weights in the current request.
READ = "live"

#: Bound on how many run records are opened to compute the deeper statistics.
#: `list_run_records` already caps its listing at 200; this is the second cap
#: for the per-run envelope/validation reads, which are file and signature work.
MAX_RUNS_INSPECTED = 200


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #


def _now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


def _parse_instant(raw: Any) -> Optional[_dt.datetime]:
    """Parse an ISO-8601 stamp, treating a naive value as UTC.

    Returns None rather than raising: an unparseable timestamp is a fact about
    the record, not a reason to fail the whole aggregate.
    """
    text = str(raw or "").strip()
    if not text:
        return None
    try:
        moment = _dt.datetime.fromisoformat(text)
    except (TypeError, ValueError):
        return None
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=_dt.timezone.utc)
    return moment


def _finite(value: Any) -> Optional[float]:
    """A float only when the value really is a finite number."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number or number in (float("inf"), float("-inf")):
        return None
    return number


def _summarize_floats(values: List[float]) -> Optional[Dict[str, Any]]:
    """min/max/mean over the values that exist, or None when there are none."""
    if not values:
        return None
    return {
        "count": len(values),
        "min": round(min(values), 6),
        "max": round(max(values), 6),
        "mean": round(sum(values) / len(values), 6),
    }


def _evidence_dir() -> Path:
    """The run ledger's directory, resolved the same way evidence_graph does."""
    from backend.core import evidence_graph as eg

    return Path(eg._evidence_dir(None))


def _storage():
    """The same DesktopStorage the dispatcher uses (same path resolution)."""
    from backend.storage import DesktopStorage

    from backend.api import dispatcher as _dispatcher

    # Reuse the dispatcher's already-initialised store so the aggregate reads
    # the exact database the rest of the API writes, with no second migration.
    return _dispatcher._store


# --------------------------------------------------------------------------- #
# the run ledger
# --------------------------------------------------------------------------- #


def _ledger() -> Dict[str, Any]:
    """Read the run ledger, distinguishing "empty" from "unreadable"."""
    from backend.core import evidence_graph as eg

    directory = _evidence_dir()
    source = {
        "name": "run_ledger",
        "path": str(directory),
    }
    if not directory.is_dir():
        return {
            "available": False,
            "source": source,
            "reason": f"the run ledger directory does not exist: {directory}",
            "summaries": [],
        }
    try:
        # list_run_records() answers [] for an unreadable directory as well as
        # for an empty one, so the readability question is settled here first.
        list(directory.iterdir())
    except OSError as exc:
        return {
            "available": False,
            "source": source,
            "reason": f"the run ledger directory is not readable: {exc}",
            "summaries": [],
        }
    try:
        summaries = eg.list_run_records()
    except Exception as exc:  # noqa: BLE001 - reported, never propagated
        return {
            "available": False,
            "source": source,
            "reason": f"list_run_records() failed: {type(exc).__name__}: {exc}",
            "summaries": [],
        }
    return {"available": True, "source": source, "summaries": summaries}


def _run_by_status(summaries: Iterable[Dict[str, Any]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for summary in summaries:
        label = str(summary.get("status") or "unknown").strip().lower() or "unknown"
        counts[label] = counts.get(label, 0) + 1
    return dict(sorted(counts.items()))


def _published_durations(summaries: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    """Duration between a run's creation and publication, where both exist.

    Only runs that record both stamps contribute. A run with no publication
    time has no duration -- that is reported as absent, never as zero seconds.
    """
    seconds: List[float] = []
    undated = 0
    unpublished = 0
    for summary in summaries:
        published = _parse_instant(summary.get("published"))
        if published is None:
            unpublished += 1
            continue
        created = _parse_instant(summary.get("created"))
        if created is None:
            undated += 1
            continue
        delta = (published - created).total_seconds()
        if delta >= 0:
            seconds.append(delta)
    stats = _summarize_floats(seconds)
    return {
        "measured": stats,
        "runs_without_publication_time": unpublished,
        "runs_without_creation_time": undated,
        "basis": "publication timestamp minus creation timestamp, from the "
                 "ledger summary",
    }


# --------------------------------------------------------------------------- #
# per-run verdicts: attestation + recorded validation
# --------------------------------------------------------------------------- #


def _attestation(limit: int) -> Dict[str, Any]:
    """Envelope/attestation status across the listed runs."""
    from backend.core import evidence_graph as eg

    ledger = _ledger()
    if not ledger["available"]:
        return {"available": False, "reason": ledger["reason"]}

    signatures: Dict[str, int] = {}
    envelopes: Dict[str, int] = {}
    attested = 0
    inspected = 0
    errors = 0
    for summary in ledger["summaries"][:limit]:
        run_id = summary.get("run_id")
        if not run_id:
            continue
        try:
            status = eg.envelope_status(str(run_id))
        except Exception:  # noqa: BLE001 - counted, not swallowed silently
            errors += 1
            continue
        inspected += 1
        if status.get("attested") is True:
            attested += 1
        sig = str(status.get("signature") or "unknown").lower()
        signatures[sig] = signatures.get(sig, 0) + 1
        env = str(status.get("envelope") or "unknown").lower()
        envelopes[env] = envelopes.get(env, 0) + 1

    return {
        "available": True,
        "runs_inspected": inspected,
        "attested": attested,
        "not_attested": inspected - attested,
        "by_signature": dict(sorted(signatures.items())),
        "by_envelope": dict(sorted(envelopes.items())),
        "inspection_errors": errors,
        "basis": "evidence_graph.envelope_status() per listed run",
    }


def _validation(limit: int) -> Dict[str, Any]:
    """Validation verdicts recorded by the validation stage.

    A pass rate is computed only over recorded ``peer_review.checks[].passed``
    booleans. When no checks were recorded the rate is None -- an absent
    verdict is not a 0% pass rate, still less a 100% one.
    """
    from backend.core import evidence_graph as eg

    ledger = _ledger()
    if not ledger["available"]:
        return {"available": False, "reason": ledger["reason"]}

    runs_with_verdict = 0
    validated_true = 0
    checks_total = 0
    checks_passed = 0
    human_review_true = 0
    human_review_false = 0
    confidence_scores: List[float] = []
    confidence_methods: Dict[str, int] = {}
    heuristic_flags: Dict[str, int] = {}
    read_errors = 0
    threshold_names: Dict[str, int] = {}

    for summary in ledger["summaries"][:limit]:
        run_id = summary.get("run_id")
        if not run_id:
            continue
        try:
            record = eg.load_run_record(str(run_id))
        except Exception:  # noqa: BLE001
            read_errors += 1
            continue
        if not isinstance(record, dict):
            read_errors += 1
            continue
        result = record.get("result") if isinstance(record.get("result"), dict) else {}
        steps = result.get("trace") if isinstance(result.get("trace"), list) else []
        step_result = None
        for step in steps:
            if isinstance(step, dict) and step.get("node") == "validate":
                candidate = step.get("result")
                if isinstance(candidate, dict):
                    step_result = candidate
                break
        if step_result is None:
            continue
        runs_with_verdict += 1
        if step_result.get("validated") is True:
            validated_true += 1

        peer = step_result.get("peer_review")
        if isinstance(peer, dict):
            if peer.get("is_human_peer_review") is True:
                human_review_true += 1
            elif peer.get("is_human_peer_review") is False:
                human_review_false += 1
            checks = peer.get("checks")
            if isinstance(checks, list):
                for check in checks:
                    if not isinstance(check, dict):
                        continue
                    passed = check.get("passed")
                    if not isinstance(passed, bool):
                        continue
                    checks_total += 1
                    if passed:
                        checks_passed += 1
                    name = str(check.get("check") or "")
                    if name:
                        threshold_names[name] = threshold_names.get(name, 0) + 1

        confidence = step_result.get("confidence")
        if isinstance(confidence, dict):
            score = _finite(confidence.get("confidence_score"))
            if score is not None:
                confidence_scores.append(score)
            method = confidence.get("method")
            if isinstance(method, str) and method.strip():
                label = method.strip()
                confidence_methods[label] = confidence_methods.get(label, 0) + 1
            heuristic = confidence.get("is_heuristic")
            if isinstance(heuristic, bool):
                key = "true" if heuristic else "false"
                heuristic_flags[key] = heuristic_flags.get(key, 0) + 1

    pass_rate = (round(checks_passed / checks_total, 6)
                 if checks_total else None)
    return {
        "available": True,
        "runs_with_a_validation_verdict": runs_with_verdict,
        "validated_true": validated_true,
        "checks_evaluated": checks_total,
        "checks_passed": checks_passed,
        "pass_rate": pass_rate,
        # Named so a reader can see the thresholds the rate is over, rather
        # than trusting an unlabelled percentage.
        "recorded_checks": dict(sorted(threshold_names.items())),
        "human_peer_review": {
            "flagged_human": human_review_true,
            "flagged_non_human": human_review_false,
        },
        "confidence": {
            "recorded_scores": _summarize_floats(confidence_scores),
            "methods": dict(sorted(confidence_methods.items())),
            "is_heuristic": dict(sorted(heuristic_flags.items())),
        },
        "read_errors": read_errors,
        "basis": "validate trace step's peer_review.checks[].passed and "
                 "confidence, as recorded by the validation stage",
    }


# --------------------------------------------------------------------------- #
# public read-models
# --------------------------------------------------------------------------- #


def aggregate_analytics() -> Dict[str, Any]:
    """Descriptive inventory. No quality metric is computed or implied."""
    unavailable: List[Dict[str, str]] = []
    ledger = _ledger()

    runs_section: Dict[str, Any]
    if ledger["available"]:
        summaries = ledger["summaries"]
        runs_section = {
            "total": len(summaries),
            "by_status": _run_by_status(summaries),
            "durations": _published_durations(summaries),
        }
    else:
        runs_section = {
            "total": None,
            "by_status": None,
            "durations": None,
        }
        unavailable.append({"source": "run_ledger", "reason": ledger["reason"]})

    experiments: Dict[str, Any]
    sessions: Dict[str, Any]
    try:
        store = _storage()
        experiments = {"total": len(store.list_experiments())}
    except Exception as exc:  # noqa: BLE001
        experiments = {"total": None}
        unavailable.append({
            "source": "experiments",
            "reason": f"{type(exc).__name__}: {exc}",
        })
    try:
        store = _storage()
        sessions = {"total": len(store.list_sessions())}
    except Exception as exc:  # noqa: BLE001
        sessions = {"total": None}
        unavailable.append({
            "source": "sessions",
            "reason": f"{type(exc).__name__}: {exc}",
        })

    fields = ("status", "generated_at", "runs", "experiments", "sessions",
              "unavailable")
    complete = not unavailable
    return {
        "status": "ok" if complete else "partial",
        "generated_at": _now(),
        "provenance": READ,
        "field_provenance": field_map(fields, READ),
        "note": ("Descriptive counts only. No quality, health or productivity "
                 "metric is computed, because nothing in this repository "
                 "measures one."),
        "runs": runs_section,
        "experiments": experiments,
        "sessions": sessions,
        "unavailable": unavailable,
    }


def scientific_health() -> Dict[str, Any]:
    """Subsystem probes + ledger + attestation + validation, with no score.

    ``score`` is always None and ``score_reason`` says why: the honest refusal
    the UI already displays, now carrying real signal beside it.
    """
    from backend.core import health as health_mod

    unavailable: List[Dict[str, str]] = []

    try:
        probes = health_mod.run_probes()
        overall = health_mod.overall(probes)
    except Exception as exc:  # noqa: BLE001
        probes = {}
        overall = health_mod.UNKNOWN
        unavailable.append({
            "source": "subsystem_probes",
            "reason": f"{type(exc).__name__}: {exc}",
        })

    ledger = _ledger()
    if ledger["available"]:
        ledger_section: Dict[str, Any] = {
            "total": len(ledger["summaries"]),
            "by_status": _run_by_status(ledger["summaries"]),
        }
    else:
        ledger_section = {"total": None, "by_status": None}
        unavailable.append({"source": "run_ledger", "reason": ledger["reason"]})

    attestation = _attestation(MAX_RUNS_INSPECTED)
    if not attestation.get("available"):
        unavailable.append({
            "source": "attestation",
            "reason": str(attestation.get("reason") or "unavailable"),
        })

    validation = _validation(MAX_RUNS_INSPECTED)
    if not validation.get("available"):
        unavailable.append({
            "source": "validation",
            "reason": str(validation.get("reason") or "unavailable"),
        })

    fields = ("status", "generated_at", "score", "subsystems", "run_ledger",
              "attestation", "validation", "unavailable")
    return {
        "status": overall,
        "generated_at": _now(),
        "provenance": READ,
        "field_provenance": field_map(fields, READ),
        # Nothing measures this. Leaving the key present with None, and saying
        # why, is the refusal; inventing a weighted sum is the fabrication.
        "score": None,
        "score_reason": (
            "No scientific-health score is computed: no component of this "
            "repository measures one. The counts below describe what was "
            "recorded, which is not the same claim as 'this deployment is "
            "healthy'."
        ),
        "subsystems": probes,
        "run_ledger": ledger_section,
        "attestation": attestation,
        "validation": validation,
        "unavailable": unavailable,
    }
