"""One vocabulary for what happened to a benchmark.

The rule this module exists to enforce
-------------------------------------
**A measurement that ran, a measurement that was scored, and a measurement that
could not be compared are three different facts. A status vocabulary that cannot
express all three will collapse two of them into a lie.**

Why a module rather than a comment
----------------------------------
The audit's P1 #8 found two competing concepts -- a scientific status
(`NOT_RUN / MEASURED / PASS / FAIL`) and a report rendering (`PASS / WARN`). In
practice the repository carried *nine* distinct status vocabularies:

    benchmark_scheduler        NOT_RUN, PASS, REGRESSION, MEASURED
    science/.../benchmark_runner PASS, FIXTURE, NOT_RUN, ERROR
    benchmarking report         FIXTURE, NOT_RUN, MEASURED
    evidence_boundary           completed, blocked, error
    society / multi_agent_society  completed
    research_manifest           VALIDATED, UNSIGNED, REVISION_REQUIRED, UNVALIDATED
    distributed/*               Success, Error, Offline        (TitleCase)
    research/campaign_analytics Completed
    research/.../curriculum     Completed

Not all of those are in scope here, and the distinction matters: `Success`/`Error`
is a *worker* status, `Completed` is a *campaign* status, and `VALIDATED`/`UNSIGNED`
is a *signature* status. Those are different axes and merging them would be its
own kind of error. What belongs on one axis is what happened to a scientific
measurement, and that is what this module defines.

The four states
---------------
``NOT_RUN``
    Nothing was measured. No implementation, or the model does not perform the
    task. Carries a ``reason`` saying which.

``MEASURED``
    Something was measured, and it is **not comparable** to the baseline it would
    be scored against. This is the state the old vocabulary could not express, and
    its absence is what caused the defect below: with only PASS and REGRESSION
    available for a measured benchmark, a consumer had to either invent a verdict
    or discard the measurement.

``PASS`` / ``REGRESSION``
    Measured, comparable, and scored against the baseline. These are the only two
    states that assert a verdict, and they are only reachable when
    ``baseline_is_comparable is True``.

``ERROR``
    The measurement was attempted and broke. Distinct from ``NOT_RUN``, which is
    a statement about scope rather than about the code.

The comparability contract
--------------------------
Comparability is not a status. It is a separate, orthogonal fact, and that is the
point: a run can be ``MEASURED`` and simultaneously not scoreable, and a consumer
needs to be able to see both without parsing prose.

``baseline_is_comparable``
    Whether ``published_baseline_fidelity`` is the same measurement this harness
    computes. ``None`` means *not established*, and is treated as **not
    comparable** -- fail-closed, because an unestablished comparison is not a
    comparison.

``runtime_is_comparable``
    The same question for wall-clock timing. It exists separately because timing
    baselines are hardware-specific in a way that fidelity baselines are not, and
    because conflating the two is what let a local run on an RTX 3050 be reported
    as a ``+6391%`` latency regression against a figure from a paper.
"""

from __future__ import annotations

from typing import Iterable, Optional

#: Nothing was measured -- no implementation, or the model cannot do the task.
NOT_RUN = "NOT_RUN"

#: Measured, but not comparable to the baseline it would be scored against.
MEASURED = "MEASURED"

#: Measured, comparable, and at or above the threshold.
PASS = "PASS"

#: Measured, comparable, and below the threshold.
REGRESSION = "REGRESSION"

#: Attempted and broke. Not the same as NOT_RUN, which is about scope.
ERROR = "ERROR"

#: A score that came from a fixture rather than a measurement. Never a pass.
FIXTURE = "FIXTURE"

#: Every state a benchmark result may be in.
BENCHMARK_STATUSES = frozenset({NOT_RUN, MEASURED, PASS, REGRESSION, ERROR, FIXTURE})

#: The states that assert a verdict. A benchmark outside these was not scored.
VERDICT_STATUSES = frozenset({PASS, REGRESSION})

#: The states in which something was actually measured.
MEASURED_STATUSES = frozenset({MEASURED, PASS, REGRESSION})

#: The states in which nothing was measured.
UNMEASURED_STATUSES = frozenset({NOT_RUN, ERROR})

#: States that mean the number came from somewhere other than a measurement.
NON_MEASUREMENT_STATUSES = frozenset({FIXTURE})


def is_measured(status: Optional[str]) -> bool:
    """Whether a status says a measurement was taken."""
    return str(status or "").strip().upper() in MEASURED_STATUSES


def is_verdict(status: Optional[str]) -> bool:
    """Whether a status asserts a scored pass or regression."""
    return str(status or "").strip().upper() in VERDICT_STATUSES


def is_unmeasured(status: Optional[str]) -> bool:
    """Whether a status says nothing was measured."""
    return str(status or "").strip().upper() in UNMEASURED_STATUSES


def is_comparable(flag: Optional[bool]) -> bool:
    """Whether a comparability flag establishes that a comparison is meaningful.

    Fail-closed. ``None`` -- not established -- is not comparable, because an
    unestablished comparison is not a comparison. Only an explicit ``True``
    permits a verdict to be issued against a baseline.
    """
    return flag is True


def describe(flags: Iterable[object]) -> str:
    """A short, stable summary of a set of statuses, for logs and reports."""
    counts = {}
    for status in flags:
        key = str(status or "UNKNOWN").upper()
        counts[key] = counts.get(key, 0) + 1
    if not counts:
        return "none"
    return ", ".join(f"{name}={counts[name]}" for name in sorted(counts))
