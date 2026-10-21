"""Shared result envelope for the benchmark reproduction pipelines.

Four pipelines in this package (`greater_than`, `copy_task`, `arithmetic`,
`factual_recall`) were stubs that fabricated their observed metrics. They are
implemented now, and this module exists so that all of them — and the two that
were already live, which follow the same shape by hand — answer with one
envelope a consumer can check without knowing which pipeline produced it.

The envelope distinguishes three states, and the distinction is the point:

``completed``
    The measurement ran on loaded weights. ``observed_metrics`` carries values,
    ``raw_traces`` carries one entry per prompt, every entry with a boolean
    outcome. The envelope is attested by
    :func:`backend.core.provenance.attest_measurement`, which is the only
    issuer of the ``live`` label -- setting it here was a claim no verifier
    could check, and ``pass_through`` would have withheld it downstream.

``unavailable``
    The pipeline ran and *the model does not do the task*, or one component of
    the measurement could not be taken. Every metric that could not be measured
    is ``None``, with a reason. A number is never substituted: ``0.0`` is a
    measurement ("none of the prompts were above chance"), so using it as a
    stand-in for "not measured" would make the two indistinguishable.

``raised LiveUnavailable``
    No weights were reachable at all, so nothing could be measured. The
    benchmark runner already turns this into ``NOT_RUN`` with the reason
    attached, which is the correct record for an environment that cannot
    measure.

Why ``raw_traces`` is not optional
---------------------------------
A headline score is an aggregate, and an aggregate hides which prompts produced
it. Every entry here carries the prompt, its target, the per-prompt boolean, and
the measurement that boolean came from. ``benchmark_runner`` uses those booleans
to build a Wilson interval, so a trace list that is present but empty is
reported as unmeasured rather than as a narrow interval around nothing.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

from backend.core.provenance import attest_measurement

#: Provenance value used when a component of the measurement was not taken.
#: There is deliberately no ``LIVE`` constant here: the ``live`` label is
#: issued by ``attest_measurement``, never written by a pipeline.
UNAVAILABLE = "unavailable"


def field_provenance(fields: Iterable[str], provenance: str) -> Dict[str, str]:
    """Map each named field to its provenance, for per-field checking.

    A single top-level ``provenance`` key says nothing about the fields inside
    it, so a caller who extracts ``result["observed_metrics"]`` loses the label.
    This is the shape the rest of the codebase already uses (see
    ``ioi_pipeline._field_map``); it is repeated here rather than imported so
    the four repaired pipelines do not depend on the IOI module.
    """
    return {str(f): provenance for f in fields}


#: Fields every envelope carries, so a consumer can check the label on each one.
_ENVELOPE_FIELDS = (
    "status",
    "observed_metrics",
    "raw_traces",
    "reason",
)


def completed(
    *,
    pipeline: str,
    paper_id: str,
    model_id: Optional[str],
    observed_metrics: Dict[str, Any],
    raw_traces: List[Dict[str, Any]],
    provenance_note: str,
    validation_eligible: bool,
    ineligible_because: Optional[List[str]] = None,
    **extra: Any,
) -> Dict[str, Any]:
    """Build the envelope for a measurement that actually ran.

    ``validation_eligible`` is a caller decision, not a default. A pipeline that
    measured something real but on a stimulus too small to support a claim (a
    handful of prompts, a task the model only barely performs) must pass False
    and say why in ``ineligible_because``.
    """
    payload: Dict[str, Any] = {
        "pipeline": pipeline,
        "paper_id": paper_id,
        "status": "completed",
        "model_id": model_id,
        "provenance_note": provenance_note,
        "observed_metrics": observed_metrics,
        "raw_traces": raw_traces,
        "n_traces": len(raw_traces),
        "validation_eligible": bool(validation_eligible),
        "publication_eligible": bool(validation_eligible),
        "ineligible_because": (ineligible_because or None),
        "mock_mode": False,
        "synthetic_fields": [],
    }
    payload.update(extra)
    return attest_measurement(payload, model_loaded=model_id or None)


def unavailable(
    *,
    pipeline: str,
    paper_id: str,
    model_id: Optional[str],
    reason: str,
    observed_metrics: Optional[Dict[str, Any]] = None,
    raw_traces: Optional[List[Dict[str, Any]]] = None,
    **extra: Any,
) -> Dict[str, Any]:
    """Build the envelope for a measurement that could not be taken.

    ``observed_metrics`` may still be supplied, and often should be: "the model
    does not perform this task, and here is the accuracy that shows it" is a
    finding worth carrying as data rather than as prose in ``reason``. Any
    metric that was not measured is present with the value ``None`` — never
    omitted, because an omitted key and a zero read the same to a caller doing
    ``metrics.get(name, 0.0)``.
    """
    payload: Dict[str, Any] = {
        "pipeline": pipeline,
        "paper_id": paper_id,
        "status": "unavailable",
        "model_id": model_id,
        "provenance": UNAVAILABLE,
        "provenance_note": (
            "No value is reported for the unmeasured fields; see reason."
        ),
        "field_provenance": field_provenance(_ENVELOPE_FIELDS, UNAVAILABLE),
        "observed_metrics": observed_metrics or {},
        "raw_traces": raw_traces or [],
        "n_traces": len(raw_traces or []),
        "validation_eligible": False,
        "publication_eligible": False,
        "reason": reason,
        "mock_mode": False,
        "synthetic_fields": [],
    }
    payload.update(extra)
    return payload
