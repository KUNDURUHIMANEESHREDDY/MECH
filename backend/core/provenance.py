"""Provenance origination — the single place a result may be called `live`.

The rule this module exists to enforce
--------------------------------------
**Provenance is originated by the layer that did the measurement. It is never
inferred by a wrapper because a function exists, imported successfully, or was
reachable.**

Two defects motivated it, both in the same trust boundary.

1. `backend/agents/executor.py:_call` did `res.setdefault("provenance", "live")`
   and then defaulted *every field* to `live` via `field_provenance`. A returned
   `{"status": "unavailable"}` therefore became a live measurement, and its
   `reason` field was relabelled `live` too.

2. `backend/api/dispatcher.py:/infer` did the same, gated only on
   `engine.is_available()`. That predicate says the module imported. It does not
   say a forward pass ran, and it says nothing about *which weights* ran.

The second was demonstrably wrong rather than merely fragile. `gpt2_engine`
loads exactly one model, hardcoded `"gpt2"`, and `infer()` echoed the caller's
`model_name` back verbatim:

    >>> gpt2_engine.infer("Hello", "gpt2-large")["model_name"]
    'gpt2-large'
    >>> gpt2_engine.infer("Hello", "gpt2-large")["d_model"]     # gpt2-large is 1280
    768                                                       # gpt2-small

So a caller asking for gpt2-large received a response labelled
`provenance: "live"` whose `model_name` said gpt2-large, while every tensor came
from gpt2-small. Nothing in the response indicated otherwise.

How it is used
--------------
Measurement layer::

    from backend.core.provenance import attest_measurement
    return attest_measurement(payload, model_loaded=_loaded_model_id(),
                              model_requested=model_name)

Wrapper / orchestrator::

    from backend.core.provenance import pass_through
    return pass_through(res)   # never invents `live`

`attest_measurement` refuses to stamp `live` on a result whose status says it
failed, so even a mistaken caller cannot manufacture a live claim.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Mapping, Optional

LIVE = "live"
SEEDED = "seeded"
REFERENCE = "reference"
UNAVAILABLE = "unavailable"

#: Statuses that mean the work did not happen. A measurement layer must not
#: stamp `live` on any of these regardless of what it was asked to attest.
_FAILURE_STATUSES = frozenset({
    "error", "failed", "failure", "unavailable", "not_run", "notrun",
    "blocked", "skipped", "timeout", "invalid", "rejected", "declined",
})

#: Keys that describe the record's own framing rather than a measured quantity,
#: and so are not defaulted to `live` when a per-field map is built.
_FRAME_KEYS = frozenset({
    "provenance", "field_provenance", "status", "reason", "error",
    "provenance_note", "model_loaded", "model_requested", "model_mismatch",
    "attested",
    "measured", "elapsed_ms", "runtime_ms", "timestamp",
})


def is_failure_status(status: Any) -> bool:
    """True when a `status` field says the work did not succeed."""
    if status is None:
        return False
    return str(status).strip().lower() in _FAILURE_STATUSES


def attest_measurement(
    result: Dict[str, Any],
    *,
    model_loaded: Optional[str] = None,
    model_requested: Optional[str] = None,
    fields: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    """Stamp `live` on a result the caller has just measured.

    Refuses when the result's own `status` reports a failure, and records which
    weights actually produced it when `model_loaded` is given. A requested model
    that differs from the loaded one is recorded as `model_mismatch`, because a
    caller asking for gpt2-large and receiving gpt2-small numbers should be able
    to see that in the record rather than having to notice the dimensions.

    An existing, more specific provenance is respected: this never overwrites a
    label the measurement layer chose deliberately.

    This function is the ONLY issuer of `attested`, which is what
    `evidence_policy._is_attested` and `evidence_graph` gate on. So it attests
    only what it measured, and never on the strength of a label it did not
    write.
    """
    if not isinstance(result, dict):
        return result

    # A failure is a failure on every path. This check used to sit *below* the
    # pre-existing-label branch, whose early `return` meant a record already
    # claiming `live` never reached it -- so `{"provenance": "live",
    # "status": "failed"}` came back attested.
    if is_failure_status(result.get("status")):
        return withhold(
            result,
            reason=(f"status was {result.get('status')!r}, so nothing was "
                    f"measured and the result cannot be attested live"),
        )

    existing = result.get("provenance")
    if isinstance(existing, str) and existing.strip():
        if existing.strip().lower() == LIVE:
            # A `live` label arriving from outside is a claim, not a
            # measurement. `provenance` is an ordinary dict key, so this
            # function cannot distinguish a deliberate choice by a measurement
            # layer from a value some caller typed in -- and treating them the
            # same is what makes the label forgeable.
            #
            # Note there is deliberately no exemption for a record that also
            # carries `attested: True`. That flag is caller-writable too, so
            # trusting it here would leave the identical hole one key over.
            # The legitimate re-wrap path does not exist: this module's own
            # contract routes wrappers and orchestrators through
            # `pass_through`, which never invents `live`. A record that really
            # was measured once already holds its attestation.
            return withhold(
                result,
                reason=("carried a 'live' provenance label that this "
                        "measurement boundary did not issue; an unattested "
                        "live label is not a measurement"),
            )
        return result

    result["provenance"] = LIVE
    result["attested"] = True
    keys = list(fields) if fields is not None else [
        k for k in result if k not in _FRAME_KEYS
    ]
    result["field_provenance"] = {
        str(key): LIVE for key in keys if str(key) not in _FRAME_KEYS
    }

    if model_loaded is not None:
        result["model_loaded"] = model_loaded
    if model_requested is not None:
        result["model_requested"] = model_requested
        result["model_mismatch"] = bool(
            model_loaded and model_requested
            and str(model_loaded) != str(model_requested)
        )
    return result


def withhold(result: Dict[str, Any], reason: str) -> Dict[str, Any]:
    """Mark a result as not measured, with the reason.

    Never overwrites an existing non-live provenance that is already more
    specific than `unavailable`.
    """
    if not isinstance(result, dict):
        return result

    # Withheld means not measured, so never attested — on every path,
    # including the early return for an already-specific non-live label.
    result["attested"] = False
    existing = str(result.get("provenance") or "").strip().lower()
    if existing and existing != LIVE:
        result.setdefault("reason", reason)
        return result

    result["provenance"] = UNAVAILABLE
    result.setdefault("status", "unavailable")
    # Keep a reason the record already carries. An engine's own explanation of
    # why it measured nothing -- "torch/transformers not installed" -- is more
    # specific than the generic "nothing attested this", and overwriting it would
    # discard the only actionable part of the record.
    if not str(result.get("reason") or "").strip():
        result["reason"] = reason
    result["field_provenance"] = {
        str(key): UNAVAILABLE
        for key in result
        if key not in {"provenance", "field_provenance"}
    }
    return result


def pass_through(result: Any, reason: Optional[str] = None) -> Any:
    """Prepare a measurement-layer result for return. Never invents `live`.

    This is what a wrapper calls. If the measurement layer attested the result,
    that label is preserved. If it did not, the result is withheld rather than
    being upgraded on the wrapper's say-so -- because "the function exists" is
    not evidence that anything was measured.
    """
    if not isinstance(result, dict):
        return result

    provenance = str(result.get("provenance") or "").strip().lower()
    if provenance:
        return result

    return withhold(
        result,
        reason=reason or (
            "The measurement layer returned this record without a provenance "
            "label, so it cannot be presented as a measurement. A wrapper must "
            "not infer 'live' from the existence of the method."
        ),
    )


# Evidence level helpers

#: Evidence levels for scientific claims, from weakest to strongest
#: OBSERVATIONAL: correlation/activation patterns observed
#: ATTRIBUTIONAL: attribution methods (attention, gradient, etc.)
#: INTERVENTIONAL: causal intervention (patching, ablation)
#: CAUSALLY_VALIDATED: passed held-out validation checks
#: REPLICATED: replicated across seeds/models/prompts
EVIDENCE_LEVELS = {
    "OBSERVATIONAL": 1,
    "ATTRIBUTIONAL": 2,
    "INTERVENTIONAL": 3,
    "CAUSALLY_VALIDATED": 4,
    "REPLICATED": 5,
}


def evidence_level_rank(level: str) -> int:
    """Return the numeric rank of an evidence level (higher = stronger)."""
    return EVIDENCE_LEVELS.get(str(level).upper(), 0)


def max_evidence_level(*levels: str) -> str:
    """Return the strongest evidence level from a list."""
    if not levels:
        return "OBSERVATIONAL"
    return max(levels, key=evidence_level_rank)


def set_evidence_level(result: Dict[str, Any], level: str) -> Dict[str, Any]:
    """Set the evidence level on a result dictionary."""
    if not isinstance(result, dict):
        return result
    level_upper = str(level).upper()
    if level_upper not in EVIDENCE_LEVELS:
        raise ValueError(f"Unknown evidence level: {level}. Valid: {list(EVIDENCE_LEVELS.keys())}")
    result["evidence_level"] = level_upper
    return result


def get_evidence_level(result: Dict[str, Any]) -> str:
    """Get the evidence level from a result, defaulting to OBSERVATIONAL."""
    if not isinstance(result, dict):
        return "OBSERVATIONAL"
    return str(result.get("evidence_level", "OBSERVATIONAL")).upper()