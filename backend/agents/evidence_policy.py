"""Fail-closed evidence eligibility rules for Research Society stages.

Scientific evidence is eligible for downstream validation or publication only
when the backend explicitly marks it as live, attests the measurement, and
opts it into both downstream uses. Missing provenance is treated as
unavailable; it is never inferred from a plausible score or a successful
process exit. A `live` label without `attested is True` is a live-looking
record, not evidence.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional


def _is_attested(payload: Any) -> bool:
    return _record(payload).get("attested") is True


_COMPLETED_STATUSES = {"completed", "complete", "ok", "passed", "success"}
_LIVE = "live"


def _record(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def status_of(payload: Any) -> str:
    return str(_record(payload).get("status", "unavailable")).strip().lower()


def provenance_of(payload: Any) -> str:
    """Return a normalized provenance label, including DTO-style mappings."""
    record = _record(payload)
    raw = record.get("provenance")
    if isinstance(raw, dict):
        raw = (raw.get("source") or raw.get("kind")
               or raw.get("type") or raw.get("status"))
    if raw is None:
        return "unavailable"
    return str(raw).strip().lower() or "unavailable"


def _is_completed(payload: Any) -> bool:
    return status_of(payload) in _COMPLETED_STATUSES


def _opted_in(payload: Any) -> bool:
    record = _record(payload)
    return (record.get("validation_eligible") is True
            and record.get("publication_eligible") is True)


def field_map(fields: Iterable[str], provenance: str) -> Dict[str, str]:
    """Create an explicit field-to-provenance map for a payload boundary."""
    label = str(provenance or "unavailable")
    return {str(field): label for field in fields}


def discovery_is_live(payload: Any) -> bool:
    """Whether a discovery result may enter validation and publication."""
    return (_is_completed(payload)
            and provenance_of(payload) == _LIVE
            and _is_attested(payload)
            and _opted_in(payload))


def validation_is_live(payload: Any) -> bool:
    """Whether a validation result may support a scientific publication."""
    record = _record(payload)
    return (discovery_is_live(record)
            and record.get("validated") is True)


def reproduction_is_live(payload: Any) -> Dict[str, Any]:
    """Return the record only when reproduction is explicitly live."""
    record = _record(payload)
    if (not _is_completed(record)
            or provenance_of(record) != _LIVE
            or not _is_attested(record)
            or record.get("mock_mode") is True):
        return {}
    return record


def gate_is_live(payload: Any) -> bool:
    """A passing gate must also carry live provenance."""
    record = _record(payload)
    return (_is_completed(record)
            and provenance_of(record) == _LIVE
            and _is_attested(record)
            and record.get("passed") is True)


def blocked_reason(payload: Any, stage: str) -> str:
    """Explain why a stage cannot contribute scientific evidence."""
    record = _record(payload)
    provenance = provenance_of(record)
    if provenance != _LIVE:
        return f"{stage} blocked: provenance '{provenance}' is not live."
    if not _is_completed(record):
        return f"{stage} blocked: status '{status_of(record)}' is not complete."
    if not _is_attested(record):
        return f"{stage} blocked: the measurement is not attested."
    if record.get("validation_eligible") is not True:
        return f"{stage} blocked: validation eligibility was not asserted."
    if record.get("publication_eligible") is not True:
        return f"{stage} blocked: publication eligibility was not asserted."
    if stage.lower().startswith("validation") and record.get("validated") is not True:
        return "Validation blocked: the live validation verdict is not passing."
    return f"{stage} blocked: required evidence fields are incomplete."


def _steps(trace: Optional[Iterable[Dict[str, Any]]], node: str):
    for step in trace or ():
        if isinstance(step, dict) and str(step.get("node", "")) == node:
            yield step


def publication_block_reason(
    trace: Optional[Iterable[Dict[str, Any]]],
    reproducibility: Any = None,
    gate: Any = None,
) -> str:
    """Check the complete Society evidence chain before publication.

    Every discovery and validation step is checked, rather than trusting the
    last or highest score.  Optional reproduction and gate arguments are still
    checked when supplied; an empty or synthetic object is not accepted.
    """
    discoveries = list(_steps(trace, "discover"))
    if not discoveries:
        return "Publication blocked: no live DiscoveryEngine result was returned."
    for step in discoveries:
        if not _is_completed(step):
            return f"Discovery blocked: stage status '{status_of(step)}' is not complete."
        # Trace steps store discoverer response fields directly (spread), not in "result"
        result = step.get("result")
        if not result:
            result = step
        if not discovery_is_live(result):
            return blocked_reason(result, "Discovery")

    validations = list(_steps(trace, "validate"))
    if not validations:
        return "Publication blocked: no validation result was returned."
    for step in validations:
        if not _is_completed(step):
            return f"Validation blocked: stage status '{status_of(step)}' is not complete."
        result = step.get("result")
        if not result:
            result = step
        if not validation_is_live(result):
            return blocked_reason(result, "Validation")

    if reproducibility is not None and not reproduction_is_live(reproducibility):
        return blocked_reason(reproducibility, "Reproduction")
    if gate is not None and not gate_is_live(gate):
        return blocked_reason(gate, "Validation gate")
    return ""
