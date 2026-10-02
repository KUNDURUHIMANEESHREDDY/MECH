"""EvidenceBoundary — the single gate every scientific result must pass through.

The problem this exists to solve: `backend/agents/evidence_policy.py` enforces
provenance on the Society workflow, but the guarded path is not the only path
through the system. Direct engines, adapters, and algorithm implementations can
still hand back a plain dict containing confident-looking numbers, and a caller
cannot tell a measured result from a fabricated one by looking at the shape.

So this module makes the boundary structural rather than advisory:

* Every result is an :class:`EvidenceResult`, never a bare dict.
* A result can only become ``provenance="live"`` by passing :meth:`EvidenceBoundary.admit`
  with an attestation that verifies. "live" is therefore *proved*, not stated.
* No algorithm can return a publishable payload. An algorithm produces a
  measurement; the boundary decides what that measurement is allowed to claim.

The rule this encodes: **a stub returning 0.94 is worse than an honest
``unavailable``.** When an algorithm cannot measure, it must say so, and this
boundary will not upgrade an excuse into a finding.
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass, field
from typing import Any, Dict, Literal, Optional

Status = Literal["completed", "blocked", "error"]
Provenance = Literal["live", "seeded", "reference", "unavailable"]

PROVENANCE_VALUES = {"live", "seeded", "reference", "unavailable"}
STATUS_VALUES = {"completed", "blocked", "error"}

# Only a live provenance may ever carry eligibility. Anything else is, by
# definition, not evidence.
_PUBLISHABLE_PROVENANCE = "live"


def _now() -> str:
    return _dt.datetime.utcnow().isoformat() + "Z"


@dataclass(frozen=True)
class RunAttestation:
    """Proof that a measurement came from a real, identified execution.

    ``live`` is only granted when this verifies. A caller cannot construct a
    passing attestation for a run that did not happen: the hashes must be
    present and real-shaped.
    """
    run_id: str
    executor_id: str
    model_id: str
    weights_sha256: str = ""
    dataset_sha256: str = ""
    code_revision: str = ""
    execution_id: str = ""

    # Hashes that are present must be well-formed; the placeholder values the
    # codebase used to ship ("sha256:8f43c...model_weights_mock", and the
    # literal string "unattested") are rejected outright.
    _REAL_SHA256 = 64

    def problems(self) -> list[str]:
        """Reasons this attestation cannot support a 'live' claim."""
        issues: list[str] = []
        if not self.run_id or not str(self.run_id).strip():
            issues.append("attestation has no run_id")
        if not self.executor_id or not str(self.executor_id).strip():
            issues.append("attestation has no executor_id")
        if not self.model_id or not str(self.model_id).strip():
            issues.append("attestation has no model_id")
        issues.extend(self._hash_problem("weights_sha256", self.weights_sha256))
        issues.extend(self._hash_problem("dataset_sha256", self.dataset_sha256))
        if not self.code_revision or not str(self.code_revision).strip():
            issues.append("attestation has no code_revision")
        return issues

    @classmethod
    def _hash_problem(cls, name: str, value: str) -> list[str]:
        raw = str(value or "").strip()
        if not raw:
            return [f"attestation has no {name}"]
        digest = raw.split(":", 1)[1] if raw.startswith("sha256:") else raw
        if "..." in digest or "mock" in digest.lower():
            return [f"{name} is a placeholder, not a real hash: {raw!r}"]
        if len(digest) != cls._REAL_SHA256:
            return [f"{name} is not a sha256 digest: {raw!r}"]
        try:
            int(digest, 16)
        except ValueError:
            return [f"{name} is not hexadecimal: {raw!r}"]
        return []

    def verifies(self) -> bool:
        return not self.problems()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "executor_id": self.executor_id,
            "model_id": self.model_id,
            "weights_sha256": self.weights_sha256,
            "dataset_sha256": self.dataset_sha256,
            "code_revision": self.code_revision,
            "execution_id": self.execution_id,
        }


@dataclass(frozen=True)
class EvidenceResult:
    """A scientific result that has passed (or failed) the boundary.

    Deliberately not a dict. Algorithms return measurements; they do not get to
    hand callers a payload that looks like a finding.
    """
    status: Status
    provenance: Provenance
    executor_id: str
    run_id: str = ""
    model_id: str = ""
    measurement: Optional[Dict[str, Any]] = None
    evidence: Optional[Dict[str, Any]] = None
    eligibility: Dict[str, Any] = field(default_factory=dict)
    reason: Optional[str] = None
    attestation: Optional[Dict[str, Any]] = None
    created_at: str = field(default_factory=_now)

    @property
    def publishable(self) -> bool:
        """True only for a live, completed, verified, explicitly eligible result."""
        return (
            self.status == "completed"
            and self.provenance == _PUBLISHABLE_PROVENANCE
            and self.eligibility.get("publication_eligible") is True
        )

    @property
    def live(self) -> bool:
        return self.provenance == _PUBLISHABLE_PROVENANCE

    def field_provenance(self, fields: tuple[str, ...]) -> Dict[str, str]:
        return {name: self.provenance for name in fields}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "provenance": self.provenance,
            "run_id": self.run_id,
            "executor_id": self.executor_id,
            "model_id": self.model_id,
            "measurement": self.measurement,
            "evidence": self.evidence,
            "eligibility": dict(self.eligibility),
            "reason": self.reason,
            "attestation": self.attestation,
            "publishable": self.publishable,
            "created_at": self.created_at,
        }

    # Dict-shaped payloads are how these results travelled before; keep the
    # shape compatible so existing consumers do not break, but only ever
    # derived from this validated record.
    def as_payload(self) -> Dict[str, Any]:
        payload = self.to_dict()
        payload["field_provenance"] = self.field_provenance(
            tuple(k for k, v in payload.items()
                  if k not in ("field_provenance", "provenance")))
        return payload


class EvidenceBoundary:
    """The one place a measurement is allowed to become a result."""

    @staticmethod
    def admit(
        *,
        executor_id: str,
        provenance: str = "unavailable",
        measurement: Optional[Dict[str, Any]] = None,
        attestation: Optional[RunAttestation] = None,
        reason: Optional[str] = None,
        evidence: Optional[Dict[str, Any]] = None,
    ) -> EvidenceResult:
        """Admit a measurement, deciding what it is allowed to claim.

        ``provenance`` is the *executor's own claim*. It is downgraded, never
        upgraded: only a verifying attestation can promote a claim to "live",
        and nothing here will promote a non-live claim on its own.
        """
        claimed = str(provenance or "unavailable").strip().lower() or "unavailable"
        if claimed not in PROVENANCE_VALUES:
            return EvidenceResult(
                status="error",
                provenance="unavailable",
                executor_id=executor_id,
                reason=(f"executor '{executor_id}' reported an unknown "
                        f"provenance {claimed!r}; treating as unavailable."),
            )

        if claimed != _PUBLISHABLE_PROVENANCE:
            # A seeded/reference/unavailable claim stays exactly that. The
            # measurement may be kept for inspection but can never be evidence.
            return EvidenceResult(
                status="blocked",
                provenance=claimed,  # type: ignore[arg-type]
                executor_id=executor_id,
                run_id=getattr(attestation, "run_id", "") if attestation else "",
                model_id=getattr(attestation, "model_id", "") if attestation else "",
                measurement=measurement,
                evidence=evidence,
                eligibility={"validation_eligible": False,
                             "publication_eligible": False},
                reason=reason or (
                    f"executor '{executor_id}' reported provenance "
                    f"'{claimed}', which is not evidence."),
                attestation=attestation.to_dict() if attestation else None,
            )

        # Claimed live. An attestation must back it.
        if attestation is None:
            return EvidenceResult(
                status="blocked",
                provenance="unavailable",
                executor_id=executor_id,
                measurement=measurement,
                eligibility={"validation_eligible": False,
                             "publication_eligible": False},
                reason=(f"executor '{executor_id}' claimed live provenance "
                        "without an attestation; a claim is not a proof."),
            )

        problems = attestation.problems()
        if problems:
            return EvidenceResult(
                status="blocked",
                provenance="unavailable",
                executor_id=executor_id,
                run_id=attestation.run_id,
                model_id=attestation.model_id,
                measurement=measurement,
                evidence=evidence,
                eligibility={"validation_eligible": False,
                             "publication_eligible": False},
                reason=("Live provenance rejected: "
                        + "; ".join(problems)),
                attestation=attestation.to_dict(),
            )

        if not isinstance(measurement, dict) or not measurement:
            return EvidenceResult(
                status="blocked",
                provenance="unavailable",
                executor_id=executor_id,
                run_id=attestation.run_id,
                model_id=attestation.model_id,
                eligibility={"validation_eligible": False,
                             "publication_eligible": False},
                reason=(f"executor '{executor_id}' is attested but produced no "
                        "measurement; an attested run that measured nothing is "
                        "not a finding."),
                attestation=attestation.to_dict(),
            )

        return EvidenceResult(
            status="completed",
            provenance="live",
            executor_id=executor_id,
            run_id=attestation.run_id,
            model_id=attestation.model_id,
            measurement=measurement,
            evidence=evidence,
            eligibility={"validation_eligible": True,
                         "publication_eligible": True},
            reason=reason,
            attestation=attestation.to_dict(),
        )

    @staticmethod
    def unavailable(executor_id: str, reason: str,
                    **kw: Any) -> EvidenceResult:
        """The honest answer when an executor cannot measure."""
        return EvidenceResult(
            status="blocked",
            provenance="unavailable",
            executor_id=executor_id,
            reason=reason,
            eligibility={"validation_eligible": False,
                         "publication_eligible": False},
            **kw,
        )

    @staticmethod
    def error(executor_id: str, reason: str, **kw: Any) -> EvidenceResult:
        return EvidenceResult(
            status="error",
            provenance="unavailable",
            executor_id=executor_id,
            reason=reason,
            eligibility={"validation_eligible": False,
                         "publication_eligible": False},
            **kw,
        )


BOUNDARY = EvidenceBoundary()
