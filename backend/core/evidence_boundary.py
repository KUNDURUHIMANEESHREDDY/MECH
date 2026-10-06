"""EvidenceBoundary — the one place a measurement is allowed to become a result.

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

Two ways this used to be theatre, and what replaced them
-------------------------------------------------------

**The attestation checked the shape of a hash, not the truth of it.**
``RunAttestation.problems()`` verified that each digest was 64 hex characters.
That proves someone typed a plausible-looking string. It does not tie the
string to the weights that produced the result, and the docstring's claim that
"a caller cannot construct a passing attestation for a run that did not happen"
was simply false:

    RunAttestation(run_id="anything", executor_id="anything", model_id="gpt2",
                   weights_sha256="<64 hex>", dataset_sha256="<64 hex>",
                   code_revision="anything")

verified, and became publishable. Now the attestation is an Ed25519 signature
over a canonical payload that includes a digest *of the measurement itself*, so
it is bound to one specific result and to one specific execution. Verification
recomputes that binding, so an attestation cannot be moved onto a different
measurement, replayed onto a second one, or re-pointed at a different model.

**``EvidenceResult`` could be constructed directly as publishable.**
``publishable`` derived from ``status``/``provenance``/``eligibility`` and never
consulted the attestation, so a caller could write

    EvidenceResult(status="completed", provenance="live", executor_id="x",
                   eligibility={"publication_eligible": True})

and get ``.publishable is True`` without ever calling ``admit()``. The class
now requires a capability that only the boundary holds, so the record's
provenance and eligibility are decided by the boundary rather than by its maker.

Honest limit on that capability: this closes the *structural* loophole — no
module can mint a live result by accident or by writing a dict literal. It is
not a defence against code that has deliberately imported the private token,
because Python offers no real private state. That attack requires intent and
code review to spot; this one showed up in an audit.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import threading
import uuid
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

Status = Literal["completed", "blocked", "error"]
Provenance = Literal["live", "seeded", "reference", "unavailable"]

PROVENANCE_VALUES = {"live", "seeded", "reference", "unavailable"}
STATUS_VALUES = {"completed", "blocked", "error"}

# Only a live provenance may ever carry eligibility. Anything else is, by
# definition, not evidence.
_PUBLISHABLE_PROVENANCE = "live"

# A sha256 digest, without its algorithm prefix, is exactly 64 hex characters.
_REAL_SHA256 = 64

# The capability that makes EvidenceResult unforgeable. Deliberately not
# exported: a caller that constructs a result without it gets an error rather
# than a result whose provenance nobody checked.
_BOUNDARY_TOKEN = object()

_integrity: Any = None


def _integrity_module():
    """Import the Ed25519 primitives on first use, not at import time.

    `backend/core` is imported by nearly everything, including the desktop app
    entry point that is meant to stay light. Pulling `backend.science` in at
    module scope would drag the statistics and reproducibility subsystems along
    with it on every `import backend.core`, so the dependency is resolved on
    first verification instead.
    """
    global _integrity
    if _integrity is None:
        from backend.science.integrity import signing

        _integrity = signing
    return _integrity


def _now() -> str:
    return _dt.datetime.utcnow().isoformat() + "Z"


def canonical_bytes(value: Any) -> bytes:
    """The one serialisation used for every digest and signature here.

    Sorted keys and fixed separators, so the same logical value always produces
    the same bytes on every platform and Python version.
    """
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      default=str).encode("utf-8")


def digest_of(value: Any) -> str:
    return "sha256:" + hashlib.sha256(canonical_bytes(value)).hexdigest()


def sha256_file(path: Any) -> str:
    """Content hash of a real file, streamed rather than read whole."""
    digest = hashlib.sha256()
    with open(Path(path), "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


class ExecutionLedger:
    """Remembers which measurement each execution id has already attested to.

    An execution happens once. Re-deriving the identical result from it is
    legitimate and idempotent, so that case is allowed. Presenting the *same*
    execution id with a *different* measurement is the replay this catches:
    without it, one real run could be stretched across any number of invented
    results.

    Bounded on purpose. A long-running server would otherwise grow this without
    limit, so the oldest entries are evicted once `_MAX_TRACKED` is reached.
    That does weaken replay detection for whatever was evicted, which is a
    deliberate trade: catching a replay from the distant past is worth less
    than not leaking memory on every admission. The bound is far above any
    realistic in-flight window.
    """

    _MAX_TRACKED = 4096

    def __init__(self) -> None:
        self._seen: "Dict[str, str]" = {}
        self._lock = threading.Lock()

    def claim(self, execution_id: str, measurement_sha256: str) -> Tuple[bool, str]:
        key = str(execution_id or "").strip()
        if not key:
            # An attestation without an execution id is already rejected by
            # `problems()`; do not silently let one through here as well.
            return True, ""
        with self._lock:
            previous = self._seen.get(key)
            if previous is not None:
                # Re-admitting the same execution with the same measurement is
                # idempotent, so it does not consume a slot or refresh order.
                return (True, "") if previous == measurement_sha256 else (False, (
                    f"execution_id {key!r} already attested measurement "
                    f"{previous}; refusing to replay it as {measurement_sha256}"))
            self._seen[key] = measurement_sha256
            while len(self._seen) > self._MAX_TRACKED:
                self._seen.pop(next(iter(self._seen)))
            return True, ""

    def reset(self) -> None:
        with self._lock:
            self._seen.clear()


LEDGER = ExecutionLedger()


@dataclass(frozen=True)
class RunAttestation:
    """Proof that a measurement came from a real, identified execution.

    ``live`` is only granted when this verifies. Verification means three
    separate things, and all three are required:

    * the digests are real-shaped, not the placeholders this codebase used to
      ship (``sha256:8f43c...model_weights_mock``, the literal ``"unattested"``);
    * an Ed25519 signature over the canonical payload verifies under the
      carried public key, whose fingerprint matches the recorded ``key_id``;
    * ``measurement_sha256`` matches the measurement actually being admitted,
      so this attestation cannot be reused for a different result.

    The signature is what makes the difference between a claim and a proof. A
    caller can construct any field values it likes -- including a perfectly
    formed 64-character hex digest -- but it cannot sign them without the
    executor's key, and verification will notice if the payload moved.
    """
    run_id: str
    executor_id: str
    model_id: str
    weights_sha256: str = ""
    dataset_sha256: str = ""
    code_revision: str = ""
    execution_id: str = ""
    # What the caller asked for, and what actually ran. Both are recorded
    # because a mismatch is a scientific error, not a cosmetic one.
    model_requested: str = ""
    model_loaded: str = ""
    config_sha256: str = ""
    measurement_sha256: str = ""
    issued_at: str = field(default_factory=_now)
    signature: str = ""
    public_key: str = ""
    key_id: str = ""
    # (label, digest) pairs naming the artifacts that were hashed. The paths
    # themselves are deliberately not stored: a persisted path can be rewritten
    # afterwards, whereas `rehash_artifacts` lets a verifier recompute against
    # the artifact store it trusts.
    artifact_bindings: Tuple[Tuple[str, str], ...] = ()

    @classmethod
    def issue(
        cls,
        *,
        run_id: str,
        executor_id: str,
        model_id: str,
        measurement: Dict[str, Any],
        model_requested: str = "",
        weights_sha256: str = "",
        dataset_sha256: str = "",
        code_revision: str = "",
        config: Optional[Dict[str, Any]] = None,
        weights_path: Any = None,
        dataset_path: Any = None,
        private_key: Any = None,
        execution_id: str = "",
        artifact_bindings: Tuple[Tuple[str, str], ...] = (),
    ) -> "RunAttestation":
        """Sign an attestation for a measurement that has just happened.

        Digests come from real files when a path is given and from the
        measurement object itself in every case. Raises `SigningUnavailable`
        when no key can be resolved: an unsigned attestation would verify as
        nothing, so the honest outcome is to fail rather than to produce one.
        """
        integrity = _integrity_module()
        bindings = [(str(label), str(value)) for label, value in artifact_bindings]

        if weights_path is not None:
            weights_sha256 = sha256_file(weights_path)
            bindings.append(("weights", weights_sha256))
        if dataset_path is not None:
            dataset_sha256 = sha256_file(dataset_path)
            bindings.append(("dataset", dataset_sha256))

        public_key = integrity.public_key_hex(private_key)
        unsigned = cls(
            run_id=run_id,
            executor_id=executor_id,
            model_id=model_id,
            weights_sha256=weights_sha256,
            dataset_sha256=dataset_sha256,
            code_revision=code_revision,
            execution_id=execution_id or uuid.uuid4().hex,
            model_requested=model_requested or model_id,
            model_loaded=model_id,
            config_sha256=digest_of(config) if config is not None else "",
            measurement_sha256=digest_of(measurement),
            public_key=public_key,
            key_id=integrity.key_id_for(public_key),
            artifact_bindings=tuple(bindings),
        )
        signature = integrity.sign(unsigned.signing_bytes(), private_key)
        # `issued_at` must be carried across explicitly. `replace()` rebuilds
        # the instance through __init__, so any field left to its
        # `default_factory` would be regenerated *after* signing -- moving the
        # payload and invalidating the signature on the way out.
        return replace(unsigned, issued_at=unsigned.issued_at,
                       signature=signature)

    def payload(self) -> Dict[str, Any]:
        """Everything the signature covers.

        `signature` itself is excluded, as it must be. Nothing else is: a
        verifier that recomputes this from the record will notice any field that
        was altered after signing, which is the property a shape check could not
        offer.
        """
        return {
            "run_id": self.run_id,
            "executor_id": self.executor_id,
            "model_id": self.model_id,
            "weights_sha256": self.weights_sha256,
            "dataset_sha256": self.dataset_sha256,
            "code_revision": self.code_revision,
            "execution_id": self.execution_id,
            "model_requested": self.model_requested,
            "model_loaded": self.model_loaded,
            "config_sha256": self.config_sha256,
            "measurement_sha256": self.measurement_sha256,
            "issued_at": self.issued_at,
            "key_id": self.key_id,
            "artifact_bindings": [[label, value]
                                  for label, value in self.artifact_bindings],
        }

    def signing_bytes(self) -> bytes:
        return canonical_bytes(self.payload())

    def problems(self) -> List[str]:
        """Reasons this attestation cannot support a 'live' claim."""
        issues: List[str] = []
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
        if not self.execution_id or not str(self.execution_id).strip():
            issues.append("attestation has no execution_id")
        issues.extend(self._hash_problem("measurement_sha256",
                                         self.measurement_sha256))
        issues.extend(self._identity_problem())
        issues.extend(self._signature_problem())
        return issues

    @staticmethod
    def _hash_problem(name: str, value: str) -> List[str]:
        raw = str(value or "").strip()
        if not raw:
            return [f"attestation has no {name}"]
        digest = raw.split(":", 1)[1] if raw.startswith("sha256:") else raw
        if "..." in digest or "mock" in digest.lower():
            return [f"{name} is a placeholder, not a real hash: {raw!r}"]
        if len(digest) != _REAL_SHA256:
            return [f"{name} is not a sha256 digest: {raw!r}"]
        try:
            int(digest, 16)
        except ValueError:
            return [f"{name} is not hexadecimal: {raw!r}"]
        return []

    def _identity_problem(self) -> List[str]:
        """Requested model and loaded model must be the same model.

        `model_id` is the model that produced the numbers. If a caller asked
        for something else, the numbers belong to a model the caller did not
        ask for, and annotating the mismatch is not enough -- the result has to
        stop being live.
        """
        issues: List[str] = []
        loaded = str(self.model_loaded or "").strip()
        requested = str(self.model_requested or "").strip()
        if not loaded:
            return ["attestation does not record which model actually ran"]
        if loaded != str(self.model_id or "").strip():
            issues.append(
                f"model identity mismatch: model_id is {self.model_id!r} but "
                f"the model that ran was {loaded!r}")
        if requested and requested != loaded:
            issues.append(
                f"model identity mismatch: caller requested {requested!r} but "
                f"{loaded!r} actually ran")
        return issues

    def _signature_problem(self) -> List[str]:
        if not str(self.signature or "").strip():
            return ["attestation carries no signature; real-shaped hashes are a "
                    "claim about a run, not proof that it happened"]
        if not str(self.public_key or "").strip():
            return ["attestation carries no public key, so its signature cannot "
                    "be checked by anyone but its maker"]
        if not str(self.key_id or "").strip():
            return ["attestation carries no key_id, so the signer is unidentifiable"]
        integrity = _integrity_module()
        expected = integrity.key_id_for(self.public_key)
        if self.key_id != expected:
            return [f"key_id {self.key_id!r} does not match the carried public "
                    f"key (expected {expected!r}); the verifying key was substituted"]
        outcome = integrity.verify(self.signing_bytes(), self.signature,
                                   self.public_key)
        if not getattr(outcome, "valid", False):
            return [f"attestation signature does not verify: "
                    f"{getattr(outcome, 'reason', 'unknown reason')}"]
        return []

    def measurement_problems(self, measurement: Any) -> List[str]:
        """Whether this attestation was made for `measurement`.

        This is the binding that stops one real execution being used to vouch
        for a different set of numbers, which a signature alone does not
        prevent: a valid signature proves who signed, not that the signed
        payload describes the result now being claimed.
        """
        if not isinstance(measurement, dict) or not measurement:
            return ["there is no measurement for the attestation to bind to"]
        if self.measurement_sha256 != digest_of(measurement):
            return ["attestation measurement_sha256 does not match the "
                    "measurement supplied; it was signed for a different result"]
        return []

    def rehash_artifacts(self, paths: Dict[str, Any]) -> List[str]:
        """Recompute recorded artifact digests against real paths.

        Not called by `admit()`: hashing multi-gigabyte checkpoints on every
        admission would be unusable, and the boundary has no way to know which
        store to trust. This is exposed instead so that a verifier holding the
        artifact store can prove the recorded digests still describe the bytes
        on disk. An empty list means every named artifact matched.
        """
        mismatches: List[str] = []
        recorded = {label: value for label, value in self.artifact_bindings}
        for label, path in (paths or {}).items():
            expected = recorded.get(label)
            if expected is None:
                mismatches.append(
                    f"artifact {label!r} was presented but the attestation does "
                    f"not record it")
                continue
            try:
                actual = sha256_file(path)
            except OSError as exc:
                mismatches.append(f"artifact {label!r} could not be read: {exc}")
                continue
            if actual != expected:
                mismatches.append(
                    f"artifact {label!r} hashes to {actual} but the attestation "
                    f"recorded {expected}")
        return mismatches

    def verifies(self, measurement: Any = None) -> bool:
        if self.problems():
            return False
        if measurement is not None and self.measurement_problems(measurement):
            return False
        return True

    def to_dict(self) -> Dict[str, Any]:
        data = self.payload()
        data["signature"] = self.signature
        data["public_key"] = self.public_key
        return data


@dataclass(frozen=True)
class EvidenceResult:
    """A scientific result that has passed (or failed) the boundary.

    Deliberately not a dict, and deliberately not constructible by callers.
    Algorithms return measurements; they do not get to hand callers a payload
    that looks like a finding. Only :class:`EvidenceBoundary` may build one, so
    the provenance and eligibility on this record are decisions rather than
    assertions.
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
    _capability: Any = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._capability is not _BOUNDARY_TOKEN:
            raise PermissionError(
                "EvidenceResult cannot be constructed directly. Its provenance "
                "and eligibility are decisions the boundary makes, so building "
                "one here would defeat the point of having a boundary. Use "
                "EvidenceBoundary.admit(), .unavailable() or .error().")
        if self.provenance == _PUBLISHABLE_PROVENANCE:
            issues = self.attestation_problems()
            if issues:
                raise ValueError(
                    "A live EvidenceResult requires a signed attestation bound "
                    "to this exact measurement: " + "; ".join(issues))

    _ATTESTATION_FIELDS = frozenset(RunAttestation.__dataclass_fields__)

    def as_attestation(self) -> Optional[RunAttestation]:
        """Rehydrate the stored attestation so it can be verified again."""
        if not isinstance(self.attestation, dict):
            return None
        usable = {k: v for k, v in self.attestation.items()
                  if k in self._ATTESTATION_FIELDS}
        try:
            return RunAttestation(**usable)
        except TypeError:
            return None

    def attestation_problems(self) -> List[str]:
        attestation = self.as_attestation()
        if attestation is None:
            return ["no verifiable attestation is recorded on this result"]
        return attestation.problems() + attestation.measurement_problems(
            self.measurement)

    def _live_but_unverified(self) -> bool:
        """Whether this claims to be live but cannot show why.

        Checked in `__post_init__` *and* in `publishable`, because a record
        whose attestation later fails to verify must stop being publishable even
        though the boundary admitted it once.
        """
        if self.provenance != _PUBLISHABLE_PROVENANCE:
            return False
        return not self.attestation_verified

    @property
    def attestation_verified(self) -> bool:
        return not self.attestation_problems()

    @property
    def publishable(self) -> bool:
        """True only for a live, completed, verified, explicitly eligible result."""
        return (
            self.status == "completed"
            and self.provenance == _PUBLISHABLE_PROVENANCE
            and self.eligibility.get("publication_eligible") is True
            and not self._live_but_unverified()
        )

    @property
    def live(self) -> bool:
        return self.provenance == _PUBLISHABLE_PROVENANCE

    def field_provenance(self, fields: tuple) -> Dict[str, str]:
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
                _capability=_BOUNDARY_TOKEN,
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
                _capability=_BOUNDARY_TOKEN,
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
                _capability=_BOUNDARY_TOKEN,
            )

        # Checked before the binding, not after: an executor that attested a run
        # but handed back nothing has measured nothing, and that is the useful
        # thing to say. Verifying the binding first would report a mismatch
        # against an empty measurement instead.
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
                _capability=_BOUNDARY_TOKEN,
            )

        problems = attestation.problems()
        if not problems:
            problems = attestation.measurement_problems(measurement)
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
                reason=("Live provenance rejected: " + "; ".join(problems)),
                attestation=attestation.to_dict(),
                _capability=_BOUNDARY_TOKEN,
            )

        claimed_execution, replay_reason = LEDGER.claim(
            attestation.execution_id, attestation.measurement_sha256)
        if not claimed_execution:
            return EvidenceResult(
                status="blocked",
                provenance="unavailable",
                executor_id=executor_id,
                run_id=attestation.run_id,
                model_id=attestation.model_id,
                measurement=measurement,
                eligibility={"validation_eligible": False,
                             "publication_eligible": False},
                reason=("Live provenance rejected: " + replay_reason),
                attestation=attestation.to_dict(),
                _capability=_BOUNDARY_TOKEN,
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
            _capability=_BOUNDARY_TOKEN,
        )

    @staticmethod
    def unavailable(executor_id: str, reason: str,
                    **kw: Any) -> EvidenceResult:
        """The honest answer when an executor cannot measure."""
        kw.pop("_capability", None)
        return EvidenceResult(
            status="blocked",
            provenance="unavailable",
            executor_id=executor_id,
            reason=reason,
            eligibility={"validation_eligible": False,
                         "publication_eligible": False},
            _capability=_BOUNDARY_TOKEN,
            **kw,
        )

    @staticmethod
    def error(executor_id: str, reason: str, **kw: Any) -> EvidenceResult:
        kw.pop("_capability", None)
        return EvidenceResult(
            status="error",
            provenance="unavailable",
            executor_id=executor_id,
            reason=reason,
            eligibility={"validation_eligible": False,
                         "publication_eligible": False},
            _capability=_BOUNDARY_TOKEN,
            **kw,
        )

    @staticmethod
    def verify_artifacts(result: EvidenceResult,
                         paths: Dict[str, Any]) -> List[str]:
        """Recompute a result's recorded artifact digests against real files.

        Kept out of `admit()` because the boundary cannot know which artifact
        store to trust, and hashing checkpoints on every admission is not
        viable. Exposed so that whoever *does* hold the store can finish the
        job; an empty list means the bytes on disk match what was attested.
        """
        attestation = result.as_attestation()
        if attestation is None:
            return ["result carries no attestation to re-verify"]
        return attestation.rehash_artifacts(paths)


BOUNDARY = EvidenceBoundary()