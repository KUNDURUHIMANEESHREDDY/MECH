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

import hashlib
import json
import logging
import os
import time
import uuid
from typing import Any, Dict, Iterable, Mapping, Optional, Union

logger = logging.getLogger(__name__)

LIVE = "live"
SEEDED = "seeded"
REFERENCE = "reference"
UNAVAILABLE = "unavailable"

#: A unique ID for this measurement boundary instance. Generated once at import.
#: All measurements from this boundary instance carry this ID for traceability.
_MEASUREMENT_BOUNDARY_ID = uuid.uuid4().hex[:16]

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
    "measurement_record",
})


# Trusted signing key management (shared with evidence_graph)
_TRUSTED_KEY_PATHS_ENV = "MECH_TRUSTED_SIGNING_KEY_PATH"
_TRUSTED_KEY_IDS: Optional[set[str]] = None
#: The env value the cache was built from. The cache is keyed on it so
#: that a changed configuration (key rotation, or tests that re-point
#: the variable) cannot keep trusting keys that are no longer configured
#: — or distrusting keys that now are. Without this, whichever test ran
#: first would decide trust for the whole session.
_TRUSTED_KEY_PATHS_CACHED: Optional[str] = None


def load_trusted_key_ids() -> set[str]:
    """Load trusted signing key IDs from MECH_TRUSTED_SIGNING_KEY_PATH.

    Returns a set of key_id strings (format: "ed25519:<16-char-hex>").
    Returns empty set if env var is not set, meaning no key is trusted
    (fail-closed: no envelope will be attested).
    """
    global _TRUSTED_KEY_IDS, _TRUSTED_KEY_PATHS_CACHED
    paths_env = os.environ.get(_TRUSTED_KEY_PATHS_ENV, "").strip()
    if _TRUSTED_KEY_IDS is not None and _TRUSTED_KEY_PATHS_CACHED == paths_env:
        return _TRUSTED_KEY_IDS
    _TRUSTED_KEY_PATHS_CACHED = paths_env

    if not paths_env:
        _TRUSTED_KEY_IDS = set()
        return _TRUSTED_KEY_IDS

    try:
        from backend.science.integrity import signing as _signing
    except Exception:
        # If signing module unavailable, trust nothing
        _TRUSTED_KEY_IDS = set()
        return _TRUSTED_KEY_IDS

    trusted = set()
    for path_str in paths_env.split(","):
        path_str = path_str.strip()
        if not path_str:
            continue
        try:
            pub_hex = _signing.public_key_hex(path_str)
            key_id = _signing.key_id_for(pub_hex)
            trusted.add(key_id)
        except Exception:
            # Individual bad key is ignored; continue with others
            continue

    _TRUSTED_KEY_IDS = trusted
    return _TRUSTED_KEY_IDS


def reset_trusted_key_ids_cache() -> None:
    """Clear the trusted key IDs cache. Used by tests when env vars change."""
    global _TRUSTED_KEY_IDS, _TRUSTED_KEY_PATHS_CACHED
    _TRUSTED_KEY_IDS = None
    _TRUSTED_KEY_PATHS_CACHED = None


def _canonical_json(data: Any) -> str:
    """Canonical JSON serialization for hashing."""
    return json.dumps(data, sort_keys=True, separators=(",", ":"), default=str)


def _hash_data(data: Any) -> str:
    """SHA-256 hash of canonical JSON, truncated to 32 hex chars."""
    return hashlib.sha256(_canonical_json(data).encode()).hexdigest()[:32]


def _create_measurement_record(
    result: Dict[str, Any],
    model_loaded: Optional[str] = None,
    model_obj: Optional[Any] = None,
    model_content_hash: Optional[str] = None,
    signing_key: Optional[Union[str, bytes, os.PathLike]] = None,
    input_fields: Optional[Iterable[str]] = None,
    output_fields: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    """Create a cryptographically bound measurement record.

    Binds model identity, inputs, outputs, execution context, and boundary ID.
    Signs the record with Ed25519 when a signing key is provided.

    `model_obj` may be any object exposing `model_fingerprint()` (preferred)
    or `model_content_hash()`; either must hash loaded weight tensors, not
    the model name. `model_content_hash` accepts a pre-computed
    weight-derived fingerprint for callers (like the engine) that hash
    weights through another path — it MUST be weight-derived, never a
    name, label, or placeholder.

    `input_fields` / `output_fields` select which retained result keys the
    input / output commitments are computed over. They default to the
    historical auto-detect rules. Whatever basis is used, the sorted key
    lists actually hashed are recorded as `input_keys` / `output_keys` so
    that `verify_payload_commitments` can recompute them later. The
    signature covers the commitments, the key lists, and the rest of the
    record.
    """
    now = time.time()

    # Model identity: name + content hash (from actual weights if available)
    #
    # `content_hash_source` is an honesty label, and the policy treats it
    # as one: `verify_measurement_record` accepts only "model_weights"
    # (hash computed from loaded weight tensors) and "explicit" (a
    # weight-derived fingerprint supplied by the calling measurement
    # layer, e.g. the engine's weights hash). "explicit" is trusted only
    # because the record is signed: without the signing key a caller
    # cannot mint a record at all, so the label cannot launder a forgery.
    # It MUST be weight-derived by the caller — never a name, label, or
    # placeholder. Anything else fails verification.
    model_identity = {"name": model_loaded or "unknown"}
    if model_content_hash:
        model_identity["content_hash"] = model_content_hash
        model_identity["content_hash_source"] = "explicit"
    elif model_obj is not None and hasattr(model_obj, "model_fingerprint"):
        # Get actual model content hash from the model object (GPT2Adapter.model_fingerprint)
        fingerprint = model_obj.model_fingerprint()
        if fingerprint:
            model_identity["content_hash"] = fingerprint
            model_identity["content_hash_source"] = "model_weights"
    elif model_obj is not None and hasattr(model_obj, "model_content_hash"):
        # Get actual model content hash from the model object (GPT2Model.model_content_hash)
        model_identity["content_hash"] = model_obj.model_content_hash()
        model_identity["content_hash_source"] = "model_weights"
    if "content_hash" not in model_identity:
        # Fallback: hash the model name (NOT the actual weights).
        # This proves nothing about which weights ran. It is recorded
        # honestly as a fallback so that display paths still have an
        # identity string, but verification rejects it: a name-derived
        # hash must never satisfy an evidence gate.
        model_identity["content_hash"] = hashlib.sha256(
            (model_loaded or "unknown").encode()
        ).hexdigest()[:32]
        model_identity["content_hash_source"] = "model_name_fallback"

    # Input commitment: explicit retained fields when given, else auto-detect.
    # An explicit list that matches nothing is a caller bug (e.g. a
    # typo'd field name) that would otherwise bind the attestation to an
    # empty basis — vacuous coverage that verifies against any payload.
    # Refuse loudly rather than minting it.
    if input_fields:
        wanted = [str(k) for k in input_fields]
        input_data = {k: result[k] for k in wanted if k in result}
        if not input_data:
            raise ValueError(
                "attestation refused: none of the declared input_fields "
                f"{wanted} is present in the result; a commitment over "
                "nothing binds nothing")
    else:
        # Input commitment: hash of the result's input-relevant fields
        input_data = {k: v for k, v in result.items() if k in ("input", "prompt", "prompt_ids", "tokens")}
        if not input_data:
            # If no explicit input fields, hash the whole result minus framing
            input_data = {k: v for k, v in result.items() if k not in _FRAME_KEYS}
    input_commitment = _hash_data(input_data)

    # Output commitment: explicit retained fields when given, else auto-detect.
    # Same fail-closed rule as inputs: a declared basis that matches
    # nothing is refused rather than hashed empty.
    if output_fields:
        wanted = [str(k) for k in output_fields]
        output_data = {k: result[k] for k in wanted if k in result}
        if not output_data:
            raise ValueError(
                "attestation refused: none of the declared output_fields "
                f"{wanted} is present in the result; a commitment over "
                "nothing binds nothing")
    else:
        # Output commitment: hash of output-relevant fields
        output_data = {k: v for k, v in result.items() if k in ("output", "result", "value", "predictions", "logits", "activations")}
        if not output_data:
            output_data = {k: v for k, v in result.items() if k not in _FRAME_KEYS}
    output_commitment = _hash_data(output_data)

    record = {
        "model_identity": model_identity,
        "input_commitment": input_commitment,
        "output_commitment": output_commitment,
        # The exact basis each commitment was computed over. Recomputation
        # reads these lists, so post-attest additions to the payload (status
        # flags, verdicts, eligibility) do not disturb verification, while
        # any change to a covered field breaks it.
        "input_keys": sorted(input_data.keys()),
        "output_keys": sorted(output_data.keys()),
        "execution_context": {
            "timestamp": now,
            "measurement_boundary_id": _MEASUREMENT_BOUNDARY_ID,
        },
    }

    # Sign the measurement record if a signing key is provided. A failure
    # here is loud (a misconfigured MECH_SIGNING_KEY_PATH must not look
    # like "signing disabled") but stays fail-closed downstream:
    # verification requires a valid trusted-key signature, so an
    # unsigned record can never satisfy an evidence gate.
    if signing_key is not None:
        try:
            from backend.science.integrity import signing as _signing
            payload = _canonical_json(record)
            signature = _signing.sign(payload, signing_key)
            public_key = _signing.public_key_hex(signing_key)
            key_id = _signing.key_id_for(public_key)
            record["signature"] = signature
            record["public_key"] = public_key
            record["key_id"] = key_id
        except Exception as exc:
            logger.warning("measurement-record signing failed: %s %s",
                           type(exc).__name__, exc)

    return record


def verify_measurement_record(record: Dict[str, Any], trusted_key_ids: Optional[set[str]] = None) -> Dict[str, Any]:
    """Verify a measurement record's cryptographic binding and signature.

    Returns a dict with:
    - valid: bool - whether all checks pass
    - checks: dict - individual check results
    - reason: str - explanation if invalid
    """
    checks = {}
    
    # Check 1: Required fields present
    required = ("model_identity", "input_commitment", "output_commitment", "execution_context")
    checks["required_fields"] = all(k in record for k in required)
    if not checks["required_fields"]:
        return {"valid": False, "checks": checks, "reason": "missing required fields"}

    # Check 2: Model identity has a WEIGHT-derived content hash.
    #
    # A hash of the model NAME proves nothing about which weights ran: a
    # swapped checkpoint keeps its model id, keeps its shapes, and passes
    # every tensor-level check while the fingerprint changes. Only
    # "model_weights" (computed from loaded weight tensors) and "explicit"
    # (a weight-derived fingerprint supplied by the calling measurement
    # layer, bound by the record signature) are accepted. "explicit" is
    # trusted only together with a valid trusted-key signature — without
    # the signing key no caller can mint a record at all.
    model_identity = record.get("model_identity", {})
    checks["has_content_hash"] = "content_hash" in model_identity
    checks["content_hash_source"] = model_identity.get("content_hash_source", "unknown")
    checks["content_hash_authentic"] = checks["content_hash_source"] in (
        "model_weights", "explicit")

    # Check 3: Execution context has boundary ID
    exec_ctx = record.get("execution_context", {})
    checks["has_boundary_id"] = "measurement_boundary_id" in exec_ctx

    # Check 4: Signature verification (if present)
    checks["has_signature"] = "signature" in record and "public_key" in record
    if checks["has_signature"]:
        try:
            from backend.science.integrity import signing as _signing
            # Verify signature against the record payload
            payload = _canonical_json({
                k: v for k, v in record.items() 
                if k not in ("signature", "public_key", "key_id")
            })
            result = _signing.verify(payload, record["signature"], record["public_key"])
            checks["signature_valid"] = result.valid
            checks["signature_reason"] = result.reason
            
            # Check if key is trusted
            if trusted_key_ids is not None:
                key_id = record.get("key_id")
                checks["key_trusted"] = key_id in trusted_key_ids if key_id else False
            else:
                checks["key_trusted"] = False
        except Exception as e:
            checks["signature_valid"] = False
            checks["signature_reason"] = f"verification error: {e}"
    else:
        checks["signature_valid"] = False
        checks["key_trusted"] = False

    # Overall validity. `key_trusted` is load-bearing, not advisory: a
    # valid signature under a key nobody trusts is a self-signed forgery,
    # and without a configured trust set nothing can be trusted
    # (fail-closed). Likewise a name-derived content hash authenticates
    # nothing about the weights.
    all_checks_pass = (
        checks["required_fields"] and
        checks["has_content_hash"] and
        checks["content_hash_authentic"] and
        checks["has_boundary_id"] and
        checks["signature_valid"] and
        checks.get("key_trusted", False)
    )
    
    if not all_checks_pass:
        failed = [k for k, v in checks.items() if v is False]
        reason = f"failed checks: {', '.join(failed)}"
        if not checks.get("signature_valid", True):
            reason += f"; signature: {checks.get('signature_reason', 'missing')}"
    else:
        reason = None

    return {"valid": all_checks_pass, "checks": checks, "reason": reason}


def verify_payload_commitments(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Recompute a payload's commitments against its measurement record.

    The record's signature proves the commitments are authentic; this
    proves they describe THIS payload. Together they close the loop:
    a self-consistent forgery fails the signature/trust check, while a
    genuine record whose outputs were altered afterwards fails here.

    The basis for each commitment is the record's own `input_keys` /
    `output_keys` (the sorted key lists actually hashed at creation), so
    post-attest additions outside the basis (eligibility flags, verdicts)
    do not disturb verification, while any change to a covered field —
    or its removal — breaks it.

    Derived payloads (e.g. the Critic's reproduction output, which maps
    rather than re-measures the pipeline run) carry the originating run
    under `pipeline_result`. When the top level lacks a basis key, the
    check falls back to that nested run — but a top-level value, when
    present, is always what is checked, so a derived payload cannot
    smuggle altered values past a genuine nested run.
    """
    if not isinstance(payload, dict):
        return {"valid": False, "reason": "payload is not a dict"}
    mr = payload.get("measurement_record")
    if not isinstance(mr, dict):
        return {"valid": False, "reason": "no measurement record to verify against"}
    input_keys = mr.get("input_keys")
    output_keys = mr.get("output_keys")
    if not isinstance(input_keys, list) or not isinstance(output_keys, list):
        return {"valid": False,
                "reason": "commitment basis not recorded (legacy record)"}

    nested = payload.get("pipeline_result")
    nested = nested if isinstance(nested, dict) else None

    def _basis(keys: list) -> Optional[Dict[str, Any]]:
        if all(k in payload for k in keys):
            return {k: payload[k] for k in keys}
        if nested is not None and all(k in nested for k in keys):
            return {k: nested[k] for k in keys}
        return None

    want_in = _basis(input_keys)
    if want_in is None:
        return {"valid": False,
                "reason": "input commitment basis missing from payload"}
    if _hash_data(want_in) != mr.get("input_commitment"):
        return {"valid": False,
                "reason": "input commitment does not match payload"}
    want_out = _basis(output_keys)
    if want_out is None:
        return {"valid": False,
                "reason": "output commitment basis missing from payload"}
    if _hash_data(want_out) != mr.get("output_commitment"):
        return {"valid": False,
                "reason": "output commitment does not match payload"}
    return {"valid": True, "reason": None}


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
    model_obj: Optional[Any] = None,
    model_content_hash: Optional[str] = None,
    signing_key: Optional[Union[str, bytes, os.PathLike, bool]] = None,
    fields: Optional[Iterable[str]] = None,
    input_fields: Optional[Iterable[str]] = None,
    output_fields: Optional[Iterable[str]] = None,
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

    Args:
        model_obj: Optional model object exposing `model_fingerprint()` or
            `model_content_hash()`. Either method must hash loaded weight
            tensors, not the model name.
        model_content_hash: Pre-computed model content hash (hex string).
            MUST be a fingerprint of actually-loaded weight artifacts, never
            a hash of a model name or other label. It is recorded with source
            "explicit" and bound by the record signature.
        signing_key: Optional Ed25519 signing key (path, bytes, or PEM). If not
            provided, reads from MECH_SIGNING_KEY_PATH environment variable.
            Set to `False` explicitly to disable signing.
        input_fields: Retained result keys the input commitment is computed
            over. Defaults to the auto-detect rules.
        output_fields: Retained result keys the output commitment is computed
            over. Defaults to the auto-detect rules.
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

    # Resolve signing key: explicit arg > env var > disabled
    if signing_key is None:
        signing_key = os.environ.get("MECH_SIGNING_KEY_PATH") or None
    elif signing_key is False:
        signing_key = None

    # Create and attach the cryptographically bound measurement record.
    # Any caller-supplied record is overwritten, never trusted: the only
    # measurement_record a verifier will ever see is one this boundary made.
    result["measurement_record"] = _create_measurement_record(
        result,
        model_loaded=model_loaded,
        model_obj=model_obj,
        model_content_hash=model_content_hash,
        signing_key=signing_key,
        input_fields=input_fields,
        output_fields=output_fields,
    )

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
    # A caller-supplied measurement record is stripped, not kept: keeping
    # it would leave a signed-looking (or signed) record on a payload this
    # boundary just refused to vouch for, and a downstream reader that
    # checks the record without checking `attested` could mistake it for
    # evidence.
    result["attested"] = False
    result.pop("measurement_record", None)
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


# --------------------------------------------------------------------------- #
# Admission: the single gate a Society result passes through.
# --------------------------------------------------------------------------- #

#: The stages the admission gate rules on. Each stage has exactly one
#: predicate, and the predicate lives in `backend.agents.evidence_policy`, so
#: the rule is stated once and no caller can supply its own.
ADMISSION_STAGES = ("discovery", "validation", "reproduction", "gate",
                    "publication")

#: How each stage names itself in a refusal. These are the labels
#: `evidence_policy.publication_block_reason` already uses, so a refusal
#: reads the same whichever path produced it.
_ADMISSION_LABELS = {
    "discovery": "Discovery",
    "validation": "Validation",
    "reproduction": "Reproduction",
    "gate": "Validation gate",
    "publication": "Publication",
}


def _evidence_policy():
    """The policy module, imported lazily: it imports this one."""
    try:
        from backend.agents import evidence_policy
    except ImportError:  # pragma: no cover - the packaged layout
        from agents import evidence_policy
    return evidence_policy


def _stage_admitted(policy, stage: str, record: Dict[str, Any]) -> bool:
    """Whether `record` clears the rule that applies to `stage`."""
    if stage == "discovery":
        return bool(policy.discovery_is_live(record))
    if stage == "validation":
        return bool(policy.validation_is_live(record))
    if stage == "reproduction":
        return bool(policy.reproduction_is_live(record))
    if stage == "gate":
        return bool(policy.gate_is_live(record))
    return False


def _evidence_steps(trace: Any) -> list:
    """The evidence-bearing steps of a trace, with their results resolved.

    A trace step stores the producing result under `result` when it has one
    and spreads it over the step otherwise, so both shapes are read here --
    the same resolution `evidence_policy.publication_block_reason` uses.
    """
    steps = []
    for step in trace or ():
        if not isinstance(step, dict):
            continue
        if str(step.get("node", "")) not in ("discover", "validate"):
            continue
        result = step.get("result")
        if not isinstance(result, dict) or not result:
            result = step
        steps.append(result)
    return steps


def _chain_provenance(policy, trace: Any) -> str:
    """The provenance of an evidence chain: its weakest evidence step."""
    labels = [policy.provenance_of(result)
              for result in _evidence_steps(trace)]
    if not labels:
        return UNAVAILABLE
    if all(label == LIVE for label in labels):
        return LIVE
    order = {LIVE: 0, SEEDED: 1, REFERENCE: 2}
    return max(labels, key=lambda label: order.get(label, 3))


def _chain_attested(policy, trace: Any) -> bool:
    """Whether every evidence-bearing step of a chain is attested."""
    results = _evidence_steps(trace)
    if not results:
        return False
    return all(policy.is_attested(result) for result in results)


def admit_result(source: Any, *, stage: str,
                 extra: Optional[Mapping[str, Any]] = None,
                 fields: Optional[Iterable[str]] = None,
                 trace: Any = None, reproducibility: Any = None,
                 gate: Any = None, status: Optional[str] = None,
                 reason: Optional[str] = None,
                 refusal: Optional[str] = None) -> Dict[str, Any]:
    """Derive a result's standing. The single admission gate.

    The Society agents each built their own envelope, so three places
    decided what "live" and "eligible" meant and a caller could assert
    either. Here all three are *derived*:

    * ``provenance`` is the label the record itself carries;
    * ``attested`` is the answer of the cryptographic check the evidence
      policy applies -- never the record's own boolean;
    * ``validation_eligible`` and ``publication_eligible`` are the stage
      predicate's answer, which requires that same cryptographic check.

    ``extra`` is merged first and the derived keys are written last, so an
    assertion inside it is overwritten rather than honoured: admission can
    only ever downgrade a claim, never promote one.

    ``status`` and ``reason`` may explain a *refusal* -- an error path, or a
    more specific reason than the default. They are ignored when the result
    is admitted, so they can never explain an admission.

    ``refusal`` forces the result to be refused even when the stage
    predicate would have admitted it. It is for the case where the source
    was a real measurement but this envelope is not one -- a live run whose
    metrics could not be mapped, say. It can only ever lower a result's
    standing.
    """
    policy = _evidence_policy()
    stage_name = str(stage or "").strip().lower()
    if stage_name not in ADMISSION_STAGES:
        raise ValueError(
            f"unknown admission stage {stage!r}; expected one of "
            f"{', '.join(ADMISSION_STAGES)}")

    record = source if isinstance(source, dict) else {}

    if stage_name == "publication":
        refusal = policy.publication_block_reason(trace, reproducibility, gate)
        admitted = not refusal
        provenance = (_chain_provenance(policy, trace) if admitted
                      else UNAVAILABLE)
        attested = _chain_attested(policy, trace) if admitted else False
        default_reason = refusal
    else:
        admitted = _stage_admitted(policy, stage_name, record)
        provenance = policy.provenance_of(record)
        attested = policy.is_attested(record)
        default_reason = ("" if admitted else
                          policy.blocked_reason(
                              record, _ADMISSION_LABELS[stage_name]))

    # A run chain that could not be completed is a refusal whatever the
    # record says about itself. The measurement may well have happened, but
    # it is not bound to a run, and an unbound chain must not be promoted.
    if admitted and not policy.run_chain_allows(record):
        chain = record.get("run_attestation")
        detail = ""
        if isinstance(chain, dict):
            detail = str(chain.get("attestation_reason") or "").strip()
        admitted = False
        attested = False
        default_reason = ("the run chain is not attested"
                          + (f": {detail}" if detail else ""))

    # A forced refusal: the source may have been a real measurement, but this
    # envelope is not one. The label drops to unavailable, because the
    # envelope -- not the source -- is what a reader sees.
    if refusal:
        admitted = False
        attested = False
        provenance = UNAVAILABLE
        default_reason = str(refusal)

    envelope: Dict[str, Any] = {}
    if isinstance(extra, Mapping):
        envelope.update(extra)

    envelope["provenance"] = provenance
    envelope["attested"] = bool(attested)
    envelope["validation_eligible"] = bool(admitted)
    envelope["publication_eligible"] = bool(admitted)
    envelope["status"] = ("completed" if admitted else str(
        status or ("blocked" if stage_name == "publication" else "unavailable")))
    if not admitted:
        envelope["reason"] = str(
            reason or default_reason
            or (f"{_ADMISSION_LABELS[stage_name]} blocked: the evidence chain "
                "did not clear the gate."))

    names = list(fields) if fields is not None else list(envelope)
    envelope["field_provenance"] = policy.field_map(names, provenance)
    return envelope


def attest_run_chain(result: Dict[str, Any], *, run_id: str, dataset: Any,
                     model_fingerprint: Any = None,
                     repo_root: Optional[Any] = None) -> Dict[str, Any]:
    """Bind a result to the run that produced it, and refuse it if the chain
    cannot be completed.

    The result keeps its own provenance and eligibility; the run attestation
    is added beside them. When the chain is incomplete the result is
    withheld rather than published beside a digest that was never computed:
    an unattested chain is not eligible for publication.
    """
    from backend.science.reproducibility.run_attestation import attest_run

    if not isinstance(result, dict):
        return result
    attestation = attest_run(run_id=run_id, dataset=dataset,
                             model_fingerprint=model_fingerprint,
                             repo_root=repo_root)
    result["run_attestation"] = attestation.to_dict()
    if not attestation.attested:
        return withhold(result, reason=(
            "the run chain is not attested: "
            + (attestation.attestation_reason or "the chain is incomplete")))
    return result