"""Scientific Response Envelope and Cryptographic Provenance Verification.

Ensures that every scientific endpoint in MECH returns an epistemically grounded envelope
with cryptographic manifest verification and tamper detection. Fails closed if provenance
is missing, unexecuted, or altered.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from backend.storage.scientific_entities import KnowledgeType

logger = logging.getLogger("MECH.science.envelope")


class ProvenanceMetadata(BaseModel):
    model_version: str = "gpt2"
    model_hash: str = ""
    dataset_version: str = "1.0.0"
    dataset_hash: str = ""
    code_version: str = "2.0.0"
    hardware: str = "cpu"
    environment: str = "PyTorch 2.4.0 (Live Hooks)"
    timestamp: float = Field(default_factory=time.time)
    manifest_sha256: str = ""


class ScientificResponseEnvelope(BaseModel):
    status: str = "SUCCESS"  # SUCCESS, UNEXECUTED_EXPERIMENT, PROVENANCE_INVALID, FAILED_RESOURCE_LIMIT, LOAD_ERROR
    data: Optional[Dict[str, Any]] = None
    manifest_id: Optional[str] = None
    manifest_sha256: Optional[str] = None
    knowledge_type: KnowledgeType = KnowledgeType.CAUSAL_EVIDENCE
    provenance: Optional[ProvenanceMetadata] = None
    integrity_status: str = "VERIFIED"  # VERIFIED, UNVERIFIED, TAMPER_DETECTED
    error: Optional[str] = None
    timestamp: float = Field(default_factory=time.time)


def compute_manifest_hash(payload: Dict[str, Any]) -> str:
    """Computes a deterministic SHA-256 hash over an experiment payload."""
    raw = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def wrap_scientific_success(
    data: Dict[str, Any],
    manifest_id: str,
    manifest_payload: Dict[str, Any],
    knowledge_type: KnowledgeType = KnowledgeType.CAUSAL_EVIDENCE,
    model_version: str = "gpt2",
    dataset_version: str = "1.0.0",
) -> ScientificResponseEnvelope:
    """Wraps scientific output in a verified envelope with manifest hash."""
    manifest_sha = compute_manifest_hash(manifest_payload)

    provenance = ProvenanceMetadata(
        model_version=model_version,
        dataset_version=dataset_version,
        manifest_sha256=manifest_sha,
    )

    return ScientificResponseEnvelope(
        status="SUCCESS",
        data=data,
        manifest_id=manifest_id,
        manifest_sha256=manifest_sha,
        knowledge_type=knowledge_type,
        provenance=provenance,
        integrity_status="VERIFIED",
    )


def wrap_unexecuted_failure(reason: str) -> ScientificResponseEnvelope:
    """Fails closed when an experiment has not been executed on live model weights."""
    return ScientificResponseEnvelope(
        status="UNEXECUTED_EXPERIMENT",
        data=None,
        integrity_status="UNVERIFIED",
        error=f"Epistemic Gate Violation: {reason}",
    )


def wrap_tamper_failure(reason: str) -> ScientificResponseEnvelope:
    """Fails closed when post-execution tampering is detected."""
    return ScientificResponseEnvelope(
        status="PROVENANCE_INVALID",
        data=None,
        integrity_status="TAMPER_DETECTED",
        error=f"Cryptographic Integrity Failure: {reason}",
    )


def verify_manifest_integrity(stored_manifest: Dict[str, Any], expected_sha256: str) -> bool:
    """Verifies that the stored manifest matches the cryptographic hash recorded during execution."""
    recalculated = compute_manifest_hash(stored_manifest)
    return recalculated == expected_sha256
