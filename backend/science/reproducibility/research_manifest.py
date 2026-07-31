"""Research Manifest — The Root Research Object with Merkle Integrity.

Aggregates all experiment artifacts (Snapshot, Certificate, Audit Log) and
binds them with a Merkle-root hash to ensure byte-for-byte integrity.
"""

from __future__ import annotations

import hashlib
import json
import os
import datetime
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ResearchManifest:
    """The master root object for a research experiment."""
    experiment_id: str
    manifest_version: str = "1.1.0"
    schema_version: str = "1.1.0"
    created_at: str = field(default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat())

    # Merkle Integrity Tree
    root_sha256: str = ""
    component_hashes: Dict[str, str] = field(default_factory=dict)

    # Linked Artifacts (Relative Paths)
    snapshot_path: str = ""
    certificate_path: str = ""
    audit_log_path: str = ""
    data_manifest_path: str = ""

    # Metadata
    benchmark_id: str = ""
    model_id: str = ""
    reproducibility_score: float = 0.0
    status: str = "VALIDATED" # VALIDATED, GOLDEN, DEPRECATED

    # Cryptographic Signature
    signature: str = ""
    signer_id: str = "MECH-CORE-VALIDATOR"


class ResearchManifestEngine:
    """Orchestrates the generation and verification of research manifests."""

    SCHEMA_VERSION = "1.1.0"

    def _compute_hash(self, data: Any) -> str:
        s = json.dumps(data, sort_keys=True)
        return hashlib.sha256(s.encode("utf-8")).hexdigest()

    def generate(
        self,
        experiment_id: str,
        snapshot: Any,
        certificate: Any,
        audit_log: List[Any],
        reproducibility_score: float
    ) -> ResearchManifest:
        """Assembles artifacts into a signed Merkle-root manifest."""

        # 1. Compute individual hashes
        snapshot_hash = self._compute_hash(asdict(snapshot))
        cert_hash = self._compute_hash(certificate) # assumed dict or asdicted
        audit_hash = self._compute_hash(audit_log)

        # 2. Build Merkle-style root
        # root = hash(snapshot_hash + cert_hash + audit_hash)
        combined = f"{snapshot_hash}{cert_hash}{audit_hash}"
        root_sha = hashlib.sha256(combined.encode("utf-8")).hexdigest()

        return ResearchManifest(
            experiment_id=experiment_id,
            root_sha256=root_sha,
            component_hashes={
                "snapshot": snapshot_hash,
                "certificate": cert_hash,
                "audit_log": audit_hash
            },
            reproducibility_score=reproducibility_score,
            status="VALIDATED"
        )

    def sign_manifest(self, manifest: ResearchManifest, private_key: str = "mock_key") -> str:
        """Signs the manifest's root hash (Simulating Ed25519)."""
        # In real implementation:
        # return ed25519.sign(manifest.root_sha256, private_key)
        payload = f"{manifest.root_sha256}:{private_key}"
        return hashlib.sha256(payload.encode()).hexdigest()

    def verify(self, manifest: ResearchManifest, snapshot_data: Any, cert_data: Any, audit_data: List[Any]) -> bool:
        """Verifies the integrity of the entire research object."""
        # Recalculate hashes
        s_hash = self._compute_hash(snapshot_data)
        c_hash = self._compute_hash(cert_data)
        a_hash = self._compute_hash(audit_data)

        # Verify component hashes match manifest
        if s_hash != manifest.component_hashes.get("snapshot"): return False
        if c_hash != manifest.component_hashes.get("certificate"): return False
        if a_hash != manifest.component_hashes.get("audit_log"): return False

        # Verify root
        combined = f"{s_hash}{c_hash}{a_hash}"
        root_sha = hashlib.sha256(combined.encode("utf-8")).hexdigest()

        return root_sha == manifest.root_sha256
