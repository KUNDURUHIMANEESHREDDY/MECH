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

from backend.science.integrity import sign, verify


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
    #: Derived by `derive_manifest_status`. Not a constant: a manifest that
    #: records a FAIL or lacks a signature says so.
    status: str = "UNVALIDATED"  # VALIDATED, UNSIGNED, REVISION_REQUIRED, UNVALIDATED
    #: The statistical verdict this status was derived from. Kept so a later
    #: re-derivation (e.g. after signing) uses the real verdict instead of
    #: assuming one.
    statistical_verdict: str = ""

    # Cryptographic Signature
    signature: str = ""
    #: Which algorithm produced `signature`. Empty means unsigned -- previously
    #: there was no way to tell an Ed25519 signature from a keyed SHA-256, because
    #: neither was recorded and only one of them was a signature.
    signature_algorithm: str = ""
    signer_id: str = "MECH-CORE-VALIDATOR"
    #: Whether the manifest is signed, and if not, why. An unsigned manifest is a
    #: legitimate outcome when no key was supplied; what is not legitimate is
    #: reporting a signature that was never made.
    signature_status: Dict[str, Any] = field(default_factory=dict)


def derive_manifest_status(verdict: str, reproducibility_score: float,
                           signed: bool) -> str:
    """Derive a manifest's status from what actually happened.

    The status used to be the literal `"VALIDATED"` in every case, whatever the
    benchmark verdict was and whether or not a signature existed. Four statuses,
    each naming a distinct real outcome:

    ``VALIDATED``    verdict PASS, high score, signature present
    ``UNSIGNED``     verdict PASS but no signature -- attestable, not attested
    ``REVISION_REQUIRED`` verdict was REVISION_REQUIRED
    ``UNVALIDATED``  verdict FAIL, absent, or not supplied
    """
    if verdict == "FAIL" or not verdict:
        return "UNVALIDATED"
    if verdict != "PASS":
        return "REVISION_REQUIRED"
    if not signed:
        return "UNSIGNED"
    return "VALIDATED" if reproducibility_score >= 80.0 else "REVISION_REQUIRED"


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
        reproducibility_score: float,
        verdict: str = "",
    ) -> ResearchManifest:
        """Assemble artifacts into a Merkle-root manifest.

        `verdict` is the statistical verdict (`PASS`, `REVISION_REQUIRED`,
        `FAIL`) from the validator. It is used to derive `status`, which was
        hardcoded to `"VALIDATED"` -- so a manifest whose benchmark FAILED, or
        which was never signed, was still labelled VALIDATED. The label was an
        assertion the pipeline made about itself rather than a summary of
        anything it measured.
        """

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
            status=derive_manifest_status(
                verdict=verdict,
                reproducibility_score=reproducibility_score,
                signed=False,
            ),
            statistical_verdict=verdict,
        )

    def sign_manifest(self, manifest: ResearchManifest,
                      private_key: Any = None) -> str:
        """Sign the manifest's root hash with Ed25519.

        This was `sha256(f"{root_sha256}:{private_key}")` with
        `private_key="mock_key"`, under a docstring reading "Simulating
        Ed25519" and a comment showing the one real line that would have been
        needed. A keyed hash is a MAC, not a signature: verifying it requires the
        secret, and the secret was a default argument in the source.

        `private_key` may instead come from `MECH_SIGNING_KEY_PATH`. There is no
        default key, so this raises rather than producing a signature that
        attests to nothing. `scientific_validator.generate_validation_artifacts`
        called the old one-argument form, so it is updated in the same change to
        pass a key through and to record when none was available.
        """
        signature = sign(manifest.root_sha256, private_key)
        manifest.signature = signature
        manifest.signature_algorithm = "Ed25519"
        manifest.signature_status = {
            "signed": True,
            "algorithm": "Ed25519",
            "reason": None,
        }
        # Promote UNSIGNED -> VALIDATED now that a signature exists. Uses the
        # verdict recorded at generation time rather than assuming PASS, so a
        # signature cannot upgrade a manifest whose benchmark failed.
        if manifest.status == "UNSIGNED":
            manifest.status = derive_manifest_status(
                verdict=manifest.statistical_verdict,
                reproducibility_score=manifest.reproducibility_score,
                signed=True,
            )
        return signature

    def verify_manifest_signature(self, manifest: ResearchManifest,
                                  public_key: Any) -> Dict[str, Any]:
        """Verify the manifest signature using the public key alone.

        Returns the `VerificationResult` dict, so "unsigned", "no key" and
        "wrong signature" remain distinguishable.
        """
        return verify(manifest.root_sha256, manifest.signature,
                      public_key).to_dict()

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
