"""Dataset Promotion Wizard — transitions a dataset to GOLDEN status.

GOLDEN means the dataset is attested: hashes recorded and matching, lineage
recorded, and signed by a key held outside the repository. This script refuses to
issue that status unless it can actually attest, because a GOLDEN marker that
was applied without a signature is worse than no marker -- it tells the next
reader the work was done when it was not.

Workflow:
  1. Verify manifest hashes, reporting each one.
  2. Validate lineage & provenance metadata.
  3. Sign with Ed25519, then verify with the public key **before** promoting.
  4. Issue the dataset certificate.
  5. Export the reproduction bundle.
  6. Write the audit event.

Signing keys are never defaulted. Supply one of:
  * a key path as argv[3], or
  * ``MECH_SIGNING_KEY_PATH`` pointing at a file outside the repository.

Usage:
    python frontend/scripts/promote_dataset.py <dataset_id> <author> [key_path]

Note on dataset ids: `golden_manifest.json` registers the dataset under the
canonical id ``IOI-Canonical-100`` while the file lives in ``datasets/ioi/``.
Both spellings now resolve to the same dataset, and the signing payload uses the
canonical id, so signing via one and verifying via the other agree.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import os
import sys

sys.path.insert(0, os.getcwd())

from backend.research_datasets.dataset_manager import DatasetManager
from backend.research_datasets.dataset_certificate import DatasetCertificateEngine
from backend.research_datasets.dataset_exporter import DatasetExporter
from backend.science.integrity import SigningUnavailable, sign, verify


def _key_path(explicit: str | None) -> str | None:
    return explicit or os.environ.get("MECH_SIGNING_KEY_PATH")


#: Module-level so a test can promote within a temporary copy. See the note in
#: `verify_golden_datasets.py` on why this is not an environment variable.
DATA_DIR = "backend/research_datasets"


def promote_dataset(dataset_id: str, author: str, key_path: str | None = None) -> int:
    print(f"Executing Promotion Workflow for: {dataset_id}")
    manager = DatasetManager(data_dir=DATA_DIR)
    cert_engine = DatasetCertificateEngine()
    exporter = DatasetExporter()

    # 1. Verify. Report each hash: the previous `except` printed "FAIL" and
    #    returned, while success printed a bare "PASS" even when the manifest
    #    recorded no hash at all.
    try:
        manager.load(dataset_id)
    except Exception as exc:
        print(f"  - Integrity: FAILED ({exc})")
        print("PROMOTION ABORTED.")
        return 1

    status = manager.last_integrity_status
    for name, state in sorted(status.get("checks", {}).items()):
        print(f"  - hash {name}: {state}")
    if not status.get("integrity_verified"):
        print("  - Integrity: NOT VERIFIED (checks were bypassed)")
        print("PROMOTION ABORTED: a bypassed load must not become GOLDEN.")
        return 1

    unrecorded = sorted(n for n, s in status.get("checks", {}).items()
                        if s == "not_recorded")
    if unrecorded:
        print(f"  - Integrity: no hash recorded for {', '.join(unrecorded)}")
        print("PROMOTION ABORTED: GOLDEN requires recorded, matching hashes. "
              "Record them in golden_manifest.json first.")
        return 1

    # 2. Metadata.
    meta = manager._manifest.get(dataset_id)
    if not meta or not meta.get("lineage") or not meta.get("provenance"):
        print("  - Metadata: FAILED (missing lineage/provenance)")
        print("PROMOTION ABORTED.")
        return 1
    print("  - Metadata: OK")

    # 3. Sign, then verify with the public key before touching the status.
    #    The old version called `sign_dataset(dataset_id)` with no key, which
    #    signed with a literal secret from the source and printed "Signature
    #    Applied" -- then set GOLDEN regardless of whether anything was verified.
    path = _key_path(key_path)
    if not path:
        print("  - Signing: no key supplied")
        print("PROMOTION ABORTED: pass a key path as argv[3] or set "
              "MECH_SIGNING_KEY_PATH. There is no default signing key, because a "
              "default key signs nothing worth trusting.")
        return 1

    try:
        signature = sign(manager._signing_payload(dataset_id, meta), path)
    except SigningUnavailable as exc:
        print(f"  - Signing: FAILED ({exc})")
        print("PROMOTION ABORTED.")
        return 1

    # What this check does and does not prove.
    #
    # The public key is derived from the key that just signed, so a successful
    # verification here is a *self-consistency* check: it catches a corrupted or
    # mis-encoded signature. It proves nothing about *who* signed, because
    # anyone holding a private key can produce a self-consistent signature. Only
    # a later check against an independently held public key -- what
    # `verify_golden_datasets.py` does -- can attest identity.
    #
    # It is still worth doing before writing GOLDEN: it means the stored bytes
    # are a valid signature over this exact payload, rather than an unverified
    # blob written next to a "signed" marker.
    from cryptography.hazmat.primitives import serialization

    try:
        with open(path, "rb") as handle:
            private_key = serialization.load_pem_private_key(handle.read(), password=None)
        public_key = private_key.public_key().public_bytes(
            serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo)
        del private_key
    except Exception:
        try:
            seed = load_raw_seed(path)
            private_key = Ed25519PrivateKey.from_private_bytes(seed)
            public_key = private_key.public_key().public_bytes(
                serialization.Encoding.PEM,
                serialization.PublicFormat.SubjectPublicKeyInfo)
            del private_key, seed
        except Exception as exc:
            print(f"  - Signing: could not derive a public key ({exc})")
            print("PROMOTION ABORTED.")
            return 1

    # `verify` returns a VerificationResult, not a dict. Subscripting it raised
    # TypeError, which meant the fail-closed path below never executed and
    # promotion crashed instead of refusing -- the worst combination, since a
    # crash reads as an unrelated bug rather than a blocked promotion.
    result = verify(manager._signing_payload(dataset_id, meta), signature, public_key)
    if not result.valid:
        print(f"  - Signature: NOT VERIFIED ({result.reason})")
        print("PROMOTION ABORTED: refusing to mark GOLDEN on an unverified signature.")
        return 1
    print(f"  - Signature: Ed25519, self-consistent ({signature[:16]}...)")
    print("    (verifies against a public key derived from the signing key; this "
          "confirms the stored bytes are a valid signature, not who signed. Run "
          "verify_golden_datasets.py with an independent public key to attest "
          "identity.)")

    meta["signature"] = signature
    meta["signature_algorithm"] = "Ed25519"
    meta["promoted_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    meta["promoted_by"] = author
    meta["status"] = "GOLDEN"
    meta["locked"] = True

    # Persist. The signature and the GOLDEN status were previously assigned to an
    # in-memory dict and never written back, so the script printed "PROMOTION
    # COMPLETE ... GOLDEN and signed" while `golden_manifest.json` kept its old
    # placeholder signature and old status. Every later verification therefore ran
    # against a manifest the promotion had never touched.
    _write_manifest(manager, meta, dataset_id)

    # 4. Certificate.
    cert = cert_engine.issue(meta)
    cert_dir = os.path.join(DATA_DIR, dataset_id)
    os.makedirs(cert_dir, exist_ok=True)
    cert_path = os.path.join(cert_dir, "dataset_certificate.json")
    with open(cert_path, "w", encoding="utf-8") as handle:
        json.dump(cert.__dict__, handle, indent=2)
    print(f"  - Certificate issued: {cert_path}")

    # 5. Bundle.
    bundle_path = exporter.export(dataset_id, manager)
    print(f"  - Reproduction bundle: {bundle_path}")

    # 6. Audit.
    manager.log_audit_event(dataset_id, "PROMOTION", author,
                            "Promoted to GOLDEN with an Ed25519 signature verified "
                            "under the public key.")

    print("\n" + "=" * 40)
    print(f"PROMOTION COMPLETE: {dataset_id} is GOLDEN and signed.")
    return 0


def _write_manifest(manager: DatasetManager, meta: dict, dataset_id: str) -> None:
    """Write the updated manifest entry back to `golden_manifest.json`.

    Rewrites only the datasets block, preserving `manifest_version`,
    `schema_version`, `created_at` and `root_signature`. A new signature changes
    the document root, so `root_signature` is recomputed over the serialized
    datasets block rather than left as `sha256:root_sig_placeholder` -- leaving a
    placeholder next to a real signature would misstate what the root attests to.
    """
    with open(manager.manifest_path, "r", encoding="utf-8") as handle:
        document = json.load(handle)

    datasets = document.get("datasets", {})
    canonical = str(meta.get("dataset_id") or dataset_id)
    entry = datasets.get(canonical) or datasets.get(dataset_id)
    if entry is None:
        raise SystemExit(f"Cannot persist: {canonical} is not in the manifest")
    entry.update(meta)

    document["datasets"] = datasets
    document["root_signature"] = "sha256:" + hashlib.sha256(
        json.dumps(datasets, sort_keys=True).encode("utf-8")
    ).hexdigest()

    with open(manager.manifest_path, "w", encoding="utf-8") as handle:
        json.dump(document, handle, indent=2)
    print(f"  - Manifest updated: {manager.manifest_path}")


def load_raw_seed(path: str) -> bytes:
    from backend.science.integrity import load_private_key

    key = load_private_key(path)
    from cryptography.hazmat.primitives import serialization

    return key.private_bytes(
        serialization.Encoding.Raw,
        serialization.PrivateFormat.Raw,
        serialization.NoEncryption(),
    )


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python frontend/scripts/promote_dataset.py "
              "<dataset_id> <author> [key_path]")
        sys.exit(2)
    sys.exit(promote_dataset(sys.argv[1], sys.argv[2],
                             sys.argv[3] if len(sys.argv) > 3 else None))