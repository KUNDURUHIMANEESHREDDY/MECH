"""Golden Dataset Verification Tool — reports what the manifest actually records.

Renamed claims, not code paths: this previously advertised "Triple-SHA hashes"
and "Signature Verification: PASS".

What it now reports, and why it differs:

* **Integrity** is reported per hash. `DatasetManager.load` returns
  `last_integrity_status` saying whether each manifest hash was `verified` or
  `not_recorded`. `golden_manifest.json` currently records *neither* bundle nor
  prompt hash (both hold placeholder digests), so "Hash Verification: PASS" was
  printed on the strength of no hash being compared. Unrecorded is now its own
  outcome, distinct from verified.
* **Signatures** require a public key. `verify_signature` used to take a
  `public_key` argument and ignore it, recomputing with a literal secret from the
  source; the key is now load-bearing, so it is required. Point
  `MECH_VERIFY_PUBLIC_KEY_PATH` at a PEM public key to check signatures. Without
  one this reports `no public key supplied` rather than a bare FAIL, because a
  missing key is not a bad signature.

Usage:
    python frontend/scripts/verify_golden_datasets.py

Exit status is 1 when any dataset fails a check that was actually performed.
Datasets whose hashes are merely unrecorded are counted separately and reported,
not silently folded into "passed".
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.getcwd())

from backend.research_datasets.dataset_manager import DatasetManager

VERIFY_PUBLIC_KEY_ENV = "MECH_VERIFY_PUBLIC_KEY_PATH"

#: Module-level so a test can point the audit at a temporary copy of the dataset
#: store. Deliberately not an environment variable: an env override would let a
#: verifier be aimed at a different dataset store than the one it reports on,
#: which is precisely the kind of quiet redirection this codebase is trying to
#: avoid. Overriding it requires editing code, which leaves a trace.
DATA_DIR = "backend/research_datasets"


def load_public_key():
    path = os.environ.get(VERIFY_PUBLIC_KEY_ENV)
    if not path:
        return None
    if not os.path.exists(path):
        raise SystemExit(f"{VERIFY_PUBLIC_KEY_ENV} points at {path}, which does not exist")
    with open(path, "rb") as handle:
        return handle.read()


def audit_datasets() -> int:
    print("Executing Golden Dataset Integrity Audit...")
    manager = DatasetManager(data_dir=DATA_DIR)
    manifest_path = os.path.join(DATA_DIR, "golden_manifest.json")

    if not os.path.exists(manifest_path):
        print("FAIL: golden_manifest.json not found.")
        return 1

    with open(manifest_path, "r", encoding="utf-8") as handle:
        manifest = json.load(handle).get("datasets", {})

    public_key = load_public_key()
    if public_key is None:
        print(f"  (no {VERIFY_PUBLIC_KEY_ENV} set; signature checks will be "
              "reported as 'no public key supplied')")

    verified = failed = unrecorded = 0
    signatures_ok = signatures_absent = signatures_failed = 0

    for ds_id, meta in manifest.items():
        print(f"\n[Audit] {ds_id} (v{meta['version']})...")

        # 1. Integrity: report each hash, do not summarise to PASS.
        try:
            manager.load(ds_id)
        except Exception as exc:
            print(f"  - Integrity: FAILED ({exc})")
            failed += 1
            continue

        status = manager.last_integrity_status
        checks = status.get("checks", {})
        for name, state in sorted(checks.items()):
            print(f"  - hash {name}: {state}")
        if status.get("integrity_verified"):
            verified += 1
        if any(state == "not_recorded" for state in checks.values()):
            unrecorded += 1
            print("  - Integrity: NOT FULLY VERIFIED (manifest records no hash "
                  "for the checks above; this is a gap in the manifest, not a "
                  "mismatch)")

        # 2. Signature.
        if public_key is None:
            print("  - Signature: NOT CHECKED (no public key supplied)")
            signatures_absent += 1
        else:
            # `verify_signature` returns a dict.
            result = manager.verify_signature(ds_id, public_key)
            if result["valid"]:
                print("  - Signature: VERIFIED (Ed25519)")
                signatures_ok += 1
            else:
                print(f"  - Signature: NOT VERIFIED ({result['reason']})")
                signatures_failed += 1

        print(f"  - Status: {meta['status']}")

    print("\n" + "=" * 40)
    print(f"Integrity: {verified}/{len(manifest)} datasets loaded with checks passing")
    if unrecorded:
        print(f"  {unrecorded} of those have hashes the manifest never recorded "
              "-- not evidence of integrity")
    if public_key is None:
        print(f"Signatures: 0 checked ({signatures_absent} skipped, no public key)")
    else:
        print(f"Signatures: {signatures_ok}/{len(manifest)} verified")
    print("=" * 40)

    if failed:
        print(f"CRITICAL: {failed} dataset(s) failed an integrity check that ran.")
        return 1
    if unrecorded:
        print("RESULT: no integrity failure, but the manifest is incomplete. "
              "Record real hashes before treating these as attested.")
        return 1
    if signatures_failed:
        print(f"RESULT: {signatures_failed} dataset(s) carry a signature that does "
              "not verify under the supplied public key. Either the key is wrong "
              "or the dataset was signed by someone else.")
        return 1
    if public_key is None:
        print("RESULT: integrity checks ran; signatures were not checked.")
        return 1
    print("RESULT: all integrity and signature checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(audit_datasets())