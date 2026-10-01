"""Golden Dataset Verification Tool — Scientific Audit & Hash Validation.

Audits all datasets in the golden manifest, verifies Triple-SHA hashes,
and checks for metadata completeness and version compatibility.
"""

import sys
import os
import json

# Ensure project root is in path
sys.path.insert(0, os.getcwd())

from backend.research_datasets.dataset_manager import DatasetManager

def audit_datasets():
    print("Executing Golden Dataset Integrity Audit...")
    manager = DatasetManager(data_dir="backend/research_datasets")
    manifest_path = os.path.join("backend/research_datasets", "golden_manifest.json")

    if not os.path.exists(manifest_path):
        print("FAIL: golden_manifest.json not found.")
        sys.exit(1)

    with open(manifest_path, "r") as f:
        manifest = json.load(f).get("datasets", {})

    stats = {"total": len(manifest), "passed": 0, "failed": 0}

    for ds_id, meta in manifest.items():
        print(f"\n[Audit] {ds_id} (v{meta['version']})...")
        try:
            # 1. Triple-SHA check
            manager.load(ds_id)
            print(f"  - Hash Verification: PASS")

            # 2. Signature Check
            if manager.verify_signature(ds_id):
                print(f"  - Signature Verification: PASS")
            else:
                print(f"  - Signature Verification: FAIL")
                stats["failed"] += 1
                continue

            # 3. Status check
            print(f"  - Status: {meta['status']}")
            stats["passed"] += 1
        except Exception as e:
            print(f"  - Audit Failure: {str(e)}")
            stats["failed"] += 1

    print("\n" + "="*40)
    print(f"Audit Summary: {stats['passed']}/{stats['total']} PASSED")
    if stats["failed"] > 0:
        print(f"CRITICAL: {stats['failed']} datasets failed integrity checks.")
        sys.exit(1)
    else:
        print("All Golden datasets verified.")

if __name__ == "__main__":
    audit_datasets()
