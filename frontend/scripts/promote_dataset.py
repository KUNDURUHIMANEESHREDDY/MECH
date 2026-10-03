"""Dataset Promotion Wizard — Transitioning to GOLDEN status.

Workflow:
1. Verify Triple-SHA Integrity.
2. Validate Lineage & Provenance Metadata.
3. Apply Asymmetric Digital Signature.
4. Issue Dataset Certificate.
"""

import sys
import os
import json

# Ensure project root is in path
sys.path.insert(0, os.getcwd())

from backend.research_datasets.dataset_manager import DatasetManager
from backend.research_datasets.dataset_certificate import DatasetCertificateEngine
from backend.research_datasets.dataset_exporter import DatasetExporter

def promote_dataset(dataset_id: str, author: str):
    print(f"Executing Promotion Workflow for: {dataset_id}")
    manager = DatasetManager(data_dir="backend/research_datasets")
    cert_engine = DatasetCertificateEngine()
    exporter = DatasetExporter()

    # 1. Fetch & Verify
    try:
        prompts = manager.load(dataset_id)
        print("  - Integrity Check: PASS")
    except Exception as e:
        print(f"  - Integrity Check: FAIL ({e})")
        return

    # 2. Metadata Check
    meta = manager._manifest.get(dataset_id)
    if not meta.get("lineage") or not meta.get("provenance"):
        print("  - Metadata Check: FAIL (Missing Lineage/Provenance)")
        return
    print("  - Metadata Check: PASS")

    # 3. Sign
    sig = manager.sign_dataset(dataset_id)
    meta["status"] = "GOLDEN"
    meta["locked"] = True
    print(f"  - Signature Applied: {sig[:16]}...")

    # 4. Issue Certificate
    cert = cert_engine.issue(meta)
    cert_path = os.path.join("backend/research_datasets", dataset_id, "dataset_certificate.json")
    with open(cert_path, "w") as f:
        json.dump(cert.__dict__, f, indent=2)
    print(f"  - Certificate Issued: {cert_path}")

    # 5. Export Reproduction Bundle
    bundle_path = exporter.export(dataset_id, manager)
    print(f"  - Reproduction Bundle Created: {bundle_path}")

    # 6. Audit Log
    manager.log_audit_event(dataset_id, "PROMOTION", author, "Promoted to GOLDEN status and exported repro bundle.")

    print("\n" + "="*40)
    print(f"PROMOTION SUCCESSFUL: {dataset_id} is now GOLDEN.")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python scripts/promote_dataset.py <dataset_id> <author>")
    else:
        promote_dataset(sys.argv[1], sys.argv[2])
