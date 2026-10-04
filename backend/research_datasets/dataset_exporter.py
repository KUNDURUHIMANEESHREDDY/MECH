"""Dataset Exporter — Generating Portable Research Reproduction Bundles.

Bundles a dataset into a ZIP containing exactly:
- dataset.json (prompts)
- metadata.json (the manifest entry)
- verify_bundle.py (standalone hash checker, shipped with the bundle)
- README.md
- REPRODUCIBILITY_BADGE.md
- CITATION.cff
- bundle_manifest.json (sha256 of each member)

This list was previously wrong: it advertised `certificate.json`,
`CHANGELOG.md` and `citation.bib`, none of which were ever written. Nothing was
packaged under those names, so the docstring described a bundle that did not
exist. The badge also claimed `GOLDEN & SIGNED` unconditionally; it is now
derived from the manifest.
"""

import os
import json
import zipfile
import hashlib
import uuid
import platform
from datetime import datetime
from typing import Any, Dict, List, Optional

VERIFY_BUNDLE_TEMPLATE = """
import json
import hashlib
import os
import sys
import platform

def verify():
    print("MECH Standalone Bundle Verifier v1.1")
    if not os.path.exists('bundle_manifest.json'):
        print("FAIL: bundle_manifest.json not found.")
        return

    with open('bundle_manifest.json', 'r') as f:
        manifest = json.load(f)

    print(f"Verifying Bundle ID: {manifest['bundle_id']}")
    print(f"Bundle UUID: {manifest.get('bundle_uuid', 'N/A')}")

    # 1. Environment Audit
    print("\\n[1/2] Environment Audit:")
    curr_os = platform.system()
    target_os = manifest.get('environment', {}).get('os', 'Any')
    print(f"  - Operating System: {curr_os} (Target: {target_os})")
    print(f"  - Python Version: {platform.python_version()} (Target: {manifest.get('environment', {}).get('python', 'Any')})")

    if curr_os != target_os and target_os != 'Any':
        print(f"  WARN: OS Mismatch. Results may vary slightly.")

    # 2. Integrity Audit
    print("\\n[2/2] Integrity Audit:")
    for file_info in manifest['files']:
        name = file_info['name']
        if not os.path.exists(name):
            print(f"  FAIL: {name} is missing.")
            continue

        with open(name, 'rb') as f:
            h = hashlib.sha256(f.read()).hexdigest()

        if h == file_info['sha256']:
            print(f"  - {name}: VERIFIED")
        else:
            print(f"  - {name}: CORRUPTED (Hash Mismatch)")

    print("\\nVerification Complete.")

if __name__ == '__main__':
    verify()
"""

class DatasetExporter:
    """Exports datasets as self-contained reproduction bundles with exhaustive fingerprinting."""

    def export(self, dataset_id: str, manager: Any, output_dir: str = "exports") -> str:
        if not os.path.exists(output_dir): os.makedirs(output_dir)

        meta = manager._manifest.get(dataset_id)
        if not meta: raise ValueError(f"Dataset {dataset_id} not found.")

        bundle_id = f"{dataset_id}_v{meta['version']}"
        bundle_uuid = str(uuid.uuid4()).upper()
        zip_name = f"{bundle_id}_repro.zip"
        zip_path = os.path.join(output_dir, zip_name)

        # 0. Health score. Was `... if hasattr(...) else {"overall": 100}` -- a missing
        #    health method produced a perfect 100% score, which then fed a badge
        #    reading "GOLDEN & SIGNED". An uncomputable score is None, and the
        #    badge says so.
        if hasattr(manager, "compute_health_score"):
            health = manager.compute_health_score(dataset_id)
        else:
            health = {}
            print("  - WARNING: DatasetManager has no compute_health_score; "
                  "the bundle will carry no health score rather than a "
                  "fabricated 100%.")

        # 1. Gather file contents
        files_to_bundle = {
            "dataset.json": json.dumps({"prompts": manager.load(dataset_id)}, indent=2),
            "metadata.json": json.dumps(meta, indent=2),
            "verify_bundle.py": VERIFY_BUNDLE_TEMPLATE,
            "README.md": f"# Reproduction Bundle: {dataset_id}\\n\\nVersion: {meta['version']}\\nUUID: {bundle_uuid}\\nHealth Score: {health.get('overall', 'not computed')}",
            "REPRODUCIBILITY_BADGE.md": self._generate_badge(
                health.get("overall"), health),
            "CITATION.cff": f"cff-version: 1.2.0\\ntitle: {dataset_id}\\nversion: {meta['version']}",
        }

        # 2. Compute Bundle Manifest with exhaustive fingerprint
        manifest_files = []
        for name, content in files_to_bundle.items():
            h = hashlib.sha256(content.encode()).hexdigest()
            manifest_files.append({"name": name, "sha256": h, "size": len(content)})

        bundle_manifest = {
            "bundle_id": bundle_id,
            "bundle_uuid": bundle_uuid,
            "created_at": datetime.now().isoformat(),
            "schema_version": "1.2.0",
            "platform_version": "39.13.1",
            "validator_version": "1.4.0",
            "environment": {
                "os": platform.system(),
                "python": platform.python_version(),
                "arch": platform.machine()
            },
            "health_score": health,
            "files": manifest_files
        }
        files_to_bundle["bundle_manifest.json"] = json.dumps(bundle_manifest, indent=2)

        # 3. Write ZIP
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as z:
            for name, content in files_to_bundle.items():
                z.writestr(name, content)

        return zip_path

    def _generate_badge(self, score: Optional[float], health: Optional[Dict[str, Any]] = None) -> str:
        """Render the reproducibility badge.

        The status line used to be the literal `**GOLDEN & SIGNED**`, printed
        unconditionally -- including for a dataset whose manifest signature was
        the placeholder `sha256:dataset_sig_placeholder`, and for a health score
        that defaulted to 100 when no health method existed. A badge is the one
        artefact likely to be copied out of context, so it must not assert
        anything the bundle did not check.

        Both `score` and the status are derived from the manifest. When the health
        method is unavailable the score is unknown and printed as such.
        """
        health = health or {}
        score_txt = "not computed" if score is None else f"{score}%"

        if score is None:
            stars = "?" * 5
        else:
            filled = min(5, max(0, int(score // 20)))
            stars = "★" * filled + "☆" * (5 - filled)

        recorded = health.get("integrity_hashes_recorded")
        is_signed = bool(health.get("signature_is_real"))

        if recorded == "0/3" and not is_signed:
            status = "**UNVERIFIED** — no hashes recorded, no signature"
        elif not is_signed:
            status = ("**UNSIGNED** — hashes recorded "
                      f"{recorded or 'unknown'}, but no verifiable signature")
        else:
            status = f"**SIGNED** — Ed25519 signature present, hashes {recorded or 'unknown'}"

        unmeasured = health.get("overall_unmeasured_weight")
        caveat = ""
        if health.get("overall_is_partial"):
            caveat = (f"\n\\n_Score covers "
                      f"{health.get('overall_weight_covered', '?')} of the weighting; "
                      f"{unmeasured} is unmeasured and excluded._")

        return (f"# REPRODUCIBILITY: {stars} {score_txt}\\n"
                f"Status: {status}{caveat}")
