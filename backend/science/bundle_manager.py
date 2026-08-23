"""Research Bundle Exporter & Importer for MECH.

Packages complete investigations, hypotheses, experiments, run records, evidence trees,
environment snapshots, and raw tensor artifacts into a self-contained ZIP archive with
cryptographic manifest verification, and reconstructs the state on import.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
import shutil
import sys
import tempfile
import time
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from backend.storage.database import DesktopStorage
from backend.storage.scientific_entities import (
    EvidenceRecord,
    Experiment,
    ExperimentRun,
    Hypothesis,
    Investigation,
    Mechanism,
    ResearchArtifact,
)

logger = logging.getLogger("MECH.science.bundle_manager")


class BundleIntegrityError(Exception):
    """Raised when a research bundle fails cryptographic checksum validation or has corrupted contents."""


class ResearchBundleManager:
    """Handles self-contained export and import of MECH research bundles."""

    def __init__(self, storage: Optional[DesktopStorage] = None) -> None:
        db_path = Path.home() / ".cache" / "neural-debugger" / "mech.db"
        self.storage = storage or DesktopStorage(db_path)
        self.storage.initialize()

    def export_bundle(self, investigation_id: str, output_zip_path: Path | str) -> str:
        """Exports an investigation, its entities, and tensor artifacts to a verifiable ZIP."""
        inv = self.storage.get_investigation(investigation_id)
        if not inv:
            raise ValueError(f"Investigation {investigation_id} not found.")

        hypotheses = self.storage.list_hypotheses(investigation_id)
        runs = self.storage.list_runs(investigation_id)
        evidence = self.storage.list_evidence(investigation_id)
        mechanisms = self.storage.list_mechanisms(investigation_id)
        artifacts = self.storage.list_artifacts(investigation_id)

        with tempfile.TemporaryDirectory() as tmpdir:
            bundle_dir = Path(tmpdir) / f"mech_bundle_{investigation_id}"
            bundle_dir.mkdir(parents=True, exist_ok=True)

            # 1. Write metadata files
            def _write_json(rel_path: str, data: Any):
                fp = bundle_dir / rel_path
                fp.parent.mkdir(parents=True, exist_ok=True)
                with open(fp, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2, sort_keys=True, default=str)

            _write_json("investigation.json", inv)
            _write_json("hypotheses.json", hypotheses)
            _write_json("runs.json", runs)
            _write_json("evidence.json", evidence)
            _write_json("mechanisms.json", mechanisms)
            _write_json("artifacts_meta.json", artifacts)

            # 2. Copy artifact payload files (tensors / manifests)
            artifacts_payload_dir = bundle_dir / "artifacts_data"
            artifacts_payload_dir.mkdir(exist_ok=True)
            for art in artifacts:
                src_file = Path(art["file_path"])
                if src_file.exists():
                    shutil.copy2(src_file, artifacts_payload_dir / src_file.name)

            # 3. Write environment snapshot
            env_info = {
                "python_version": sys.version,
                "pytorch_version": torch.__version__,
                "platform": platform.platform(),
                "os": os.name,
                "exported_at": time.time(),
            }
            _write_json("environment.json", env_info)

            # 4. Generate manifest with SHA256 checksums of every file
            manifest: Dict[str, str] = {}
            for root, _, files in os.walk(bundle_dir):
                for f in files:
                    full_p = Path(root) / f
                    rel_p = str(full_p.relative_to(bundle_dir)).replace("\\", "/")
                    hasher = hashlib.sha256()
                    with open(full_p, "rb") as fh:
                        while chunk := fh.read(65536):
                            hasher.update(chunk)
                    manifest[rel_p] = hasher.hexdigest()

            _write_json("manifest.json", {"files": manifest, "investigation_id": investigation_id})

            # 5. Pack into ZIP
            out_p = Path(output_zip_path)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(str(out_p), "w", zipfile.ZIP_DEFLATED) as zf:
                for root, _, files in os.walk(bundle_dir):
                    for f in files:
                        full_p = Path(root) / f
                        arcname = str(full_p.relative_to(bundle_dir)).replace("\\", "/")
                        zf.write(full_p, arcname)

            logger.info("Exported research bundle for %s to %s", investigation_id, out_p)
            return str(out_p)

    def import_bundle(self, zip_path: Path | str) -> Dict[str, Any]:
        """Validates manifest checksums and imports all research entities into SQLite."""
        zip_p = Path(zip_path)
        if not zip_p.exists():
            raise FileNotFoundError(f"Bundle file not found: {zip_p}")

        with tempfile.TemporaryDirectory() as tmpdir:
            extract_dir = Path(tmpdir) / "extracted"
            with zipfile.ZipFile(str(zip_p), "r") as zf:
                zf.extractall(extract_dir)

            manifest_path = extract_dir / "manifest.json"
            if not manifest_path.exists():
                raise BundleIntegrityError("Invalid bundle: missing manifest.json")

            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest_data = json.load(f)

            file_checksums = manifest_data.get("files", {})

            # Validate SHA-256 checksum of every file
            for rel_path, expected_sha in file_checksums.items():
                target = extract_dir / rel_path
                if not target.exists():
                    raise BundleIntegrityError(f"Missing file in bundle: {rel_path}")
                hasher = hashlib.sha256()
                with open(target, "rb") as fh:
                    while chunk := fh.read(65536):
                        hasher.update(chunk)
                current_sha = hasher.hexdigest()
                if current_sha != expected_sha:
                    raise BundleIntegrityError(
                        f"TAMPER DETECTED in bundle file {rel_path}! Expected: {expected_sha}, Got: {current_sha}"
                    )

            # Reconstruct DB records
            def _read_json(rel_p: str) -> Any:
                fp = extract_dir / rel_p
                if fp.exists():
                    with open(fp, "r", encoding="utf-8") as f:
                        return json.load(f)
                return None

            inv = _read_json("investigation.json")
            if inv:
                self.storage.save_investigation(inv)

            for hyp in _read_json("hypotheses.json") or []:
                self.storage.save_hypothesis(hyp)

            for r in _read_json("runs.json") or []:
                self.storage.save_run(r)

            for evi in _read_json("evidence.json") or []:
                self.storage.save_evidence(evi)

            for m in _read_json("mechanisms.json") or []:
                self.storage.save_mechanism(m)

            # Copy artifact files to local storage
            target_artifact_dir = Path.home() / ".cache" / "neural-debugger" / "artifacts" / "tensors"
            target_artifact_dir.mkdir(parents=True, exist_ok=True)

            for art in _read_json("artifacts_meta.json") or []:
                src = extract_dir / "artifacts_data" / Path(art["file_path"]).name
                if src.exists():
                    dest = target_artifact_dir / src.name
                    shutil.copy2(src, dest)
                    art["file_path"] = str(dest)
                self.storage.save_artifact(art)

            logger.info("Imported research bundle %s successfully.", inv.get("id"))
            return inv


# Global bundle manager singleton
bundle_manager = ResearchBundleManager()
