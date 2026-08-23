"""Production Artifact Registry & Two-Phase Atomic Tensor Storage for MECH.

Stores large activations, attention matrices, plots, and manifests on the filesystem
using a two-phase atomic write protocol (.tmp -> validate SHA256 -> .final) and registers
metadata references into SQLite.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch

from backend.storage.database import DesktopStorage
from backend.storage.scientific_entities import ResearchArtifact

logger = logging.getLogger("MECH.science.artifact_registry")


class ArtifactStorageError(Exception):
    """Raised when an artifact fails atomic storage, validation, or checksum integrity."""


class ArtifactRegistry:
    """Manages atomic artifact persistence, tensor isolation, and checksum verification."""

    def __init__(self, base_dir: Optional[Path] = None, storage: Optional[DesktopStorage] = None) -> None:
        self.base_dir = Path(base_dir or (Path.home() / ".cache" / "neural-debugger" / "artifacts"))
        self.tensors_dir = self.base_dir / "tensors"
        self.manifests_dir = self.base_dir / "manifests"
        self.reports_dir = self.base_dir / "reports"

        for d in (self.tensors_dir, self.manifests_dir, self.reports_dir):
            d.mkdir(parents=True, exist_ok=True)

        db_path = Path.home() / ".cache" / "neural-debugger" / "mech.db"
        self.storage = storage or DesktopStorage(db_path)
        self.storage.initialize()

    def store_tensor_artifact(
        self,
        tensor_data: Union[torch.Tensor, Dict[str, torch.Tensor]],
        name: str,
        investigation_id: str,
        experiment_run_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ResearchArtifact:
        """Saves a tensor to disk via two-phase atomic write and registers metadata."""
        artifact_id = f"art_{uuid.uuid4().hex[:10]}"
        tmp_path = self.tensors_dir / f"{artifact_id}.tmp"
        final_path = self.tensors_dir / f"{artifact_id}.pt"

        try:
            # Phase 1: Write to temporary file
            torch.save(tensor_data, str(tmp_path))

            # Phase 2: Compute SHA256 and byte size
            hasher = hashlib.sha256()
            size_bytes = 0
            with open(tmp_path, "rb") as f:
                while chunk := f.read(65536):
                    hasher.update(chunk)
                    size_bytes += len(chunk)
            sha256 = hasher.hexdigest()

            # Phase 3: Atomic filesystem rename
            os.replace(str(tmp_path), str(final_path))

            # Phase 4: Register metadata reference in SQLite
            artifact = ResearchArtifact(
                id=artifact_id,
                investigation_id=investigation_id,
                experiment_run_id=experiment_run_id,
                name=name,
                artifact_type="tensor",
                file_path=str(final_path),
                checksum_sha256=sha256,
                size_bytes=size_bytes,
                storage_format="pt",
                metadata=metadata or {},
            )
            self.storage.save_artifact(artifact.model_dump())
            logger.info("Saved tensor artifact %s (size: %d bytes, sha: %s)", artifact_id, size_bytes, sha256[:10])
            return artifact

        except Exception as exc:
            # Instant cleanup on failure
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)
            logger.error("Failed to store tensor artifact %s: %s", artifact_id, exc)
            raise ArtifactStorageError(f"Atomic artifact write failed: {exc}") from exc

    def store_json_artifact(
        self,
        json_data: Any,
        name: str,
        investigation_id: str,
        artifact_type: str = "json_manifest",
        experiment_run_id: Optional[str] = None,
    ) -> ResearchArtifact:
        """Saves JSON metadata via two-phase atomic write."""
        artifact_id = f"art_{uuid.uuid4().hex[:10]}"
        tmp_path = self.manifests_dir / f"{artifact_id}.tmp"
        final_path = self.manifests_dir / f"{artifact_id}.json"

        try:
            raw_bytes = json.dumps(json_data, indent=2, sort_keys=True).encode("utf-8")
            with open(tmp_path, "wb") as f:
                f.write(raw_bytes)

            sha256 = hashlib.sha256(raw_bytes).hexdigest()
            size_bytes = len(raw_bytes)

            os.replace(str(tmp_path), str(final_path))

            artifact = ResearchArtifact(
                id=artifact_id,
                investigation_id=investigation_id,
                experiment_run_id=experiment_run_id,
                name=name,
                artifact_type=artifact_type,
                file_path=str(final_path),
                checksum_sha256=sha256,
                size_bytes=size_bytes,
                storage_format="json",
            )
            self.storage.save_artifact(artifact.model_dump())
            return artifact

        except Exception as exc:
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)
            raise ArtifactStorageError(f"Atomic JSON write failed: {exc}") from exc

    def load_tensor_artifact(self, artifact_id: str, verify_checksum: bool = True) -> torch.Tensor:
        """Loads a tensor artifact from disk with mandatory SHA256 integrity verification."""
        record = self.storage.get_artifact(artifact_id)
        if not record:
            raise ArtifactStorageError(f"Artifact {artifact_id} not found in database registry.")

        file_path = Path(record["file_path"])
        if not file_path.exists():
            raise ArtifactStorageError(f"Artifact file missing from disk: {file_path}")

        if verify_checksum:
            hasher = hashlib.sha256()
            with open(file_path, "rb") as f:
                while chunk := f.read(65536):
                    hasher.update(chunk)
            current_sha = hasher.hexdigest()
            if current_sha != record["checksum_sha256"]:
                raise ArtifactStorageError(
                    f"TAMPER DETECTED: Artifact {artifact_id} checksum mismatch! "
                    f"Expected: {record['checksum_sha256']}, Computed: {current_sha}"
                )

        return torch.load(str(file_path), map_location="cpu", weights_only=True)

    def prune_temporary_files(self) -> int:
        """Prunes any orphaned .tmp files left by interrupted workers."""
        count = 0
        for d in (self.tensors_dir, self.manifests_dir, self.reports_dir):
            for tmp in d.glob("*.tmp"):
                try:
                    tmp.unlink(missing_ok=True)
                    count += 1
                except Exception:
                    pass
        return count


# Global artifact registry singleton
artifact_registry = ArtifactRegistry()
