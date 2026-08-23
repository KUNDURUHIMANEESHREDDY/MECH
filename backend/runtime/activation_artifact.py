"""First-Class Activation Artifact System for Out-of-Core Mechanistic Research.

Encapsulates intermediate representations, residual stream tensors, and hook states
with explicit storage tiering (GPU VRAM -> CPU RAM -> Disk CAS/mmap cache).
"""

from __future__ import annotations

import datetime as _dt
import os
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import torch

DEFAULT_ACTIVATION_CACHE_DIR = Path(__file__).resolve().parent.parent.parent / "storage" / "activations_cache"


import hashlib
import json

@dataclass
class ActivationArtifact:
    """A first-class activation tensor container with tiered residency and provenance."""
    artifact_id: str
    model_id: str
    layer: int
    sequence_length: int
    hidden_dimension: int
    dtype: str
    current_location: str  # "vram" | "ram" | "disk"
    tensor_payload: Optional[torch.Tensor] = None
    disk_path: Optional[str] = None
    intervention_context: Optional[Dict[str, Any]] = None
    sequence_position: Optional[int] = None
    experiment_hash: Optional[str] = None
    source_runtime: str = "out_of_core"
    artifact_hash: str = ""
    created_at_utc: str = field(default_factory=lambda: _dt.datetime.now(_dt.timezone.utc).isoformat())

    @classmethod
    def from_tensor(
        cls,
        tensor: torch.Tensor,
        model_id: str,
        layer: int,
        intervention_context: Optional[Dict[str, Any]] = None,
        sequence_position: Optional[int] = None,
        experiment_hash: Optional[str] = None,
        source_runtime: str = "out_of_core",
    ) -> ActivationArtifact:
        """Constructs an artifact from a live torch.Tensor with deterministic SHA-256 hash."""
        device_type = "vram" if tensor.is_cuda else "ram"
        shape = tensor.shape
        seq_len = shape[1] if len(shape) >= 2 else 1
        d_model = shape[-1]
        art_id = f"act_{uuid.uuid4().hex[:12]}"

        # Compute deterministic SHA-256 artifact hash
        meta = {
            "model_id": model_id,
            "layer": layer,
            "sequence_length": seq_len,
            "hidden_dimension": d_model,
            "dtype": str(tensor.dtype),
            "sequence_position": sequence_position,
            "experiment_hash": experiment_hash,
            "source_runtime": source_runtime,
        }
        raw_bytes = tensor.detach().cpu().numpy().tobytes()
        h = hashlib.sha256(json.dumps(meta, sort_keys=True).encode("utf-8"))
        h.update(raw_bytes)
        art_hash = h.hexdigest()

        return cls(
            artifact_id=art_id,
            model_id=model_id,
            layer=layer,
            sequence_length=seq_len,
            hidden_dimension=d_model,
            dtype=str(tensor.dtype),
            current_location=device_type,
            tensor_payload=tensor,
            intervention_context=intervention_context,
            sequence_position=sequence_position,
            experiment_hash=experiment_hash,
            source_runtime=source_runtime,
            artifact_hash=art_hash,
        )

    def spill_to_cpu(self) -> None:
        """Moves tensor payload from GPU VRAM to host CPU RAM."""
        if self.tensor_payload is not None and self.tensor_payload.is_cuda:
            self.tensor_payload = self.tensor_payload.detach().cpu()
            self.current_location = "ram"

    def spill_to_disk(self, cache_dir: Optional[Path | str] = None) -> None:
        """Spills tensor payload from RAM/VRAM to disk cache to free system memory."""
        if self.tensor_payload is None and self.disk_path is not None:
            return  # Already on disk

        cdir = Path(cache_dir or DEFAULT_ACTIVATION_CACHE_DIR)
        os.makedirs(cdir, exist_ok=True)
        file_path = cdir / f"{self.artifact_id}.pt"

        cpu_tensor = self.tensor_payload.detach().cpu() if self.tensor_payload is not None else None
        if cpu_tensor is not None:
            torch.save(cpu_tensor, str(file_path))
            self.disk_path = str(file_path)
            self.tensor_payload = None  # Free memory
            self.current_location = "disk"

    def get_tensor(self, target_device: Optional[torch.device | str] = None) -> torch.Tensor:
        """Restores and returns the tensor payload on the requested device."""
        if self.tensor_payload is not None:
            t = self.tensor_payload
        elif self.disk_path is not None and os.path.exists(self.disk_path):
            t = torch.load(self.disk_path, weights_only=True)
            self.tensor_payload = t
            self.current_location = "ram"
        else:
            raise ValueError(f"Activation artifact '{self.artifact_id}' has no tensor or disk backing.")

        if target_device is not None:
            t = t.to(target_device)
            if str(target_device).startswith("cuda"):
                self.current_location = "vram"
            elif str(target_device) == "cpu":
                self.current_location = "ram"

        return t

    def to_dict(self) -> Dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "artifact_hash": self.artifact_hash,
            "model_id": self.model_id,
            "layer": self.layer,
            "sequence_length": self.sequence_length,
            "hidden_dimension": self.hidden_dimension,
            "dtype": self.dtype,
            "current_location": self.current_location,
            "disk_path": self.disk_path,
            "sequence_position": self.sequence_position,
            "experiment_hash": self.experiment_hash,
            "source_runtime": self.source_runtime,
            "intervention_context": self.intervention_context,
            "created_at_utc": self.created_at_utc,
        }

