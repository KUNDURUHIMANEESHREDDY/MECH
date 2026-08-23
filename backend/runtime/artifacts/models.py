"""Execution Artifact & Provenance Data Models.

Defines immutable containers for cached intermediate computations,
including full lineage, provenance, parent artifact digests, and hardware metadata.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import logging
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import torch

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Provenance:
    """Complete lineage and execution context for a computation."""
    model_id: str
    weights_digest: str = "unknown"
    tokenizer_digest: str = "unknown"
    precision: str = "float32"
    device: str = "cpu"
    operation: str = "forward"
    operation_params: Dict[str, Any] = field(default_factory=dict)
    framework_version: str = "torch"
    engine_commit: str = "v1.0"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def compute_digest(self) -> str:
        """Deterministic SHA-256 digest of execution provenance."""
        data = {
            "model_id": self.model_id,
            "weights_digest": self.weights_digest,
            "tokenizer_digest": self.tokenizer_digest,
            "precision": self.precision,
            "operation": self.operation,
            "operation_params": self.operation_params,
        }
        raw = json.dumps(data, sort_keys=True)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()


@dataclass
class ArtifactMetadata:
    """Metadata describing tensor properties, location, and tags."""
    name: str
    component: str  # e.g., "residual", "mlp", "attention", "sae_latents", "logits"
    layer: Optional[int] = None
    head: Optional[int] = None
    shape: Tuple[int, ...] = ()
    dtype: str = "float32"
    device: str = "cpu"
    session_id: str = ""
    prompt_id: str = ""
    created_at: float = field(default_factory=time.time)
    size_bytes: int = 0
    tags: List[str] = field(default_factory=list)
    custom: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["shape"] = list(self.shape)
        return d


@dataclass
class ExecutionArtifact:
    """Immutable execution artifact with CAS key and dependency links."""
    artifact_id: str  # Deterministic SHA-256 CAS key
    metadata: ArtifactMetadata
    provenance: Provenance
    parent_ids: List[str] = field(default_factory=list)  # Dependency hashes
    tensor: Optional[torch.Tensor] = None
    data: Optional[Any] = None  # For non-tensor structured data (e.g. circuit graphs, token lists)
    storage_path: Optional[str] = None  # On-disk L2 path if offloaded

    def is_tensor(self) -> bool:
        return self.tensor is not None

    def get_tensor(self) -> Optional[torch.Tensor]:
        if self.tensor is not None:
            return self.tensor
        if self.storage_path and self.storage_path.endswith((".pt", ".safetensors", ".bin")):
            # Lazy load from storage
            try:
                loaded = torch.load(self.storage_path, weights_only=True, map_location="cpu")
                return loaded
            except (OSError, RuntimeError, ValueError) as exc:
                logger.debug("Failed to lazy-load tensor from %s: %s", self.storage_path, exc)
                return None
        return None

    def to_summary(self) -> Dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "name": self.metadata.name,
            "component": self.metadata.component,
            "layer": self.metadata.layer,
            "shape": list(self.metadata.shape),
            "dtype": self.metadata.dtype,
            "size_bytes": self.metadata.size_bytes,
            "parent_ids": self.parent_ids,
            "operation": self.provenance.operation,
            "created_at": self.metadata.created_at,
        }
