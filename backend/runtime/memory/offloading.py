"""Memory Offloading Manager.

Handles serialization and deserialization of massive activation tensors
from GPU VRAM directly to NVMe-backed storage using memory mapping and torch serialization.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
from typing import Any, Dict, List, Optional

import torch

logger = logging.getLogger("MECH.offload")


class NVMeOffloadManager:
    """Offloads tensor streams to NVMe to prevent OOM on 70B+ models."""

    def __init__(self, scratch_dir: str = "backend/storage/scratch/offload") -> None:
        self.scratch_dir = os.path.abspath(scratch_dir)
        os.makedirs(self.scratch_dir, exist_ok=True)

    def write_tensor(self, session_id: str, layer: int, name: str, tensor: torch.Tensor) -> str:
        """Serializes a tensor to disk."""
        filepath = os.path.join(self.scratch_dir, f"{session_id}_L{layer}_{name}.pt")
        torch.save(tensor.detach().cpu(), filepath)
        return filepath

    def read_tensor(self, filepath: str, map_location: str = "cpu") -> torch.Tensor:
        """Deserializes a tensor back into memory."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Offloaded tensor not found: {filepath}")
        return torch.load(filepath, weights_only=True, map_location=map_location)

    def write_chunk(self, session_id: str, layer: int, chunk_idx: int, activations: List[Dict[str, Any]]) -> str:
        """Serializes a chunk of activations (metadata/structured records) to disk."""
        filepath = os.path.join(self.scratch_dir, f"{session_id}_L{layer}_chunk{chunk_idx}.jsonl")
        with open(filepath, "w", encoding="utf-8") as f:
            for act in activations:
                f.write(json.dumps(act) + "\n")
        return filepath

    def read_chunk(self, filepath: str) -> List[Dict[str, Any]]:
        """Deserializes a chunk back into memory."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Offloaded chunk not found: {filepath}")
        results = []
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    results.append(json.loads(line))
        return results

    def cleanup_session(self, session_id: str) -> None:
        """Removes all offloaded tensors for a completed session."""
        if not os.path.exists(self.scratch_dir):
            return
        for filename in os.listdir(self.scratch_dir):
            if filename.startswith(session_id):
                try:
                    os.remove(os.path.join(self.scratch_dir, filename))
                except OSError:
                    pass

    def clear_all(self) -> None:
        """Cleans entire scratch directory."""
        if os.path.exists(self.scratch_dir):
            shutil.rmtree(self.scratch_dir)
            os.makedirs(self.scratch_dir, exist_ok=True)
