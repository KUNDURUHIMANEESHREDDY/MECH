"""Memory Offloading Manager.

Handles serialization and deserialization of massive activation tensors
from GPU VRAM directly to NVMe-backed storage using memory mapping.
"""

from __future__ import annotations

import os
import mmap
import json
from typing import Any, Dict, List


class NVMeOffloadManager:
    """Offloads tensor streams to NVMe to prevent OOM on 70B+ models."""

    def __init__(self, scratch_dir: str = "/tmp/antigravity_offload") -> None:
        self.scratch_dir = scratch_dir
        if not os.path.exists(self.scratch_dir):
            os.makedirs(self.scratch_dir, exist_ok=True)

    def write_chunk(self, session_id: str, layer: int, chunk_idx: int, activations: List[Dict[str, Any]]) -> str:
        """Serializes a chunk of activations to disk.
        
        In a real PyTorch system, this would use safetensors or torch.save.
        Here we mock it using JSON lines.
        """
        filepath = os.path.join(self.scratch_dir, f"{session_id}_L{layer}_chunk{chunk_idx}.jsonl")
        
        # Write to disk
        with open(filepath, "w", encoding="utf-8") as f:
            for act in activations:
                f.write(json.dumps(act) + "\n")
                
        return filepath

    def read_chunk(self, filepath: str) -> List[Dict[str, Any]]:
        """Deserializes a chunk back into memory (e.g. CPU RAM)."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Offloaded chunk not found: {filepath}")
            
        results = []
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                results.append(json.loads(line))
        return results

    def cleanup_session(self, session_id: str) -> None:
        """Removes all offloaded tensors for a completed session."""
        for filename in os.listdir(self.scratch_dir):
            if filename.startswith(session_id):
                try:
                    os.remove(os.path.join(self.scratch_dir, filename))
                except OSError:
                    pass
