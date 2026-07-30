"""Multi-GPU Model Partitioning & Placement Planner.

Calculates Tensor Parallel (TP) shard boundaries based on ModelSpec constraints.
"""

from __future__ import annotations

from typing import Any, Dict, Optional
from backend.science.models.adapter_base import ModelSpec


class TensorParallelPlanner:
    """Plans tensor-parallel sharding across multiple GPUs for 70B+ models."""

    def __init__(self, num_gpus: int = 8) -> None:
        self.num_gpus = num_gpus

    def calculate_shard_mapping(self, spec: ModelSpec) -> Dict[str, Any]:
        """Calculates which GPU owns which attention heads and MLP slices.
        
        Example: A 70B model with 64 heads across 8 GPUs -> 8 heads per GPU.
        """
        if spec.num_heads % self.num_gpus != 0:
            raise ValueError(f"Number of heads ({spec.num_heads}) must be divisible by num_gpus ({self.num_gpus}) for strict TP.")
            
        heads_per_shard = spec.num_heads // self.num_gpus
        mlp_dim_per_shard = spec.d_mlp // self.num_gpus
        
        mapping = {}
        for rank in range(self.num_gpus):
            mapping[f"gpu_{rank}"] = {
                "head_start": rank * heads_per_shard,
                "head_end": (rank + 1) * heads_per_shard - 1,
                "mlp_start": rank * mlp_dim_per_shard,
                "mlp_end": (rank + 1) * mlp_dim_per_shard - 1,
            }
            
        return {
            "strategy": "TensorParallel",
            "world_size": self.num_gpus,
            "shard_mapping": mapping
        }

    def get_rank_for_head(self, head_index: int, spec: ModelSpec) -> str:
        """Determines which GPU rank owns a specific attention head."""
        heads_per_shard = spec.num_heads // self.num_gpus
        rank = head_index // heads_per_shard
        return f"gpu_{rank}"
        
    def get_rank_for_mlp(self, neuron_index: int, spec: ModelSpec) -> str:
        """Determines which GPU rank owns a specific MLP neuron."""
        mlp_dim_per_shard = spec.d_mlp // self.num_gpus
        rank = neuron_index // mlp_dim_per_shard
        return f"gpu_{rank}"


# Alias expected by backend.runtime.engine
MultiGPUPlacementPlanner = TensorParallelPlanner
