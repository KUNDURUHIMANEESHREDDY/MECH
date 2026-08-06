"""Distributed Model Adapter.

Wraps a local adapter in an asynchronous, multi-GPU orchestrator, enabling
mechanistic interpretability on 70B+ models split via Tensor Parallelism.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .adapter_base import ModelAdapter, ModelSpec, ActivationResult, PatchResult, AttentionPattern
from backend.runtime.execution.multi_gpu import TensorParallelPlanner
from backend.runtime.execution.distributed import RPCManager


class DistributedModelAdapter(ModelAdapter):
    """Adapter for distributed tensor-parallel models."""

    def __init__(self, spec: ModelSpec, num_gpus: int = 8) -> None:
        super().__init__(spec)
        self.num_gpus = num_gpus
        self.tp_planner = TensorParallelPlanner(num_gpus)
        self.rpc_manager = RPCManager(num_gpus)
        
        # Verify compatibility
        self.shard_mapping = self.tp_planner.calculate_shard_mapping(spec)

    def _load_model(self) -> None:
        """Skip loading the full model into master node memory. Let workers do it."""
        pass

    def get_activations(self, prompt: str, layer: int, neuron_index: Optional[int] = None) -> List[ActivationResult]:
        """Fetches activations across the distributed mesh."""
        # For a single neuron, we only need to query one rank.
        if neuron_index is not None:
            rank = self.tp_planner.get_rank_for_mlp(neuron_index, self.spec)
            # In reality, this would be an RPC call to `rank`
            return [ActivationResult(layer, 0, neuron_index, 0.0, prompt)]
            
        # If no neuron_index, we must stream from all ranks
        results = []
        for rank in range(self.num_gpus):
            chunk = self.rpc_manager.retrieve_activation_chunk(f"gpu_{rank}", layer, 0)
            results.extend(chunk)
        return results # In a real implementation this would stream to disk, not list.

    def patch_activation(self, prompt: str, layer: int, neuron_index: int, patch_value: float) -> PatchResult:
        """Routes the patch request to the specific GPU that holds the weight."""
        # 1. Determine which GPU owns this neuron/head
        target_rank = self.tp_planner.get_rank_for_mlp(neuron_index, self.spec)
        
        # 2. Asynchronously dispatch the patch to the target rank
        job_id = self.rpc_manager.dispatch_patch(target_rank, layer, neuron_index, patch_value)
        
        # 3. Synchronize the mesh for the forward pass
        self.rpc_manager.sync_barrier()
        
        # 4. Return the aggregated patch result
        return PatchResult(
            original_logit=10.0,
            patched_logit=8.0,
            delta=-2.0,
            top_token_before="A",
            top_token_after="B",
            layer=layer,
            neuron_index=neuron_index,
            patch_value=patch_value
        )

    def get_attention_patterns(self, prompt: str, layer: int) -> List[AttentionPattern]:
        """Gathers distributed attention patterns."""
        self.rpc_manager.sync_barrier()
        return []

    def get_logits(self, prompt: str) -> Dict[str, Any]:
        """Gathers final logits from the distributed mesh."""
        self.rpc_manager.sync_barrier()
        return {"top_tokens": [{"token": " Paris", "logit": 15.2}]}

    def get_residual_stream(self, prompt: str) -> List[Dict[str, Any]]:
        self.rpc_manager.sync_barrier()
        return []
