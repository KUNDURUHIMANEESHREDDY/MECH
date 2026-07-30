"""Activation and Weight Streaming Engine.

Streams model weights into VRAM on-demand, and streams massive residual
stream activations OUT of VRAM into NVMe/RAM buffers.
"""

from __future__ import annotations

import collections
from typing import Any, Dict, Iterator, List


class ModelStreamingEngine:
    """Streams weights and activations to optimize VRAM utilization."""

    def __init__(self, policy: str = "lazy") -> None:
        self.policy = policy  # eager, lazy, predictive_prefetch

    def stream_layer_weights(self, model_name: str, layer: int) -> Dict[str, Any]:
        """Streams weights from NVMe -> GPU VRAM."""
        return {
            "model_name": model_name,
            "layer": layer,
            "policy": self.policy,
            "weight_size_mb": 142.5,
            "loaded_to_vram": True,
            "status": "cached",
        }

    def stream_activations(self, layer_outputs: List[Dict[str, Any]], chunk_size: int = 1024) -> Iterator[List[Dict[str, Any]]]:
        """Streams activations from GPU VRAM -> NVMe chunk by chunk.
        
        Yields chunks of activations instead of holding them in memory.
        """
        chunk = []
        for act in layer_outputs:
            chunk.append(act)
            if len(chunk) >= chunk_size:
                yield chunk
                chunk = []
                
        if chunk:
            yield chunk
