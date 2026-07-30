"""Activation Patching & PatchSet Engine.

Implements activation interventions: replace, add, multiply, zero, mask, custom operations,
and intervention history logging, with full support for distributed routing.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


class ActivationPatch:
    """Represents a single activation intervention on a layer/component/neuron."""

    def __init__(
        self,
        session_id: str,
        layer: int,
        component: str,
        neuron_index: int,
        operation: str = "replace",
        value: float = 0.0,
    ) -> None:
        valid_ops = {"replace", "add", "multiply", "zero", "mask", "custom"}
        if operation not in valid_ops:
            raise ValueError(f"Invalid operation {operation}. Must be one of {valid_ops}")

        self.session_id = session_id
        self.layer = layer
        self.component = component
        self.neuron_index = neuron_index
        self.operation = operation
        self.value = value

    def apply(self, original_activation: float) -> float:
        if self.operation == "replace":
            return self.value
        if self.operation == "add":
            return original_activation + self.value
        if self.operation == "multiply":
            return original_activation * self.value
        if self.operation == "zero":
            return 0.0
        if self.operation == "mask":
            return original_activation if self.value > 0 else 0.0
        # custom
        return self.value

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "layer": self.layer,
            "component": self.component,
            "neuron_index": self.neuron_index,
            "operation": self.operation,
            "value": self.value,
            "is_distributed": False
        }


class DistributedPatch(ActivationPatch):
    """An activation patch aware of its destination in a tensor-parallel mesh."""
    
    def __init__(self, target_rank: str, **kwargs) -> None:
        super().__init__(**kwargs)
        self.target_rank = target_rank
        
    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d["is_distributed"] = True
        d["target_rank"] = self.target_rank
        return d


class PatchSet:
    """Container for managing multiple simultaneous activation interventions."""

    def __init__(self, patches: Optional[List[ActivationPatch]] = None) -> None:
        self.patches: List[ActivationPatch] = patches or []

    def add_patch(self, patch: ActivationPatch) -> None:
        self.patches.append(patch)

    def apply_patches(self, layer: int, component: str, neuron_index: int, activation: float) -> float:
        current = activation
        for patch in self.patches:
            if patch.layer == layer and patch.component == component and patch.neuron_index == neuron_index:
                current = patch.apply(current)
        return current
        
    def get_patches_for_rank(self, rank: str) -> List[DistributedPatch]:
        """Returns only patches destined for a specific distributed worker."""
        return [p for p in self.patches if isinstance(p, DistributedPatch) and p.target_rank == rank]

    def to_list(self) -> List[Dict[str, Any]]:
        return [p.to_dict() for p in self.patches]
