"""In-memory activation cache keyed by activation name."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Iterable, Iterator, Optional

import torch


@dataclass(frozen=True)
class ActivationMetadata:
    """Metadata describing one captured tensor."""

    name: str
    kind: str
    layer: Optional[int]
    shape: tuple[int, ...]
    dtype: str
    device: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ActivationRecord:
    """Captured tensor plus lookup metadata."""

    tensor: torch.Tensor
    metadata: ActivationMetadata


class ActivationCache:
    """Stores detached activations captured during a forward pass."""

    def __init__(self, *, clone_tensors: bool = True, move_to_cpu: bool = True) -> None:
        self._records: Dict[str, ActivationRecord] = {}
        self._clone_tensors = clone_tensors
        self._move_to_cpu = move_to_cpu

    def store(
        self,
        name: str,
        tensor: torch.Tensor,
        *,
        kind: str,
        layer: Optional[int] = None,
    ) -> None:
        captured = tensor.detach()
        if self._move_to_cpu:
            captured = captured.cpu()
        if self._clone_tensors:
            captured = captured.clone()

        metadata = ActivationMetadata(
            name=name,
            kind=kind,
            layer=layer,
            shape=tuple(captured.shape),
            dtype=str(captured.dtype),
            device=str(captured.device),
        )
        self._records[name] = ActivationRecord(tensor=captured, metadata=metadata)

    def get(self, name: str) -> ActivationRecord:
        return self._records[name]

    def keys(self) -> Iterable[str]:
        return self._records.keys()

    def values(self) -> Iterable[ActivationRecord]:
        return self._records.values()

    def items(self) -> Iterable[tuple[str, ActivationRecord]]:
        return self._records.items()

    def tensors(self) -> dict[str, torch.Tensor]:
        return {name: record.tensor for name, record in self._records.items()}

    def metadata(self) -> dict[str, dict[str, object]]:
        return {
            name: record.metadata.to_dict()
            for name, record in self._records.items()
        }

    def __contains__(self, name: object) -> bool:
        return name in self._records

    def __iter__(self) -> Iterator[str]:
        return iter(self._records)

    def __len__(self) -> int:
        return len(self._records)
