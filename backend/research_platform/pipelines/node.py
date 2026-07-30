"""Graph Pipeline Node Base Interface."""

from __future__ import annotations

from typing import Any, Dict


class PipelineNode:
    """Base interface for graph pipeline nodes."""

    def __init__(self, node_id: str, node_type: str) -> None:
        self.node_id = node_id
        self.node_type = node_type

    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError

    def validate(self, inputs: Dict[str, Any]) -> bool:
        return True

    def serialize(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "node_type": self.node_type,
        }

    def metadata(self) -> Dict[str, Any]:
        return {
            "name": self.__class__.__name__,
            "version": "1.0.0",
        }
