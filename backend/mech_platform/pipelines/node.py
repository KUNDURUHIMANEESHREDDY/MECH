"""Backward-compatible re-export of PipelineNode from backend.research_platform.pipelines.node."""

from backend.research_platform.pipelines.node import (
    PipelineNode,
    NodeStatus,
)

__all__ = ["PipelineNode", "NodeStatus"]
