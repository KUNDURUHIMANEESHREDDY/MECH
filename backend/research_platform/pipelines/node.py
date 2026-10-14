"""Graph Pipeline Node Base Interface.

This file used to be a byte-identical copy of
``backend/mech_platform/pipelines/node.py``. Two copies of a pipeline
node interface is not a convenience: an interface change in one and not
the other breaks callers that import the other.

Rather than maintain two copies, this module delegates. The implementation
lives in ``backend.mech_platform.pipelines.node``.
"""

from __future__ import annotations

from backend.mech_platform.pipelines.node import PipelineNode

__all__ = ["PipelineNode"]