"""Graph Pipeline Execution Engine.

This file used to be a byte-identical copy of
``backend/mech_platform/pipelines/engine.py``. Two copies of a pipeline
engine is not a convenience: a stage execution bug fixed in one and not the
other is a bug that can be triggered by importing the other.

Rather than patch the same logic twice and leave two implementations to drift
again, this module delegates. The implementation lives in
``backend.mech_platform.pipelines.engine``.
"""

from __future__ import annotations

from backend.mech_platform.pipelines.engine import GraphPipelineEngine

__all__ = ["GraphPipelineEngine"]