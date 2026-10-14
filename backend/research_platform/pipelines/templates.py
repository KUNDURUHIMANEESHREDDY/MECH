"""Built-in Interpretability Pipeline Templates.

This file used to be a byte-identical copy of
``backend/mech_platform/pipelines/templates.py``. Two copies of pipeline
templates is not a convenience: a template change in one and not the other
is a pipeline that behaves differently depending on the import path.

Rather than maintain two copies, this module delegates. The implementation
lives in ``backend.mech_platform.pipelines.templates``.
"""

from __future__ import annotations

from backend.mech_platform.pipelines.templates import PipelineTemplates

__all__ = ["PipelineTemplates"]