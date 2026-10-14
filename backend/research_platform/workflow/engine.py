"""Research Workflow State Machine Engine.

This file used to be a byte-identical copy of
``backend/mech_platform/workflow/engine.py``. Two copies of a workflow
engine is not a convenience: a validation rule fixed in one and not the other
is a rule that can be bypassed by importing the other.

Rather than patch the same logic twice and leave two implementations to drift
again, this module delegates. The implementation lives in
``backend.mech_platform.workflow.engine``.
"""

from __future__ import annotations

from backend.mech_platform.workflow.engine import WorkflowEngine

__all__ = ["WorkflowEngine"]