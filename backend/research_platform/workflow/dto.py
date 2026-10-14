"""Research Workflow DTO and State Machine Definitions.

This file used to be a byte-identical copy of
``backend/mech_platform/workflow/dto.py``. Two copies of a workflow DTO
is not a convenience: a state definition that differs between them is a
state machine that behaves differently depending on the import path.

Rather than maintain two copies, this module delegates. The implementation
lives in ``backend.mech_platform.workflow.dto``.
"""

from __future__ import annotations

from backend.mech_platform.workflow.dto import WorkflowState, ResearchWorkflowDTO

__all__ = ["WorkflowState", "ResearchWorkflowDTO"]