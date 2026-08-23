"""Backward-compatible re-export of Workflow DTOs from backend.research_platform.workflow.dto."""

from backend.research_platform.workflow.dto import (
    ResearchWorkflowDTO,
    WorkflowState,
)

__all__ = ["ResearchWorkflowDTO", "WorkflowState"]
