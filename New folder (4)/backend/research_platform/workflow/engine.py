"""Research Workflow State Machine Engine."""

from __future__ import annotations

import datetime as _dt
from typing import Any, Dict, Optional
from .dto import ResearchWorkflowDTO, WorkflowState


class WorkflowEngine:
    """State machine managing research workflow lifecycle transitions."""

    TRANSITIONS = {
        WorkflowState.DRAFT: [WorkflowState.QUESTION],
        WorkflowState.QUESTION: [WorkflowState.HYPOTHESIS],
        WorkflowState.HYPOTHESIS: [WorkflowState.EXPERIMENT],
        WorkflowState.EXPERIMENT: [WorkflowState.RUNNING],
        WorkflowState.RUNNING: [WorkflowState.OBSERVATION],
        WorkflowState.OBSERVATION: [WorkflowState.ANALYSIS],
        WorkflowState.ANALYSIS: [WorkflowState.CONCLUSION],
        WorkflowState.CONCLUSION: [WorkflowState.PUBLISHED],
    }

    def __init__(self) -> None:
        self._workflows: Dict[str, ResearchWorkflowDTO] = {}

    def create_workflow(self, workflow_id: str, title: str, question: str = "") -> Dict[str, Any]:
        wf = ResearchWorkflowDTO(workflow_id=workflow_id, title=title, question=question)
        self._workflows[workflow_id] = wf
        return wf.to_dict()

    def transition_state(self, workflow_id: str, target_state: str) -> Dict[str, Any]:
        wf = self._workflows.get(workflow_id)
        if not wf:
            raise KeyError(f"Workflow {workflow_id} not found")

        wf.state = target_state
        wf.updated_at = _dt.datetime.utcnow().isoformat() + "Z"
        return wf.to_dict()

    def get_workflow(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        wf = self._workflows.get(workflow_id)
        return wf.to_dict() if wf else None
