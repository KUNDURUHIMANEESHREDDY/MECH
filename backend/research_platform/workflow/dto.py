"""Research Workflow DTO and State Machine Definitions."""

from __future__ import annotations

import datetime as _dt
from typing import Any, Dict, List, Optional


class WorkflowState:
    DRAFT = "Draft"
    QUESTION = "Question"
    HYPOTHESIS = "Hypothesis"
    EXPERIMENT = "Experiment"
    RUNNING = "Running"
    OBSERVATION = "Observation"
    ANALYSIS = "Analysis"
    CONCLUSION = "Conclusion"
    PUBLISHED = "Published"


class ResearchWorkflowDTO:
    """DTO representing a structured research workflow lifecycle."""

    def __init__(
        self,
        workflow_id: str,
        title: str,
        question: str = "",
        state: str = WorkflowState.DRAFT,
    ) -> None:
        self.workflow_id = workflow_id
        self.title = title
        self.question = question
        self.state = state
        self.hypotheses: List[Dict[str, Any]] = []
        self.experiments: List[Dict[str, Any]] = []
        self.observations: List[Dict[str, Any]] = []
        self.conclusions: List[Dict[str, Any]] = []
        self.artifacts: List[Dict[str, Any]] = []
        self.created_at = _dt.datetime.utcnow().isoformat() + "Z"
        self.updated_at = self.created_at

    def to_dict(self) -> Dict[str, Any]:
        return {
            "workflow_id": self.workflow_id,
            "title": self.title,
            "question": self.question,
            "state": self.state,
            "hypotheses": self.hypotheses,
            "experiments": self.experiments,
            "observations": self.observations,
            "conclusions": self.conclusions,
            "artifacts": self.artifacts,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }
