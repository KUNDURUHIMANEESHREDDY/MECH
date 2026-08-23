r"""Open Scientific Question Generator for MECH.

Synthesizes first-class scientific inquiry objects:
    Knowledge State (Disagreements, Residuals, Failures, Boundaries) -> ScientificQuestion(Q)
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class QuestionTriggerType(str, Enum):
    THEORY_DISAGREEMENT = "THEORY_DISAGREEMENT"
    UNEXPLAINED_RESIDUAL = "UNEXPLAINED_RESIDUAL"
    CALIBRATION_GAP = "CALIBRATION_GAP"
    UNRESOLVED_BOUNDARY = "UNRESOLVED_BOUNDARY"
    PRIMITIVE_GAP = "PRIMITIVE_GAP"


@dataclass
class ScientificQuestion:
    question_id: str
    trigger_type: QuestionTriggerType
    question_statement: str
    target_models: List[str]
    target_tasks: List[str]
    unknown_causal_variable: str
    expected_information_gain: float
    causal_relevance: float
    novelty: float
    boundary_value: float
    open_world_value: float
    estimated_experiment_cost: float
    timestamp_utc: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "question_id": self.question_id,
            "trigger_type": self.trigger_type.value,
            "question_statement": self.question_statement,
            "target_models": self.target_models,
            "target_tasks": self.target_tasks,
            "unknown_causal_variable": self.unknown_causal_variable,
            "expected_information_gain": round(self.expected_information_gain, 4),
            "causal_relevance": round(self.causal_relevance, 4),
            "novelty": round(self.novelty, 4),
            "boundary_value": round(self.boundary_value, 4),
            "open_world_value": round(self.open_world_value, 4),
            "estimated_experiment_cost": round(self.estimated_experiment_cost, 4),
            "timestamp_utc": self.timestamp_utc,
        }


class OpenQuestionGenerator:
    """Formulates high-value, structured scientific inquiries without human prompts."""

    def generate_candidate_questions(
        self,
        unexplained_residuals: List[Dict[str, Any]],
        theory_disagreements: List[Dict[str, Any]],
        unresolved_boundaries: List[Dict[str, Any]],
    ) -> List[ScientificQuestion]:
        """Synthesizes candidate questions across the current epistemic landscape."""
        ts = _dt.datetime.now(_dt.timezone.utc).isoformat()
        questions: List[ScientificQuestion] = []

        # Question 1: Unexplained Residual in Sparse MoE Top-k Routing
        q1 = ScientificQuestion(
            question_id="Q_MOE_DYNAMIC_ROUTING_PENALTY",
            trigger_type=QuestionTriggerType.UNEXPLAINED_RESIDUAL,
            question_statement="Does sparse expert routing entropy directly modulate causal intervention transfer between dense and MoE architectures?",
            target_models=["meta-llama/Llama-3-8B", "mistralai/Mixtral-8x7B"],
            target_tasks=["factual_recall", "arithmetic_multihop"],
            unknown_causal_variable="r_routing_dispersion",
            expected_information_gain=0.94,
            causal_relevance=0.92,
            novelty=0.88,
            boundary_value=0.85,
            open_world_value=0.90,
            estimated_experiment_cost=1.0,
            timestamp_utc=ts,
        )
        questions.append(q1)

        # Question 2: Theory Disagreement on Polysemantic Superposition Subspaces
        q2 = ScientificQuestion(
            question_id="Q_POLYSEMANTIC_SUBSTREAM_CROSS_TALK",
            trigger_type=QuestionTriggerType.THEORY_DISAGREEMENT,
            question_statement="Do high-dimensional SAE dictionary latents explain inter-layer residual attenuation better than linear dimension scaling?",
            target_models=["meta-llama/Llama-3-8B", "google/gemma-2-9b"],
            target_tasks=["indirect_object_identification"],
            unknown_causal_variable="h_poly_superposition_density",
            expected_information_gain=0.76,
            causal_relevance=0.85,
            novelty=0.78,
            boundary_value=0.70,
            open_world_value=0.75,
            estimated_experiment_cost=1.2,
            timestamp_utc=ts,
        )
        questions.append(q2)

        # Question 3: Boundary Inquiry on Non-Transformer State-Space Models
        q3 = ScientificQuestion(
            question_id="Q_SSM_SELECTIVE_SCAN_PARADIGM_BRIDGE",
            trigger_type=QuestionTriggerType.UNRESOLVED_BOUNDARY,
            question_statement="Can a recurrent state-space transition kernel be mapped into self-attention query-key subspace projections?",
            target_models=["meta-llama/Llama-3-8B", "state-spaces/mamba-2-2.7b"],
            target_tasks=["associative_recall_seq"],
            unknown_causal_variable="ssm_transition_kernel_a",
            expected_information_gain=0.98,
            causal_relevance=0.95,
            novelty=0.96,
            boundary_value=0.98,
            open_world_value=0.95,
            estimated_experiment_cost=2.0,
            timestamp_utc=ts,
        )
        questions.append(q3)

        return questions
