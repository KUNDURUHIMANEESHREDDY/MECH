r"""Scientific Priority Engine & Open-World Utility Optimizer for MECH.

Ranks scientific questions by multi-factor expected utility:
    U(Q) = [ EIG(Q) * CausalRelevance(Q) * Novelty(Q) * BoundaryValue(Q) * OpenWorldValue(Q) ] / Cost(Q)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from .open_question_generator import ScientificQuestion


@dataclass
class PrioritizedQuestionRecord:
    question: ScientificQuestion
    utility_score: float
    rank: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "question": self.question.to_dict(),
            "utility_score": round(self.utility_score, 4),
            "rank": self.rank,
        }


class ScientificPriorityEngine:
    """Calculates scientific inquiry utility and prioritizes research agendas."""

    def prioritize_questions(
        self,
        questions: List[ScientificQuestion],
    ) -> List[PrioritizedQuestionRecord]:
        """Calculates U(Q) and sorts questions in descending priority order."""
        scored: List[Tuple[ScientificQuestion, float]] = []

        for q in questions:
            numerator = (
                q.expected_information_gain
                * q.causal_relevance
                * q.novelty
                * q.boundary_value
                * q.open_world_value
            )
            denominator = max(0.1, q.estimated_experiment_cost)
            utility = numerator / denominator
            scored.append((q, utility))

        scored.sort(key=lambda x: x[1], reverse=True)

        records: List[PrioritizedQuestionRecord] = []
        for rank, (q, u) in enumerate(scored, start=1):
            records.append(PrioritizedQuestionRecord(question=q, utility_score=u, rank=rank))

        return records
