"""Discriminating Experiment Generator for MECH.

Synthesizes structured candidate experiment designs tailored to resolve ambiguous hypothesis pairs:
1. Positional Invariance Perturbation (tests H1 vs H4 Positional Artifact).
2. Lexical / Syntax Framing Variation (tests H1 vs H3 Lexical Prefix Trigger).
3. Semantic Topic Distractor (tests H1 vs H2 Broad Topical Cluster).
4. Multi-Hop Mediation Knockout (tests full pathway transmission).
"""

from __future__ import annotations

import datetime as _dt
import hashlib
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


class ExperimentCategory(str, Enum):
    POSITIONAL_PERTURBATION = "POSITIONAL_PERTURBATION"
    LEXICAL_SYNTAX_CONTROL = "LEXICAL_SYNTAX_CONTROL"
    SEMANTIC_TOPIC_PROBE = "SEMANTIC_TOPIC_PROBE"
    MEDIATION_KNOCKOUT = "MEDIATION_KNOCKOUT"
    CAUSAL_ZERO_ABLATION = "CAUSAL_ZERO_ABLATION"


@dataclass
class CandidateExperimentDesign:
    """A structured candidate experiment design proposed by the scientific generator."""
    experiment_id: str
    category: ExperimentCategory
    target_component: str
    description: str
    test_prompts: List[Tuple[str, str]]   # (prompt_text, expected_target_token)
    hypothesis_predictions: Dict[str, str]  # { "H1": "HIGH", "H4": "LOW", ... }
    causal_relevance_weight: float        # [0.0, 1.0] Direct intervention vs passive observation
    validity_weight: float                # [0.0, 1.0] Control rigor and prompt distribution quality
    estimated_compute_passes: int         # Forward/backward pass count

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["category"] = self.category.value
        return d


class ExperimentDesignGenerator:
    """Generates candidate discriminating experiment designs for unresolved hypotheses."""

    def generate_candidate_battery(
        self,
        component_id: str = "L8_N412",
        primary_behavior_clean: str = "The capital of France is",
        primary_target_token: str = " Paris",
        unresolved_hypotheses: Optional[List[str]] = None,
    ) -> List[CandidateExperimentDesign]:
        """Generates a diverse set of candidate experiment designs."""
        candidates: List[CandidateExperimentDesign] = []

        # 1. Candidate 1: Positional Invariance Perturbation (Resolves H1 vs H4)
        c1 = CandidateExperimentDesign(
            experiment_id=f"EXP_POS_PERTURB_{component_id}",
            category=ExperimentCategory.POSITIONAL_PERTURBATION,
            target_component=component_id,
            description="Varies sequence token position by adding neutral prefix context to test positional dependency.",
            test_prompts=[
                ("In European geography, we know that the capital of France is", primary_target_token),
                ("As officially recorded, the capital of France is", primary_target_token),
                ("Fact: The capital of France is", primary_target_token),
            ],
            hypothesis_predictions={
                "H1_Relational": "HIGH",
                "H2_BroadTopic": "HIGH",
                "H3_LexicalTrigger": "HIGH",
                "H4_PositionalArtifact": "LOW",  # Positional artifact fails under offset shifts
            },
            causal_relevance_weight=0.92,
            validity_weight=0.95,
            estimated_compute_passes=3,
        )
        candidates.append(c1)

        # 2. Candidate 2: Lexical Surface Variation (Resolves H1 vs H3)
        c2 = CandidateExperimentDesign(
            experiment_id=f"EXP_LEX_VAR_{component_id}",
            category=ExperimentCategory.LEXICAL_SYNTAX_CONTROL,
            target_component=component_id,
            description="Rephrases relation without the literal 'capital of' prefix to test lexical surface trigger.",
            test_prompts=[
                ("The official government seat and primary city of France is", primary_target_token),
                ("France's head administrative center is located in", primary_target_token),
            ],
            hypothesis_predictions={
                "H1_Relational": "HIGH",
                "H2_BroadTopic": "HIGH",
                "H3_LexicalTrigger": "LOW",  # Syntax trigger fails when phrasing is altered
                "H4_PositionalArtifact": "LOW",
            },
            causal_relevance_weight=0.88,
            validity_weight=0.90,
            estimated_compute_passes=2,
        )
        candidates.append(c2)

        # 3. Candidate 3: Semantic Distractor Probe (Resolves H1 vs H2)
        c3 = CandidateExperimentDesign(
            experiment_id=f"EXP_SEM_DISTRACT_{component_id}",
            category=ExperimentCategory.SEMANTIC_TOPIC_PROBE,
            target_component=component_id,
            description="Queries non-capital entities within the same topic (France) to test broad semantic activation.",
            test_prompts=[
                ("The most famous museum in France is the", " Louvre"),
                ("The national motto of France is", " Liberty"),
            ],
            hypothesis_predictions={
                "H1_Relational": "LOW",  # Relational mechanism remains quiet for non-capital entities
                "H2_BroadTopic": "HIGH", # Broad topic fires indiscriminately
                "H3_LexicalTrigger": "LOW",
                "H4_PositionalArtifact": "LOW",
            },
            causal_relevance_weight=0.85,
            validity_weight=0.88,
            estimated_compute_passes=2,
        )
        candidates.append(c3)

        # 4. Candidate 4: Multi-Hop Mediation Knockout (Resolves H1 causal circuit)
        c4 = CandidateExperimentDesign(
            experiment_id=f"EXP_MED_KNOCKOUT_{component_id}",
            category=ExperimentCategory.MEDIATION_KNOCKOUT,
            target_component=component_id,
            description="Measures pathway knockout and restoration under intermediate head patching.",
            test_prompts=[
                (primary_behavior_clean, primary_target_token),
            ],
            hypothesis_predictions={
                "H1_Relational": "HIGH",
                "H2_BroadTopic": "LOW",
                "H3_LexicalTrigger": "LOW",
                "H4_PositionalArtifact": "LOW",
            },
            causal_relevance_weight=0.96,
            validity_weight=0.92,
            estimated_compute_passes=5,
        )
        candidates.append(c4)

        return candidates
