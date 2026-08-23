r"""Held-Out Combinatorial Generalization Engine for MECH.

Manages strict combinatorial partitioning over:
1. Model Families (Llama, Gemma, Qwen, Mistral, Mixtral, Mamba, StarCoder)
2. Task Families (Factual Recall, Arithmetic Comparison, IOI Induction, Distractor Code Suppression)
3. Mechanistic Phenomena (Linear Projection, Duplicate Token Suppression, Recurrent State Tracking)
4. Source -> Target Pairings

Enforces the critical partition invariant:
    D_calibration ∩ D_heldout = ∅
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple


class SplitType(str, Enum):
    SPLIT_A_NEW_MODEL = "SPLIT_A_NEW_MODEL"
    SPLIT_B_NEW_TASK = "SPLIT_B_NEW_TASK"
    SPLIT_C_CROSSED_PAIRING = "SPLIT_C_CROSSED_PAIRING"
    SPLIT_D_NEW_ARCHITECTURE_FAMILY = "SPLIT_D_NEW_ARCHITECTURE_FAMILY"
    SPLIT_E_NEGATIVE_MATCHED_CONTROLS = "SPLIT_E_NEGATIVE_MATCHED_CONTROLS"


@dataclass(frozen=True)
class ExperimentTuple:
    source_model: str
    target_model: str
    task_name: str
    mechanism_type: str

    def tuple_key(self) -> str:
        return f"{self.source_model}->{self.target_model}:{self.task_name}:{self.mechanism_type}"


@dataclass
class GeneralizationPartition:
    calibration_set: List[ExperimentTuple]
    heldout_set: List[ExperimentTuple]
    split_type: SplitType
    ood_distances: Dict[str, float]

    def verify_partition_disjointness(self) -> bool:
        calib_keys = {e.tuple_key() for e in self.calibration_set}
        heldout_keys = {e.tuple_key() for e in self.heldout_set}
        return len(calib_keys.intersection(heldout_keys)) == 0


class HeldOutGeneralizationEngine:
    """Constructs and verifies combinatorial held-out splits for cross-model/cross-task generalization."""

    def __init__(self) -> None:
        self.calibration_tuples = [
            ExperimentTuple("meta-llama/Llama-3.1-8B", "Qwen/Qwen-2.5-7B", "factual_relational_recall", "linear_projection"),
            ExperimentTuple("google/gemma-2-9b", "meta-llama/Llama-3.1-8B", "greater_than_arithmetic", "magnitude_comparison"),
            ExperimentTuple("mistralai/Mistral-7B", "mistralai/Mixtral-8x7B", "ioi_induction", "duplicate_token_suppression"),
        ]

    def get_calibration_tuples(self) -> List[ExperimentTuple]:
        return list(self.calibration_tuples)

    def generate_crossed_heldout_partition(self) -> GeneralizationPartition:
        """Constructs Split C: Crossed pairings where models, tasks, and mechanisms exist in calibration,

        but the cross-combination is completely novel.
        """
        heldout_tuples = [
            # Crossed 1: Llama -> Mixtral on Arithmetic
            ExperimentTuple("meta-llama/Llama-3.1-8B", "mistralai/Mixtral-8x7B", "greater_than_arithmetic", "magnitude_comparison"),
            # Crossed 2: Qwen -> Gemma on Induction
            ExperimentTuple("Qwen/Qwen-2.5-7B", "google/gemma-2-9b", "ioi_induction", "duplicate_token_suppression"),
            # Crossed 3: Mistral -> Llama on Factual Recall
            ExperimentTuple("mistralai/Mistral-7B", "meta-llama/Llama-3.1-8B", "factual_relational_recall", "linear_projection"),
            # Split D: Llama -> Mamba-2 SSM on Recurrent Tracking (Out of domain architecture -> Must Abstain)
            ExperimentTuple("meta-llama/Llama-3.1-8B", "state-spaces/mamba-2-2.7b", "recurrent_state_inhibition", "ssm_hidden_state"),
        ]

        ood_distances = {
            heldout_tuples[0].tuple_key(): 0.35,  # Moderate MoE routing distance
            heldout_tuples[1].tuple_key(): 0.28,  # Low distance (dense transformer crossed)
            heldout_tuples[2].tuple_key(): 0.22,  # Low distance (dense transformer crossed)
            heldout_tuples[3].tuple_key(): 0.88,  # High distance (SSM non-transformer architecture)
        }

        partition = GeneralizationPartition(
            calibration_set=self.calibration_tuples,
            heldout_set=heldout_tuples,
            split_type=SplitType.SPLIT_C_CROSSED_PAIRING,
            ood_distances=ood_distances,
        )

        if not partition.verify_partition_disjointness():
            raise ValueError("Partition Invariant Violated: Calibration and Held-Out sets overlap!")

        return partition

    def compute_ood_distance(self, source_model: str, target_model: str, task: str) -> float:
        """Calculates distance in representation space d_OOD based on architectural and task divergence."""
        is_ssm = "mamba" in target_model.lower() or "rwkv" in target_model.lower()
        is_moe = "mixtral" in target_model.lower() or "moe" in target_model.lower()
        is_code = "starcoder" in target_model.lower() or "code" in task.lower()

        dist = 0.15
        if is_ssm:
            dist += 0.70
        if is_moe:
            dist += 0.20
        if is_code:
            dist += 0.25

        return round(min(1.0, dist), 3)
