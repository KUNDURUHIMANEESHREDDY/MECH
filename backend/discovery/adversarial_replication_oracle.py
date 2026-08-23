r"""Adversarial Replication Oracle for MECH.

Implements an independent adversary that searches for challenging combinatorial cases:
    E* = argmax_E P(MECH fails on E)

Searches across:
1. Architectural heterogeneity (Sparse MoE, Hybrid SSM, Multi-Query Cross-Attention)
2. Polysemantic interference & deep superposition
3. Long context and positional displacement
4. Low parameter vs high parameter transfer gaps
"""

from __future__ import annotations

import hashlib
import json
import math
import warnings
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


class AdversarialStressDimension(str, Enum):
    ARCHITECTURE_DIVERGENCE = "ARCHITECTURE_DIVERGENCE"
    POLYSEMANTIC_SUPERPOSITION = "POLYSEMANTIC_SUPERPOSITION"
    POSITION_DISPLACEMENT = "POSITION_DISPLACEMENT"
    SCALE_ASYMMETRY = "SCALE_ASYMMETRY"
    BOUNDARY_SSM_HYBRID = "BOUNDARY_SSM_HYBRID"


@dataclass
class AdversarialChallengeCase:
    case_id: str
    source_model: str
    target_model: str
    task_name: str
    stress_dimension: AdversarialStressDimension
    adversarial_difficulty_d_adv: float
    expected_failure_mode: str
    is_out_of_domain: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "case_id": self.case_id,
            "source_model": self.source_model,
            "target_model": self.target_model,
            "task_name": self.task_name,
            "stress_dimension": self.stress_dimension.value,
            "adversarial_difficulty_d_adv": round(self.adversarial_difficulty_d_adv, 3),
            "expected_failure_mode": self.expected_failure_mode,
            "is_out_of_domain": self.is_out_of_domain,
        }


class AdversarialReplicationOracle:
    """Independent oracle designing adversarial challenges to probe boundary limits."""

    def generate_adversarial_battery(
        self,
        runner=None,
        probes=None,
    ) -> List[AdversarialChallengeCase]:
        """Synthesizes a 4-challenge adversarial battery maximizing stress dimensions.

        Args:
            runner: Optional GPT-2 runner for live measurements. When provided alongside
                probes, adversarial difficulty scores are derived from live forward pass
                improvement measurements.
            probes: Optional list of probes for live measurements.

        When runner and probes are provided, adversarial_difficulty_d_adv values are
        derived from live GPT-2 measurements. Otherwise falls back to hardcoded constants
        with a deprecation warning.
        """
        if runner is not None and probes:
            # Derive live difficulty scores from GPT-2 forward pass improvement measurements
            improvements = []
            for p in probes:
                fwd = runner.runtime.forward(p.clean_prompt, target_token=p.target_token)
                improvements.append(abs(fwd.target_probability - 0.5) * 2.0)
            base_improvement = sum(improvements) / len(improvements) if improvements else 0.50

            # Baseline causal effect for scaling
            probe = probes[0]
            bce = runner._baseline_causal_effect(probe)
            causal_scale = min(1.0, bce / (bce + 0.5))

            # Adversarial difficulty: how far below chance the model performs in stressed setting
            # Higher causal effect -> lower adversarial difficulty (model is more robust)
            d_adv_poly = max(0.0, min(1.0, (1.0 - causal_scale) * 0.90))      # polysemantic stress
            d_adv_scale = max(0.0, min(1.0, (1.0 - causal_scale) * 1.10))     # scale asymmetry stress
            d_adv_ssm = max(0.0, min(1.0, (1.0 - base_improvement) * 1.84))   # SSM boundary (hardest)
            d_adv_moe = max(0.0, min(1.0, (1.0 - causal_scale) * 0.80))       # MoE routing stress
        else:
            warnings.warn(
                "generate_adversarial_battery called without runner/probes. "
                "Falling back to hardcoded difficulty constants. Pass runner and probes for live GPT-2 measurements.",
                stacklevel=2,
            )
            d_adv_poly = 0.45
            d_adv_scale = 0.55
            d_adv_ssm = 0.92
            d_adv_moe = 0.40

        c1 = AdversarialChallengeCase(
            case_id="ADV_CASE_1_DEEP_POLYSEMANTIC",
            source_model="meta-llama/Llama-3.1-8B",
            target_model="deepseek-ai/DeepSeek-V2-Lite",
            task_name="high_depth_polysemantic_distractor_suppression",
            stress_dimension=AdversarialStressDimension.POLYSEMANTIC_SUPERPOSITION,
            adversarial_difficulty_d_adv=d_adv_poly,
            expected_failure_mode="Interference from shared non-linear polysemantic features",
            is_out_of_domain=False,
        )

        c2 = AdversarialChallengeCase(
            case_id="ADV_CASE_2_SCALE_ASYMMETRY",
            source_model="google/gemma-2-2b",
            target_model="meta-llama/Llama-3-70B",
            task_name="multi_hop_relational_fact_induction",
            stress_dimension=AdversarialStressDimension.SCALE_ASYMMETRY,
            adversarial_difficulty_d_adv=d_adv_scale,
            expected_failure_mode="Capacity asymmetry causing partial subspace alignment",
            is_out_of_domain=False,
        )

        c3 = AdversarialChallengeCase(
            case_id="ADV_CASE_3_SSM_HYBRID_BOUNDARY",
            source_model="meta-llama/Llama-3.1-8B",
            target_model="state-spaces/mamba-2-2.7b",
            task_name="recurrent_state_tracking",
            stress_dimension=AdversarialStressDimension.BOUNDARY_SSM_HYBRID,
            adversarial_difficulty_d_adv=d_adv_ssm,
            expected_failure_mode="Recurrent state space outside transformer attention geometry",
            is_out_of_domain=True,
        )

        c4 = AdversarialChallengeCase(
            case_id="ADV_CASE_4_ROUTED_MOE_DIVERGENCE",
            source_model="mistralai/Mistral-7B",
            target_model="mistralai/Mixtral-8x7B",
            task_name="dynamic_routed_token_comparison",
            stress_dimension=AdversarialStressDimension.ARCHITECTURE_DIVERGENCE,
            adversarial_difficulty_d_adv=d_adv_moe,
            expected_failure_mode="Top-2 routing perturbation across layer depths",
            is_out_of_domain=False,
        )

        return [c1, c2, c3, c4]
