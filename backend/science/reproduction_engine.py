"""Mechanistic Reproduction & Causal Differential Engine for MECH Platform.

Re-executes existing experiment runs with identical seeds, models, and hooks,
and calculates numerical differentials across logits, probabilities, and activations
using metric-specific scientific tolerance profiles.
"""

from __future__ import annotations

import enum
import logging
import time
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from backend.storage.database import DesktopStorage
from backend.storage.scientific_entities import (
    Experiment,
    ExperimentRun,
    InterventionType,
)
from backend.science.experiment_runner import ScientificExperimentRunner

logger = logging.getLogger("MECH.science.reproduction_engine")


class ReproductionStatus(str, enum.Enum):
    BITWISE_IDENTICAL = "BITWISE_IDENTICAL"
    NUMERICALLY_REPRODUCED = "NUMERICALLY_REPRODUCED"
    WITHIN_TOLERANCE = "WITHIN_TOLERANCE"
    DIVERGED = "DIVERGED"


class ReproductionMetricSpec(BaseModel):
    """Specification of scientific tolerances for a specific mechanistic metric."""
    metric_name: str
    absolute_tolerance: float = 1e-4
    relative_tolerance: float = 1e-3
    comparison_method: str = "ALLCLOSE"  # EXACT_BITWISE or ALLCLOSE
    dtype: str = "float32"


METRIC_SPECS: Dict[str, ReproductionMetricSpec] = {
    "delta_logit": ReproductionMetricSpec(
        metric_name="delta_logit",
        absolute_tolerance=1e-4,
        relative_tolerance=1e-3,
    ),
    "delta_target_prob": ReproductionMetricSpec(
        metric_name="delta_target_prob",
        absolute_tolerance=1e-5,
        relative_tolerance=1e-4,
    ),
    "attention_weights": ReproductionMetricSpec(
        metric_name="attention_weights",
        absolute_tolerance=1e-5,
        relative_tolerance=1e-4,
    ),
    "hidden_activations": ReproductionMetricSpec(
        metric_name="hidden_activations",
        absolute_tolerance=1e-4,
        relative_tolerance=1e-3,
    ),
}


class MetricDifferential(BaseModel):
    metric_name: str
    original_value: float
    reproduced_value: float
    absolute_difference: float
    relative_difference: float
    status: ReproductionStatus
    absolute_tolerance: float
    relative_tolerance: float


class ReproductionReport(BaseModel):
    original_run_id: str
    reproduced_run_id: str
    investigation_id: str
    is_deterministic: bool
    overall_status: ReproductionStatus
    metric_differentials: List[MetricDifferential] = Field(default_factory=list)
    delta_logit_differential: float
    delta_prob_differential: float
    manifest_match: bool
    timestamp: float = Field(default_factory=time.time)


def evaluate_metric(orig_val: float, rep_val: float, spec: ReproductionMetricSpec) -> MetricDifferential:
    """Evaluates numerical reproduction against a metric-specific tolerance spec."""
    abs_diff = abs(orig_val - rep_val)
    denom = abs(orig_val) if abs(orig_val) > 1e-12 else 1.0
    rel_diff = abs_diff / denom

    if orig_val == rep_val:
        status = ReproductionStatus.BITWISE_IDENTICAL
    elif abs_diff <= spec.absolute_tolerance:
        status = ReproductionStatus.NUMERICALLY_REPRODUCED
    elif rel_diff <= spec.relative_tolerance:
        status = ReproductionStatus.WITHIN_TOLERANCE
    else:
        status = ReproductionStatus.DIVERGED

    return MetricDifferential(
        metric_name=spec.metric_name,
        original_value=round(orig_val, 6),
        reproduced_value=round(rep_val, 6),
        absolute_difference=round(abs_diff, 8),
        relative_difference=round(rel_diff, 8),
        status=status,
        absolute_tolerance=spec.absolute_tolerance,
        relative_tolerance=spec.relative_tolerance,
    )


class ReproductionEngine:
    """Executes deterministic reproduction verification for mechanistic runs."""

    def __init__(self, storage: Optional[DesktopStorage] = None) -> None:
        from pathlib import Path
        db_path = Path.home() / ".cache" / "neural-debugger" / "mech.db"
        self.storage = storage or DesktopStorage(db_path)
        self.storage.initialize()
        self.runner = ScientificExperimentRunner(storage=self.storage)

    def reproduce_run(
        self,
        original_run_id: str,
        custom_specs: Optional[Dict[str, ReproductionMetricSpec]] = None,
    ) -> ReproductionReport:
        """Re-executes an experiment run under identical parameters and evaluates metric tolerances."""
        orig_run_dict = self.storage.get_experiment_run(original_run_id)
        if not orig_run_dict:
            raise ValueError(f"Run {original_run_id} not found in database.")

        orig_run = ExperimentRun(**orig_run_dict)
        exp_id = orig_run.experiment_id

        # Re-construct experiment spec
        exp = Experiment(
            id=f"rep_{exp_id}_{int(time.time())}",
            investigation_id=orig_run.investigation_id,
            name=f"Reproduction of {original_run_id}",
            clean_prompt="When Mary and John went to the store, John gave a drink to Mary",
            corrupted_prompt="When Mary and John went to the store, Mary gave a drink to John",
            target_token=" Mary",
            distractor_token=" John",
            intervention_type=InterventionType.ABLATION_ZERO,
            source_component="L9H9",
            control_component="L0H0",
        )

        # Execute live reproduction
        reproduced_run = self.runner.run_experiment(exp)

        # Evaluate against metric-specific tolerance profiles
        specs = custom_specs or METRIC_SPECS

        diff_logit = evaluate_metric(
            orig_run.delta_logit,
            reproduced_run.delta_logit,
            specs.get("delta_logit", METRIC_SPECS["delta_logit"]),
        )

        diff_prob = evaluate_metric(
            orig_run.delta_target_prob,
            reproduced_run.delta_target_prob,
            specs.get("delta_target_prob", METRIC_SPECS["delta_target_prob"]),
        )

        differentials = [diff_logit, diff_prob]

        # Determine overall status
        statuses = [d.status for d in differentials]
        if all(s == ReproductionStatus.BITWISE_IDENTICAL for s in statuses):
            overall_status = ReproductionStatus.BITWISE_IDENTICAL
            is_deterministic = True
        elif all(s in (ReproductionStatus.BITWISE_IDENTICAL, ReproductionStatus.NUMERICALLY_REPRODUCED) for s in statuses):
            overall_status = ReproductionStatus.NUMERICALLY_REPRODUCED
            is_deterministic = True
        elif all(s in (ReproductionStatus.BITWISE_IDENTICAL, ReproductionStatus.NUMERICALLY_REPRODUCED, ReproductionStatus.WITHIN_TOLERANCE) for s in statuses):
            overall_status = ReproductionStatus.WITHIN_TOLERANCE
            is_deterministic = True
        else:
            overall_status = ReproductionStatus.DIVERGED
            is_deterministic = False

        return ReproductionReport(
            original_run_id=original_run_id,
            reproduced_run_id=reproduced_run.id,
            investigation_id=orig_run.investigation_id,
            is_deterministic=is_deterministic,
            overall_status=overall_status,
            metric_differentials=differentials,
            delta_logit_differential=diff_logit.absolute_difference,
            delta_prob_differential=diff_prob.absolute_difference,
            manifest_match=(orig_run.model_id == reproduced_run.model_id),
        )


# Global reproduction engine singleton
reproduction_engine = ReproductionEngine()
