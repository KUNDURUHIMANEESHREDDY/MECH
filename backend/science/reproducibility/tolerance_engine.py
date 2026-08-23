"""Numerical Tolerance Engine for Scientific Reproduction Verification.

Evaluates reproduction runs using rigorous component-level floating point
tolerances and exact rank assertions rather than a coarse percentage score.
"""

from __future__ import annotations

import datetime as _dt
import math
from typing import Any, Dict, List, Optional, Tuple

from .types import (
    ComponentToleranceResult,
    ImmutableExperimentRun,
    ReproductionComparisonReport,
)

# Standard scientific numerical tolerances for FP32/FP16 transformer executions
DEFAULT_ABS_TOLERANCE = 1e-4
DEFAULT_REL_TOLERANCE = 1e-3
CONTROL_ABS_TOLERANCE = 5e-4
MEDIATION_ABS_TOLERANCE = 1e-3


def _compare_float(
    name: str,
    expected: float,
    observed: float,
    abs_tol: float = DEFAULT_ABS_TOLERANCE,
    rel_tol: float = DEFAULT_REL_TOLERANCE,
) -> ComponentToleranceResult:
    abs_diff = abs(observed - expected)
    denominator = max(abs(expected), 1e-9)
    rel_diff = abs_diff / denominator

    passed = (abs_diff <= abs_tol) or (rel_diff <= rel_tol)
    if abs_diff == 0.0:
        status = "EXACT"
        details = f"Exact match (0.0 difference)"
    elif passed:
        status = "WITHIN_TOLERANCE"
        details = f"Within tolerance (abs: {abs_diff:.2e} <= {abs_tol:.2e}, rel: {rel_diff:.2e} <= {rel_tol:.2e})"
    else:
        status = "MISMATCH"
        details = f"Exceeded tolerance (abs: {abs_diff:.2e} > {abs_tol:.2e}, rel: {rel_diff:.2e} > {rel_tol:.2e})"

    return ComponentToleranceResult(
        metric_name=name,
        expected_value=round(expected, 6),
        observed_value=round(observed, 6),
        abs_diff=round(abs_diff, 6),
        rel_diff=round(rel_diff, 6),
        tolerance_threshold=f"abs<={abs_tol:.1e} | rel<={rel_tol:.1e}",
        passed=passed,
        status=status,
        details=details,
    )


def _compare_exact(name: str, expected: Any, observed: Any) -> ComponentToleranceResult:
    passed = expected == observed
    status = "EXACT" if passed else "MISMATCH"
    details = f"Exact equality check: '{expected}' == '{observed}'" if passed else f"Mismatch: expected '{expected}', got '{observed}'"

    return ComponentToleranceResult(
        metric_name=name,
        expected_value=expected,
        observed_value=observed,
        abs_diff=None,
        rel_diff=None,
        tolerance_threshold="EXACT_EQUALITY",
        passed=passed,
        status=status,
        details=details,
    )


class ReproductionToleranceEngine:
    """Compares original experiment runs against reproduction runs using strict tolerances."""

    @staticmethod
    def compare_runs(
        original: ImmutableExperimentRun,
        reproduction: ImmutableExperimentRun,
        duration_ms: float = 0.0,
    ) -> ReproductionComparisonReport:
        components: List[ComponentToleranceResult] = []

        orig_m = original.measurements
        repro_m = reproduction.measurements

        # 1. Target rank (Strict exact integer match)
        if "clean_rank" in orig_m and "clean_rank" in repro_m:
            components.append(_compare_exact("clean_target_rank", orig_m["clean_rank"], repro_m["clean_rank"]))
        if "intervened_rank" in orig_m and "intervened_rank" in repro_m:
            components.append(_compare_exact("intervened_target_rank", orig_m["intervened_rank"], repro_m["intervened_rank"]))

        # 2. Clean and intervened logits
        if "clean_logit" in orig_m and "clean_logit" in repro_m:
            components.append(_compare_float("clean_logit", float(orig_m["clean_logit"]), float(repro_m["clean_logit"])))
        if "intervened_logit" in orig_m and "intervened_logit" in repro_m:
            components.append(_compare_float("intervened_logit", float(orig_m["intervened_logit"]), float(repro_m["intervened_logit"])))

        # 3. Delta logit / causal effect (Δz)
        if "delta_logit" in orig_m and "delta_logit" in repro_m:
            components.append(_compare_float("delta_logit_causal_effect", float(orig_m["delta_logit"]), float(repro_m["delta_logit"])))

        # 4. Probabilities
        if "clean_probability" in orig_m and "clean_probability" in repro_m:
            components.append(_compare_float("clean_probability", float(orig_m["clean_probability"]), float(repro_m["clean_probability"])))
        if "intervened_probability" in orig_m and "intervened_probability" in repro_m:
            components.append(_compare_float("intervened_probability", float(orig_m["intervened_probability"]), float(repro_m["intervened_probability"])))

        # 5. Provenance: Mediation rescue & null distribution
        components.append(_compare_float(
            "mediation_rescue_fraction",
            original.provenance_chain.mediation_rescue_fraction,
            reproduction.provenance_chain.mediation_rescue_fraction,
            abs_tol=MEDIATION_ABS_TOLERANCE,
        ))
        components.append(_compare_float(
            "null_distribution_percentile",
            original.provenance_chain.null_distribution_percentile,
            reproduction.provenance_chain.null_distribution_percentile,
            abs_tol=1e-2,
        ))

        # 6. Evidence tier (Exact qualitative tier match)
        components.append(_compare_exact(
            "final_evidence_tier",
            original.provenance_chain.final_evidence_tier,
            reproduction.provenance_chain.final_evidence_tier,
        ))

        # Model, Environment, and Strategy matching
        model_matched = (
            original.model.model_id == reproduction.model.model_id
            and original.model.weights_hash == reproduction.model.weights_hash
        )
        env_matched = (
            original.environment.pytorch_version == reproduction.environment.pytorch_version
            and original.environment.python_version.split()[0] == reproduction.environment.python_version.split()[0]
        )

        orig_strat = original.runtime_strategy
        repro_strat = reproduction.runtime_strategy
        strategy_matched = (
            orig_strat.execution_runtime == repro_strat.execution_runtime
            and orig_strat.max_active_layers == repro_strat.max_active_layers
            and orig_strat.quantization == repro_strat.quantization
        )

        strategy_div: Optional[Dict[str, Any]] = None
        if not strategy_matched:
            strategy_div = {
                "original_runtime": orig_strat.execution_runtime,
                "reproduction_runtime": repro_strat.execution_runtime,
                "original_quantization": orig_strat.quantization,
                "reproduction_quantization": repro_strat.quantization,
                "original_max_active_layers": orig_strat.max_active_layers,
                "reproduction_max_active_layers": repro_strat.max_active_layers,
            }

        reproduction_category = "STRICT_IDENTICAL_RUNTIME" if strategy_matched else "CROSS_STRATEGY_REPLICATION"

        all_passed = all(c.passed for c in components)
        passed_count = sum(1 for c in components if c.passed)

        if all_passed:
            prefix = "REPRODUCED WITHIN NUMERICAL TOLERANCE (IDENTICAL RUNTIME)" if strategy_matched else "REPLICATED (CROSS-STRATEGY) WITHIN NUMERICAL TOLERANCE"
            verdict = f"{prefix} ({passed_count}/{len(components)} components verified)"
        else:
            failed = [c.metric_name for c in components if not c.passed]
            verdict = f"REPRODUCTION DIVERGENCE: components failed tolerances: {', '.join(failed)}"


        return ReproductionComparisonReport(
            original_run_id=original.run_id,
            reproduction_run_id=reproduction.run_id,
            timestamp_utc=_dt.datetime.now(_dt.timezone.utc).isoformat(),
            overall_reproduced=all_passed,
            numerical_tolerance_verdict=verdict,
            component_comparisons=components,
            execution_duration_ms=duration_ms,
            model_matched=model_matched,
            environment_matched=env_matched,
            reproduction_category=reproduction_category,
            strategy_matched=strategy_matched,
            strategy_divergence_summary=strategy_div,
        )

