"""Uncertainty decisions and quality scores must not come from defaults.

A second wave of the defect fixed in
test_confidence_defaults_are_not_fabricated.py. The shared shape is the same --
a parameter default standing in for a measurement -- but here the fabricated
defaults *composed into a decision*, which is worse than a wrong number sitting
in a field. Individually each looked unremarkable.

evaluate_uncertainty had four:

    confidence_score=0.92, uncertainty_interval=[0.88, 0.95],
    sample_size=5, variance=0.02

With all four defaults the branch chain reached
`0.92 >= publication_confidence (0.85)` and `width 0.07 <= max_interval_width
(0.15)`, returning "Enough evidence" / **"Publish"**. So calling the method with
no arguments published on four invented numbers -- the precise outcome the
`evidence_missing` branch exists to prevent, reachable by omitting arguments.

compute_quality_score had six: novelty 0.90, reproducibility 0.95, confidence
0.92, benchmark_performance 0.94, interpretability 0.88, cross_model_support
0.89. Together they produced `overall_quality_score ~0.91` and
`quality_grade: "A+"` from a call that supplied no evidence.

Left deliberately alone: `UncertaintyPolicy.publication_confidence` (0.85) and
`rejection_confidence` (0.60). Those are decision thresholds in a class
documented as "threshold policy for research decisions" -- a rule about when to
publish, not a claim about the world. Changing them would weaken a policy, not
fix a fabrication. Pinned below so they are not later confused with the
fabricated inputs sitting beside them.
"""

from __future__ import annotations

import re

import pytest


# ── evaluate_uncertainty ────────────────────────────────────────────────────

def test_evaluate_uncertainty_with_no_arguments_does_not_publish():
    """The regression this file exists for: no args used to mean "Publish"."""
    from backend.research_platform.autonomous.uncertainty_manager import (
        UncertaintyManagerEngine,
    )

    decision = UncertaintyManagerEngine().evaluate_uncertainty()

    assert decision["action"] != "Publish", (
        "omitting every argument produced a publish decision on invented inputs"
    )
    assert decision["action"] == "More experiments"
    assert decision["decision"] == "Validation unavailable"
    assert decision["evidence_missing"] is True
    assert decision["confidence_score"] is None
    assert decision["uncertainty_interval"] is None


@pytest.mark.parametrize(
    "partial",
    [
        {"confidence_score": 0.95},
        {"uncertainty_interval": (0.80, 0.95)},
        {"confidence_score": 0.95, "uncertainty_interval": (0.80, 0.95)},
        {"confidence_score": 0.95, "uncertainty_interval": (0.80, 0.95), "sample_size": 5},
        # A single-token interval cannot produce a width, so it is not evidence.
        {"confidence_score": 0.95, "uncertainty_interval": (0.9,), "sample_size": 5, "variance": 0.02},
    ],
)
def test_partial_evidence_is_treated_as_missing(partial):
    """A confident number without an interval is not a publication."""
    from backend.research_platform.autonomous.uncertainty_manager import (
        UncertaintyManagerEngine,
    )

    decision = UncertaintyManagerEngine().evaluate_uncertainty(**partial)
    assert decision["evidence_missing"] is True, partial
    assert decision["action"] == "More experiments", partial


def test_supplied_evidence_still_reaches_every_decision_branch():
    """The fail-closed default must not have disabled the policy."""
    from backend.research_platform.autonomous.uncertainty_manager import (
        UncertaintyManagerEngine,
    )

    manager = UncertaintyManagerEngine()

    published = manager.evaluate_uncertainty(
        confidence_score=0.95, uncertainty_interval=(0.80, 0.95),
        sample_size=5, variance=0.02,
    )
    assert published["action"] == "Publish"
    assert published["evidence_missing"] is False

    rejected = manager.evaluate_uncertainty(
        confidence_score=0.30, uncertainty_interval=(0.20, 0.40),
        sample_size=5, variance=0.02,
    )
    assert rejected["action"] == "Reject"

    thin = manager.evaluate_uncertainty(
        confidence_score=0.90, uncertainty_interval=(0.80, 0.95),
        sample_size=1, variance=0.02,
    )
    assert thin["action"] == "More experiments"
    assert thin["decision"] == "Weak evidence"

    conflicted = manager.evaluate_uncertainty(
        confidence_score=0.90, uncertainty_interval=(0.80, 0.95),
        sample_size=5, variance=0.9,
    )
    assert conflicted["action"] == "Debate"


def test_evidence_missing_flag_alone_still_short_circuits():
    from backend.research_platform.autonomous.uncertainty_manager import (
        UncertaintyManagerEngine,
    )

    decision = UncertaintyManagerEngine().evaluate_uncertainty(
        confidence_score=0.99, uncertainty_interval=(0.99, 0.995),
        sample_size=50, variance=0.0,
        evidence_missing=True,
    )
    assert decision["action"] == "More experiments"
    assert decision["evidence_missing"] is True


def test_policy_thresholds_are_policy_not_fabrication():
    """Pinned so they are not 'fixed' alongside the fabricated inputs.

    0.85 / 0.60 are thresholds in `UncertaintyPolicy`, documented as a
    "Configurable and versioned threshold policy for research decisions". They
    are rules for when to publish, not assertions that anything was measured.
    """
    from backend.research_platform.autonomous.uncertainty_manager import (
        UncertaintyPolicy,
    )

    assert UncertaintyPolicy.publication_confidence == 0.85
    assert UncertaintyPolicy.rejection_confidence == 0.60
    assert UncertaintyPolicy.min_samples == 2


def test_evaluate_uncertainty_signature_has_no_fabricated_defaults():
    import inspect

    from backend.research_platform.autonomous.uncertainty_manager import (
        UncertaintyManagerEngine,
    )

    params = inspect.signature(UncertaintyManagerEngine.evaluate_uncertainty).parameters
    for name in ("confidence_score", "uncertainty_interval", "sample_size", "variance"):
        assert params[name].default is None, (
            f"{name} still defaults to {params[name].default!r}"
        )


# ── compute_quality_score ───────────────────────────────────────────────────

def test_quality_score_with_no_dimensions_is_not_graded():
    from backend.interpretability.discovery.discovery_quality_score import (
        DiscoveryQualityScoreEngine,
    )

    result = DiscoveryQualityScoreEngine().compute_quality_score()

    assert result["overall_quality_score"] is None
    assert result["quality_grade"] is None, "an A+ was produced from six invented dimensions"
    assert result["measured"] is False
    assert result["provenance"] == "unavailable"
    for key in ("confidence_score", "scientific_quality_score", "expected_impact_score"):
        assert result[key] is None, key
    assert "no measurement supplied" in result["reason"]


def test_quality_score_lists_exactly_which_dimensions_are_missing():
    from backend.interpretability.discovery.discovery_quality_score import (
        DiscoveryQualityScoreEngine,
    )

    result = DiscoveryQualityScoreEngine().compute_quality_score(novelty=0.9)
    reason = result["reason"]
    for missing in ("reproducibility", "confidence", "benchmark_performance",
                    "interpretability", "cross_model_support"):
        assert missing in reason, f"{missing} not named in the reason"
    assert "novelty" not in reason, "a supplied dimension was reported as missing"


def test_quality_score_still_grades_when_all_dimensions_are_supplied():
    from backend.interpretability.discovery.discovery_quality_score import (
        DiscoveryQualityScoreEngine,
    )

    result = DiscoveryQualityScoreEngine().compute_quality_score(
        novelty=0.9, reproducibility=0.95, confidence=0.92,
        benchmark_performance=0.94, interpretability=0.88, cross_model_support=0.89,
    )
    assert result["measured"] is True
    assert result["provenance"] == "live"
    assert result["quality_grade"] == "A+"
    assert result["overall_quality_score"] == pytest.approx(0.92, abs=0.01)


def test_quality_score_signature_has_no_fabricated_defaults():
    import inspect

    from backend.interpretability.discovery.discovery_quality_score import (
        DiscoveryQualityScoreEngine,
    )

    params = inspect.signature(
        DiscoveryQualityScoreEngine.compute_quality_score
    ).parameters
    for name in ("novelty", "reproducibility", "confidence",
                 "benchmark_performance", "interpretability", "cross_model_support"):
        assert params[name].default is None, f"{name} defaults to {params[name].default!r}"


# ── Source-level guard ──────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "module_path",
    [
        "backend.research_platform.autonomous.uncertainty_manager",
        "backend.interpretability.discovery.discovery_quality_score",
        "backend.interpretability.discovery.discovery_result",
        "backend.interpretability.sae.feature_dictionary",
        "backend.research_platform.autonomous.knowledge_base",
        "backend.research_platform.autonomous.engine",
    ],
)
def test_no_measurement_parameter_defaults_to_a_plausible_float(module_path):
    """Catch any measurement-shaped *parameter* reverting to a literal default.

    Parsed with the AST rather than grepped, because the same names appear
    legitimately in two different roles and a regex cannot tell them apart:

      * `QualityScoreWeights.novelty = 0.25` and friends -- dataclass *weights*
        that sum to 1.0. A weighting scheme is a policy choice, like the
        publication thresholds, and must keep its defaults.
      * `compute_quality_score(novelty=0.9)` -- a *measurement input*. A default
        here is a fabricated observation.

    Only `FunctionDef`/`AsyncFunctionDef` arguments are inspected, so dataclass
    fields and module constants are out of scope by construction.
    """
    import ast
    import importlib
    import inspect

    measurement_names = {
        "confidence", "confidence_score", "uncertainty", "uncertainty_interval",
        "sample_size", "variance", "novelty", "reproducibility",
        "benchmark_performance", "interpretability", "cross_model_support",
    }

    module = importlib.import_module(module_path)
    tree = ast.parse(inspect.getsource(module))

    offenders = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        args = node.args
        positional = list(args.posonlyargs) + list(args.args) + list(args.kwonlyargs)
        for arg, default in zip(positional, [None] * (len(positional) - len(args.defaults))
                                + list(args.defaults)):
            if arg.arg not in measurement_names or default is None:
                continue
            if isinstance(default, ast.Constant) and default.value is None:
                continue
            offenders.append(f"{module_path}:{node.lineno} {node.name}({arg.arg}=...)")

    assert not offenders, (
        "measurement parameters still have invented defaults:\n  "
        + "\n  ".join(sorted(offenders))
    )


def test_autonomous_engine_does_not_store_a_hardcoded_fact():
    """`execute_goal` stored a fixed neuron at 0.95 on every call."""
    from source_assert import executable_source

    import backend.research_platform.autonomous.engine as eng

    src = executable_source(eng)

    # The specific fabricated claim, and the fabricated narrative.
    assert "L8_N402" not in src, (
        "execute_goal is storing a hardcoded neuron/function claim again"
    )
    assert "cleanly" not in src, (
        "execute_goal asserts clean execution unconditionally again"
    )
    assert not re.search(r"utility_score\s*=\s*0\.\d", src), (
        "execute_goal is inventing a utility score again"
    )
