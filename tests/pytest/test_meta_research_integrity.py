"""Integrity of the meta-research layer's policy and performance accounting.

These modules lost their test coverage when `legacy_dispatcher.py` was removed
upstream: it was a 147-route dict registry that was never mounted into FastAPI
and existed only so the test suite had something to call. Deleting it was
correct, but it also deleted the only tests covering three real bugs found and
fixed on this branch:

* ``policy_repository`` raised ``TypeError: round(None)`` when asked to store an
  unmeasured policy, and shipped a baseline policy with ``success_rate=0.91`` /
  ``cost=4.12`` / ``runtime=9.0`` presented as measurements -- then used that
  baseline as the comparison reference in ``compare_policies``.
* ``meta_research_engine.__init__`` seeded a fabricated campaign (127 hypotheses
  tested, 41 discoveries, 13 publications, confidence 0.91) describing a run
  that never happened, and every aggregate in ``analyze_meta_performance`` summed
  it. Separately, ``c.get("published_count", 1)`` counted one publication for
  every campaign that recorded none.
* ``confidence_scorer`` emitted "Inputs were not supplied" whenever any one of
  three metrics was defaulted, which is false and hides which term is fabricated.

The tests below exercise the modules directly, so this coverage does not depend
on the deleted dispatcher coming back.
"""
import pytest

from backend.interpretability.discovery.confidence_scorer import (
    PlatformConfidenceEngine,
)
from backend.research_platform.meta.meta_research_engine import MetaResearchEngine
from backend.research_platform.meta.policy_repository import PolicyRepository


# ── PolicyRepository: unmeasured must be storable and must stay unmeasured ──

def test_unmeasured_policy_is_storable():
    """round(None) raised, so an honest policy could not be saved at all."""
    saved = PolicyRepository().save_policy(
        domain="circuit_discovery",
        workflow=["Run SAE Inspection"],
        success_rate=None,
        average_cost_usd=None,
        average_runtime_min=None,
    )
    assert saved["version"] == "2.0.0"
    assert saved["success_rate"] is None
    assert saved["metrics_measured"] is False


def test_baseline_policy_carries_no_invented_metrics():
    """It shipped success_rate=0.91 / cost=4.12 / runtime=9.0 as if measured."""
    baseline = PolicyRepository().get_latest_policy("circuit_discovery")
    assert baseline["policy_id"] == "pol_cd_v1"
    assert baseline["success_rate"] is None
    assert baseline["average_cost_usd"] is None
    assert baseline["average_runtime_min"] is None
    assert baseline["metrics_measured"] is False


def test_numbers_without_a_source_are_not_measured():
    saved = PolicyRepository().save_policy(
        domain="bench", workflow=["w"], success_rate=0.99,
        average_cost_usd=1.0, average_runtime_min=1.0,
    )
    assert saved["success_rate"] == 0.99
    assert saved["metrics_measured"] is False, "an unsourced claim counted as evidence"
    assert saved["metrics_source"] is None


def test_sourced_metrics_count_as_measured():
    saved = PolicyRepository().save_policy(
        domain="bench", workflow=["w"], success_rate=0.8,
        average_cost_usd=2.0, average_runtime_min=4.0,
        metrics_source="run_42",
    )
    assert saved["metrics_measured"] is True
    assert saved["metrics_source"] == "run_42"


def test_comparison_against_unmeasured_baseline_is_undefined():
    """Not a small delta -- an absence of one."""
    repo = PolicyRepository()
    repo.save_policy(
        domain="circuit_discovery", workflow=["w"], success_rate=0.95,
        average_cost_usd=3.5, average_runtime_min=8.0,
        metrics_source="run_x",
    )
    comp = repo.compare_policies("pol_cd_v1", "pol_circuit_discovery_v2")

    assert comp["success_rate_delta"] is None
    assert comp["improvement_detected"] is None
    assert comp["metrics_measured"] is False
    assert comp["provenance"] == "unavailable"
    assert "success_rate" in comp["unmeasured_fields"]
    assert comp["reason"]


def test_comparison_works_when_both_sides_are_measured():
    repo = PolicyRepository()
    a = repo.save_policy(domain="bench", workflow=["a"], success_rate=0.71,
                         average_cost_usd=5.0, average_runtime_min=10.0,
                         metrics_source="run_a")
    b = repo.save_policy(domain="bench", workflow=["b"], success_rate=0.88,
                         average_cost_usd=3.0, average_runtime_min=6.0,
                         metrics_source="run_b")
    comp = repo.compare_policies(a["policy_id"], b["policy_id"])

    assert comp["metrics_measured"] is True
    assert comp["provenance"] == "live"
    assert comp["success_rate_delta"] == round(0.88 - 0.71, 4)
    assert comp["cost_delta_usd"] == round(3.0 - 5.0, 2)
    assert comp["runtime_delta_min"] == round(6.0 - 10.0, 2)
    assert comp["improvement_detected"] is True
    assert comp["unmeasured_fields"] == []


def test_compare_policies_still_reports_missing_ids():
    comp = PolicyRepository().compare_policies("nope", "also_nope")
    assert comp["status"] == "PolicyNotFound"


# ── MetaResearchEngine: aggregates must come only from real campaigns ──

def test_engine_does_not_seed_a_phantom_campaign():
    engine = MetaResearchEngine()
    assert engine.campaign_history == [], "shipped a fabricated campaign record"
    assert engine.strategy_adaptation_factor is None, "a literal 1.05, unmeasured"


def test_aggregates_reflect_only_recorded_campaigns():
    engine = MetaResearchEngine()
    engine.record_campaign_performance(
        campaign_id="c1", topic="IOI", hypotheses_tested=10,
        discoveries_count=8, compute_used_vram_gb=16.0, failures_count=2,
    )
    summary = engine.analyze_meta_performance()["campaign_summary"]

    assert summary["total_campaigns"] == 1
    assert summary["hypotheses_tested"] == 10
    assert summary["discoveries_count"] == 8
    assert summary["rejected_count"] == 2


def test_campaign_without_published_count_publishes_nothing():
    """c.get("published_count", 1) counted a publication for every campaign."""
    engine = MetaResearchEngine()
    engine.record_campaign_performance(
        campaign_id="c1", topic="IOI", hypotheses_tested=4,
        discoveries_count=1, compute_used_vram_gb=1.0,
    )
    assert engine.analyze_meta_performance()["campaign_summary"]["published_count"] == 0


def test_unmeasured_aggregates_are_none_not_literals():
    engine = MetaResearchEngine()
    engine.record_campaign_performance(
        campaign_id="c1", topic="IOI", hypotheses_tested=4,
        discoveries_count=1, compute_used_vram_gb=1.0,
    )
    summary = engine.analyze_meta_performance()["campaign_summary"]

    # Were 8.4 / 0.91 / two fixed workflow strings, reported as aggregates over
    # the campaign history. Nothing here times a run or scores a confidence.
    assert summary["average_runtime_min"] is None
    assert summary["average_confidence"] is None
    assert summary["best_workflow"] is None
    assert summary["worst_workflow"] is None
    assert summary["metrics_measured"] is False
    assert summary["reason"]


def test_fresh_engine_reports_no_success_rate():
    analysis = MetaResearchEngine().analyze_meta_performance()
    assert analysis["overall_success_rate_measured"] is False
    assert analysis["campaign_summary"]["total_campaigns"] == 0


def test_success_rate_is_derived_from_real_counts():
    engine = MetaResearchEngine()
    engine.record_campaign_performance(
        campaign_id="c1", topic="IOI", hypotheses_tested=10,
        discoveries_count=8, compute_used_vram_gb=16.0, failures_count=2,
    )
    analysis = engine.analyze_meta_performance()
    assert analysis["overall_success_rate"] == round(8 / 10, 4)
    assert analysis["overall_success_rate_measured"] is True


def test_closed_loop_saves_a_policy_without_inventing_metrics():
    loop = MetaResearchEngine().run_closed_feedback_loop("camp_x")
    assert loop["status"] == "ClosedFeedbackLoopCompleted"
    assert loop["policy_metrics_measured"] is False
    assert loop["provenance"] == "unavailable"
    assert loop["publication_eligible"] is False
    assert loop["reason"]


# ── ConfidenceScorer: name the inputs that were assumed ──

def test_no_inputs_names_every_assumed_field():
    result = PlatformConfidenceEngine().score_confidence()
    assert result["inputs_assumed"] is True
    assert result["assumed_inputs"] == [
        "evidence_count", "reproducibility_score", "variance"
    ]
    assert result["provenance"] == "unavailable"
    assert result["reason"]


def test_partial_inputs_name_only_the_missing_one():
    result = PlatformConfidenceEngine().score_confidence(
        evidence_count=8, reproducibility_score=0.98,
    )
    assert result["inputs_assumed"] is True
    assert result["assumed_inputs"] == ["variance"], (
        "a blanket 'inputs were not supplied' is false when two of three "
        "were supplied, and hides which term is fabricated"
    )
    assert "variance" in result["reason"]
    assert result["evidence_count"] == 8
    assert result["reproducibility_score"] == 0.98


def test_all_inputs_supplied_means_no_default_in_the_arithmetic():
    result = PlatformConfidenceEngine().score_confidence(
        evidence_count=8, reproducibility_score=0.98, variance=0.04,
    )
    assert result["inputs_assumed"] is False
    assert result["assumed_inputs"] == []
    assert result["provenance"] == "reference"
    assert result["confidence_score"] > 0.90
    assert result["reliability_rating"] == "High"


@pytest.mark.parametrize("kwargs", [
    {},
    {"evidence_count": 8},
    {"evidence_count": 8, "reproducibility_score": 0.98},
    {"evidence_count": 8, "reproducibility_score": 0.98, "variance": 0.04},
])
def test_scoring_is_never_publishable(kwargs):
    """A weighted sum over supplied numbers is not evidence, either way."""
    result = PlatformConfidenceEngine().score_confidence(**kwargs)
    assert result["publication_eligible"] is False
    assert result["validation_eligible"] is False


# ── Guard against the literals coming back ──

def test_no_fabricated_literals_remain_in_these_modules():
    from source_assert import executable_source

    from backend.research_platform.meta import (
        meta_research_engine,
        policy_repository,
    )

    for module in (policy_repository, meta_research_engine):
        # Parsed, not grepped: the docstrings explaining these removals contain
        # the literals themselves, which is exactly how an earlier version of
        # this guard failed.
        code = executable_source(module)
        for literal in ("127", "0.91", "8.4", "4.12", "1.05"):
            assert literal not in code, f"{module.__name__} reintroduced {literal}"

    # The default that counted a publication for every campaign.
    engine_code = executable_source(meta_research_engine)
    assert '"published_count", 1' not in engine_code

    # round() on an unmeasured policy raised TypeError.
    assert "round(success_rate" not in executable_source(policy_repository)
