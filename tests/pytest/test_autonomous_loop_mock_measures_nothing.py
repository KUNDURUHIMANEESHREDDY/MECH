"""A mock-adapter campaign records no measurements.

`AutonomousResearchLoop` gated `iteration_measured` on
`report.confidence is not None`. ACDC always reports confidence 0.0 --
calibrated to "no confidence", never a measurement -- while declaring
`provenance.measured: False` when its sweep evaluated nothing. The 0.0
counted as evidence: the iteration was recorded measured, the trace
claimed measured stages, and only the confidence arithmetic (driven
downward by the zeros) accidentally kept the verdict honest.

The loop now asks the report whether it measured. A mock campaign must
therefore end exactly like a no-adapter one: nothing measured, no
confidence, hypothesis unevaluated.
"""

from __future__ import annotations

import pytest


def _goal():
    from backend.interpretability.discovery.discovery_planner import (
        ResearchGoal,
    )

    return ResearchGoal(goal_id="g", description="d", model_id="gpt2-small")


def _mock_adapter():
    from backend.science.models.gpt2_adapter import GPT2Adapter

    return GPT2Adapter(variant="small", mock_mode=True)


def test_mock_campaign_measures_nothing():
    from backend.interpretability.discovery.autonomous_research_loop import (
        AutonomousResearchLoop,
    )

    report = AutonomousResearchLoop(
        adapter=_mock_adapter(), confidence_threshold=0.90, max_iterations=5
    ).run_campaign(_goal()).to_dict()

    assert report["final_composite_confidence"] is None
    assert report["target_confidence_reached"] is False
    assert report["evidence_gathered"] == 0

    trace = report["reasoning_trace"]
    assert trace["measured"] is False
    assert trace["final_confidence"] is None
    assert trace["selected_hypothesis_id"] is None
    assert trace["hypotheses"][0]["status"] == "unevaluated"
    for experiment in trace["experiments"]:
        assert experiment["measured"] is False, experiment
        assert experiment["verdict"] == "unevaluated"


def test_hardcoded_zero_confidence_is_not_evidence():
    """The precise regression: ACDC's constant 0.0 must not mark measured."""
    from backend.interpretability.discovery.algorithms import get_algorithm

    engine = get_algorithm("acdc", _mock_adapter())
    result = engine.run(dataset={
        "id": "ioi_0001",
        "clean": "John and Mary went to the store, John gave a drink to Mary",
        "corrupted": "John and Mary went to the store, Mary gave a drink to John",
        "target_token": " Mary",
    })

    assert result.confidence == 0.0
    assert result.provenance.get("measured") is False
