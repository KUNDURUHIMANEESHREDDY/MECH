"""Guards on the discovery cluster: no plausible number without a measurement.

Every test here corresponds to a specific defect that was present. The common
shape is a result describing itself rather than the measurement: a fabricated
fallback, a constant, a formula that cannot fail, or a field whose name does not
match the quantity it holds.

Covered defects
---------------
* `discovery_planner` returned a complete fabricated IOI circuit (L9H9 as the Name
  Mover Head, edges 0.92/0.95, confidences 0.95/0.98) and `composite_confidence
  = 0.95` when no algorithm ran, with a summary reading "Validated mechanism".
* `discovery_planner` read falsification as `statistics.get("validated", True)`,
  so an unrun or unmeasured falsification counted as a pass -- and the initial
  state was `falsification_passed = True` before anything had run.
* `autonomous_research_loop` climbed a synthetic confidence by 0.08 per iteration
  with no adapter, from 0.50 to any threshold, then emitted `STOP_THRESHOLD_MET`.
* `autonomous_research_loop` hardcoded `semantic_confidence: 0.75` and
  `replication_score: 0.95` for a single-model, single-seed campaign.
* `training_dynamics_engine` ignored its `checkpoint_results` and returned
  `birth_step=500`, `stabilization_step=2500`, `is_stable=True` and a four-entry
  timeline with confidences 0.72/0.88/0.94.
* `concept_evolution_engine` emitted `("Names", "Cities", "ancestor", 0.72)` for
  every layer pair regardless of input, and `persistence = 0.94` always.
* `attribution_patching` read a neuron dimension as if it were a head, read token
  position 0 (identical across a shared-prefix pair, so delta_x was exactly 0.0
  for all 144 components), substituted `metric_delta / (|c| + |r|)` for a
  gradient -- algebraically degenerate to exactly `metric_delta` whenever the two
  activations straddle zero -- and fabricated activations as `else 0.5` / `else
  0.1` when unreadable.
* `transcoders` simulated an encoder and decoder that do not exist and clamped
  FVE into [0.80, 0.99].
* `feature_auto_interpreter` derived SAE activations from a substring match on two
  hardcoded feature indices, assigned `concept_family` by testing whether the
  generated label contained "Paris", and returned a constant `confidence: 0.85`.
* `debate_engine` returned `hypothesis_a` as `consensus_winner` for any arguments
  and `confidence: 0.93` always.
* `hypothesis_generator` ignored `context_prompt` and returned two fixed
  hypotheses about L8_H9 (an induction head -- it is this project's IOI name-mover
  head) and SAE feature #1402 (no SAE exists), with invented evidence.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

from tests.pytest.source_assert import doc_and_comments, executable_source

ROOT = Path(__file__).resolve().parents[2]

CLUSTER = [
    "backend/interpretability/discovery/discovery_planner.py",
    "backend/interpretability/discovery/autonomous_research_loop.py",
    "backend/interpretability/discovery/training_dynamics_engine.py",
    "backend/interpretability/discovery/concept_evolution_engine.py",
    "backend/interpretability/discovery/feature_auto_interpreter.py",
    "backend/interpretability/discovery/algorithms/attribution_patching.py",
    "backend/interpretability/discovery/algorithms/transcoders.py",
    "backend/research_platform/autonomous/debate_engine.py",
    "backend/research_platform/autonomous/hypothesis_generator.py",
]


def _source(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8")


# ── discovery_planner ───────────────────────────────────────────────────────

def _planner():
    from backend.interpretability.discovery.discovery_planner import (
        AutonomousDiscoveryPlanner,
        ResearchGoal,
    )
    return AutonomousDiscoveryPlanner, ResearchGoal


def test_unrun_campaign_reports_no_circuit_and_no_confidence():
    """It used to return a fabricated three-node IOI graph and 0.95."""
    Planner, Goal = _planner()
    claim = Planner().execute_campaign(
        Goal(goal_id="g", description="Find the IOI name mover head",
             model_id="gpt2-small", require_falsification=True)
    ).to_dict()

    assert claim["composite_confidence"] is None
    graph = claim["unified_graph"]
    assert graph["nodes"] == []
    assert graph["edges"] == []
    assert graph["score"] is None
    assert graph["measured"] is False
    assert "No mechanism was established" in claim["summary_claim"]


def test_absent_falsification_is_not_a_passed_falsification():
    """`get("validated", True)` plus an initial `falsification_passed = True`
    meant a campaign with no falsification stage reported a passed one."""
    Planner, Goal = _planner()
    claim = Planner().execute_campaign(
        Goal(goal_id="g", description="d", model_id="gpt2-small")
    ).to_dict()

    assert claim["falsification_passed"] is None
    assert claim["falsification_established"] is False
    assert claim["falsification_reason"]


def test_claim_id_is_stable_format_not_a_process_salted_hash():
    """It was `hash(goal_id + str(time.time()))`, different every call *and*
    every process. An identifier that changes on every read cannot identify."""
    from backend.interpretability.discovery.discovery_planner import _claim_id

    first = _claim_id("goal-1")
    assert first.startswith("claim_")
    assert len(first) == len("claim_") + 16
    assert all(c in "0123456789abcdef" for c in first[len("claim_"):])
    # A pure function of the goal: repeatable within a run, across processes, and
    # across a clock-tick boundary. (An intermediate version hashed
    # `utcnow()`, which on Windows ticks at ~15.6 ms -- so it agreed within a tick
    # and disagreed across one, which is why this assertion was flaky.)
    assert _claim_id("goal-1") == first
    assert _claim_id("goal-2") != first

    # And stable across interpreter processes, which the old salted `hash` was not.
    import subprocess
    import sys as _sys
    out = subprocess.run(
        [_sys.executable, "-c",
         "import sys; sys.path[:0] = ['backend', '.'];"
         "from backend.interpretability.discovery.discovery_planner import _claim_id;"
         "print(_claim_id('goal-1'))"],
        capture_output=True, text=True, cwd=str(ROOT), timeout=180)
    assert out.stdout.strip().splitlines()[-1] == first, (
        f"claim id differs across processes: {out.stdout.strip()!r} vs {first!r}")

    source = _source("backend/interpretability/discovery/discovery_planner.py")
    assert "hash(" not in executable_source(source)
    # The timestamp must not be part of the handle.
    assert "utcnow()" not in executable_source(
        source[source.index("def _claim_id"):source.index("class AutonomousDiscoveryPlanner")])


# ── autonomous_research_loop ───────────────────────────────────────────────

def test_loop_without_an_adapter_measures_nothing():
    """It used to add 0.08 per iteration from 0.50 and report a clean stop."""
    from backend.interpretability.discovery.autonomous_research_loop import (
        AutonomousResearchLoop,
        ResearchGoal,
    )

    report = AutonomousResearchLoop(
        adapter=None, confidence_threshold=0.90, max_iterations=5
    ).run_campaign(ResearchGoal(goal_id="g", description="d",
                                model_id="gpt2-small")).to_dict()

    assert report["final_composite_confidence"] is None
    assert report["target_confidence_reached"] is False
    assert report["target_confidence_assessed"] is False
    assert report["evidence_gathered"] == 0

    trace = report["reasoning_trace"]
    assert trace["measured"] is False
    assert trace["final_confidence"] is None
    assert trace["selected_hypothesis_id"] is None
    assert trace["hypotheses"][0]["status"] == "unevaluated"
    for experiment in trace["experiments"]:
        assert experiment["verdict"] == "unevaluated"
        assert experiment["measured"] is False


def test_loop_never_claims_replication_or_a_computed_semantic_confidence():
    """Both were literals: 0.75 and 0.95, on a single-model single-seed run."""
    from backend.interpretability.discovery.autonomous_research_loop import (
        AutonomousResearchLoop,
        ResearchGoal,
    )

    report = AutonomousResearchLoop(adapter=None).run_campaign(
        ResearchGoal(goal_id="g", description="d", model_id="gpt2-small")
    ).to_dict()
    hypothesis = report["reasoning_trace"]["hypotheses"][0]

    assert hypothesis["replication_score"] is None
    assert hypothesis["replication_performed"] is False
    assert hypothesis["semantic_confidence"] is None
    assert hypothesis["replication_reason"]


# ── training_dynamics_engine ───────────────────────────────────────────────

def test_concept_birth_needs_checkpoints():
    """It returned birth 500, stabilization 2500, is_stable True for any input,
    including the empty list the audit script passed."""
    from backend.interpretability.discovery.training_dynamics_engine import (
        TrainingDynamicsEngine,
    )

    result = TrainingDynamicsEngine().track_concept_birth("European Capitals", [])

    assert result["measured"] is False
    assert result["birth_step"] is None
    assert result["stabilization_step"] is None
    assert result["is_stable"] is False
    assert result["evolution_timeline"] == []
    assert result["n_checkpoints_observed"] == 0
    assert result["reason"]


def test_concept_timeline_is_computed_from_the_supplied_checkpoints():
    from backend.interpretability.discovery.training_dynamics_engine import (
        TrainingDynamicsEngine,
    )

    checkpoints = [
        {"step": 100, "score": 0.31},
        {"step": 500, "score": 0.74},
        {"step": 1000, "score": 0.86, "concept_id": "cap_5013"},
        {"step": 2500, "score": 0.91},
        {"step": 5000, "score": 0.93},
    ]
    result = TrainingDynamicsEngine().track_concept_birth(
        "European Capitals", checkpoints)

    assert result["measured"] is True
    assert result["birth_step"] == 1000
    assert result["stabilization_step"] == 1000
    # One timeline entry per examined checkpoint, each carrying its own score --
    # and no invented checkpoints at steps 100/500/1000/5000 beyond these.
    assert [entry["step"] for entry in result["evolution_timeline"]] == [
        100, 500, 1000, 2500, 5000]
    assert [entry["confidence"] for entry in result["evolution_timeline"]] == [
        0.31, 0.74, 0.86, 0.91, 0.93]
    # A real concept id from the input is preferred over `TEMP-...`.
    events = TrainingDynamicsEngine()
    events.track_concept_birth("European Capitals", checkpoints)
    assert any(e["concept_id"] == "cap_5013" for e in events.get_events())


def test_learning_velocity_needs_two_checkpoints_and_is_not_a_constant():
    """It returned 0.65 for any domain, ignoring the argument entirely."""
    from backend.interpretability.discovery.training_dynamics_engine import (
        TrainingDynamicsEngine,
    )

    engine = TrainingDynamicsEngine()

    nothing = engine.analyze_learning_velocity("Math")
    assert nothing["measured"] is False
    assert nothing["velocity_per_1k_steps"] is None

    rising = engine.analyze_learning_velocity("Math", [
        {"step": 0, "score": 0.2}, {"step": 1000, "score": 0.6}])
    falling = engine.analyze_learning_velocity("Math", [
        {"step": 0, "score": 0.9}, {"step": 1000, "score": 0.3}])

    assert rising["measured"] is True and rising["velocity_per_1k_steps"] > 0
    assert falling["measured"] is True and falling["velocity_per_1k_steps"] < 0
    assert rising["velocity_per_1k_steps"] != falling["velocity_per_1k_steps"]


# ── concept_evolution_engine ───────────────────────────────────────────────

def test_layer_progression_uses_the_actual_concepts():
    """It emitted ("Names", "Cities", "ancestor", 0.72) for every pair."""
    from backend.interpretability.discovery.concept_evolution_engine import (
        ConceptEvolutionEngine,
    )

    edges = ConceptEvolutionEngine().analyze_layer_progression({
        0: ["Python", "Basketball"],
        1: ["Python", "Orbital mechanics"],
    })

    assert edges, "a readable pair must produce edges"
    for edge in edges:
        assert (edge.source_concept, edge.target_concept) != ("Names", "Cities")
        assert not (edge.type == "ancestor"
                    and (edge.source_concept, edge.target_concept)
                    == ("Names", "Cities"))

    shared = [e for e in edges if e.type == "shared"]
    assert [e.source_concept for e in shared] == ["Python"]
    assert shared[0].alignment_score is not None
    assert shared[0].measured is True

    # Different input, different edges.
    other = ConceptEvolutionEngine().analyze_layer_progression({
        0: ["Jazz"], 1: ["Jazz", "Bach"]})
    assert [e.source_concept for e in other if e.type == "shared"] == ["Jazz"]


def test_concept_evolution_refuses_rather_than_guessing():
    from backend.interpretability.discovery.concept_evolution_engine import (
        ConceptEvolutionEngine,
    )

    drift = ConceptEvolutionEngine().analyze_cross_model_drift(None, ["x"])
    assert drift["measured"] is False
    assert drift["drift_score"] is None
    assert drift["persistence"] is None

    persistence = ConceptEvolutionEngine().compute_concept_persistence("Python", [])
    assert persistence["measured"] is False
    assert persistence["persistence"] is None


def test_drift_reads_concepts_from_a_container_key():
    """Regression: `{"concepts": [...]}` used to yield a concept literally
    named "concepts", because the dict's keys were treated as the names."""
    from backend.interpretability.discovery.concept_evolution_engine import (
        ConceptEvolutionEngine,
    )

    drift = ConceptEvolutionEngine().analyze_cross_model_drift(
        {"concepts": ["Python", "Basketball", "Jazz"]},
        {"concepts": ["Python", "TensorFlow", "Jazz"]},
    )
    assert drift["measured"] is True
    assert drift["shared_concepts"] == ["Jazz", "Python"]
    assert drift["novel_concepts"] == ["TensorFlow"]
    assert drift["extinct_concepts"] == ["Basketball"]
    assert "concepts" not in drift["shared_concepts"]


# ── attribution_patching ───────────────────────────────────────────────────

def _attribution_report():
    """A report from a run that can actually measure.

    `mock_mode=True` has no tokenizer and no weights, so the algorithm takes its
    fail-closed path and the statistics block describes the refusal rather than a
    sweep. The sweep-level assertions need the real path, so these use live
    weights.
    """
    from backend.interpretability.discovery.algorithms.attribution_patching import (
        AttributionPatchingAlgorithm,
    )
    from backend.science.models.gpt2_adapter import GPT2Adapter

    return AttributionPatchingAlgorithm(
        GPT2Adapter(variant="small", mock_mode=False)).run({
            "id": "IOI-Canonical-100",
            "clean": "When John and Mary went to the store, John gave a drink to",
            "corrupted": "When John and Mary went to the store, Mary gave a drink to John",
            "target_token": " Mary",
            "baseline_token": " John",
        })


def test_attribution_reports_its_search_space_honestly():
    """`min(4, num_heads)` examined 48 of 144 components under a declared
    `search_space` of "all_components"."""
    stats = _attribution_report().statistics

    assert stats["heads_examined_per_layer"] == stats["heads_in_model"]
    assert stats["heads_skipped"] == 0
    assert stats["search_space_complete"] is True
    assert stats["total_components_analyzed"] == (
        stats["heads_in_model"] ** 2)


def test_attribution_never_fabricates_an_activation():
    """`else 0.5` / `else 0.1` gave every unreadable component a delta of
    exactly 0.4 and an attribution of 0.0138, just over the 0.01 threshold."""
    source = executable_source(
        _source("backend/interpretability/discovery/algorithms/attribution_patching.py"))
    assert "else 0.5" not in source
    assert "else 0.1" not in source

    stats = _attribution_report().statistics
    assert stats["components_unreadable"] == 0, (
        "every head was readable, so none may be reported as unreadable")


def test_attribution_closing_edge_is_structural_not_certain():
    """It carried `weight: 1.0, confidence: 1.0`, asserting that the top
    component accounts for the output exactly."""
    report = _attribution_report()

    structural = [e for e in report.graph["edges"] if e.get("structural")]
    assert structural, "the closing edge must still exist to close the graph"
    for edge in structural:
        assert edge["weight"] is None
        assert edge["confidence"] is None
        assert edge["weight_reason"]

    for edge in report.graph["edges"]:
        assert not (edge["weight"] == 1.0 and edge["confidence"] == 1.0)


def test_attribution_scores_are_not_all_the_same_number():
    """The gradient substitution collapsed to exactly `metric_delta` for every
    component whose clean and corrupted activations straddled zero -- 9 of the
    previous top 10 -- leaving the ranking decided by rounding."""
    report = _attribution_report()
    scores = [a["attribution_score"]
              for a in report.evidence["top_attributions"]]

    assert scores, "a real IOI pair must produce attributions"
    assert len(set(scores)) > 1, "attributions are degenerate again"


def test_attribution_does_not_substitute_a_ratio_for_a_gradient():
    """`metric_delta / (|c| + |r|)` collapses to exactly `metric_delta` when
    the two activations straddle zero, because |c - r| == |c| + |r|."""
    source = executable_source(
        _source("backend/interpretability/discovery/algorithms/attribution_patching.py"))
    # The old expression, with any spacing.
    compact = source.replace(" ", "")
    assert "metric_delta/max(1e-4,abs(c_val)+abs(cor_val))" not in compact
    # It must actually take a gradient.
    assert "capture_head_outputs_with_grad" in source


def test_attribution_requires_predictable_prompt_shape():
    """A clean prompt already containing its own answer makes the next-token
    metric trivially satisfied -- which is how the bundled IOI dataset stores
    prompts."""
    from backend.interpretability.discovery.algorithms.attribution_patching import (
        AttributionPatchingAlgorithm,
    )
    from backend.science.models.gpt2_adapter import GPT2Adapter

    report = AttributionPatchingAlgorithm(
        GPT2Adapter(variant="small", mock_mode=False)).run({
            "id": "x",
            "clean": "When John and Mary went to the store, John gave a drink to Mary",
            "corrupted": "When John and Mary went to the store, Mary gave a drink to John",
            "target": " Mary",
        })

    assert report.statistics["measured"] is False
    assert report.confidence is None
    assert report.provenance["unavailable_reason"]
    assert "already ends with the target" in report.provenance["unavailable_reason"]


# ── transcoders ────────────────────────────────────────────────────────────

def test_transcoders_refuses_rather_than_simulating():
    """It simulated encoder and decoder responses from an arithmetic ramp and
    clamped FVE into [0.80, 0.99], so a bad reconstruction could not be
    reported."""
    from backend.interpretability.discovery.algorithms.transcoders import (
        TranscoderAlgorithm,
    )
    from backend.science.models.gpt2_adapter import GPT2Adapter

    report = TranscoderAlgorithm(
        GPT2Adapter(variant="small", mock_mode=False)).run(
        {"id": "x", "clean": "When John and Mary went to the store"})

    stats = report.statistics
    assert stats["measured"] is False
    assert stats["fve_variance_explained"] is None
    assert stats["l0_sparsity"] is None
    assert stats["active_features_count"] is None
    assert report.confidence is None
    assert report.graph["nodes"] == [] and report.graph["edges"] == []
    assert report.graph["score"] is None
    assert report.evidence["missing_components"]

    source = _source("backend/interpretability/discovery/algorithms/transcoders.py")
    tree = ast.parse(source)
    # AST rather than text: the docstring and the refusal reason both name
    # `0.4 + 0.04 * feature_index` in order to say it is no longer computed.
    # A *computation* is a BinOp; a string literal is a Constant. This matches
    # only the former.
    for node in ast.walk(tree):
        if not isinstance(node, ast.BinOp):
            continue
        text = ast.unparse(node).replace(" ", "")
        assert not (text.startswith("0.4+0.04") or text.startswith("0.92+0.05")), (
            f"transcoders.py: the simulated ramp is computed again at line "
            f"{node.lineno}: {ast.unparse(node)}")


# ── feature_auto_interpreter ───────────────────────────────────────────────

def test_cluster_without_measured_activations_is_not_described():
    from backend.interpretability.discovery.feature_auto_interpreter import (
        FeatureAutoInterpreter,
    )

    result = FeatureAutoInterpreter().interpret_cluster("C1", [1042, 1043])

    assert result["measured"] is False
    assert result["semantic_name"] is None
    assert result["concept_family"] is None
    assert result["confidence"] is None
    assert result["reason"]


def test_concept_family_is_supplied_not_inferred_from_a_substring():
    """`"Geography" if "Paris" in name else "Syntax"` filed every non-Paris
    cluster under Syntax."""
    source = executable_source(
        _source("backend/interpretability/discovery/feature_auto_interpreter.py"))
    assert '"Geography" if' not in source.replace(" ", "")
    assert "in name else" not in source.replace(" ", "")

    from backend.interpretability.discovery.feature_auto_interpreter import (
        FeatureAutoInterpreter,
    )

    activations = {1042: [{"prompt": "Paris is lovely", "activation": 4.6}]}
    unspecified = FeatureAutoInterpreter().interpret_cluster(
        "C1", [1042], activations=activations)
    assert unspecified["concept_family"] is None

    supplied = FeatureAutoInterpreter().interpret_cluster(
        "C1", [1042], activations=activations, concept_family="Geography")
    assert supplied["concept_family"] == "Geography"
    assert supplied["confidence"] is None
    assert supplied["verified"] is False
    assert supplied["auto_suggested"] is True


def test_campaign_statistics_are_computed_not_fixed():
    """It returned campaign_id "REPR-772", a fixed timestamp, polysemanticity
    0.12 and stability 0.94 regardless of input."""
    from backend.interpretability.discovery.feature_auto_interpreter import (
        FeatureAutoInterpreter,
    )

    report = FeatureAutoInterpreter().generate_discovery_report({})
    stats = report["global_statistics"]
    assert stats["polysemanticity"] is None
    assert stats["stability"] is None
    assert report["measured"] is False
    assert report["campaign_id"] is None
    # A real timestamp, not 2026-07-28T21:35:00Z.
    assert not report["timestamp"].startswith("2026-07-28")


# ── debate_engine ──────────────────────────────────────────────────────────

def test_debate_without_evidence_names_no_winner():
    """It returned hypothesis_a and confidence 0.93 for any arguments."""
    from backend.research_platform.autonomous.debate_engine import (
        ScientificDebateEngine,
    )

    result = ScientificDebateEngine().debate_hypotheses("A", "B")

    assert result["verdict"] == "no_consensus"
    assert result["consensus_winner"] is None
    assert result["winning_side"] is None
    assert result["confidence"] is None
    assert result["debate_rounds"] == 0
    assert result["counter_evidence_evaluated"] is False


def test_debate_verdict_follows_the_evidence_not_argument_order():
    from backend.research_platform.autonomous.debate_engine import (
        ScientificDebateEngine,
    )

    engine = ScientificDebateEngine()
    strong_a = [{"support": 0.8}, {"support": 0.7}]

    forwards = engine.debate_hypotheses(
        "A", "B", evidence_a=strong_a, evidence_b=[{"support": 0.3}],
        counter_evidence_a=[{"claim": "x"}], rounds=2)
    swapped = engine.debate_hypotheses(
        "B", "A", evidence_a=strong_a, evidence_b=[{"support": 0.3}],
        counter_evidence_a=[{"claim": "x"}], rounds=2)

    # Same evidence, different argument order: the same side must win. The old
    # engine returned the first argument in both cases.
    assert forwards["winning_side"] == swapped["winning_side"] == "a"
    assert forwards["verdict"] == swapped["verdict"] == "consensus"

    # And the reverse assignment flips the side.
    reversed_side = engine.debate_hypotheses(
        "A", "B", evidence_b=strong_a, evidence_a=[{"support": 0.3}],
        counter_evidence_b=[{"claim": "x"}])
    assert reversed_side["winning_side"] == "b"


def test_debate_distinguishes_tie_insufficient_and_untested():
    from backend.research_platform.autonomous.debate_engine import (
        ScientificDebateEngine,
    )

    engine = ScientificDebateEngine()

    tie = engine.debate_hypotheses("A", "B",
                                   evidence_a=[{"support": 0.5}],
                                   evidence_b=[{"support": 0.5}])
    assert tie["verdict"] == "tie"
    assert tie["confidence"] is None

    thin = engine.debate_hypotheses("A", "B",
                                    evidence_a=[{"support": 0.9}],
                                    evidence_b=[{"support": 0.1}])
    assert thin["verdict"] == "insufficient_evidence"
    assert thin["confidence"] is None

    # Two arguments at a tiny scale clear the same bar as two at 0.8: the bar is
    # a count, not a sum, so it cannot be passed or failed by rescaling.
    tiny = engine.debate_hypotheses("A", "B",
                                    evidence_a=[{"support": 0.02},
                                                {"support": 0.03}],
                                    evidence_b=[{"support": 0.01}])
    assert tiny["verdict"] == "consensus"

    # A winner nobody argued against is discounted and flagged.
    countered = engine.debate_hypotheses("A", "B", evidence_a=strong_pair(),
                                         evidence_b=[{"support": 0.1}],
                                         counter_evidence_a=[{"claim": "x"}])
    uncountered = engine.debate_hypotheses("A", "B", evidence_a=strong_pair(),
                                           evidence_b=[{"support": 0.1}])
    assert uncountered["winner_is_untested"] is True
    assert uncountered["confidence"] < countered["confidence"]


def strong_pair() -> List[Dict[str, Any]]:
    return [{"support": 0.9}, {"support": 0.9}]


# ── hypothesis_generator ───────────────────────────────────────────────────

def test_generator_invents_nothing_without_findings():
    """It ignored context_prompt and returned two fixed hypotheses about L8_H9
    and SAE feature #1402, with invented evidence and confidences."""
    from backend.research_platform.autonomous.hypothesis_generator import (
        HypothesisGeneratorEngine,
    )

    engine = HypothesisGeneratorEngine()
    assert engine.generate_hypotheses("The capital of France is") == []
    assert "No findings" in engine.last_generation_note


def test_hypothesis_records_carry_only_checkable_evidence():
    from backend.research_platform.autonomous.hypothesis_generator import (
        HypothesisGeneratorEngine,
    )

    engine = HypothesisGeneratorEngine()
    records = engine.generate_hypotheses("IOI", findings=[{
        "statement": "Head L9H9 carries the name-mover role in IOI.",
        "evidence": [
            {"claim": "logit recovery when patched", "value": 0.8035,
             "source": "acdc.logit_recovery_fidelity"},
            # Uncheckable: no number, so a reader cannot assess it.
            {"claim": "High activation on 'Mary'"},
        ],
        "suggested_experiment": "head-level ablation on IOI prompts",
        "provenance": "live",
    }])

    assert len(records) == 1
    record = records[0]
    assert record["n_evidence_items"] == 1, "the uncheckable item must be dropped"
    assert record["evidence"][0]["value"] == 0.8035
    assert record["hypothesis_status"] == "evidence_backed"
    assert record["verified"] is False
    assert record["context_prompt"] == "IOI"


def test_unmeasured_hypothesis_is_speculative_and_carries_no_confidence():
    from backend.research_platform.autonomous.hypothesis_generator import (
        HypothesisGeneratorEngine,
    )

    records = HypothesisGeneratorEngine().generate_hypotheses(findings=[{
        "statement": "Some head might matter.",
        "evidence": [{"claim": "weight", "value": 0.5}],
        "provenance": "speculative",
    }])

    assert records[0]["hypothesis_status"] == "speculative"
    assert records[0]["confidence"] is None
    assert records[0]["confidence_reason"]


def test_hypothesis_ids_are_unique():
    from backend.research_platform.autonomous.hypothesis_generator import (
        HypothesisGeneratorEngine,
    )

    engine = HypothesisGeneratorEngine()
    records = engine.generate_hypotheses(
        findings=[{"statement": "a"}, {"statement": "b"}])
    ids = {r["hypothesis_id"] for r in records}
    assert len(ids) == 2, "ids were the fixed literals hyp_gen_1 / hyp_gen_2"


# ── cluster-wide source guards ─────────────────────────────────────────────

def test_no_module_in_the_cluster_returns_a_literal_confidence():
    """`confidence: <float literal>` in a returned dict, with no computation
    feeding it."""
    offenders: List[str] = []
    for rel in CLUSTER:
        tree = ast.parse(_source(rel))
        for node in ast.walk(tree):
            # Dict literals inside a `return` -- i.e. a produced record.
            if not isinstance(node, ast.Return) or not isinstance(node.value, ast.Dict):
                continue
            for key, value in zip(node.value.keys, node.value.values):
                if not (isinstance(key, ast.Constant)
                        and isinstance(key.value, str)):
                    continue
                if "confidence" not in key.value:
                    continue
                if isinstance(value, ast.Constant) and isinstance(value.value, float):
                    offenders.append(f"{rel}:{node.lineno} {key.value}")
    assert not offenders, "literal confidence in a returned record: " + "; ".join(offenders)


def test_no_module_in_the_cluster_hardcodes_the_prior_fabrications():
    """The specific literals that were removed, so they cannot return."""
    forbidden = [
        # discovery_planner's fabricated IOI circuit
        '"Name Mover Head (L9H9)"',
        # concept_evolution_engine's fixed lineage edge
        '"Names", "Cities", "ancestor"',
        # training_dynamics_engine's fixed birth/stabilization
        "birth_step = 500",
        "stabilization_step = 2500",
        # hypothesis_generator's two fixed hypotheses
        "hyp_gen_1",
        "SAE Feature #1402",
        # ai_scientist_engine's debate pairing, including the invented neuron
        "L8_N402 mediates",
    ]
    offenders: List[str] = []
    for rel in CLUSTER + ["backend/research_platform/autonomous/ai_scientist_engine.py",
                          "backend/interpretability/discovery/run_representation_audit.py"]:
        executable = executable_source(_source(rel))
        for needle in forbidden:
            if needle in executable:
                offenders.append(f"{rel}: {needle}")
    assert not offenders, "prior fabrication returned: " + "; ".join(offenders)


def test_dag_nodes_without_an_adapter_are_not_marked_executed():
    """`{"algorithm": ..., "confidence": 0.95}` plus `node.executed = True`
    regardless -- so an unrun node reported a fabricated 0.95 *and* was marked
    executed, which is what lets `get_topological_levels` treat it as satisfied
    for its dependants."""
    from backend.interpretability.discovery.dag_discovery_planner import (
        DynamicDAGPlanner,
    )
    from backend.interpretability.discovery.discovery_planner import ResearchGoal

    report = DynamicDAGPlanner(adapter=None).execute_dag(ResearchGoal(
        goal_id="g", description="d", model_id="gpt2-small", dataset_name="ioi"))

    assert report["nodes_executed"] == 0
    assert report["nodes_measured"] == 0
    assert report["measured"] is False

    for level in report["execution_logs"]:
        for node in level["executed_nodes"]:
            assert node["executed"] is False
            assert node["measured"] is False


# ── MultiAgentResearchSociety ──────────────────────────────────────────────

def test_society_consensus_is_derived_not_asserted():
    """It returned `consensus_reached: True` and `society_status: "Completed"`
    for every goal, from seven hardcoded role dicts with no agents behind them."""
    from backend.research_platform.autonomous.multi_agent_society import (
        MultiAgentResearchSociety,
    )

    class Gate:
        def __init__(self, result):
            self._result = result

        def run_blocking(self, goal):
            return self._result

    # Raises: reported as a failure, with the exception text.
    class Boom:
        def run_blocking(self, goal):
            raise RuntimeError("planner unavailable")

    failed = MultiAgentResearchSociety(society=Boom()).run_society_collaboration("g")
    assert failed["consensus_reached"] is False
    assert failed["society_status"] == "failed"
    assert "planner unavailable" in failed["error"]
    assert failed["reason"]

    # Not a dict: unavailable, not a fabricated success.
    weird = MultiAgentResearchSociety(
        society=Gate("all good")).run_society_collaboration("g")
    assert weird["society_status"] == "unavailable"
    assert weird["consensus_reached"] is False

    # Completed but not live / not eligible: each clause must withhold consensus.
    for result, clause in (
        ({"status": "blocked", "provenance": "live",
          "validation_eligible": True, "publication_eligible": True},
         "not 'completed'"),
        ({"status": "completed", "provenance": "unavailable",
          "validation_eligible": True, "publication_eligible": True},
         "live provenance"),
        ({"status": "completed", "provenance": "live",
          "validation_eligible": False, "publication_eligible": True},
         "validation was not eligible"),
        ({"status": "completed", "provenance": "live",
          "validation_eligible": True, "publication_eligible": False},
         "publication was not eligible"),
    ):
        outcome = MultiAgentResearchSociety(
            society=Gate(result)).run_society_collaboration("g")
        assert outcome["consensus_reached"] is False, result
        assert clause in outcome["consensus_basis"], outcome["consensus_basis"]

    # Only a fully live, completed, doubly-eligible run reaches consensus.
    agreed = MultiAgentResearchSociety(society=Gate({
        "status": "completed", "provenance": "live",
        "validation_eligible": True, "publication_eligible": True,
    })).run_society_collaboration("g")
    assert agreed["consensus_reached"] is True


def test_society_does_not_invent_hypotheses_to_debate():
    """The Society runs a workflow; it does not produce two rival claims.
    `ai_scientist_engine` must not receive a fabricated pair."""
    from backend.research_platform.autonomous.multi_agent_society import (
        MultiAgentResearchSociety,
    )

    class Gate:
        def run_blocking(self, goal):
            return {"status": "completed", "provenance": "live",
                    "validation_eligible": True, "publication_eligible": True}

    society = MultiAgentResearchSociety(society=Gate())
    result = society.run_society_collaboration("g")
    assert "hypothesis_a" not in result
    assert "hypothesis_b" not in result
    assert "does not generate rival hypotheses" in result["hypotheses_source"]

    society.set_hypotheses("A", "B", evidence_a=[{"support": 0.9}])
    supplied = society.run_society_collaboration("g")
    assert supplied["hypothesis_a"] == "A"
    assert supplied["hypotheses_source"] == "supplied by caller"


def test_the_five_unimplemented_modules_still_say_so():
    """None of these may quietly start reporting numbers again."""
    from backend.science.reproducibility.copy_task_pipeline import (
        CopyTaskPipeline,
    )
    from backend.science.reproducibility.arithmetic_pipeline import (
        ArithmeticPipeline,
    )
    from backend.science.reproducibility.factual_recall_pipeline import (
        FactualRecallPipeline,
    )
    from backend.science.models.adapter_base import LiveUnavailable

    for factory in (CopyTaskPipeline, ArithmeticPipeline, FactualRecallPipeline):
        with pytest.raises(LiveUnavailable):
            factory().run()