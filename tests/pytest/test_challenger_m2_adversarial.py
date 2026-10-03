"""Adversarial stress-testing suite for Milestone 2 deliverables.

Empirically challenges:
1. IOI reproduction pipeline with diverse permutations, unusual syntax, alternative verbs,
   unregistered name pairs, and boundary inputs (empty/zero prompts).
2. AI Scientist Engine under degraded, empty, non-dict, and malformed validation inputs.
3. Discovery Lifecycle transition consistency, backward permissiveness, unhandled exceptions,
   and hypothesis outcome independence.
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
BACKEND_DIR = ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from backend.interpretability.discovery.discovery_engine import DiscoveryEngine
from backend.interpretability.discovery.discovery_lifecycle import (
    VALID_DISCOVERY_STATES,
    DiscoveryLifecycleState,
)
from backend.research_platform.autonomous.ai_scientist_engine import AIScientistEngine
from backend.research_platform.autonomous.uncertainty_manager import UncertaintyPolicy
from backend.science.models.gpt2_adapter import GPT2Adapter
from backend.science.reproducibility.ioi_pipeline import IOIReproductionPipeline


# ============================================================================
# 1. IOI REPRODUCTION PIPELINE & MOCK TOKEN ADAPTER ADVERSARIAL CHALLENGES
# ============================================================================

class TestIOIPipelineAdversarial:
    """Stress tests and edge-case evaluations for IOIReproductionPipeline and GPT2Adapter."""

    def test_ioi_pipeline_zero_prompts_handling(self):
        """Challenge: What happens if n_prompts=0 is requested?

        Does the pipeline fail closed or raise unhandled ZeroDivisionError / IndexError?
        """
        pipeline = IOIReproductionPipeline(mock_mode=True)
        # Empirical test: check if n_prompts=0 raises ZeroDivisionError or IndexError
        with pytest.raises((ZeroDivisionError, IndexError, ValueError)) as excinfo:
            pipeline.run(n_prompts=0)
        assert excinfo.type in (ZeroDivisionError, IndexError, ValueError)

    def test_gpt2_adapter_mock_does_not_synthesize_ioi_for_arbitrary_names(self):
        """Mock mode must not invent an IOI answer for arbitrary name pairs.

        A name-parsing heuristic would let 'When Xavier and Yolanda ... gave a
        drink to' return 'Yolanda', which reads exactly like a measured
        indirect-object finding while being pure fixture text. The mock only
        recognises its exact prefixes and labels everything it returns as
        synthetic and ineligible.
        """
        adapter = GPT2Adapter(variant="small", mock_mode=True)
        prompt = "When Xavier and Yolanda went to the store, Xavier gave a drink to"
        logits = adapter.get_logits(prompt)

        assert "top_tokens" in logits
        assert logits["provenance"] == "seeded"
        assert logits["validation_eligible"] is False
        assert logits["publication_eligible"] is False
        assert "Not evidence" in logits["reason"]
        tokens = [t["token"].strip() for t in logits["top_tokens"]]
        assert "Yolanda" not in tokens

    def test_gpt2_adapter_unregistered_names_without_when_fallback(self):
        """Arbitrary names without a recognised prefix fall through, still labelled."""
        adapter = GPT2Adapter(variant="small", mock_mode=True)
        prompt = "Xavier and Yolanda went to the store, Xavier gave a drink to"
        logits = adapter.get_logits(prompt)
        assert "top_tokens" in logits
        assert logits["top_token"] == " the"
        assert logits["provenance"] == "seeded"
        assert logits["publication_eligible"] is False

    def test_gpt2_adapter_recognised_prefix_is_still_labelled_synthetic(self):
        """Even a matched mock fixture must not look publishable."""
        adapter = GPT2Adapter(variant="small", mock_mode=True)
        logits = adapter.get_logits("When Mary and John went")
        assert logits["top_token"] == "to"
        assert logits["provenance"] == "seeded"
        assert logits["validation_eligible"] is False
        assert logits["publication_eligible"] is False

    def test_gpt2_adapter_mock_response_cannot_pass_evidence_policy(self):
        """A mock response must fail the discovery eligibility gate."""
        from backend.agents.evidence_policy import discovery_is_live

        adapter = GPT2Adapter(variant="small", mock_mode=True)
        logits = adapter.get_logits("When Mary and John went")
        assert discovery_is_live(logits) is False

    def test_gpt2_adapter_alternative_verb_corrupted_inversion(self):
        r"""An alternative verb must not yield a fabricated IOI resolution.

        'When Alice and Bob went to the store, Bob handed a drink to' is not a
        recognised fixture, so the mock returns its generic fallback. What
        matters here is that nothing name-shaped is invented to look like a
        measured answer.
        """
        adapter = GPT2Adapter(variant="small", mock_mode=True)
        prompt = "When Alice and Bob went to the store, Bob handed a drink to"
        logits = adapter.get_logits(prompt)
        assert logits["provenance"] == "seeded"
        assert logits["publication_eligible"] is False
        tokens = [t["token"].strip() for t in logits["top_tokens"]]
        assert "Bob" not in tokens and "Alice" not in tokens

    def test_gpt2_adapter_empty_and_whitespace_prompts(self):
        """Challenge: Extreme prompt inputs (empty string, whitespace, non-alphabetic)."""
        adapter = GPT2Adapter(variant="small", mock_mode=True)
        for extreme in ["", "   ", "\n\t", "12345!@#$%^&*()"]:
            logits = adapter.get_logits(extreme)
            assert "top_tokens" in logits
            assert logits["top_token"] == " the"
            acts = adapter.get_activations(extreme, layer=0)
            assert len(acts) > 0
            attn = adapter.get_attention_patterns(extreme, layer=0)
            assert len(attn) == adapter.spec.num_heads

    def test_gpt2_adapter_patch_activation_bounds(self):
        """Challenge: Patching with extreme values (very large, negative, zero)."""
        adapter = GPT2Adapter(variant="small", mock_mode=True)
        for val in [0.0, -100.0, 99999.0]:
            patch = adapter.patch_activation("When Alice and Bob went", layer=0, neuron_index=0, patch_value=val)
            assert patch.patched_logit == val
            assert patch.layer == 0
            assert patch.neuron_index == 0


# ============================================================================
# 2. AI SCIENTIST ENGINE DEGRADED / MALFORMED VALIDATION CHALLENGES
# ============================================================================

class TestAIScientistEngineAdversarial:
    """Stress tests for AIScientistEngine under degraded, empty, non-dict validation responses."""

    def test_ai_scientist_engine_empty_dict_validation(self):
        """Challenge: Validation engine returns an empty dict {}.

        Engine must handle missing 'confidence' gracefully with fallbacks.
        """
        engine = AIScientistEngine()
        with patch.object(engine.validation_engine, "validate_discovery", return_value={}):
            campaign = engine.run_scientific_campaign("Test question")
            assert campaign["status"] == "ScientificCampaignCompleted"
            assert "uncertainty_decision" in campaign
            assert campaign["uncertainty_decision"]["action"] in ("Publish", "More experiments", "Debate", "Reject")

    def test_ai_scientist_engine_validation_confidence_none(self):
        """Validation returns {'confidence': None}: must fail closed, not crash.

        Previously this was val_res['confidence']['confidence_score'], which
        raised TypeError. There is no confidence evidence here, so the
        campaign must not be able to publish or reject the hypothesis.
        """
        engine = AIScientistEngine()
        with patch.object(engine.validation_engine, "validate_discovery", return_value={"confidence": None}):
            campaign = engine.run_scientific_campaign("Test question")
            assert campaign["status"] == "ScientificCampaignCompleted"
            decision = campaign["uncertainty_decision"]
            assert decision["evidence_missing"] is True
            assert decision["ready_for_publication"] is False
            assert decision["action"] == "More experiments"

    def test_ai_scientist_engine_validation_confidence_empty_dict(self):
        """Validation returns {'confidence': {}}: fails closed."""
        engine = AIScientistEngine()
        with patch.object(engine.validation_engine, "validate_discovery", return_value={"confidence": {}}):
            campaign = engine.run_scientific_campaign("Test question")
            assert campaign["status"] == "ScientificCampaignCompleted"
            decision = campaign["uncertainty_decision"]
            assert decision["evidence_missing"] is True
            assert decision["ready_for_publication"] is False

    def test_ai_scientist_engine_validation_confidence_non_dict_string(self):
        """Validation returns {'confidence': 'High'}.

        A bare string is not a confidence block. This previously raised
        AttributeError ("'High' has no attribute 'get'") and aborted the
        campaign; it must now be treated as absent evidence.
        """
        engine = AIScientistEngine()
        with patch.object(engine.validation_engine, "validate_discovery", return_value={"confidence": "High"}):
            campaign = engine.run_scientific_campaign("Test question")
            assert campaign["status"] == "ScientificCampaignCompleted"
            decision = campaign["uncertainty_decision"]
            assert decision["evidence_missing"] is True
            assert decision["ready_for_publication"] is False

    def test_ai_scientist_engine_validation_confidence_non_dict_float(self):
        """Validation returns {'confidence': 0.95}.

        A bare float is not a confidence block and must not be mistaken for a
        score. This previously raised AttributeError; an absent measurement
        must fail closed rather than be read as a passing one.
        """
        engine = AIScientistEngine()
        with patch.object(engine.validation_engine, "validate_discovery", return_value={"confidence": 0.95}):
            campaign = engine.run_scientific_campaign("Test question")
            assert campaign["status"] == "ScientificCampaignCompleted"
            decision = campaign["uncertainty_decision"]
            assert decision["evidence_missing"] is True
            assert decision["ready_for_publication"] is False

    def test_ai_scientist_engine_validation_confidence_score_none(self):
        """Validation returns {'confidence': {'confidence_score': None}}.

        A None score previously reached a float comparison and raised
        TypeError("'<' not supported"). It must now be treated as missing.
        """
        engine = AIScientistEngine()
        with patch.object(engine.validation_engine, "validate_discovery", return_value={"confidence": {"confidence_score": None}}):
            campaign = engine.run_scientific_campaign("Test question")
            assert campaign["status"] == "ScientificCampaignCompleted"
            decision = campaign["uncertainty_decision"]
            assert decision["evidence_missing"] is True
            assert decision["ready_for_publication"] is False

    def test_ai_scientist_engine_validation_malformed_interval_single_item(self):
        """Validation returns a one-element uncertainty interval.

        interval[1] - interval[0] previously raised IndexError. A malformed
        interval is unusable evidence, so the campaign must fail closed.
        """
        engine = AIScientistEngine()
        with patch.object(engine.validation_engine, "validate_discovery", return_value={
            "confidence": {"confidence_score": 0.9, "uncertainty_interval": [0.9]}
        }):
            campaign = engine.run_scientific_campaign("Test question")
            assert campaign["status"] == "ScientificCampaignCompleted"
            decision = campaign["uncertainty_decision"]
            assert decision["evidence_missing"] is True
            assert decision["ready_for_publication"] is False

    def test_ai_scientist_engine_valid_confidence_can_publish(self):
        """Control: a well-formed confidence block is still honoured.

        The fail-closed cases above must not have silently disabled the
        publish path -- real evidence has to reach it.
        """
        engine = AIScientistEngine()
        with patch.object(engine.validation_engine, "validate_discovery", return_value={
            "confidence": {"confidence_score": 0.9, "uncertainty_interval": [0.88, 0.95]}
        }):
            campaign = engine.run_scientific_campaign("Test question")
            decision = campaign["uncertainty_decision"]
            assert decision["evidence_missing"] is False
            assert decision["action"] == "Publish"
            assert decision["ready_for_publication"] is True

    def test_ai_scientist_engine_closed_loop_more_experiments_trigger(self):
        """Verify the planner loop triggers on 'More experiments' from the sample-size branch.

        Requires real confidence evidence: with none, the campaign fails closed
        on 'Validation unavailable' and never reaches the sample-size branch.
        """
        engine = AIScientistEngine()
        # min_samples=10 against the campaign's declared sample size of 5.
        strict_policy = UncertaintyPolicy(min_samples=10)
        with patch.object(engine.validation_engine, "validate_discovery", return_value={
            "confidence": {"confidence_score": 0.9, "uncertainty_interval": [0.88, 0.95]}
        }):
            campaign = engine.run_scientific_campaign("Test question", policy=strict_policy)
        assert campaign["uncertainty_decision"]["evidence_missing"] is False
        assert campaign["uncertainty_decision"]["action"] == "More experiments"
        assert campaign["planner_feedback_loop"] is not None
        assert campaign["planner_feedback_loop"]["triggered"] is True
        assert campaign["planner_feedback_loop"]["target_action"] == "experiment_recommender"
        assert "next_recommended_experiment" in campaign["planner_feedback_loop"]

    def test_ai_scientist_engine_closed_loop_debate_trigger(self):
        """Verify the debate loop triggers on 'Debate' from the variance branch.

        Needs real confidence evidence plus a debate_variance below the
        campaign's declared variance of 0.02. With no evidence the campaign
        fails closed instead -- absence of evidence is not conflicting evidence.
        """
        engine = AIScientistEngine()
        debate_policy = UncertaintyPolicy(debate_variance=0.001)
        with patch.object(engine.validation_engine, "validate_discovery", return_value={
            "confidence": {"confidence_score": 0.9, "uncertainty_interval": [0.88, 0.95]}
        }):
            campaign = engine.run_scientific_campaign("Test question", policy=debate_policy)
        assert campaign["uncertainty_decision"]["evidence_missing"] is False
        assert campaign["uncertainty_decision"]["action"] == "Debate"
        assert campaign["planner_feedback_loop"] is not None
        assert campaign["planner_feedback_loop"]["triggered"] is True
        assert campaign["planner_feedback_loop"]["target_action"] == "scientific_debate"
        assert "followup_debate" in campaign["planner_feedback_loop"]

    def test_ai_scientist_engine_no_evidence_fails_closed(self):
        """Control: with no live validator connected the campaign cannot publish.

        The real validation engine returns an 'unavailable' envelope, so the
        campaign must not reach Publish or Reject on absent evidence.
        """
        engine = AIScientistEngine()
        campaign = engine.run_scientific_campaign("Test question")
        decision = campaign["uncertainty_decision"]
        assert decision["evidence_missing"] is True
        assert decision["ready_for_publication"] is False
        assert decision["action"] == "More experiments"


# ============================================================================
# 3. DISCOVERY LIFECYCLE TRANSITION CONSISTENCY ADVERSARIAL CHALLENGES
# ============================================================================

class TestDiscoveryLifecycleAdversarial:
    """Stress tests for DiscoveryEngine lifecycle state transitions and invariants."""

    def test_lifecycle_canonical_progression(self):
        """Verify standard forward progression through all 6 valid states."""
        lifecycle = DiscoveryLifecycleState(discovery_id="disc_test_1", title="Test Discovery")
        assert lifecycle.state == "Candidate"

        order = [
            "Evidence Collection",
            "Validation",
            "Confidence",
            "Knowledge",
            "Publication",
        ]
        for st in order:
            lifecycle.transition_to(st, f"Entering {st}")
            assert lifecycle.state == st

        assert len(lifecycle.history) == 6
        assert [h["state"] for h in lifecycle.history] == ["Candidate"] + order

    def test_lifecycle_rejects_invalid_state_name(self):
        """Verify ValueError is raised when transitioning to an unregistered state name."""
        lifecycle = DiscoveryLifecycleState(discovery_id="disc_test_2", title="Test Discovery")
        with pytest.raises(ValueError) as excinfo:
            lifecycle.transition_to("Archived")
        assert "Invalid discovery state transition" in str(excinfo.value)

        with pytest.raises(ValueError) as excinfo:
            lifecycle.transition_to("Failed")
        assert "Invalid discovery state transition" in str(excinfo.value)

    def test_lifecycle_allows_backwards_and_arbitrary_jumps(self):
        """Challenge: Does DiscoveryLifecycleState prevent backwards or skipped transitions?

        Empirical test reveals that state transitions do not enforce forward-only DAG order.
        Any state in VALID_DISCOVERY_STATES can be transitioned to at any time.
        """
        lifecycle = DiscoveryLifecycleState(discovery_id="disc_test_3", title="Test Discovery")
        # Jump straight to Publication from Candidate
        lifecycle.transition_to("Publication", "Immediate publication")
        assert lifecycle.state == "Publication"

        # Backward transition from Publication to Candidate
        lifecycle.transition_to("Candidate", "Demoted back to candidate")
        assert lifecycle.state == "Candidate"

    def test_discovery_engine_empty_and_whitespace_hypothesis(self):
        """Empty or whitespace hypotheses are rejected, not run.

        These previously flowed straight through the orchestrator and produced
        an ordinary-looking result -- including reaching Publication. A
        hypothesis with no content cannot produce evidence, so it must not
        start a lifecycle at all.
        """
        engine = DiscoveryEngine()
        for empty_stmt in ["", "   ", "\t\n"]:
            with pytest.raises(ValueError) as excinfo:
                engine.discover_and_orchestrate(empty_stmt)
            assert "non-empty string" in str(excinfo.value)
        assert engine.discoveries == {}

    def test_discovery_engine_none_hypothesis_rejected(self):
        """A None hypothesis raises a clear ValueError, not an opaque TypeError.

        hypothesis_statement[:50] used to raise
        TypeError: 'NoneType' object is not subscriptable.
        """
        engine = DiscoveryEngine()
        with pytest.raises(ValueError) as excinfo:
            engine.discover_and_orchestrate(None)
        assert "non-empty string" in str(excinfo.value)

    def test_discovery_engine_executor_failure_does_not_leave_orphaned_state(self):
        """A raising live executor must not leave a half-advanced discovery.

        Previously the synthetic cross-model stage raised mid-run and left the
        discovery stranded in 'Validation' with no failure record.
        """
        engine = DiscoveryEngine()
        with patch(
            "backend.interpretability.discovery.live_discovery.LiveIOIDiscovery.run",
            side_effect=RuntimeError("executor exploded"),
        ):
            with patch(
                "backend.interpretability.discovery.live_discovery.LiveIOIDiscovery.available",
                return_value=True,
            ):
                with pytest.raises(RuntimeError):
                    engine.discover_and_orchestrate("L8_N402 mediates IOI capital retrieval")

        # The failure is recorded in the lifecycle rather than silently stranding it.
        disc = list(engine.discoveries.values())[0]
        assert disc.failed is True
        assert "executor exploded" in disc.failure_reason
        assert disc.history[-1]["failed"] is True

    def test_discovery_engine_never_reaches_publication_without_live_evidence(self):
        """Control: this orchestrator never advances a discovery to Publication.

        The removed synthetic pipeline advanced every discovery to Publication
        regardless of the hypothesis test outcome. The live orchestrator stops
        at Validation and hands off to Society validation instead.
        """
        engine = DiscoveryEngine()
        res = engine.discover_and_orchestrate("A completely false hypothesis")

        assert res["lifecycle"]["state"] != "Publication"
        assert res["status"] in ("completed", "unavailable", "blocked", "error")
        # Eligibility may only be asserted when the evidence is explicitly live
        # and completed. Anything else must be ineligible with a reason.
        if res.get("provenance") == "live" and res.get("status") == "completed":
            # Live and completed is not sufficient for eligibility. The executor
            # derives the flags from prompt count and interaction sampling, and
            # marks a four-prompt run validation-ineligible. This branch used to
            # assert `validation_eligible is True`, pinning the unconditional
            # flags the executor used to emit.
            #
            # The invariant worth keeping is that the lifecycle agrees with the
            # flags: the orchestrator must not sit at Validation while the record
            # says the result is not eligible for validation.
            assert res["measured"] is True
            assert res["publication_eligible"] is False
            if res["validation_eligible"] is False:
                assert res["ineligible_because"], "an ineligible result must say why"
                assert res["lifecycle"]["state"] != "Validation"
            else:
                assert res["lifecycle"]["state"] == "Validation"
        else:
            assert res["validation_eligible"] is False
            assert res["publication_eligible"] is False
            assert res["reason"]
