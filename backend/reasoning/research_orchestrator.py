"""
Scientific Research Orchestrator

The brain/orchestrator between the raw scientific engine (Agent 1) and the UI (Agent 2).

Owns:
- Hypothesis generation from behavior/observations
- Experiment planning from hypotheses
- Candidate prioritization based on evidence
- Hypothesis lifecycle management
- Falsification testing
- Experiment selection for research loops
- Evidence reasoning and chain organization

IMPORTANT: This module NEVER fabricates experimental evidence.
It calls Agent 1 (scientific engine) for all actual measurements.
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("MECH.reasoning.orchestrator")


# ---------------------------------------------------------------------------
# Hypothesis Lifecycle States
# ---------------------------------------------------------------------------
class HypothesisState(str, Enum):
    PROPOSED = "PROPOSED"
    PLANNED = "PLANNED"
    TESTING = "TESTING"
    REPLICATION = "REPLICATION"
    EVALUATION = "EVALUATION"
    SUPPORTED = "SUPPORTED"
    REFUTED = "REFUTED"
    INSUFFICIENT = "INSUFFICIENT"


# ---------------------------------------------------------------------------
# State Mapping: HypothesisState ↔ HypothesisStatus
# ---------------------------------------------------------------------------
def state_to_status(state: HypothesisState) -> str:
    """
    Maps orchestrator HypothesisState to database HypothesisStatus.
    
    The orchestrator tracks detailed lifecycle states (PROPOSED, PLANNED, TESTING, etc.)
    while the database uses simpler evidence-based statuses.
    """
    mapping = {
        HypothesisState.PROPOSED: "UNTESTED",
        HypothesisState.PLANNED: "UNTESTED",
        HypothesisState.TESTING: "UNTESTED",
        HypothesisState.REPLICATION: "UNTESTED",
        HypothesisState.EVALUATION: "INCONCLUSIVE",
        HypothesisState.SUPPORTED: "SUPPORTED",
        HypothesisState.REFUTED: "FALSIFIED",
        HypothesisState.INSUFFICIENT: "INCONCLUSIVE",
    }
    return mapping.get(state, "UNTESTED")


def status_to_state(status: str) -> HypothesisState:
    """
    Maps database HypothesisStatus to orchestrator HypothesisState.
    
    When loading hypotheses from the database, we map the simpler status
    back to the orchestrator's lifecycle state.
    """
    mapping = {
        "UNTESTED": HypothesisState.PROPOSED,
        "SUPPORTED": HypothesisState.SUPPORTED,
        "PARTIALLY_SUPPORTED": HypothesisState.EVALUATION,
        "CONTRADICTED": HypothesisState.REFUTED,
        "FALSIFIED": HypothesisState.REFUTED,
        "INCONCLUSIVE": HypothesisState.INSUFFICIENT,
    }
    return mapping.get(status, HypothesisState.PROPOSED)


def sync_hypothesis_state_to_db(hypothesis: "Hypothesis", storage: Any) -> None:
    """
    Synchronizes the orchestrator's hypothesis state to the database.
    
    This ensures the database HypothesisStatus stays in sync with
    the orchestrator's HypothesisState lifecycle.
    """
    try:
        db_hyp = storage.get_hypothesis(hypothesis.id)
        if db_hyp:
            db_hyp["status"] = state_to_status(hypothesis.state)
            storage.save_hypothesis(db_hyp)
    except Exception as e:
        logger.warning("Failed to sync hypothesis state to DB: %s", e)


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------
@dataclass
class Hypothesis:
    """A falsifiable mechanistic claim about model behavior."""
    id: str
    title: str
    statement: str
    target_component: str
    prediction: str
    falsification_condition: str
    state: HypothesisState = HypothesisState.PROPOSED
    confidence: float = 0.0
    evidence_count_supporting: int = 0
    evidence_count_contradicting: int = 0
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExperimentSpecification:
    """A complete experiment plan derived from a hypothesis."""
    id: str
    hypothesis_id: str
    name: str
    description: str
    clean_prompt: str
    corrupted_prompt: Optional[str]
    target_token: str
    distractor_token: Optional[str]
    source_component: str
    intervention_type: str
    control_components: List[str]
    repeats: int
    seed: int
    metrics: List[str]
    created_at: float = field(default_factory=time.time)


@dataclass
class EvidenceChain:
    """Organized evidence supporting or contradicting a hypothesis."""
    hypothesis_id: str
    supporting_evidence: List[Dict[str, Any]]
    contradicting_evidence: List[Dict[str, Any]]
    total_effect_size: float
    specificity_ratio: float
    replication_count: int
    overall_assessment: str


@dataclass
class CandidatePriority:
    """Priority ranking for a candidate component."""
    component: str
    priority_score: float
    reason: str
    evidence_level: str
    recommended_action: str


@dataclass
class ResearchLoopState:
    """State of an active research loop."""
    investigation_id: str
    current_hypothesis_id: Optional[str]
    experiment_queue: List[str]
    completed_experiments: List[str]
    iteration: int
    max_iterations: int
    status: str


# ---------------------------------------------------------------------------
# Scientific Research Orchestrator
# ---------------------------------------------------------------------------
class ScientificResearchOrchestrator:
    """
    Orchestrates the scientific research process between Agent 1 and Agent 2.
    
    This module NEVER fabricates evidence. It:
    1. Generates hypotheses from observations
    2. Plans experiments from hypotheses
    3. Sends specifications to Agent 1 for execution
    4. Analyzes Agent 1's results
    5. Manages hypothesis lifecycle
    6. Designs falsification tests
    """
    
    def __init__(self, storage=None, experiment_runner=None, llm_engine=None):
        self.storage = storage
        self.experiment_runner = experiment_runner
        self.llm_engine = llm_engine
        self._hypotheses: Dict[str, Hypothesis] = {}
        self._experiment_specs: Dict[str, ExperimentSpecification] = {}
        self._research_loops: Dict[str, ResearchLoopState] = {}
    
    # -----------------------------------------------------------------------
    # Hypothesis Generation
    # -----------------------------------------------------------------------
    def generate_hypotheses(
        self,
        behavior: str,
        observation: str,
        research_question: str,
        target_components: List[str],
        max_hypotheses: int = 3,
    ) -> List[Hypothesis]:
        """
        Generate testable hypotheses from behavior, observation, and research question.
        
        Uses LLM engine if available for richer hypothesis generation,
        falls back to template-based generation otherwise.
        
        Args:
            behavior: The model behavior being investigated
            observation: What was observed about the behavior
            research_question: The core research question
            target_components: Components to focus on (e.g., ["L9H9", "L7H9"])
            max_hypotheses: Maximum number of hypotheses to generate
        
        Returns:
            List of PROPOSED hypotheses (not yet tested)
        """
        hypotheses = []
        
        # Try LLM-based generation first
        if self.llm_engine:
            try:
                feature_report = {
                    "behavior": behavior,
                    "observation": observation,
                    "research_question": research_question,
                    "target_components": target_components,
                }
                llm_hypotheses = self.llm_engine.generate_hypotheses(
                    feature_report=feature_report,
                    max_candidates=max_hypotheses,
                )
                
                for i, llm_hyp in enumerate(llm_hypotheses[:max_hypotheses]):
                    component = target_components[i] if i < len(target_components) else target_components[0] if target_components else "L9H9"
                    hyp = Hypothesis(
                        id=f"hyp_{uuid.uuid4().hex[:8]}",
                        title=f"LLM Hypothesis {i+1}: {component} role in {behavior}",
                        statement=llm_hyp.get("description", f"Component {component} contributes to {behavior}"),
                        target_component=component,
                        prediction=f"Ablating {component} will reduce target logit by >0.5",
                        falsification_condition=f"Ablating {component} produces Δlogit < 0.3 or equal effect on random control",
                        state=HypothesisState.PROPOSED,
                        confidence=llm_hyp.get("initial_confidence", 0.5),
                        metadata={
                            "behavior": behavior,
                            "observation": observation,
                            "research_question": research_question,
                            "source": "llm",
                        },
                    )
                    hypotheses.append(hyp)
                    self._hypotheses[hyp.id] = hyp
                
                logger.info("Generated %d hypotheses via LLM engine", len(hypotheses))
                return hypotheses
                
            except Exception as e:
                logger.warning("LLM hypothesis generation failed, falling back to templates: %s", e)
        
        # Fallback: template-based generation
        for i, component in enumerate(target_components[:max_hypotheses]):
            hyp = Hypothesis(
                id=f"hyp_{uuid.uuid4().hex[:8]}",
                title=f"H{i+1}: {component} role in {behavior}",
                statement=f"Component {component} contributes to {behavior} via {observation}",
                target_component=component,
                prediction=f"Ablating {component} will reduce target logit by >0.5",
                falsification_condition=f"Ablating {component} produces Δlogit < 0.3 or equal effect on random control",
                state=HypothesisState.PROPOSED,
                metadata={
                    "behavior": behavior,
                    "observation": observation,
                    "research_question": research_question,
                    "source": "template",
                },
            )
            hypotheses.append(hyp)
            self._hypotheses[hyp.id] = hyp
        
        return hypotheses
    
    # -----------------------------------------------------------------------
    # Experiment Planning
    # -----------------------------------------------------------------------
    def plan_experiment(
        self,
        hypothesis: Hypothesis,
        clean_prompt: str,
        corrupted_prompt: Optional[str] = None,
        target_token: str = "",
        distractor_token: Optional[str] = None,
        repeats: int = 10,
        seed: int = 42,
    ) -> ExperimentSpecification:
        """
        Convert a hypothesis into a complete experiment specification.
        
        The plan follows the scientific method:
        1. Baseline measurement
        2. Intervention on target component
        3. Matched control (same layer, different head)
        4. Random control (different layer)
        5. Repeat N times
        6. Measure logit difference
        7. Compare effect sizes
        
        Args:
            hypothesis: The hypothesis to test
            clean_prompt: The clean prompt for the experiment
            corrupted_prompt: Optional corrupted prompt for patching
            target_token: Target token to measure
            distractor_token: Optional distractor token
            repeats: Number of trial repetitions
            seed: Random seed for reproducibility
        
        Returns:
            Complete experiment specification ready for Agent 1
        """
        # Generate control components
        layer = self._extract_layer(hypothesis.target_component)
        head = self._extract_head(hypothesis.target_component)
        
        control_components = []
        if head is not None:
            # Same-layer, different head (matched control)
            for h in range(12):
                if h != head:
                    control_components.append(f"L{layer}H{h}")
                    if len(control_components) >= 2:
                        break
        
        # Random control (different layer)
        random_layer = (layer + 3) % 12
        control_components.append(f"L{random_layer}H{head or 0}")
        
        spec = ExperimentSpecification(
            id=f"exp_{uuid.uuid4().hex[:8]}",
            hypothesis_id=hypothesis.id,
            name=f"Test: {hypothesis.title}",
            description=f"Testing hypothesis: {hypothesis.statement}",
            clean_prompt=clean_prompt,
            corrupted_prompt=corrupted_prompt,
            target_token=target_token,
            distractor_token=distractor_token,
            source_component=hypothesis.target_component,
            intervention_type="ABLATION_ZERO",
            control_components=control_components,
            repeats=repeats,
            seed=seed,
            metrics=["delta_logit", "delta_prob", "effect_size", "specificity"],
        )
        
        # Update hypothesis state
        hypothesis.state = HypothesisState.PLANNED
        hypothesis.updated_at = time.time()
        
        self._experiment_specs[spec.id] = spec
        return spec
    
    # -----------------------------------------------------------------------
    # Falsification Testing
    # -----------------------------------------------------------------------
    def design_falsification_test(
        self,
        hypothesis: Hypothesis,
        clean_prompt: str,
        target_token: str,
    ) -> ExperimentSpecification:
        """
        Design a falsification test for a hypothesis.
        
        Instead of asking "How can I prove this?", we ask:
        "What experiment could disprove this?"
        
        The falsification test:
        1. Identifies the strongest prediction
        2. Designs a test that would fail if the hypothesis is wrong
        3. Includes controls that would show effect if hypothesis is correct
        
        Args:
            hypothesis: The hypothesis to falsify
            clean_prompt: The clean prompt for testing
            target_token: Target token to measure
        
        Returns:
            Experiment specification designed to falsify the hypothesis
        """
        # Extract the strongest prediction from the hypothesis
        prediction = hypothesis.prediction
        
        # Design a test that would DISPROVE the hypothesis if it's wrong
        # If the hypothesis is correct, ablating the target should reduce logit
        # If the hypothesis is wrong, ablating the target should have no effect
        
        falsification_spec = ExperimentSpecification(
            id=f"fals_{uuid.uuid4().hex[:8]}",
            hypothesis_id=hypothesis.id,
            name=f"Falsification: {hypothesis.title}",
            description=f"Attempting to DISPROVE: {hypothesis.statement}",
            clean_prompt=clean_prompt,
            corrupted_prompt=None,
            target_token=target_token,
            distractor_token=None,
            source_component=hypothesis.target_component,
            intervention_type="ABLATION_ZERO",
            control_components=[f"L0H0", f"L6H6"],  # Known irrelevant controls
            repeats=20,  # More repetitions for falsification
            seed=42,
            metrics=["delta_logit", "delta_prob", "effect_size", "specificity", "p_value"],
        )
        
        return falsification_spec
    
    # -----------------------------------------------------------------------
    # Hypothesis Lifecycle Management
    # -----------------------------------------------------------------------
    def update_hypothesis_state(
        self,
        hypothesis_id: str,
        new_state: HypothesisState,
        evidence: Optional[Dict[str, Any]] = None,
    ) -> Hypothesis:
        """
        Update the state of a hypothesis based on evidence.
        
        Lifecycle: PROPOSED → PLANNED → TESTING → REPLICATION → EVALUATION → SUPPORTED/REFUTED/INSUFFICIENT
        
        Args:
            hypothesis_id: ID of the hypothesis to update
            new_state: New state to transition to
            evidence: Optional evidence record from Agent 1
        
        Returns:
            Updated hypothesis
        """
        hypothesis = self._hypotheses.get(hypothesis_id)
        if not hypothesis:
            raise ValueError(f"Hypothesis {hypothesis_id} not found")
        
        # Validate state transition
        # Per spec: PROPOSED -> PLANNED -> TESTING -> REPLICATION -> EVALUATION
        # REPLICATION is required before EVALUATION
        valid_transitions = {
            HypothesisState.PROPOSED: [HypothesisState.PLANNED],
            HypothesisState.PLANNED: [HypothesisState.TESTING],
            HypothesisState.TESTING: [HypothesisState.REPLICATION],  # Must replicate before evaluation
            HypothesisState.REPLICATION: [HypothesisState.EVALUATION],
            HypothesisState.EVALUATION: [
                HypothesisState.SUPPORTED,
                HypothesisState.REFUTED,
                HypothesisState.INSUFFICIENT,
            ],
        }
        
        if new_state not in valid_transitions.get(hypothesis.state, []):
            raise ValueError(
                f"Invalid state transition: {hypothesis.state} → {new_state}"
            )
        
        # Update evidence counts if evidence provided
        if evidence:
            if evidence.get("supports_hypothesis", False):
                hypothesis.evidence_count_supporting += 1
            else:
                hypothesis.evidence_count_contradicting += 1
        
        # Update confidence based on evidence ratio
        total_evidence = hypothesis.evidence_count_supporting + hypothesis.evidence_count_contradicting
        if total_evidence > 0:
            hypothesis.confidence = hypothesis.evidence_count_supporting / total_evidence
        
        hypothesis.state = new_state
        hypothesis.updated_at = time.time()
        
        # Sync state to database if storage is available
        if self.storage:
            sync_hypothesis_state_to_db(hypothesis, self.storage)
        
        return hypothesis
    
    # -----------------------------------------------------------------------
    # Candidate Prioritization
    # -----------------------------------------------------------------------
    def prioritize_candidates(
        self,
        candidates: List[str],
        evidence_records: List[Dict[str, Any]],
    ) -> List[CandidatePriority]:
        """
        Prioritize candidate components based on existing evidence.
        
        Args:
            candidates: List of candidate component names
            evidence_records: List of evidence records from Agent 1
        
        Returns:
            Prioritized list of candidates with recommendations
        """
        priorities = []
        
        for candidate in candidates:
            # Find evidence for this candidate
            candidate_evidence = [
                e for e in evidence_records
                if e.get("target_component") == candidate
            ]
            
            # Calculate priority score
            supporting = sum(1 for e in candidate_evidence if e.get("supports_hypothesis"))
            contradicting = sum(1 for e in candidate_evidence if not e.get("supports_hypothesis"))
            total = supporting + contradicting
            
            if total == 0:
                score = 0.5  # No evidence, neutral priority
                level = "UNTESTED"
                action = "INVESTIGATE"
            elif supporting > contradicting:
                score = supporting / total
                level = "SUPPORTED"
                action = "REPLICATE"
            elif contradicting > supporting:
                score = contradicting / total
                level = "CONTRADICTED"
                action = "FALSIFY"
            else:
                score = 0.5
                level = "INCONCLUSIVE"
                action = "INVESTIGATE"
            
            priorities.append(CandidatePriority(
                component=candidate,
                priority_score=score,
                reason=f"{supporting} supporting, {contradicting} contradicting evidence",
                evidence_level=level,
                recommended_action=action,
            ))
        
        # Sort by priority score (highest first)
        priorities.sort(key=lambda x: x.priority_score, reverse=True)
        
        return priorities
    
    # -----------------------------------------------------------------------
    # Evidence Reasoning
    # -----------------------------------------------------------------------
    def reason_about_evidence(
        self,
        hypothesis_id: str,
        evidence_records: List[Dict[str, Any]],
    ) -> EvidenceChain:
        """
        Organize evidence into a chain and explain support/contradiction.
        
        Args:
            hypothesis_id: ID of the hypothesis to reason about
            evidence_records: List of evidence records from Agent 1
        
        Returns:
            Organized evidence chain with assessment
        """
        hypothesis = self._hypotheses.get(hypothesis_id)
        if not hypothesis:
            raise ValueError(f"Hypothesis {hypothesis_id} not found")
        
        supporting = [e for e in evidence_records if e.get("supports_hypothesis")]
        contradicting = [e for e in evidence_records if not e.get("supports_hypothesis")]
        
        # Calculate aggregate metrics
        total_effect = sum(e.get("metric_value", 0) for e in supporting)
        avg_effect = total_effect / len(supporting) if supporting else 0
        
        # Specificity: ratio of target effect to max control effect
        control_effects = [e.get("control_value", 0) for e in evidence_records if e.get("control_value")]
        max_control = max(control_effects) if control_effects else 0
        specificity = avg_effect / max_control if max_control > 0 and len(evidence_records) > 0 else 0.0
        
        # Replication count
        replication_count = len(set(e.get("experiment_run_id") for e in evidence_records if e.get("experiment_run_id")))
        
        # Overall assessment
        if len(supporting) > len(contradicting) and specificity > 2.0:
            assessment = "EVIDENCE_SUPPORTS"
        elif len(contradicting) > len(supporting):
            assessment = "EVIDENCE_CONTRADICTS"
        else:
            assessment = "INSUFFICIENT_EVIDENCE"
        
        return EvidenceChain(
            hypothesis_id=hypothesis_id,
            supporting_evidence=supporting,
            contradicting_evidence=contradicting,
            total_effect_size=avg_effect,
            specificity_ratio=specificity,
            replication_count=replication_count,
            overall_assessment=assessment,
        )
    
    # -----------------------------------------------------------------------
    # Research Loop Controller
    # -----------------------------------------------------------------------
    def start_research_loop(
        self,
        investigation_id: str,
        initial_hypothesis_id: str,
        max_iterations: int = 10,
    ) -> ResearchLoopState:
        """
        Start a research loop for iterative experimentation.
        
        The loop:
        1. Run experiment
        2. Analyze results
        3. Identify uncertainty
        4. Design next experiment
        5. Repeat
        
        Args:
            investigation_id: ID of the investigation
            initial_hypothesis_id: Starting hypothesis
            max_iterations: Maximum loop iterations
        
        Returns:
            Initial research loop state
        """
        loop = ResearchLoopState(
            investigation_id=investigation_id,
            current_hypothesis_id=initial_hypothesis_id,
            experiment_queue=[],
            completed_experiments=[],
            iteration=0,
            max_iterations=max_iterations,
            status="RUNNING",
        )
        
        self._research_loops[investigation_id] = loop
        return loop
    
    def select_next_experiment(
        self,
        investigation_id: str,
        available_experiments: List[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        """
        Select the next experiment to run based on current state.
        
        Considers:
        - Current hypothesis state
        - Existing evidence
        - Uncertainty reduction potential
        - Control requirements
        
        Args:
            investigation_id: ID of the investigation
            available_experiments: List of available experiment specifications
        
        Returns:
            Selected experiment or None if loop should end
        """
        loop = self._research_loops.get(investigation_id)
        if not loop:
            return None
        
        if loop.iteration >= loop.max_iterations:
            loop.status = "COMPLETED"
            return None
        
        if not available_experiments:
            return None
        
        # Simple selection: pick experiments that test the current hypothesis
        # or provide controls
        current_hyp = self._hypotheses.get(loop.current_hypothesis_id)
        
        if current_hyp:
            # Prioritize experiments testing current hypothesis
            relevant = [
                e for e in available_experiments
                if e.get("hypothesis_id") == loop.current_hypothesis_id
            ]
            if relevant:
                return relevant[0]
        
        # Otherwise, pick the first available
        return available_experiments[0] if available_experiments else None
    
    # -----------------------------------------------------------------------
    # Helper Methods
    # -----------------------------------------------------------------------
    def _extract_layer(self, component: str) -> int:
        """Extract layer number from component string like 'L9H9'."""
        if component.startswith("L"):
            parts = component.split("H")
            return int(parts[0][1:])
        return 0
    
    def _extract_head(self, component: str) -> Optional[int]:
        """Extract head number from component string like 'L9H9'."""
        if "H" in component:
            parts = component.split("H")
            return int(parts[1])
        return None
    
    def evaluate_with_llm(
        self,
        hypothesis_id: str,
        experiment_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Use LLM to evaluate whether experiment results support or falsify a hypothesis.
        
        This provides a natural language interpretation of the quantitative results.
        
        Args:
            hypothesis_id: ID of the hypothesis to evaluate
            experiment_result: The experiment result from Agent 1
            
        Returns:
            LLM evaluation with type (supportive/falsifying/inconclusive) and rationale
        """
        hypothesis = self._hypotheses.get(hypothesis_id)
        if not hypothesis:
            raise ValueError(f"Hypothesis {hypothesis_id} not found")
        
        if not self.llm_engine:
            # Fallback: simple heuristic evaluation
            delta_logit = experiment_result.get("delta_logit", 0.0)
            specificity = experiment_result.get("specificity_ratio", 0.0)
            
            if delta_logit > 0.5 and specificity > 2.0:
                return {"type": "supportive", "rationale": f"Strong effect: Δlogit={delta_logit:.2f}, specificity={specificity:.1f}x"}
            elif delta_logit < 0.1:
                return {"type": "falsifying", "rationale": f"Weak effect: Δlogit={delta_logit:.2f} (below threshold)"}
            else:
                return {"type": "inconclusive", "rationale": f"Moderate effect: Δlogit={delta_logit:.2f}, needs more evidence"}
        
        try:
            evaluation = self.llm_engine.evaluate_experiment(
                hypothesis=hypothesis.statement,
                experiment_result=experiment_result,
            )
            return evaluation
        except Exception as e:
            logger.warning("LLM evaluation failed: %s", e)
            return {"type": "inconclusive", "rationale": f"LLM evaluation failed: {e}"}
    
    def critique_hypothesis(self, hypothesis_id: str) -> Dict[str, Any]:
        """
        Use LLM to generate a skeptic's critique of a hypothesis.
        
        This helps identify weaknesses, confounding variables, and alternative explanations.
        
        Args:
            hypothesis_id: ID of the hypothesis to critique
            
        Returns:
            LLM critique with critique text and alternative explanation
        """
        hypothesis = self._hypotheses.get(hypothesis_id)
        if not hypothesis:
            raise ValueError(f"Hypothesis {hypothesis_id} not found")
        
        if not self.llm_engine:
            return {
                "critique": "No LLM engine available for critique.",
                "alternative_explanation": "Consider that the observed effect may be due to confounding variables.",
            }
        
        try:
            critique = self.llm_engine.generate_critique(hypothesis=hypothesis.statement)
            return critique
        except Exception as e:
            logger.warning("LLM critique failed: %s", e)
            return {"critique": f"LLM critique failed: {e}", "alternative_explanation": ""}
    
    # -----------------------------------------------------------------------
    # Research Loop Step
    # -----------------------------------------------------------------------
    def step_research_loop(
        self,
        investigation_id: str,
        experiment_result: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Execute one step of the research loop.
        
        Steps through: OBSERVATION → HYPOTHESIS → PLAN → EXPERIMENT → RESULT → 
                       FALSIFICATION → REPLICATION → EVALUATION → NEXT EXPERIMENT
        
        Args:
            investigation_id: ID of the investigation
            experiment_result: Optional result from the previous experiment (Agent 1)
        
        Returns:
            Next experiment recommendation or loop completion status
        """
        loop = self._research_loops.get(investigation_id)
        if not loop:
            raise ValueError(f"Research loop {investigation_id} not found")
        
        if loop.status != "RUNNING":
            return {"status": loop.status, "message": "Loop is not running"}
        
        if loop.iteration >= loop.max_iterations:
            loop.status = "COMPLETED"
            return {"status": "COMPLETED", "message": "Maximum iterations reached"}
        
        # Process previous experiment result if provided
        if experiment_result:
            loop.completed_experiments.append(loop.current_hypothesis_id or "")
            
            # Update hypothesis state based on result
            if loop.current_hypothesis_id:
                hypothesis = self._hypotheses.get(loop.current_hypothesis_id)
                if hypothesis:
                    # Evaluate the result
                    evaluation = self.evaluate_with_llm(loop.current_hypothesis_id, experiment_result)
                    
                    # Determine next state
                    if evaluation.get("type") == "supportive":
                        if hypothesis.state == HypothesisState.TESTING:
                            loop.current_state = HypothesisState.REPLICATION
                        elif hypothesis.state == HypothesisState.REPLICATION:
                            loop.current_state = HypothesisState.EVALUATION
                    elif evaluation.get("type") == "falsifying":
                        loop.current_state = HypothesisState.REFUTED
                    # If inconclusive, stay in current state
        
        loop.iteration += 1
        
        # Generate next experiment recommendation
        recommendation = self._generate_next_experiment_recommendation(loop)
        
        return {
            "status": "RUNNING",
            "iteration": loop.iteration,
            "max_iterations": loop.max_iterations,
            "current_hypothesis_id": loop.current_hypothesis_id,
            "next_experiment": recommendation,
            "completed_experiments": len(loop.completed_experiments),
        }
    
    def _generate_next_experiment_recommendation(
        self,
        loop: ResearchLoopState,
    ) -> Optional[Dict[str, Any]]:
        """Generate recommendation for the next experiment."""
        hypothesis = self._hypotheses.get(loop.current_hypothesis_id)
        if not hypothesis:
            return None
        
        # Based on hypothesis state, recommend next action
        if hypothesis.state == HypothesisState.PROPOSED:
            return {
                "action": "PLAN_EXPERIMENT",
                "hypothesis_id": hypothesis.id,
                "message": f"Plan experiment to test: {hypothesis.title}",
            }
        elif hypothesis.state == HypothesisState.PLANNED:
            return {
                "action": "RUN_EXPERIMENT",
                "hypothesis_id": hypothesis.id,
                "message": f"Execute experiment for: {hypothesis.title}",
            }
        elif hypothesis.state == HypothesisState.TESTING:
            return {
                "action": "FALSIFICATION_TEST",
                "hypothesis_id": hypothesis.id,
                "message": f"Design falsification test for: {hypothesis.title}",
            }
        elif hypothesis.state == HypothesisState.REPLICATION:
            return {
                "action": "REPLICATE",
                "hypothesis_id": hypothesis.id,
                "message": f"Replicate results for: {hypothesis.title}",
            }
        elif hypothesis.state == HypothesisState.EVALUATION:
            return {
                "action": "EVALUATE_EVIDENCE",
                "hypothesis_id": hypothesis.id,
                "message": f"Evaluate cumulative evidence for: {hypothesis.title}",
            }
        
        return None
    
    def auto_research_loop(
        self,
        investigation_id: str,
        max_iterations: int = 10,
    ) -> Dict[str, Any]:
        """
        Run the research loop automatically until completion or max iterations.
        
        This generates a sequence of experiments without requiring manual intervention.
        
        Args:
            investigation_id: ID of the investigation
            max_iterations: Maximum number of iterations
        
        Returns:
            Complete research loop results
        """
        loop = self._research_loops.get(investigation_id)
        if not loop:
            raise ValueError(f"Research loop {investigation_id} not found")
        
        results = []
        
        for i in range(max_iterations):
            if loop.status != "RUNNING":
                break
            
            # Generate next experiment
            recommendation = self._generate_next_experiment_recommendation(loop)
            if not recommendation:
                break
            
            results.append({
                "iteration": loop.iteration + 1,
                "recommendation": recommendation,
                "hypothesis_state": loop.current_state.value if hasattr(loop, 'current_state') else "UNKNOWN",
            })
            
            # Simulate step (in real usage, this would call Agent 1)
            loop.iteration += 1
            loop.completed_experiments.append(loop.current_hypothesis_id or "")
        
        loop.status = "COMPLETED"
        
        return {
            "status": "COMPLETED",
            "iterations": loop.iteration,
            "results": results,
            "hypotheses_evaluated": len(set(r["recommendation"].get("hypothesis_id") for r in results if r.get("recommendation"))),
        }
    
    # -----------------------------------------------------------------------
    # Discovery Bridge
    # -----------------------------------------------------------------------
    def process_discovery_report(
        self,
        discovery_report: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Process Agent 1's CircuitDiscoveryReport and generate research directions.
        
        Consumes:
        - CircuitDiscoveryReport
        - CandidateResult
        
        Generates:
        - Candidate hypotheses
        - Experiment specifications
        - Investigation priorities
        
        Args:
            discovery_report: The discovery report from Agent 1
        
        Returns:
            Research directions with hypotheses and experiment specs
        """
        candidates = discovery_report.get("candidates", [])
        behavior = discovery_report.get("behavior", "unknown")
        observation = discovery_report.get("observation", "No observation provided")
        
        # Generate hypotheses from candidates
        hypotheses = []
        for candidate in candidates[:3]:  # Limit to top 3
            component = candidate.get("component", "L0H0")
            score = candidate.get("score", 0.0)
            
            hyp = Hypothesis(
                id=f"hyp_disc_{uuid.uuid4().hex[:8]}",
                title=f"Discovery: {component} role in {behavior}",
                statement=f"Component {component} (score: {score:.2f}) contributes to {behavior}",
                target_component=component,
                prediction=f"Ablating {component} will reduce target logit by >0.3",
                falsification_condition=f"Ablating {component} produces Δlogit < 0.1 or equal effect on control",
                state=HypothesisState.PROPOSED,
                metadata={
                    "behavior": behavior,
                    "observation": observation,
                    "discovery_score": score,
                    "source": "circuit_discovery",
                },
            )
            hypotheses.append(hyp)
            self._hypotheses[hyp.id] = hyp
        
        # Generate experiment specifications
        experiment_specs = []
        for hyp in hypotheses:
            spec = self.plan_experiment(
                hypothesis=hyp,
                clean_prompt=discovery_report.get("clean_prompt", ""),
                target_token=discovery_report.get("target_token", ""),
                repeats=10,
            )
            experiment_specs.append({
                "id": spec.id,
                "hypothesis_id": spec.hypothesis_id,
                "name": spec.name,
                "source_component": spec.source_component,
            })
        
        # Prioritize candidates
        priorities = self.prioritize_candidates(
            candidates=[c.get("component", "") for c in candidates],
            evidence_records=discovery_report.get("evidence", []),
        )
        
        return {
            "hypotheses": [
                {
                    "id": h.id,
                    "title": h.title,
                    "statement": h.statement,
                    "state": h.state.value,
                }
                for h in hypotheses
            ],
            "experiment_specs": experiment_specs,
            "priorities": [
                {
                    "component": p.component,
                    "priority_score": p.priority_score,
                    "recommended_action": p.recommended_action,
                }
                for p in priorities
            ],
            "research_question": f"What is the mechanistic basis of {behavior}?",
        }
