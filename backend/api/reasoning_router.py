"""
Reasoning Agent API Router

Endpoints for the Scientific Research Orchestrator.
These endpoints expose hypothesis generation, experiment planning,
falsification testing, and evidence reasoning to the frontend.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

logger = logging.getLogger("MECH.api.reasoning_router")

router = APIRouter(prefix="/api/v1/reasoning", tags=["reasoning"])


# ---------------------------------------------------------------------------
# Request/Response Models
# ---------------------------------------------------------------------------
class GenerateHypothesesRequest(BaseModel):
    behavior: str
    observation: str
    research_question: str
    target_components: List[str]
    max_hypotheses: int = 3


class PlanExperimentRequest(BaseModel):
    hypothesis_id: str
    clean_prompt: str
    corrupted_prompt: Optional[str] = None
    target_token: str = ""
    distractor_token: Optional[str] = None
    repeats: int = 10
    seed: int = 42


class FalsificationTestRequest(BaseModel):
    hypothesis_id: str
    clean_prompt: str
    target_token: str


class UpdateHypothesisStateRequest(BaseModel):
    hypothesis_id: str
    new_state: str
    evidence: Optional[Dict[str, Any]] = None


class PrioritizeCandidatesRequest(BaseModel):
    candidates: List[str]
    evidence_records: List[Dict[str, Any]]


class ReasonEvidenceRequest(BaseModel):
    hypothesis_id: str
    evidence_records: List[Dict[str, Any]]


class StartResearchLoopRequest(BaseModel):
    investigation_id: str
    initial_hypothesis_id: str
    max_iterations: int = 10


class IngestDiscoveryReportRequest(BaseModel):
    report: Dict[str, Any]  # CircuitDiscoveryReport.to_dict()
    investigation_id: str = "auto_discovery"
    auto_generate_hypotheses: bool = True
    auto_plan_experiments: bool = True


class IngestCandidateListRequest(BaseModel):
    candidates: List[str]  # List of component IDs like ["L9H9", "L8H7"]
    clean_prompt: str
    target_token: str
    investigation_id: str = "manual_candidates"
    auto_generate_hypotheses: bool = True


class LLMEvaluateRequest(BaseModel):
    hypothesis_id: str
    experiment_result: Dict[str, Any]


class LLMCritiqueRequest(BaseModel):
    hypothesis_id: str


# ---------------------------------------------------------------------------
# Orchestrator Instance
# ---------------------------------------------------------------------------
_orchestrator = None


def _get_orchestrator():
    global _orchestrator
    if _orchestrator is None:
        from backend.reasoning.research_orchestrator import ScientificResearchOrchestrator
        _orchestrator = ScientificResearchOrchestrator()
    return _orchestrator


# ---------------------------------------------------------------------------
# Hypothesis Generation
# ---------------------------------------------------------------------------
@router.post("/hypotheses/generate")
def generate_hypotheses(payload: GenerateHypothesesRequest) -> Dict[str, Any]:
    """
    Generate testable hypotheses from behavior, observation, and research question.
    
    Returns a list of PROPOSED hypotheses (not yet tested).
    """
    orchestrator = _get_orchestrator()
    
    hypotheses = orchestrator.generate_hypotheses(
        behavior=payload.behavior,
        observation=payload.observation,
        research_question=payload.research_question,
        target_components=payload.target_components,
        max_hypotheses=payload.max_hypotheses,
    )
    
    return {
        "status": "success",
        "hypotheses": [
            {
                "id": h.id,
                "title": h.title,
                "statement": h.statement,
                "target_component": h.target_component,
                "prediction": h.prediction,
                "falsification_condition": h.falsification_condition,
                "state": h.state.value,
                "confidence": h.confidence,
                "metadata": h.metadata,
            }
            for h in hypotheses
        ],
        "count": len(hypotheses),
    }


# ---------------------------------------------------------------------------
# Experiment Planning
# ---------------------------------------------------------------------------
@router.post("/experiments/plan")
def plan_experiment(payload: PlanExperimentRequest) -> Dict[str, Any]:
    """
    Convert a hypothesis into a complete experiment specification.
    
    The plan includes:
    1. Baseline measurement
    2. Intervention on target component
    3. Matched control (same layer, different head)
    4. Random control (different layer)
    5. Repeat N times
    6. Measure logit difference
    7. Compare effect sizes
    """
    orchestrator = _get_orchestrator()
    
    # Get the hypothesis
    hypothesis = orchestrator._hypotheses.get(payload.hypothesis_id)
    if not hypothesis:
        raise HTTPException(status_code=404, detail=f"Hypothesis {payload.hypothesis_id} not found")
    
    spec = orchestrator.plan_experiment(
        hypothesis=hypothesis,
        clean_prompt=payload.clean_prompt,
        corrupted_prompt=payload.corrupted_prompt,
        target_token=payload.target_token,
        distractor_token=payload.distractor_token,
        repeats=payload.repeats,
        seed=payload.seed,
    )
    
    return {
        "status": "success",
        "experiment_spec": {
            "id": spec.id,
            "hypothesis_id": spec.hypothesis_id,
            "name": spec.name,
            "description": spec.description,
            "clean_prompt": spec.clean_prompt,
            "corrupted_prompt": spec.corrupted_prompt,
            "target_token": spec.target_token,
            "distractor_token": spec.distractor_token,
            "source_component": spec.source_component,
            "intervention_type": spec.intervention_type,
            "control_components": spec.control_components,
            "repeats": spec.repeats,
            "seed": spec.seed,
            "metrics": spec.metrics,
        },
    }


# ---------------------------------------------------------------------------
# Falsification Testing
# ---------------------------------------------------------------------------
@router.post("/hypotheses/falsification")
def design_falsification_test(payload: FalsificationTestRequest) -> Dict[str, Any]:
    """
    Design a falsification test for a hypothesis.
    
    Instead of asking "How can I prove this?", we ask:
    "What experiment could disprove this?"
    """
    orchestrator = _get_orchestrator()
    
    hypothesis = orchestrator._hypotheses.get(payload.hypothesis_id)
    if not hypothesis:
        raise HTTPException(status_code=404, detail=f"Hypothesis {payload.hypothesis_id} not found")
    
    spec = orchestrator.design_falsification_test(
        hypothesis=hypothesis,
        clean_prompt=payload.clean_prompt,
        target_token=payload.target_token,
    )
    
    return {
        "status": "success",
        "falsification_spec": {
            "id": spec.id,
            "hypothesis_id": spec.hypothesis_id,
            "name": spec.name,
            "description": spec.description,
            "clean_prompt": spec.clean_prompt,
            "source_component": spec.source_component,
            "control_components": spec.control_components,
            "repeats": spec.repeats,
            "metrics": spec.metrics,
        },
    }


# ---------------------------------------------------------------------------
# Hypothesis Lifecycle
# ---------------------------------------------------------------------------
@router.put("/hypotheses/state")
def update_hypothesis_state(payload: UpdateHypothesisStateRequest) -> Dict[str, Any]:
    """
    Update the state of a hypothesis based on evidence.
    
    Lifecycle: PROPOSED → PLANNED → TESTING → REPLICATION → EVALUATION → SUPPORTED/REFUTED/INSUFFICIENT
    """
    orchestrator = _get_orchestrator()
    
    from backend.reasoning.research_orchestrator import HypothesisState
    
    try:
        new_state = HypothesisState(payload.new_state)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid state: {payload.new_state}")
    
    try:
        hypothesis = orchestrator.update_hypothesis_state(
            hypothesis_id=payload.hypothesis_id,
            new_state=new_state,
            evidence=payload.evidence,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    return {
        "status": "success",
        "hypothesis": {
            "id": hypothesis.id,
            "title": hypothesis.title,
            "state": hypothesis.state.value,
            "confidence": hypothesis.confidence,
            "evidence_count_supporting": hypothesis.evidence_count_supporting,
            "evidence_count_contradicting": hypothesis.evidence_count_contradicting,
        },
    }


# ---------------------------------------------------------------------------
# Candidate Prioritization
# ---------------------------------------------------------------------------
@router.post("/candidates/prioritize")
def prioritize_candidates(payload: PrioritizeCandidatesRequest) -> Dict[str, Any]:
    """
    Prioritize candidate components based on existing evidence.
    """
    orchestrator = _get_orchestrator()
    
    priorities = orchestrator.prioritize_candidates(
        candidates=payload.candidates,
        evidence_records=payload.evidence_records,
    )
    
    return {
        "status": "success",
        "priorities": [
            {
                "component": p.component,
                "priority_score": p.priority_score,
                "reason": p.reason,
                "evidence_level": p.evidence_level,
                "recommended_action": p.recommended_action,
            }
            for p in priorities
        ],
    }


# ---------------------------------------------------------------------------
# Evidence Reasoning
# ---------------------------------------------------------------------------
@router.post("/evidence/reason")
def reason_about_evidence(payload: ReasonEvidenceRequest) -> Dict[str, Any]:
    """
    Organize evidence into a chain and explain support/contradiction.
    """
    orchestrator = _get_orchestrator()
    
    try:
        chain = orchestrator.reason_about_evidence(
            hypothesis_id=payload.hypothesis_id,
            evidence_records=payload.evidence_records,
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    
    return {
        "status": "success",
        "evidence_chain": {
            "hypothesis_id": chain.hypothesis_id,
            "supporting_count": len(chain.supporting_evidence),
            "contradicting_count": len(chain.contradicting_evidence),
            "total_effect_size": chain.total_effect_size,
            "specificity_ratio": chain.specificity_ratio,
            "replication_count": chain.replication_count,
            "overall_assessment": chain.overall_assessment,
        },
    }


# ---------------------------------------------------------------------------
# Research Loop
# ---------------------------------------------------------------------------
@router.post("/research-loop/start")
def start_research_loop(payload: StartResearchLoopRequest) -> Dict[str, Any]:
    """
    Start a research loop for iterative experimentation.
    """
    orchestrator = _get_orchestrator()
    
    loop = orchestrator.start_research_loop(
        investigation_id=payload.investigation_id,
        initial_hypothesis_id=payload.initial_hypothesis_id,
        max_iterations=payload.max_iterations,
    )
    
    return {
        "status": "success",
        "loop": {
            "investigation_id": loop.investigation_id,
            "current_hypothesis_id": loop.current_hypothesis_id,
            "iteration": loop.iteration,
            "max_iterations": loop.max_iterations,
            "status": loop.status,
        },
    }


# ---------------------------------------------------------------------------
# Get Hypotheses
# ---------------------------------------------------------------------------
@router.get("/hypotheses")
def list_hypotheses() -> Dict[str, Any]:
    """List all hypotheses managed by the orchestrator."""
    orchestrator = _get_orchestrator()
    
    hypotheses = list(orchestrator._hypotheses.values())
    
    return {
        "hypotheses": [
            {
                "id": h.id,
                "title": h.title,
                "statement": h.statement,
                "target_component": h.target_component,
                "state": h.state.value,
                "confidence": h.confidence,
                "evidence_count_supporting": h.evidence_count_supporting,
                "evidence_count_contradicting": h.evidence_count_contradicting,
            }
            for h in hypotheses
        ],
        "count": len(hypotheses),
    }


# ---------------------------------------------------------------------------
# Discovery Bridge: Agent 1 → Agent 2
# ---------------------------------------------------------------------------
@router.post("/discoveries/ingest-report")
def ingest_discovery_report(payload: IngestDiscoveryReportRequest) -> Dict[str, Any]:
    """
    Ingests a CircuitDiscoveryReport from Agent 1 and generates
    hypotheses + experiment specifications for Agent 2.
    
    This is the bridge that connects Agent 1's discovery outputs
    to Agent 2's research pipeline.
    """
    from backend.reasoning.discovery_bridge import DiscoveryBridge
    
    bridge = DiscoveryBridge(orchestrator=_get_orchestrator())
    result = bridge.ingest_discovery_report(
        report=payload.report,
        investigation_id=payload.investigation_id,
        auto_generate_hypotheses=payload.auto_generate_hypotheses,
        auto_plan_experiments=payload.auto_plan_experiments,
    )
    
    return {"status": "success", **result}


@router.post("/discoveries/ingest-candidates")
def ingest_candidate_list(payload: IngestCandidateListRequest) -> Dict[str, Any]:
    """
    Ingests a list of candidate component IDs from Agent 1
    and generates hypotheses for investigation.
    """
    from backend.reasoning.discovery_bridge import DiscoveryBridge
    
    bridge = DiscoveryBridge(orchestrator=_get_orchestrator())
    result = bridge.ingest_candidate_list(
        candidates=payload.candidates,
        clean_prompt=payload.clean_prompt,
        target_token=payload.target_token,
        investigation_id=payload.investigation_id,
        auto_generate_hypotheses=payload.auto_generate_hypotheses,
    )
    
    return {"status": "success", **result}


# ---------------------------------------------------------------------------
# Hypothesis → Experiment Pipeline
# ---------------------------------------------------------------------------
@router.post("/hypotheses/{hyp_id}/to-experiment")
def hypothesis_to_experiment(
    hyp_id: str,
    clean_prompt: str = "",
    target_token: str = "",
    repeats: int = 10,
) -> Dict[str, Any]:
    """
    Automatically converts a hypothesis into a complete experiment specification.
    
    This is the automated pipeline that takes a PROPOSED or PLANNED hypothesis
    and creates the full experiment spec ready for Agent 1 execution.
    """
    orchestrator = _get_orchestrator()
    
    hypothesis = orchestrator._hypotheses.get(hyp_id)
    if not hypothesis:
        raise HTTPException(status_code=404, detail=f"Hypothesis {hyp_id} not found")
    
    # Use hypothesis defaults if not provided
    if not clean_prompt:
        clean_prompt = hypothesis.metadata.get("clean_prompt", "When Mary and John went to the store, John gave a drink to")
    if not target_token:
        target_token = hypothesis.metadata.get("target_token", " Mary")
    
    spec = orchestrator.plan_experiment(
        hypothesis=hypothesis,
        clean_prompt=clean_prompt,
        target_token=target_token,
        repeats=repeats,
    )
    
    return {
        "status": "success",
        "experiment_spec": {
            "id": spec.id,
            "hypothesis_id": spec.hypothesis_id,
            "name": spec.name,
            "description": spec.description,
            "clean_prompt": spec.clean_prompt,
            "corrupted_prompt": spec.corrupted_prompt,
            "target_token": spec.target_token,
            "distractor_token": spec.distractor_token,
            "source_component": spec.source_component,
            "intervention_type": spec.intervention_type,
            "control_components": spec.control_components,
            "repeats": spec.repeats,
            "seed": spec.seed,
            "metrics": spec.metrics,
        },
        "hypothesis_state": hypothesis.state.value,
    }


# ---------------------------------------------------------------------------
# Research Loop Progression (Step + Auto)
# ---------------------------------------------------------------------------
@router.post("/research-loop/step")
def step_research_loop(investigation_id: str) -> Dict[str, Any]:
    """
    Executes one iteration of the research loop:
    1. Select next experiment based on current hypothesis state
    2. Execute the experiment via Agent 1
    3. Analyze results
    4. Update hypothesis state
    5. Return updated loop state
    """
    orchestrator = _get_orchestrator()
    from backend.reasoning.research_orchestrator import HypothesisState
    
    loop = orchestrator._research_loops.get(investigation_id)
    if not loop:
        raise HTTPException(status_code=404, detail=f"No active research loop for investigation {investigation_id}")
    
    if loop.status != "RUNNING":
        return {"status": "loop_not_running", "loop_status": loop.status}
    
    if loop.iteration >= loop.max_iterations:
        loop.status = "COMPLETED"
        return {"status": "loop_completed", "iteration": loop.iteration}
    
    # Get current hypothesis
    hypothesis = orchestrator._hypotheses.get(loop.current_hypothesis_id)
    if not hypothesis:
        loop.status = "ERROR"
        return {"status": "error", "message": f"Hypothesis {loop.current_hypothesis_id} not found"}
    
    # Select next experiment
    from backend.storage.database import DesktopStorage
    storage = DesktopStorage()
    storage.initialize()
    available_experiments = storage.list_experiments()
    
    selected_experiment = orchestrator.select_next_experiment(
        investigation_id=investigation_id,
        available_experiments=available_experiments,
    )
    
    if not selected_experiment:
        loop.status = "COMPLETED"
        return {"status": "no_experiments_available", "iteration": loop.iteration}
    
    # Transition hypothesis to TESTING
    try:
        if hypothesis.state == HypothesisState.PLANNED:
            orchestrator.update_hypothesis_state(
                hypothesis_id=hypothesis.id,
                new_state=HypothesisState.TESTING,
            )
    except ValueError:
        pass  # Already in TESTING or later state
    
    # Execute experiment via Agent 1
    from backend.services import gpt2_engine
    experiment_result = None
    
    if gpt2_engine.is_available():
        try:
            from backend.science.experiment_runner import ScientificExperimentRunner
            from backend.storage.scientific_entities import Experiment
            
            runner = ScientificExperimentRunner(storage=storage)
            exp = Experiment(**selected_experiment)
            run = runner.run_experiment(exp)
            experiment_result = run.model_dump()
        except Exception as e:
            logger.exception("Experiment execution failed in research loop: %s", e)
    
    # Increment iteration
    loop.iteration += 1
    loop.completed_experiments.append(selected_experiment.get("id", f"exp_{loop.iteration}"))
    
    # Evaluate hypothesis if we have results
    if experiment_result:
        try:
            from backend.science.hypothesis_engine import HypothesisEngine
            engine = HypothesisEngine(storage=storage)
            evaluation = engine.evaluate_hypothesis(
                hypothesis_id=hypothesis.id,
                investigation_id=investigation_id,
            )
            
            # Update hypothesis state based on evaluation
            status = evaluation.get("status", "UNTESTED")
            if status == "SUPPORTED":
                try:
                    orchestrator.update_hypothesis_state(
                        hypothesis_id=hypothesis.id,
                        new_state=HypothesisState.EVALUATION,
                    )
                    orchestrator.update_hypothesis_state(
                        hypothesis_id=hypothesis.id,
                        new_state=HypothesisState.SUPPORTED,
                    )
                except ValueError:
                    pass
            elif status in ("CONTRADICTED", "FALSIFIED"):
                try:
                    orchestrator.update_hypothesis_state(
                        hypothesis_id=hypothesis.id,
                        new_state=HypothesisState.EVALUATION,
                    )
                    orchestrator.update_hypothesis_state(
                        hypothesis_id=hypothesis.id,
                        new_state=HypothesisState.REFUTED,
                    )
                except ValueError:
                    pass
        except Exception as e:
            logger.warning("Hypothesis evaluation failed: %s", e)
    
    return {
        "status": "step_completed",
        "iteration": loop.iteration,
        "max_iterations": loop.max_iterations,
        "hypothesis": {
            "id": hypothesis.id,
            "title": hypothesis.title,
            "state": hypothesis.state.value,
            "confidence": hypothesis.confidence,
        },
        "experiment_executed": experiment_result is not None,
        "loop_status": loop.status,
    }


@router.post("/research-loop/auto")
def auto_run_research_loop(
    investigation_id: str,
    max_iterations: int = 5,
    clean_prompt: str = "When Mary and John went to the store, John gave a drink to",
    target_token: str = " Mary",
) -> Dict[str, Any]:
    """
    Runs the research loop automatically for N iterations.
    
    Each iteration:
    1. Selects next experiment
    2. Executes via Agent 1
    3. Analyzes results
    4. Updates hypothesis state
    5. Designs next experiment if needed
    """
    orchestrator = _get_orchestrator()
    
    # Start loop if not already running
    if investigation_id not in orchestrator._research_loops:
        # Find or create a hypothesis
        hypotheses = list(orchestrator._hypotheses.values())
        if not hypotheses:
            # Create a default hypothesis
            hyp = Hypothesis(
                id=f"hyp_{uuid.uuid4().hex[:8]}",
                title="Default Investigation Hypothesis",
                statement="Investigating model behavior on the provided prompt.",
                target_component="L9H9",
                prediction="Intervening on L9H9 will affect target logit.",
                falsification_condition="No significant effect observed.",
                state=HypothesisState.PROPOSED,
                metadata={
                    "clean_prompt": clean_prompt,
                    "target_token": target_token,
                },
            )
            orchestrator._hypotheses[hyp.id] = hyp
            initial_hyp_id = hyp.id
        else:
            initial_hyp_id = hypotheses[0].id
        
        loop = orchestrator.start_research_loop(
            investigation_id=investigation_id,
            initial_hypothesis_id=initial_hyp_id,
            max_iterations=max_iterations,
        )
    else:
        loop = orchestrator._research_loops[investigation_id]
        loop.max_iterations = max_iterations
    
    # Run iterations
    results = []
    for i in range(max_iterations):
        step_result = step_research_loop(investigation_id)
        results.append(step_result)
        
        if step_result.get("loop_status") != "RUNNING":
            break
    
    return {
        "status": "auto_loop_completed",
        "iterations_run": len(results),
        "results": results,
        "final_state": {
            "investigation_id": investigation_id,
            "iteration": loop.iteration,
            "status": loop.status,
        },
    }


# ---------------------------------------------------------------------------
# LLM-Powered Evaluation & Critique
# ---------------------------------------------------------------------------
@router.post("/hypotheses/llm-evaluate")
def llm_evaluate_hypothesis(payload: LLMEvaluateRequest) -> Dict[str, Any]:
    """
    Use LLM to evaluate whether experiment results support or falsify a hypothesis.
    
    Provides natural language interpretation of quantitative results.
    """
    orchestrator = _get_orchestrator()
    
    try:
        evaluation = orchestrator.evaluate_with_llm(
            hypothesis_id=payload.hypothesis_id,
            experiment_result=payload.experiment_result,
        )
        return {"status": "success", "evaluation": evaluation}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.post("/hypotheses/llm-critique")
def llm_critique_hypothesis(payload: LLMCritiqueRequest) -> Dict[str, Any]:
    """
    Use LLM to generate a skeptic's critique of a hypothesis.
    
    Identifies weaknesses, confounding variables, and alternative explanations.
    """
    orchestrator = _get_orchestrator()
    
    try:
        critique = orchestrator.critique_hypothesis(hypothesis_id=payload.hypothesis_id)
        return {"status": "success", "critique": critique}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.post("/hypotheses/{hyp_id}/full-evaluation")
def full_hypothesis_evaluation(hyp_id: str) -> Dict[str, Any]:
    """
    Complete evaluation pipeline for a hypothesis:
    1. Run evidence-based evaluation (HypothesisEngine)
    2. Run LLM-based evaluation (if available)
    3. Run skeptic's critique (if available)
    4. Synthesize findings
    
    Returns comprehensive evaluation with all perspectives.
    """
    orchestrator = _get_orchestrator()
    
    hypothesis = orchestrator._hypotheses.get(hyp_id)
    if not hypothesis:
        raise HTTPException(status_code=404, detail=f"Hypothesis {hyp_id} not found")
    
    result = {
        "hypothesis_id": hyp_id,
        "title": hypothesis.title,
        "statement": hypothesis.statement,
        "state": hypothesis.state.value,
        "confidence": hypothesis.confidence,
        "evidence_count_supporting": hypothesis.evidence_count_supporting,
        "evidence_count_contradicting": hypothesis.evidence_count_contradicting,
        "evaluations": {},
    }
    
    # Evidence-based evaluation
    from backend.storage.database import DesktopStorage
    storage = DesktopStorage()
    storage.initialize()
    
    investigation_id = hypothesis.metadata.get("investigation_id", "default")
    try:
        from backend.science.hypothesis_engine import HypothesisEngine
        engine = HypothesisEngine(storage=storage)
        evidence_eval = engine.evaluate_hypothesis(hyp_id, investigation_id)
        result["evaluations"]["evidence"] = evidence_eval
    except Exception as e:
        result["evaluations"]["evidence"] = {"error": str(e)}
    
    # LLM-based evaluation
    try:
        llm_eval = orchestrator.evaluate_with_llm(
            hypothesis_id=hyp_id,
            experiment_result={"delta_logit": 0.0, "specificity_ratio": 0.0},  # Placeholder
        )
        result["evaluations"]["llm"] = llm_eval
    except Exception as e:
        result["evaluations"]["llm"] = {"error": str(e)}
    
    # Critique
    try:
        critique = orchestrator.critique_hypothesis(hyp_id)
        result["evaluations"]["critique"] = critique
    except Exception as e:
        result["evaluations"]["critique"] = {"error": str(e)}
    
    # Synthesize
    evaluations = result["evaluations"]
    if "evidence" in evaluations and evaluations["evidence"].get("status") == "SUPPORTED":
        result["synthesis"] = "Evidence-based evaluation supports this hypothesis."
    elif "llm" in evaluations and evaluations["llm"].get("type") == "supportive":
        result["synthesis"] = "LLM evaluation supports this hypothesis."
    elif "critique" in evaluations and evaluations["critique"].get("critique"):
        result["synthesis"] = "Skeptic has raised concerns. Consider designing falsification tests."
    else:
        result["synthesis"] = "Insufficient evidence for definitive evaluation."
    
    return {"status": "success", **result}
