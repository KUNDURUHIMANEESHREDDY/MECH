"""
Discovery Bridge: Agent 1 Discovery Outputs → Agent 2 Research Inputs

This module bridges the gap between Agent 1's mechanistic discovery engine
and Agent 2's research orchestration pipeline.

When Agent 1 discovers candidates (head scan, neuron scan, layer scan),
this bridge:
1. Converts discovered candidates into testable hypotheses
2. Generates experiment specifications from those hypotheses
3. Prioritizes candidates for investigation
4. Feeds everything into the research loop

IMPORTANT: This module NEVER fabricates evidence. It only structures
Agent 1's real discoveries into Agent 2's research format.
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import Any, Dict, List, Optional

from backend.reasoning.research_orchestrator import (
    ExperimentSpecification,
    Hypothesis,
    HypothesisState,
    ScientificResearchOrchestrator,
)

logger = logging.getLogger("MECH.reasoning.discovery_bridge")


class DiscoveryBridge:
    """
    Bridges Agent 1's discovery outputs into Agent 2's research pipeline.
    
    Takes a CircuitDiscoveryReport (or list of candidates) from Agent 1
    and converts them into hypotheses and experiment specifications
    that Agent 2 can reason about and execute.
    """
    
    def __init__(self, orchestrator: Optional[ScientificResearchOrchestrator] = None):
        self.orchestrator = orchestrator or ScientificResearchOrchestrator()
    
    def ingest_discovery_report(
        self,
        report: Dict[str, Any],
        investigation_id: str = "auto_discovery",
        auto_generate_hypotheses: bool = True,
        auto_plan_experiments: bool = True,
    ) -> Dict[str, Any]:
        """
        Ingests a CircuitDiscoveryReport from Agent 1 and creates
        hypotheses + experiment specifications for Agent 2.
        
        Args:
            report: CircuitDiscoveryReport.to_dict() from Agent 1
            investigation_id: Investigation to associate discoveries with
            auto_generate_hypotheses: Whether to auto-create hypotheses from candidates
            auto_plan_experiments: Whether to auto-create experiment specs from hypotheses
            
        Returns:
            Summary of ingested discoveries, generated hypotheses, and planned experiments
        """
        candidates = report.get("top_candidates", [])
        validated_circuit = report.get("validated_circuit", [])
        model_id = report.get("model_id", "unknown")
        clean_prompt = report.get("clean_prompt", "")
        target_token = report.get("target_token", "")
        
        result = {
            "investigation_id": investigation_id,
            "model_id": model_id,
            "candidates_ingested": len(candidates),
            "validated_components": len(validated_circuit),
            "hypotheses_generated": [],
            "experiments_planned": [],
            "priorities": [],
        }
        
        # Extract component IDs from candidates
        candidate_components = [c.get("component_id", "") for c in candidates if c.get("component_id")]
        
        # Prioritize candidates based on evidence
        evidence_records = []
        for vc in validated_circuit:
            evidence_records.append({
                "target_component": vc.get("component_id", ""),
                "supports_hypothesis": vc.get("evidence_tier") in ("CAUSALLY_VERIFIED", "SUPPORTED", "WEAKLY_SUPPORTED"),
                "metric_value": abs(vc.get("causal_effect_ci", [0, 0])[1] - vc.get("causal_effect_ci", [0, 0])[0]),
                "control_value": 0.0,
            })
        
        if candidate_components:
            priorities = self.orchestrator.prioritize_candidates(
                candidates=candidate_components,
                evidence_records=evidence_records,
            )
            result["priorities"] = [
                {
                    "component": p.component,
                    "priority_score": p.priority_score,
                    "evidence_level": p.evidence_level,
                    "recommended_action": p.recommended_action,
                    "reason": p.reason,
                }
                for p in priorities
            ]
        
        # Generate hypotheses from top candidates
        if auto_generate_hypotheses:
            hypotheses = self._generate_hypotheses_from_candidates(
                candidates=candidates,
                validated_circuit=validated_circuit,
                investigation_id=investigation_id,
                clean_prompt=clean_prompt,
                target_token=target_token,
            )
            result["hypotheses_generated"] = [
                {
                    "id": h.id,
                    "title": h.title,
                    "statement": h.statement,
                    "target_component": h.target_component,
                    "state": h.state.value,
                }
                for h in hypotheses
            ]
            
            # Plan experiments from hypotheses
            if auto_plan_experiments and hypotheses:
                for hyp in hypotheses[:3]:  # Top 3 hypotheses
                    spec = self.orchestrator.plan_experiment(
                        hypothesis=hyp,
                        clean_prompt=clean_prompt,
                        target_token=target_token,
                        repeats=10,
                    )
                    result["experiments_planned"].append({
                        "id": spec.id,
                        "hypothesis_id": spec.hypothesis_id,
                        "name": spec.name,
                        "source_component": spec.source_component,
                        "intervention_type": spec.intervention_type,
                        "control_components": spec.control_components,
                        "repeats": spec.repeats,
                    })
        
        logger.info(
            "Discovery bridge: ingested %d candidates, generated %d hypotheses, planned %d experiments",
            len(candidates),
            len(result["hypotheses_generated"]),
            len(result["experiments_planned"]),
        )
        
        return result
    
    def _generate_hypotheses_from_candidates(
        self,
        candidates: List[Dict[str, Any]],
        validated_circuit: List[Dict[str, Any]],
        investigation_id: str,
        clean_prompt: str,
        target_token: str,
    ) -> List[Hypothesis]:
        """
        Generates testable hypotheses from discovered candidates.
        
        Each candidate becomes a hypothesis with:
        - A specific prediction about what will happen when intervened
        - A falsification condition
        - Links to supporting evidence from validation
        """
        hypotheses = []
        
        for candidate in candidates[:5]:  # Top 5 candidates
            comp_id = candidate.get("component_id", "")
            comp_type = candidate.get("component_type", "unknown")
            functional_role = candidate.get("functional_role", "Unknown role")
            delta_logit = candidate.get("delta_logit", 0.0)
            indirect_effect = candidate.get("indirect_effect", 0.0)
            validation_status = candidate.get("validation_status", "UNTESTED")
            
            # Generate prediction based on candidate type and role
            if comp_type == "attention_head":
                if "Name Mover" in functional_role:
                    prediction = (
                        f"Zero-ablating {comp_id} will reduce target token logit by >0.5, "
                        f"as this head causally retrieves the indirect object."
                    )
                    falsification = (
                        f"Zero-ablation of {comp_id} produces Δlogit < 0.3 or "
                        f"no significant effect compared to random control head."
                    )
                elif "Induction" in functional_role:
                    prediction = (
                        f"Zero-ablating {comp_id} will reduce target logit by >0.3, "
                        f"as this head routes information for the target behavior."
                    )
                    falsification = (
                        f"Zero-ablation of {comp_id} produces Δlogit < 0.2 or "
                        f"equal effect on matched-norm control."
                    )
                else:
                    prediction = (
                        f"Zero-ablating {comp_id} will reduce target logit by >0.2, "
                        f"as this head contributes to the observed behavior."
                    )
                    falsification = (
                        f"Zero-ablation of {comp_id} produces Δlogit < 0.1 or "
                        f"effect is within control noise floor."
                    )
            elif comp_type == "neuron":
                prediction = (
                    f"Zero-ablating neuron {comp_id} will reduce target logit by >0.2, "
                    f"as this neuron has high DLA score ({delta_logit:.2f})."
                )
                falsification = (
                    f"Zero-ablation of {comp_id} produces Δlogit < 0.1 or "
                    f"effect is within control noise floor."
                )
            else:
                prediction = (
                    f"Intervening on {comp_id} will reduce target logit by >0.2."
                )
                falsification = (
                    f"Intervention on {comp_id} produces Δlogit < 0.1."
                )
            
            # Create hypothesis
            hyp = Hypothesis(
                id=f"hyp_{uuid.uuid4().hex[:8]}",
                title=f"Hypothesis: {functional_role} via {comp_id}",
                statement=(
                    f"Component {comp_id} ({comp_type}, {functional_role}) "
                    f"causally contributes to the observed behavior. "
                    f"Observed Δlogit = {delta_logit:.2f}, indirect effect = {indirect_effect:.2f}."
                ),
                target_component=comp_id,
                prediction=prediction,
                falsification_condition=falsification,
                state=HypothesisState.PROPOSED,
                confidence=0.5 if validation_status == "UNTESTED" else 0.7,
                metadata={
                    "component_type": comp_type,
                    "functional_role": functional_role,
                    "delta_logit": delta_logit,
                    "indirect_effect": indirect_effect,
                    "validation_status": validation_status,
                    "investigation_id": investigation_id,
                },
            )
            
            # Add evidence counts from validation
            if validation_status in ("CAUSALLY_VERIFIED", "SUPPORTED"):
                hyp.evidence_count_supporting = 1
            elif validation_status == "REFUTED":
                hyp.evidence_count_contradicting = 1
            
            hypotheses.append(hyp)
            self.orchestrator._hypotheses[hyp.id] = hyp
        
        return hypotheses
    
    def ingest_candidate_list(
        self,
        candidates: List[str],
        clean_prompt: str,
        target_token: str,
        investigation_id: str = "manual_candidates",
        auto_generate_hypotheses: bool = True,
    ) -> Dict[str, Any]:
        """
        Ingests a simple list of candidate component IDs and generates hypotheses.
        
        This is the lightweight version for when the user manually specifies
        candidates to investigate (e.g., from the UI).
        """
        result = {
            "investigation_id": investigation_id,
            "candidates_ingested": len(candidates),
            "hypotheses_generated": [],
            "experiments_planned": [],
        }
        
        # Create minimal candidate dicts
        candidate_dicts = []
        for comp_id in candidates:
            candidate_dicts.append({
                "component_id": comp_id,
                "component_type": "attention_head" if "H" in comp_id else "neuron",
                "functional_role": "Unknown (manual candidate)",
                "delta_logit": 0.0,
                "indirect_effect": 0.0,
                "validation_status": "UNTESTED",
            })
        
        if auto_generate_hypotheses:
            hypotheses = self._generate_hypotheses_from_candidates(
                candidates=candidate_dicts,
                validated_circuit=[],
                investigation_id=investigation_id,
                clean_prompt=clean_prompt,
                target_token=target_token,
            )
            result["hypotheses_generated"] = [
                {
                    "id": h.id,
                    "title": h.title,
                    "statement": h.statement,
                    "target_component": h.target_component,
                    "state": h.state.value,
                }
                for h in hypotheses
            ]
            
            # Plan experiments
            for hyp in hypotheses[:3]:
                spec = self.orchestrator.plan_experiment(
                    hypothesis=hyp,
                    clean_prompt=clean_prompt,
                    target_token=target_token,
                    repeats=10,
                )
                result["experiments_planned"].append({
                    "id": spec.id,
                    "hypothesis_id": spec.hypothesis_id,
                    "name": spec.name,
                    "source_component": spec.source_component,
                    "intervention_type": spec.intervention_type,
                    "control_components": spec.control_components,
                    "repeats": spec.repeats,
                })
        
        return result
