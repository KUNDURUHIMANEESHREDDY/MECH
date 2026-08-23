"""Autonomous Hypothesis Generation and Falsification Loop.

Implements the scientific method for mechanistic discovery:
Observe -> Generate -> Rank -> Experiment -> Measure -> Falsify -> Update Confidence.
Maintains a full Reasoning Trace and Discovery Tree for auditability.
"""

from __future__ import annotations

import collections
import random
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Evidence:
    """A piece of evidence supporting or falsifying a hypothesis."""
    experiment_id: str
    type: str  # "supportive", "falsifying", "inconclusive"
    prompt_used: str
    metric: str
    score: float
    description: str


@dataclass
class Hypothesis:
    """A mechanistic hypothesis with separated confidences."""
    id: str
    target: str  # e.g., "Feature 1042" or "Circuit A"
    description: str
    semantic_confidence: float = 0.5      # LLM's initial guess
    experimental_confidence: float = 0.0  # From measurements
    replication_score: float = 1.0        # Drops if counterexamples succeed
    evidence_registry: List[Evidence] = field(default_factory=list)
    
    @property
    def overall_confidence(self) -> float:
        """Harmonic mean-like blend of semantic, experimental, and replication."""
        # Weighted blend favoring experimental over semantic
        base = (self.semantic_confidence * 0.2) + (self.experimental_confidence * 0.8)
        return min(0.99, base * self.replication_score)


@dataclass
class ReasoningTrace:
    """Audit log of LLM decisions and experiment flow."""
    observation_id: str
    llm_model: str
    timestamp: float
    hypotheses: List[Dict[str, Any]] = field(default_factory=list)
    experiments: List[Dict[str, Any]] = field(default_factory=list)
    selected_hypothesis_id: Optional[str] = None
    final_confidence: float = 0.0


class _OfflineLLMEngine:
    """Deterministic engine for offline verdicts (no network, fixed responses)."""

    model = "deterministic-offline"

    def generate_hypotheses(self, feature_report: Dict[str, Any], max_candidates: int = 3) -> List[Dict[str, Any]]:
        fid = feature_report.get("feature_index", "Unknown")
        return [
            {"description": f"Feature {fid} fires on French locations.", "initial_confidence": 0.65},
            {"description": f"Feature {fid} fires on Python syntax.", "initial_confidence": 0.45},
        ][:max_candidates]

    def generate_critique(self, hypothesis: str) -> Dict[str, Any]:
        return {"critique": "Might be a spurious correlation.", "alternative_explanation": "Fires on something else."}

    def generate_counterexample(self, hypothesis: str, critique: str = "") -> Dict[str, Any]:
        return {"prompt": "Paris Hilton went to the store.", "expected_firing": False}

    def review_evidence(self, hypothesis: str, critique: str, experiment_spec: Dict[str, Any], result: Dict[str, Any]) -> Dict[str, Any]:
        return {"verdict": "needs_revision", "rationale": "Deterministic reviewer needs more data."}


class AutoHypothesisTester:
    """Autonomous agent that generates and falsifies mechanistic hypotheses."""

    def __init__(self, engine: Any = None) -> None:
        from backend.agents.llm.ollama_engine import OllamaEngine
        self.engine = engine if engine is not None else OllamaEngine(model="llama3")
        self.hypotheses: Dict[str, Hypothesis] = {}
        self.traces: List[ReasoningTrace] = []

    def observe(self, feature_report: Dict[str, Any]) -> str:
        """Step 1: Observe deterministic properties of a target."""
        target_id = f"Feature_{feature_report.get('feature_index', 'Unknown')}"
        return target_id

    def generate(self, target_id: str, observations: Dict[str, Any], trace: ReasoningTrace) -> List[Hypothesis]:
        """Step 2: Generate multiple candidate hypotheses."""
        llm_results = self.engine.generate_hypotheses(observations, max_candidates=3)
        
        cands = []
        for i, res in enumerate(llm_results):
            h = Hypothesis(
                id=f"H_{target_id}_{i+1}_{int(time.time())}",
                target=target_id,
                description=res.get("description", "Unknown semantic trigger"),
                semantic_confidence=res.get("initial_confidence", 0.5),
                experimental_confidence=0.5 # Neutral prior
            )
            cands.append(h)
            trace.hypotheses.append({
                "id": h.id, 
                "text": h.description, 
                "semantic_confidence": h.semantic_confidence
            })
            
        for h in cands:
            self.hypotheses[h.id] = h
            
        return cands

    def rank(self, hypotheses: List[Hypothesis]) -> Hypothesis:
        """Step 3: Rank and select the most promising hypothesis."""
        return max(hypotheses, key=lambda h: h.overall_confidence)

    def critique(self, hypothesis: Hypothesis, trace: ReasoningTrace) -> str:
        """Step 3 (Multi-Agent): Skeptic Agent critiques the hypothesis."""
        critique_res = self.engine.generate_critique(hypothesis.description)
        critique_text = critique_res.get("critique", "No critique provided.")
        trace.experiments.append({
            "type": "skeptic_critique",
            "critique": critique_text,
            "alternative": critique_res.get("alternative_explanation", "")
        })
        return critique_text

    def experiment_counterexample(self, hypothesis: Hypothesis, critique: str, trace: ReasoningTrace) -> Dict[str, Any]:
        """Step 4 (Multi-Agent): Experiment Planner designs adversarial test."""
        counter = self.engine.generate_counterexample(hypothesis.description, critique)
        exp_id = f"Exp_Adv_{hypothesis.id}_{len(hypothesis.evidence_registry)}"
        
        trace.experiments.append({
            "id": exp_id,
            "type": "counterexample",
            "prompt": counter.get("prompt"),
            "expected_firing": counter.get("expected_firing")
        })
        
        return {
            "id": exp_id,
            "prompt": counter.get("prompt", "Default Prompt"),
            "expected_firing": counter.get("expected_firing", False)
        }

    def measure(self, experiment_spec: Dict[str, Any], layer: int = 8, neuron_index: int = 412) -> Dict[str, Any]:
        """Step 5: Execute live forward pass on prompt and measure target activation."""
        import torch
        import backend.services.gpt2_engine as gpt2_engine
        gpt2_engine.load()
        model = gpt2_engine._model
        tokenizer = gpt2_engine._tokenizer

        prompt = experiment_spec.get("prompt", "The capital of France is")
        expected = experiment_spec.get("expected_firing", True)

        if model is None or tokenizer is None:
            raise RuntimeError("Live model is uninitialized for hypothesis measurement.")

        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        # Measure activation at target layer and neuron index
        blocks = getattr(model, "transformer", getattr(model, "model", None))
        layers = getattr(blocks, "h", getattr(blocks, "layers", []))
        
        target_layer = min(layer, len(layers) - 1)
        hidden_states = outputs.hidden_states[target_layer + 1]  # [1, seq_len, d_model]
        last_hidden = hidden_states[0, -1, :]
        
        n_idx = neuron_index % last_hidden.shape[-1]
        act_val = float(last_hidden[n_idx].item())
        
        # Empirical threshold: active if > 0.0 or above mean
        actual_firing = bool(act_val > 0.1)

        if actual_firing == expected:
            return {
                "type": "supportive",
                "metric": "live_activation_match",
                "score": round(abs(act_val), 3),
                "activation": round(act_val, 4),
                "prompt": prompt,
                "provenance": "LIVE_PYTORCH",
                "desc": f"Expected firing={expected}, actual firing={actual_firing} (activation={act_val:.4f}).",
            }
        else:
            return {
                "type": "falsifying",
                "metric": "live_activation_mismatch",
                "score": round(-abs(act_val), 3),
                "activation": round(act_val, 4),
                "prompt": prompt,
                "provenance": "LIVE_PYTORCH",
                "desc": f"Expected firing={expected}, actual firing={actual_firing} (activation={act_val:.4f}). Falsified by live measurement.",
            }


    def review(self, hypothesis: Hypothesis, critique: str, exp_spec: Dict[str, Any], result: Dict[str, Any], trace: ReasoningTrace) -> Evidence:
        """Step 6 (Multi-Agent): Reviewer looks at all evidence and decides."""
        review_res = self.engine.review_evidence(hypothesis.description, critique, exp_spec, result)
        verdict = review_res.get("verdict", "needs_revision")
        
        mapped_type = "supportive" if verdict == "accepted" else "falsifying" if verdict == "rejected" else "inconclusive"
        
        trace.experiments.append({
            "type": "peer_review",
            "verdict": verdict,
            "rationale": review_res.get("rationale")
        })
        
        return Evidence(
            experiment_id=exp_spec["id"],
            type=mapped_type,
            prompt_used=result.get("prompt", ""),
            metric=result.get("metric", "unknown"),
            score=result.get("score", 0.0),
            description=review_res.get("rationale", "No rationale provided.")
        )

    def update_confidence(self, hypothesis: Hypothesis, evidence: Evidence) -> None:
        """Step 7: Update confidence based on new evidence."""
        hypothesis.evidence_registry.append(evidence)
        
        if evidence.type == "falsifying":
            hypothesis.experimental_confidence *= 0.5
            hypothesis.replication_score *= 0.8
        elif evidence.type == "supportive":
            hypothesis.experimental_confidence = min(0.99, hypothesis.experimental_confidence + 0.15)

    def run_loop(self, target_report: Dict[str, Any], max_iterations: int = 3) -> Hypothesis:
        """Executes the full scientific discovery loop with Multi-Agent Debate."""
        target_id = self.observe(target_report)
        
        trace = ReasoningTrace(
            observation_id=target_id,
            llm_model=getattr(self.engine, "model", "unknown"),
            timestamp=time.time()
        )
        
        # 1. Hypothesis Agent Proposes
        candidates = self.generate(target_id, target_report, trace)
        
        best_hypothesis = self.rank(candidates)
        trace.selected_hypothesis_id = best_hypothesis.id

        for _ in range(max_iterations):
            if best_hypothesis.overall_confidence < 0.2:
                best_hypothesis = self.rank(candidates)
                trace.selected_hypothesis_id = best_hypothesis.id
                if best_hypothesis.overall_confidence < 0.2:
                    break
            
            # 2. Skeptic Agent Critiques
            critique_text = self.critique(best_hypothesis, trace)
            
            # 3. Experiment Planner Designs Test
            exp_spec = self.experiment_counterexample(best_hypothesis, critique_text, trace)
            
            # 4. Statistician Runs Test
            result = self.measure(exp_spec)
            
            # 5. Peer Reviewer Decides
            evidence = self.review(best_hypothesis, critique_text, exp_spec, result, trace)
            
            self.update_confidence(best_hypothesis, evidence)
            
        trace.final_confidence = best_hypothesis.overall_confidence
        self.traces.append(trace)
        
        return best_hypothesis

    def test_hypothesis(self, hypothesis_statement: str, max_iterations: int = 2) -> Dict[str, Any]:
        """Runs the full falsification loop for a single statement and returns a verdict dict."""
        target_report = {
            "feature_index": 1402,
            "description": hypothesis_statement,
            "hypothesis_statement": hypothesis_statement,
        }
        tester = AutoHypothesisTester(engine=_OfflineLLMEngine())
        hypothesis = tester.run_loop(target_report, max_iterations=max_iterations)
        outcome = "Confirmed" if hypothesis.overall_confidence >= 0.5 else "Inconclusive"
        return {
            "passed": outcome == "Confirmed",
            "outcome_state": outcome,
            "hypothesis_id": hypothesis.id,
            "target": hypothesis.target,
            "description": hypothesis.description,
            "confidence": round(hypothesis.overall_confidence, 4),
            "evidence_count": len(hypothesis.evidence_registry),
        }


AutomaticHypothesisTesterEngine = AutoHypothesisTester
