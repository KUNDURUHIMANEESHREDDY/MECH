"""Autonomous Hypothesis Generation and Falsification Loop.

Implements the scientific method for mechanistic discovery:
Observe -> Generate -> Rank -> Experiment -> Measure -> Falsify -> Update Confidence.
Maintains a full Reasoning Trace and Discovery Tree for auditability.
"""

from __future__ import annotations

import collections
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from backend.core.identifiers import content_id


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
    """Fail-closed offline engine when no LLM is configured."""

    model = "deterministic-offline"

    def generate_hypotheses(self, feature_report: Dict[str, Any], max_candidates: int = 3) -> List[Dict[str, Any]]:
        return [{
            "status": "unavailable",
            "provenance": "unavailable",
            "reason": "Offline LLM engine cannot generate hypotheses without network access.",
            "description": feature_report.get("description") or feature_report.get("hypothesis_statement") or "Unspecified hypothesis",
            "initial_confidence": 0.0,
        }]

    def generate_critique(self, hypothesis: str) -> Dict[str, Any]:
        return {
            "status": "unavailable",
            "provenance": "unavailable",
            "reason": "Offline LLM engine cannot generate critiques.",
            "critique": "",
            "alternative_explanation": "",
        }

    def generate_counterexample(self, hypothesis: str, critique: str = "") -> Dict[str, Any]:
        return {
            "status": "unavailable",
            "provenance": "unavailable",
            "reason": "Offline LLM engine cannot generate counterexamples.",
            "prompt": "",
            "expected_firing": False,
        }

    def review_evidence(self, hypothesis: str, critique: str, experiment_spec: Dict[str, Any], result: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "status": "unavailable",
            "provenance": "unavailable",
            "verdict": "inconclusive",
            "rationale": "Offline LLM engine cannot review evidence.",
        }


class AutoHypothesisTester:
    """Autonomous agent that generates and falsifies mechanistic hypotheses."""

    def __init__(self, engine: Any = None) -> None:
        self.engine = engine
        self.hypotheses: Dict[str, Hypothesis] = {}
        self.traces: List[ReasoningTrace] = []

    def observe(self, feature_report: Dict[str, Any]) -> str:
        """Step 1: Observe deterministic properties of a target."""
        target_id = f"Feature_{feature_report.get('feature_index', 'Unknown')}"
        return target_id

    def generate(self, target_id: str, observations: Dict[str, Any], trace: ReasoningTrace) -> List[Hypothesis]:
        """Step 2: Generate candidate hypotheses from observations."""
        if self.engine is None:
            desc = observations.get("description") or observations.get("hypothesis_statement")
            if not desc:
                return []
            h = Hypothesis(
                id=f"H_{target_id}_1_{int(time.time())}",
                target=target_id,
                description=desc,
                semantic_confidence=0.5,
                experimental_confidence=0.0,
            )
            self.hypotheses[h.id] = h
            trace.hypotheses.append({
                "id": h.id,
                "text": h.description,
                "semantic_confidence": h.semantic_confidence,
            })
            return [h]

        llm_results = self.engine.generate_hypotheses(observations, max_candidates=3)
        cands = []
        for i, res in enumerate(llm_results):
            if not isinstance(res, dict) or res.get("status") == "unavailable" or res.get("provenance") == "unavailable":
                continue
            desc = res.get("description")
            if not desc:
                continue
            h = Hypothesis(
                id=f"H_{target_id}_{i+1}_{int(time.time())}",
                target=target_id,
                description=desc,
                semantic_confidence=res.get("initial_confidence", 0.5),
                experimental_confidence=0.5,
            )
            cands.append(h)
            trace.hypotheses.append({
                "id": h.id,
                "text": h.description,
                "semantic_confidence": h.semantic_confidence,
            })

        for h in cands:
            self.hypotheses[h.id] = h

        return cands

    def rank(self, hypotheses: List[Hypothesis]) -> Hypothesis:
        """Step 3: Rank and select the most promising hypothesis."""
        if not hypotheses:
            raise ValueError("No candidate hypotheses available to rank.")
        return max(hypotheses, key=lambda h: h.overall_confidence)

    def critique(self, hypothesis: Hypothesis, trace: ReasoningTrace) -> str:
        """Step 3 (Multi-Agent): Skeptic Agent critiques the hypothesis."""
        if self.engine is None:
            critique_text = "No LLM engine available to formulate skeptic critique."
            trace.experiments.append({
                "type": "skeptic_critique",
                "critique": critique_text,
                "alternative": "",
                "provenance": "unavailable",
            })
            return critique_text

        critique_res = self.engine.generate_critique(hypothesis.description)
        critique_text = critique_res.get("critique", "No critique provided.")
        trace.experiments.append({
            "type": "skeptic_critique",
            "critique": critique_text,
            "alternative": critique_res.get("alternative_explanation", ""),
            # A critique is a semantic judgement, not a measurement of the
            # model, so it is never `live` on the strength of the engine
            # having answered. The engine's own label is used when it gives
            # one; otherwise the answer is a reference, not a measurement.
            "provenance": critique_res.get("provenance") or (
                "unavailable" if critique_res.get("status") == "unavailable"
                else "reference"),
        })
        return critique_text

    def experiment_counterexample(self, hypothesis: Hypothesis, critique: str, trace: ReasoningTrace) -> Dict[str, Any]:
        """Step 4 (Multi-Agent): Experiment Planner designs adversarial test."""
        exp_id = f"Exp_Adv_{hypothesis.id}_{len(hypothesis.evidence_registry)}"
        if self.engine is None:
            prompt = hypothesis.description
            expected = True
        else:
            counter = self.engine.generate_counterexample(hypothesis.description, critique)
            prompt = counter.get("prompt") or hypothesis.description
            expected = counter.get("expected_firing", False)

        trace.experiments.append({
            "id": exp_id,
            "type": "counterexample",
            "prompt": prompt,
            "expected_firing": expected,
        })
        return {
            "id": exp_id,
            "prompt": prompt,
            "expected_firing": expected,
        }

    def measure(self, experiment_spec: Dict[str, Any]) -> Dict[str, Any]:
        """Step 5 (Multi-Agent): Statistician runs real model forward pass and collects data."""
        prompt = experiment_spec.get("prompt", "")
        expected = experiment_spec.get("expected_firing", False)

        try:
            from backend.services import gpt2_engine
            if gpt2_engine.is_available():
                gpt2_engine.load()
                fwd = gpt2_engine.run_prompt(prompt)
                if fwd.get("status") == "ok":
                    next_tok = fwd.get("next_token", "")
                    top5 = fwd.get("top5", [])
                    top_prob = top5[0].get("prob", 0.0) if top5 else 0.0
                    return {
                        "type": "measured",
                        "metric": "next_token_prediction",
                        "score": float(top_prob),
                        "top_token": next_tok,
                        "top5": [t.get("token") for t in top5],
                        "prompt": prompt,
                        "desc": f"Measured model forward pass on {prompt!r} -> predicted {next_tok!r} (prob={top_prob:.4f}).",
                        "provenance": "live",
                        "measured": True,
                    }
        except Exception:
            pass

        return {
            "type": "inconclusive",
            "metric": "activation_measurement_unavailable",
            "score": 0.0,
            "prompt": prompt,
            "desc": f"Expected firing={expected}; model engine unavailable to execute forward pass.",
            "provenance": "unavailable",
            "measured": False,
        }

    def review(self, hypothesis: Hypothesis, critique: str, exp_spec: Dict[str, Any], result: Dict[str, Any], trace: ReasoningTrace) -> Evidence:
        """Step 6 (Multi-Agent): Reviewer looks at measured data and decides."""
        if self.engine is not None:
            review_res = self.engine.review_evidence(hypothesis.description, critique, exp_spec, result)
            verdict = review_res.get("verdict", "needs_revision")
            mapped_type = "supportive" if verdict == "accepted" else "falsifying" if verdict == "rejected" else "inconclusive"
            desc = review_res.get("rationale", "No rationale provided.")
        else:
            if result.get("measured") and result.get("score", 0.0) > 0.05:
                mapped_type = "supportive"
                desc = result.get("desc", "Measurement indicates model activation on prompt.")
                verdict = "accepted"
            elif result.get("measured"):
                mapped_type = "falsifying"
                desc = result.get("desc", "Measurement indicates near-zero probability.")
                verdict = "rejected"
            else:
                mapped_type = "inconclusive"
                desc = "Measurement was unavailable."
                verdict = "needs_revision"

        trace.experiments.append({
            "type": "peer_review",
            "verdict": verdict,
            "rationale": desc,
        })

        return Evidence(
            experiment_id=exp_spec["id"],
            type=mapped_type,
            prompt_used=result.get("prompt", ""),
            metric=result.get("metric", "unknown"),
            score=result.get("score", 0.0),
            description=desc,
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
        """Executes the scientific discovery loop."""
        target_id = self.observe(target_report)
        
        trace = ReasoningTrace(
            observation_id=target_id,
            llm_model=getattr(self.engine, "model", "none"),
            timestamp=time.time(),
        )
        
        candidates = self.generate(target_id, target_report, trace)
        if not candidates:
            h = Hypothesis(
                id=f"H_{target_id}_unavailable",
                target=target_id,
                description=target_report.get("description") or target_report.get("hypothesis_statement") or "No hypothesis available",
                semantic_confidence=0.0,
                experimental_confidence=0.0,
                replication_score=0.0,
            )
            trace.final_confidence = 0.0
            self.traces.append(trace)
            return h

        best_hypothesis = self.rank(candidates)
        trace.selected_hypothesis_id = best_hypothesis.id

        for _ in range(max_iterations):
            critique_text = self.critique(best_hypothesis, trace)
            exp_spec = self.experiment_counterexample(best_hypothesis, critique_text, trace)
            result = self.measure(exp_spec)
            evidence = self.review(best_hypothesis, critique_text, exp_spec, result, trace)
            self.update_confidence(best_hypothesis, evidence)
            
        trace.final_confidence = best_hypothesis.overall_confidence
        self.traces.append(trace)
        
        return best_hypothesis

    def test_hypothesis(self, hypothesis_statement: str, max_iterations: int = 2) -> Dict[str, Any]:
        """Runs the falsification test for a hypothesis statement against real model forward passes."""
        from backend.services import gpt2_engine

        # The id names the hypothesis statement, so this is content identity:
        # the same statement must get the name it has in every process, or a
        # citation written today stops resolving tomorrow. `hash()` is salted
        # per interpreter, so it could never do that.
        hypothesis_id = content_id(hypothesis_statement, prefix="hyp_")
        if not gpt2_engine.is_available():
            return {
                "passed": False,
                "outcome_state": "Unavailable",
                "hypothesis_id": hypothesis_id,
                "target": "gpt2",
                "description": hypothesis_statement,
                "confidence": None,
                "evidence_count": 0,
                "provenance": "unavailable",
                "measured": False,
                "reason": "Real model forward passes unavailable: gpt2_engine is not available.",
            }

        load_res = gpt2_engine.load()
        if not isinstance(load_res, dict) or load_res.get("status") != "loaded":
            return {
                "passed": False,
                "outcome_state": "Unavailable",
                "hypothesis_id": hypothesis_id,
                "target": "gpt2",
                "description": hypothesis_statement,
                "confidence": None,
                "evidence_count": 0,
                "provenance": "unavailable",
                "measured": False,
                "reason": f"Model failed to load: {load_res}",
            }

        import re
        quotes = re.findall(r"['\"](.*?)['\"]", hypothesis_statement)
        prompt = None
        expected_target = None

        if len(quotes) >= 2:
            prompt = quotes[0].strip()
            expected_target = quotes[1].strip()
        elif len(quotes) == 1:
            prompt = quotes[0].strip()
        else:
            prompt = hypothesis_statement.strip()

        if not expected_target:
            m = re.search(r"\b(?:predicts|outputs|yields|generates|to)\s+['\"]?([A-Za-z0-9_]+)['\"]?", hypothesis_statement, re.IGNORECASE)
            if m:
                expected_target = m.group(1).strip()

        try:
            fwd = gpt2_engine.run_prompt(prompt)
        except Exception as exc:
            return {
                "passed": False,
                "outcome_state": "Inconclusive",
                "hypothesis_id": hypothesis_id,
                "target": "gpt2",
                "description": hypothesis_statement,
                "confidence": None,
                "evidence_count": 0,
                "provenance": "unavailable",
                "measured": False,
                "reason": f"Forward pass error: {exc}",
            }

        if fwd.get("status") != "ok":
            return {
                "passed": False,
                "outcome_state": "Inconclusive",
                "hypothesis_id": hypothesis_id,
                "target": "gpt2",
                "description": hypothesis_statement,
                "confidence": None,
                "evidence_count": 0,
                "provenance": "unavailable",
                "measured": False,
                "reason": f"Forward pass failed: {fwd.get('error', 'unknown error')}",
            }

        next_tok = fwd.get("next_token", "").strip()
        top5 = fwd.get("top5", [])
        top5_tokens = [t.get("token", "").strip() for t in top5]
        top_prob = top5[0].get("prob", 0.0) if top5 else 0.0

        evidence_item = {
            "prompt": prompt,
            "next_token": next_tok,
            "top5": top5_tokens,
            "top_prob": top_prob,
            "measured": True,
        }

        if expected_target:
            clean_expected = expected_target.strip().lower()
            clean_pred = next_tok.lower()
            clean_top5 = [t.lower() for t in top5_tokens]

            matching_prob = 0.0
            for t in top5:
                if t.get("token", "").strip().lower() == clean_expected:
                    matching_prob = float(t.get("prob", 0.0))
                    break

            if clean_pred == clean_expected or clean_expected in clean_top5[:2]:
                passed = True
                outcome = "Confirmed"
                conf = round(matching_prob if matching_prob > 0 else float(top_prob), 4)
            else:
                passed = False
                outcome = "Refuted"
                conf = round(1.0 - float(top_prob), 4)
        else:
            passed = True
            outcome = "Measured"
            conf = round(float(top_prob), 4)

        return {
            "passed": passed,
            "outcome_state": outcome,
            "hypothesis_id": hypothesis_id,
            "target": "gpt2",
            "description": hypothesis_statement,
            "confidence": conf,
            "evidence_count": 1,
            "provenance": "live",
            "measured": True,
            "evidence": [evidence_item],
        }


AutomaticHypothesisTesterEngine = AutoHypothesisTester
