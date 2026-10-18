"""Ollama-backed LLM Engine.

Implements the LLMEngine interface using Python's built-in urllib to make
REST calls to a local Ollama instance, avoiding the need for API keys or
external dependencies.
"""

from __future__ import annotations

import json
import urllib.request
import urllib.error
from typing import Any, Dict, List

from .engine_base import LLMEngine
from .prompts import HYPOTHESIS_GENERATION_PROMPT, EXPERIMENT_EVALUATION_PROMPT


class OllamaEngine(LLMEngine):
    """LLM Engine using a local Ollama instance via urllib."""

    #: Default socket timeout for a generation call. Bounded on purpose: an
    #: unreachable host must fail closed in seconds, not hang the Society run
    #: that asked for a hypothesis. Overridable per instance, and the engine
    #: factory passes `MECH_OLLAMA_TIMEOUT` through here.
    DEFAULT_TIMEOUT = 10.0

    def __init__(self, model: str = "llama3",
                 endpoint: str = "http://localhost:11434/v1/chat/completions",
                 timeout: float = DEFAULT_TIMEOUT) -> None:
        self.model = model
        self.endpoint = endpoint
        self.timeout = float(timeout)

    def _call_api(self, prompt: str) -> str:
        """Helper to make the REST call to local Ollama."""
        headers = {
            "Content-Type": "application/json"
        }
        
        data = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2
        }
        
        req = urllib.request.Request(
            self.endpoint, 
            data=json.dumps(data).encode("utf-8"), 
            headers=headers
        )
        
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                result = json.loads(response.read().decode("utf-8"))
                return result["choices"][0]["message"]["content"].strip()
        except urllib.error.URLError as e:
            raise RuntimeError(f"Ollama API call failed. Is Ollama running on {self.endpoint}? Error: {e}")

    def _parse_json_response(self, text: str) -> Any:
        """Safely parses a JSON response, handling markdown code blocks if present."""
        text = text.strip()
        if text.startswith("```json"):
            text = text[7:]
        elif text.startswith("```"):
            text = text[3:]
            
        if text.endswith("```"):
            text = text[:-3]
            
        return json.loads(text.strip())

    def generate_hypotheses(self, feature_report: Dict[str, Any], max_candidates: int = 3) -> List[Dict[str, Any]]:
        prompt = HYPOTHESIS_GENERATION_PROMPT.format(
            max_candidates=max_candidates,
            feature_report_json=json.dumps(feature_report, indent=2)
        )
        
        try:
            response_text = self._call_api(prompt)
            hypotheses = self._parse_json_response(response_text)
            candidates = hypotheses if isinstance(hypotheses, list) else [hypotheses]
            accepted = [
                c for c in candidates
                if isinstance(c, dict) and bool(str(c.get("description") or "").strip())
            ]
            if not accepted:
                return [{
                    "status": "unavailable",
                    "provenance": "unavailable",
                    "reason": "Ollama generation returned no valid hypotheses with descriptions",
                }]
            return accepted[:max_candidates]
        except Exception as e:
            return [{
                "status": "unavailable",
                "provenance": "unavailable",
                "reason": f"Ollama generation failed: {e}",
            }]

    def evaluate_experiment(self, hypothesis: str, experiment_result: Dict[str, Any]) -> Dict[str, Any]:
        prompt = EXPERIMENT_EVALUATION_PROMPT.format(
            hypothesis=hypothesis,
            experiment_result_json=json.dumps(experiment_result, indent=2)
        )
        
        try:
            response_text = self._call_api(prompt)
            result = self._parse_json_response(response_text)
            if not isinstance(result, dict) or "error" in result or not ("type" in result or "rationale" in result):
                return {
                    "status": "unavailable",
                    "provenance": "unavailable",
                    "type": "inconclusive",
                    "reason": f"Ollama evaluation returned invalid response: {result}",
                    "rationale": f"Evaluation unavailable: {result}",
                }
            return result
        except Exception as e:
            return {
                "status": "unavailable",
                "provenance": "unavailable",
                "type": "inconclusive",
                "reason": f"Ollama evaluation failed: {e}",
                "rationale": f"Ollama evaluation failed: {e}",
            }

    def generate_critique(self, hypothesis: str) -> Dict[str, Any]:
        from .prompts import SKEPTIC_CRITIQUE_PROMPT
        prompt = SKEPTIC_CRITIQUE_PROMPT.format(hypothesis=hypothesis)
        try:
            response_text = self._call_api(prompt)
            result = self._parse_json_response(response_text)
            if not isinstance(result, dict) or "error" in result or not ("critique" in result or "alternative_explanation" in result):
                return {
                    "status": "unavailable",
                    "provenance": "unavailable",
                    "reason": f"Ollama critique returned invalid response: {result}",
                    "critique": "",
                    "alternative_explanation": "",
                }
            return result
        except Exception as e:
            return {
                "status": "unavailable",
                "provenance": "unavailable",
                "reason": f"Ollama critique failed: {e}",
                "critique": "",
                "alternative_explanation": "",
            }

    def generate_counterexample(self, hypothesis: str, critique: str = "") -> Dict[str, Any]:
        from .prompts import COUNTEREXAMPLE_GENERATION_PROMPT
        prompt = COUNTEREXAMPLE_GENERATION_PROMPT.format(hypothesis=hypothesis, critique=critique)
        try:
            response_text = self._call_api(prompt)
            result = self._parse_json_response(response_text)
            if not isinstance(result, dict) or "error" in result or not ("prompt" in result or "expected_firing" in result):
                return {
                    "status": "unavailable",
                    "provenance": "unavailable",
                    "reason": f"Ollama counterexample returned invalid response: {result}",
                    "prompt": "",
                    "expected_firing": False,
                }
            return result
        except Exception as e:
            return {
                "status": "unavailable",
                "provenance": "unavailable",
                "reason": f"Ollama counterexample failed: {e}",
                "prompt": "",
                "expected_firing": False,
            }
            
    def review_evidence(self, hypothesis: str, critique: str, experiment_spec: Dict[str, Any], result: Dict[str, Any]) -> Dict[str, Any]:
        from .prompts import REVIEWER_CONSENSUS_PROMPT
        prompt = REVIEWER_CONSENSUS_PROMPT.format(
            hypothesis=hypothesis,
            critique=critique,
            prompt=experiment_spec.get("prompt", ""),
            expected=experiment_spec.get("expected_firing", False),
            actual=(result.get("type") == "supportive")
        )
        try:
            response_text = self._call_api(prompt)
            result = self._parse_json_response(response_text)
            if not isinstance(result, dict) or "error" in result or not ("verdict" in result or "rationale" in result):
                return {
                    "status": "unavailable",
                    "provenance": "unavailable",
                    "verdict": "inconclusive",
                    "rationale": f"Ollama review returned invalid response: {result}",
                    "reason": f"Ollama review returned invalid response: {result}",
                }
            return result
        except Exception as e:
            return {
                "status": "unavailable",
                "provenance": "unavailable",
                "verdict": "inconclusive",
                "rationale": f"Ollama review failed: {e}",
                "reason": f"Ollama review failed: {e}",
            }

    # --- Mocks replaced with fail-closed behavior to prevent fabricated scientific reasoning ---

    def _mock_generate(self, feature_report: Dict[str, Any], max_candidates: int) -> List[Dict[str, Any]]:
        return [{
            "status": "unavailable",
            "provenance": "unavailable",
            "reason": "Mock hypothesis generation is disabled. Ensure Ollama is running.",
        }]
        
    def _mock_evaluate(self, hypothesis: str, experiment_result: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "status": "unavailable",
            "provenance": "unavailable",
            "type": "inconclusive",
            "reason": "Mock evaluation is disabled. Ensure Ollama is running.",
            "rationale": "Mock evaluation is disabled.",
        }
        
    def _mock_counterexample(self, hypothesis: str) -> Dict[str, Any]:
        return {
            "status": "unavailable",
            "provenance": "unavailable",
            "prompt": "",
            "expected_firing": False,
            "reason": "Mock counterexample generation is disabled. Ensure Ollama is running.",
        }
