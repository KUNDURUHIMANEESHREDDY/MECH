"""OpenAI-backed LLM Engine.

Implements the LLMEngine interface using Python's built-in urllib to make
REST calls to the OpenAI API, adhering to the zero-dependency requirement.
"""

from __future__ import annotations

import json
import os
import urllib.request
import urllib.error
from typing import Any, Dict, List

from .engine_base import LLMEngine
from .prompts import HYPOTHESIS_GENERATION_PROMPT, EXPERIMENT_EVALUATION_PROMPT


class OpenAIEngine(LLMEngine):
    """LLM Engine using OpenAI's chat completions API via urllib."""

    def __init__(self, model: str = "gpt-4o") -> None:
        self.api_key = os.environ.get("OPENAI_API_KEY")
        self.model = model
        self.endpoint = "https://api.openai.com/v1/chat/completions"

    def _call_api(self, prompt: str) -> str:
        """Helper to make the REST call to OpenAI."""
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set.")

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
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
            with urllib.request.urlopen(req) as response:
                result = json.loads(response.read().decode("utf-8"))
                return result["choices"][0]["message"]["content"].strip()
        except urllib.error.URLError as e:
            raise RuntimeError(f"OpenAI API call failed: {e}")

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
        if not self.api_key:
            return [{
                "status": "unavailable",
                "provenance": "unavailable",
                "reason": "OPENAI_API_KEY environment variable is not set.",
            }]
            
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
                    "reason": "OpenAI generation returned no valid hypotheses with descriptions",
                }]
            return accepted[:max_candidates]
        except Exception as e:
            return [{
                "status": "unavailable",
                "provenance": "unavailable",
                "reason": f"OpenAI API call failed: {e}",
            }]

    def evaluate_experiment(self, hypothesis: str, experiment_result: Dict[str, Any]) -> Dict[str, Any]:
        if not self.api_key:
            return {
                "status": "unavailable",
                "provenance": "unavailable",
                "type": "inconclusive",
                "reason": "OPENAI_API_KEY environment variable is not set.",
                "rationale": "Evaluation unavailable: OPENAI_API_KEY is not set.",
            }
            
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
                    "reason": f"OpenAI evaluation returned invalid response: {result}",
                    "rationale": f"Evaluation unavailable: {result}",
                }
            return result
        except Exception as e:
            return {
                "status": "unavailable",
                "provenance": "unavailable",
                "type": "inconclusive",
                "reason": f"OpenAI API call failed: {e}",
                "rationale": f"Evaluation failed: {e}",
            }

    def generate_critique(self, hypothesis: str) -> Dict[str, Any]:
        if not self.api_key:
            return {
                "status": "unavailable",
                "provenance": "unavailable",
                "reason": "OPENAI_API_KEY environment variable is not set.",
                "critique": "",
                "alternative_explanation": "",
            }
            
        from .prompts import SKEPTIC_CRITIQUE_PROMPT
        prompt = SKEPTIC_CRITIQUE_PROMPT.format(hypothesis=hypothesis)
        try:
            response_text = self._call_api(prompt)
            result = self._parse_json_response(response_text)
            if not isinstance(result, dict) or "error" in result or not ("critique" in result or "alternative_explanation" in result):
                return {
                    "status": "unavailable",
                    "provenance": "unavailable",
                    "reason": f"OpenAI critique returned invalid response: {result}",
                    "critique": "",
                    "alternative_explanation": "",
                }
            return result
        except Exception as e:
            return {
                "status": "unavailable",
                "provenance": "unavailable",
                "reason": f"OpenAI API call failed: {e}",
                "critique": "",
                "alternative_explanation": "",
            }

    def generate_counterexample(self, hypothesis: str, critique: str = "") -> Dict[str, Any]:
        if not self.api_key:
            return {
                "status": "unavailable",
                "provenance": "unavailable",
                "reason": "OPENAI_API_KEY environment variable is not set.",
                "prompt": "",
                "expected_firing": False,
            }
            
        from .prompts import COUNTEREXAMPLE_GENERATION_PROMPT
        prompt = COUNTEREXAMPLE_GENERATION_PROMPT.format(hypothesis=hypothesis, critique=critique)
        
        try:
            response_text = self._call_api(prompt)
            result = self._parse_json_response(response_text)
            if not isinstance(result, dict) or "error" in result or not ("prompt" in result or "expected_firing" in result):
                return {
                    "status": "unavailable",
                    "provenance": "unavailable",
                    "reason": f"OpenAI counterexample returned invalid response: {result}",
                    "prompt": "",
                    "expected_firing": False,
                }
            return result
        except Exception as e:
            return {
                "status": "unavailable",
                "provenance": "unavailable",
                "reason": f"OpenAI API call failed: {e}",
                "prompt": "",
                "expected_firing": False,
            }
            
    def review_evidence(self, hypothesis: str, critique: str, experiment_spec: Dict[str, Any], result: Dict[str, Any]) -> Dict[str, Any]:
        if not self.api_key:
            return {
                "status": "unavailable",
                "provenance": "unavailable",
                "verdict": "inconclusive",
                "rationale": "Review unavailable: OPENAI_API_KEY is not set.",
                "reason": "OPENAI_API_KEY environment variable is not set.",
            }
            
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
                    "rationale": f"OpenAI review returned invalid response: {result}",
                    "reason": f"OpenAI review returned invalid response: {result}",
                }
            return result
        except Exception as e:
            return {
                "status": "unavailable",
                "provenance": "unavailable",
                "verdict": "inconclusive",
                "rationale": f"Review failed: {e}",
                "reason": f"OpenAI API call failed: {e}",
            }

    # --- Mocks replaced with fail-closed behavior to prevent fabricated scientific reasoning ---

    def _mock_generate(self, feature_report: Dict[str, Any], max_candidates: int) -> List[Dict[str, Any]]:
        return [{
            "status": "unavailable",
            "provenance": "unavailable",
            "reason": "Mock hypothesis generation is disabled. Configure OPENAI_API_KEY for live generation.",
        }]
        
    def _mock_evaluate(self, hypothesis: str, experiment_result: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "status": "unavailable",
            "provenance": "unavailable",
            "type": "inconclusive",
            "reason": "Mock evaluation is disabled. Configure OPENAI_API_KEY for live evaluation.",
            "rationale": "Mock evaluation is disabled.",
        }
        
    def _mock_counterexample(self, hypothesis: str) -> Dict[str, Any]:
        return {
            "status": "unavailable",
            "provenance": "unavailable",
            "prompt": "",
            "expected_firing": False,
            "reason": "Mock counterexample generation is disabled. Configure OPENAI_API_KEY for live counterexamples.",
        }
