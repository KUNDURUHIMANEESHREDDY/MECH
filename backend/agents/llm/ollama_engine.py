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
from backend.utils.url_validator import validate_url


class OllamaEngine(LLMEngine):
    """LLM Engine using a local Ollama instance via urllib."""

    def __init__(self, model: str = "llama3", endpoint: str = "http://localhost:11434/v1/chat/completions") -> None:
        self.model = model
        self.endpoint = validate_url(endpoint, allow_private=True, allow_localhost=True)

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
            with urllib.request.urlopen(req) as response:
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
            if isinstance(hypotheses, list):
                return hypotheses[:max_candidates]
            return [hypotheses] # In case it returns a single object
        except Exception as e:
            print(f"Warning: Ollama generation failed ({e}). Falling back to mock.")
            return self._mock_generate(feature_report, max_candidates)

    def evaluate_experiment(self, hypothesis: str, experiment_result: Dict[str, Any]) -> Dict[str, Any]:
        prompt = EXPERIMENT_EVALUATION_PROMPT.format(
            hypothesis=hypothesis,
            experiment_result_json=json.dumps(experiment_result, indent=2)
        )
        
        try:
            response_text = self._call_api(prompt)
            return self._parse_json_response(response_text)
        except Exception as e:
            print(f"Warning: Ollama evaluation failed ({e}). Falling back to mock.")
            return self._mock_evaluate(hypothesis, experiment_result)

    def generate_critique(self, hypothesis: str) -> Dict[str, Any]:
        from .prompts import SKEPTIC_CRITIQUE_PROMPT
        prompt = SKEPTIC_CRITIQUE_PROMPT.format(hypothesis=hypothesis)
        try:
            response_text = self._call_api(prompt)
            return self._parse_json_response(response_text)
        except Exception as e:
            print(f"Warning: Ollama critique failed ({e}). Falling back to mock.")
            return {"critique": "Might be a spurious correlation.", "alternative_explanation": "Fires on something else."}

    def generate_counterexample(self, hypothesis: str, critique: str = "") -> Dict[str, Any]:
        from .prompts import COUNTEREXAMPLE_GENERATION_PROMPT
        prompt = COUNTEREXAMPLE_GENERATION_PROMPT.format(hypothesis=hypothesis, critique=critique)
        try:
            response_text = self._call_api(prompt)
            return self._parse_json_response(response_text)
        except Exception as e:
            print(f"Warning: Ollama counterexample failed ({e}). Falling back to mock.")
            return self._mock_counterexample(hypothesis)
            
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
            return self._parse_json_response(response_text)
        except Exception as e:
            print(f"Warning: Ollama review failed ({e}). Falling back to mock.")
            return {"verdict": "needs_revision", "rationale": "Mock reviewer says needs more data."}

    # --- Mocks for fallback if Ollama isn't running ---

    def _mock_generate(self, feature_report: Dict[str, Any], max_candidates: int) -> List[Dict[str, Any]]:
        fid = feature_report.get("feature_index", "Unknown")
        return [
            {
                "description": f"Feature {fid} fires on French locations (Ollama Mock).",
                "initial_confidence": 0.65
            },
            {
                "description": f"Feature {fid} fires on Python syntax (Ollama Mock).",
                "initial_confidence": 0.45
            }
        ][:max_candidates]
        
    def _mock_evaluate(self, hypothesis: str, experiment_result: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "type": "supportive",
            "rationale": "Mock evaluation: result aligns with the hypothesis (Ollama Mock)."
        }
        
    def _mock_counterexample(self, hypothesis: str) -> Dict[str, Any]:
        return {
            "prompt": "Paris Hilton went to the store.",
            "expected_firing": False
        }
