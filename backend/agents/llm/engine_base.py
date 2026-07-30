"""Base LLM Engine Interface.

Defines the contract for interacting with Large Language Models in a 
zero-dependency environment.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List


class LLMEngine(ABC):
    """Abstract interface for LLM interaction."""

    @abstractmethod
    def generate_hypotheses(self, feature_report: Dict[str, Any], max_candidates: int = 3) -> List[Dict[str, Any]]:
        """Analyzes a deterministic feature report and returns candidate hypotheses.
        
        Args:
            feature_report: The deterministic metrics (n-grams, histograms, etc.)
            max_candidates: Maximum number of hypotheses to generate.
            
        Returns:
            A list of dictionaries, each containing:
                - description: A string explaining the semantic trigger.
                - initial_confidence: A float between 0.0 and 1.0.
        """
        ...
        
    @abstractmethod
    def evaluate_experiment(self, hypothesis: str, experiment_result: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluates whether an experiment result supports or falsifies a hypothesis.
        
        Args:
            hypothesis: The semantic description of the hypothesis.
            experiment_result: The measurement from the experiment.
            
        Returns:
            A dictionary containing:
                - type: "supportive", "falsifying", or "inconclusive"
                - rationale: The LLM's explanation for the evaluation.
        """
        ...

    @abstractmethod
    def generate_critique(self, hypothesis: str) -> Dict[str, Any]:
        """Generates a skeptic's critique of a hypothesis.
        
        Args:
            hypothesis: The semantic description of the hypothesis.
            
        Returns:
            A dictionary containing "critique" and "alternative_explanation".
        """
        ...
        
    @abstractmethod
    def generate_counterexample(self, hypothesis: str, critique: str = "") -> Dict[str, Any]:
        """Generates a prompt designed to test the skeptic's critique.
        
        Args:
            hypothesis: The semantic description of the hypothesis.
            critique: The skeptic's critique.
            
        Returns:
            A dictionary containing "prompt" and "expected_firing".
        """
        ...
        
    @abstractmethod
    def review_evidence(self, hypothesis: str, critique: str, experiment_spec: Dict[str, Any], result: Dict[str, Any]) -> Dict[str, Any]:
        """Acts as a peer reviewer evaluating the hard data.
        
        Returns:
            A dictionary containing "verdict" and "rationale".
        """
        ...
