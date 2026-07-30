from abc import ABC, abstractmethod
from typing import Dict, Any, List
from ..science.statistics.statistical_validator import StatisticalValidator
from ..science.statistics.statistical_protocol import StatisticalProtocol

class LandmarkReproduction(ABC):
    """
    Base class for reproducing landmark Mechanistic Interpretability papers.
    Ensures that every reproduction follows a strict statistical protocol.
    """
    def __init__(self, experiment_id: str):
        self.experiment_id = experiment_id
        
        # Default protocol for landmark reproductions: 
        # Highly rigorous, permutation-based, high power requirement.
        self.protocol = StatisticalProtocol(
            version="landmark_v1.0",
            alpha=0.01,
            correction_method="BH",
            permutation_count=10000,
            power_threshold=0.90,
            effect_size_metric="cohens_d"
        )
        self.validator = StatisticalValidator(experiment_id, self.protocol)

    @property
    @abstractmethod
    def paper_reference(self) -> Dict[str, str]:
        """Returns metadata about the original paper."""
        pass

    @abstractmethod
    def run(self, model_name: str = "gpt2-small", **kwargs) -> Dict[str, Any]:
        """Executes the reproduction pipeline."""
        pass

    def validate_reproduction(self, results: Dict[str, Any], expected_effect_size: float) -> bool:
        """
        Validates if the reproduction achieved the expected statistical power
        and effect size compared to the original paper.
        """
        actual_effect = results.get("effect_sizes", {}).get(self.protocol.effect_size_metric, 0)
        p_value = results.get("p_value_raw", 1.0)
        power = results.get("power", {}).get("observed_power", 0.0)
        
        is_significant = p_value < self.protocol.alpha
        is_powered = power >= self.protocol.power_threshold
        # Check if effect size is at least 80% of the expected effect
        effect_matches = actual_effect >= (0.8 * expected_effect_size)
        
        return is_significant and is_powered and effect_matches
