import json
from dataclasses import dataclass, asdict
from typing import Dict, Any

@dataclass
class StatisticalProtocol:
    """
    Standardized Statistical Protocol for Reproducibility.
    Every benchmark can cite a versioned protocol.
    """
    version: str = "v1.0"
    alpha: float = 0.05
    correction_method: str = "BH"
    permutation_count: int = 10000
    bootstrap_method: str = "percentile"
    confidence_level: float = 0.95
    effect_size_metric: str = "cohens_d"
    power_threshold: float = 0.80
    random_seed: int = 42
    stopping_rule: str = "sequential_power_and_significance"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self, filepath: str):
        """
        Exports the protocol to a JSON file.
        """
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_json(cls, filepath: str) -> "StatisticalProtocol":
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls(**data)
