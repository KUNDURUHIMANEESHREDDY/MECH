"""Automated Circuit Discovery Engine."""

from __future__ import annotations

import datetime as _dt
import os
from typing import Any, Dict, List, Optional

from backend.science.models.adapter_base import ModelAdapter
from .algorithms import get_algorithm
from backend.datasets.dataset_manager import DatasetManager


class CircuitDiscoveryEngine:
    """Discovers circuits using registered discovery algorithms."""

    def __init__(self, adapter: Optional[ModelAdapter] = None) -> None:
        self.adapter = adapter
        # Point to the datasets directory relative to this file
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        self.dataset_manager = DatasetManager(os.path.join(base_dir, "datasets"))

    def discover_circuit(self, dataset_name: str = "ioi", prompt_id: str = "ioi_0001", algorithm_name: str = "acdc") -> Dict[str, Any]:
        """Discovers a circuit using the specified registered algorithm and dataset."""
        if self.adapter:
            try:
                prompts = self.dataset_manager.load(dataset_name)
                if not prompts:
                    return {"error": f"Dataset '{dataset_name}' is empty."}
                dataset_item = next((p for p in prompts if isinstance(p, dict) and p.get("id") == prompt_id), prompts[0])
                
                engine = get_algorithm(algorithm_name, self.adapter)
                report = engine.run(dataset=dataset_item)
                return report.to_dict()
            except Exception as e:
                return {"error": str(e)}
            
        # Fail-closed abstention when adapter is not provided
        return {
            "error": "EXECUTION_REQUIRED",
            "message": "Live model adapter is required for circuit discovery. Synthetic fallback generation is prohibited.",
            "status": "abstained",
            "dataset_id": f"{dataset_name}_{prompt_id}",
            "timestamp": _dt.datetime.utcnow().isoformat() + "Z",
        }
