"""Automated Circuit Discovery Engine."""

from __future__ import annotations

import datetime as _dt
import os
from typing import Any, Dict, List, Optional

from backend.science.models.adapter_base import ModelAdapter
from .algorithms import get_algorithm
from backend.research_datasets.dataset_manager import DatasetManager


class CircuitDiscoveryEngine:
    """Discovers circuits using registered discovery algorithms."""

    def __init__(self, adapter: Optional[ModelAdapter] = None) -> None:
        self.adapter = adapter
        # Point at the research datasets directory relative to this file.
        # Renamed from "datasets" to "research_datasets": a backend/datasets/
        # package shadowed HuggingFace's `datasets` whenever backend/ was on
        # sys.path, which is how the desktop app launches the backend.
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        self.dataset_manager = DatasetManager(
            os.path.join(base_dir, "research_datasets"))

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
            
        # Fallback Mock for UI
        cid = f"circ_mock_{hash(dataset_name) & 0xffffffff:08x}"
        return {
            "algorithm": "mock",
            "dataset_id": f"{dataset_name}_{prompt_id}",
            "model_id": "mock",
            "runtime_ms": 150.0,
            "statistics": {},
            "evidence": {},
            "confidence": 0.95,
            "graph": {
                "nodes": [
                    {"id": "T_0", "type": "Token", "label": "Prompt"},
                    {"id": "N_L8_N402", "type": "Neuron", "label": "L8_N402 (IOI)"},
                    {"id": "P_0", "type": "Prediction", "label": "Target"},
                ],
                "edges": [
                    {"source": "T_0", "target": "N_L8_N402", "weight": 0.88, "confidence": 0.94},
                    {"source": "N_L8_N402", "target": "P_0", "weight": 0.95, "confidence": 0.98},
                ],
                "score": 0.945
            },
            "artifacts": [],
            "provenance": {},
            "timestamp": _dt.datetime.utcnow().isoformat() + "Z",
        }
