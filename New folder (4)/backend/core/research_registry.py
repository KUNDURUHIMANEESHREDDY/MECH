"""Global Research Registry."""

from __future__ import annotations

import datetime as _dt
from typing import Any, Dict, List


class ResearchRegistry:
    """Global registry indexing experiments, discoveries, benchmarks, datasets, publications, and plugins."""

    def __init__(self) -> None:
        self.catalog: Dict[str, Dict[str, Any]] = {
            "exp_1": {"id": "exp_1", "type": "Experiment", "name": "IOI Circuit Tracing", "version": "1.0.0"},
            "disc_1": {"id": "disc_1", "type": "Discovery", "name": "Layer 8 Induction Head", "version": "1.2.0"},
            "bench_1": {"id": "bench_1", "type": "Benchmark", "name": "IOI Benchmark Suite", "version": "2.0.0"},
            "ds_1": {"id": "ds_1", "type": "Dataset", "name": "IOI Prompts v1", "version": "1.0.0"},
            "pub_1": {"id": "pub_1", "type": "Publication", "name": "IOI Paper Package", "version": "1.0.0"},
        }

    def register_item(self, item_id: str, item_type: str, name: str, version: str = "1.0.0") -> Dict[str, Any]:
        entry = {
            "id": item_id,
            "type": item_type,
            "name": name,
            "version": version,
            "registered_at": _dt.datetime.utcnow().isoformat() + "Z",
        }
        self.catalog[item_id] = entry
        return entry

    def list_catalog(self, item_type: str | None = None) -> List[Dict[str, Any]]:
        items = list(self.catalog.values())
        if item_type:
            return [i for i in items if i["type"].lower() == item_type.lower()]
        return items
