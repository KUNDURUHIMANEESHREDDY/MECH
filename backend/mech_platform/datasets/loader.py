"""Dataset Loader & Streaming Engine."""

from __future__ import annotations

from typing import Any, Dict, List, Optional


class DatasetLoader:
    """Loader and streaming engine for research datasets."""

    DATASETS = {
        "openwebtext": {
            "name": "OpenWebText",
            "samples": 8000000,
            "license": "MIT",
            "tokenizer": "GPT-2 BPE",
        },
        "the_pile": {
            "name": "The Pile",
            "samples": 21000000,
            "license": "CC-BY-4.0",
            "tokenizer": "GPT-NeoX",
        },
        "wikitext": {
            "name": "WikiText-103",
            "samples": 103000,
            "license": "CC-BY-SA-3.0",
            "tokenizer": "GPT-2 BPE",
        },
        "custom": {
            "name": "Custom User Corpus",
            "samples": 5000,
            "license": "Proprietary",
            "tokenizer": "Custom",
        },
    }

    def list_datasets(self) -> List[Dict[str, Any]]:
        return [{"dataset_id": k, **v} for k, v in self.DATASETS.items()]

    def stream_samples(self, dataset_id: str = "openwebtext", limit: int = 5) -> List[Dict[str, Any]]:
        ds = self.DATASETS.get(dataset_id, self.DATASETS["openwebtext"])
        samples = []
        for i in range(limit):
            samples.append({
                "sample_id": i + 1,
                "dataset": ds["name"],
                "text": f"Sample text #{i + 1} from {ds['name']} dataset for activation probes.",
            })
        return samples
