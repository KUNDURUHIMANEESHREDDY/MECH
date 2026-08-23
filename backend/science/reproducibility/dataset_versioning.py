"""Dataset Version Manifest.

Records the exact dataset + tokenizer + library + seed configuration
for every reproduction run, so results are fully reproducible.
"""

from __future__ import annotations

import datetime as _dt
import platform
import sys
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional
import logging
logger = logging.getLogger(__name__)



@dataclass
class DatasetVersionManifest:
    """Complete snapshot of everything needed to reproduce a pipeline run."""
    manifest_id: str
    paper_id: str
    pipeline_name: str
    model_id: str
    hf_repo_id: str
    dataset_name: str
    dataset_version: str
    dataset_split: str
    dataset_num_examples: int
    tokenizer_id: str
    tokenizer_version: str
    tokenizer_revision: str
    random_seed: int
    python_version: str
    transformers_version: str
    torch_version: str
    numpy_version: str
    platform_info: str
    os_info: str
    cuda_version: str
    gpu_name: str
    git_commit_sha: str           # populated from environment or "unknown"
    pipeline_version: str
    model_revision: str
    recorded_at: str = field(default_factory=lambda: _dt.datetime.now(_dt.timezone.utc).isoformat())


def _safe_version(pkg: str) -> str:
    try:
        import importlib.metadata
        return importlib.metadata.version(pkg)
    except Exception as exc:  # noqa: BLE001
        logger.debug("Swallowed exception: %s", exc)
        return "not-installed"


class DatasetVersioningEngine:
    """Creates and stores version manifests for each reproduction run."""

    def __init__(self) -> None:
        self._manifests: Dict[str, DatasetVersionManifest] = {}

    def create_manifest(
        self,
        paper_id: str,
        pipeline_name: str,
        model_id: str,
        hf_repo_id: str,
        dataset_name: str,
        dataset_version: str = "1.0.0",
        dataset_split: str = "test",
        dataset_num_examples: int = 100,
        tokenizer_id: str = "gpt2",
        tokenizer_revision: str = "main",
        model_revision: str = "main",
        pipeline_version: str = "1.0.0",
        random_seed: int = 42,
        git_commit_sha: str = "unknown",
    ) -> DatasetVersionManifest:
        sha = os.environ.get("GIT_COMMIT_SHA", git_commit_sha)
        manifest_id = f"manifest_{paper_id}_{pipeline_name}_{int(_dt.datetime.now(_dt.timezone.utc).timestamp())}"
        
        # Capture GPU details if torch is available
        cuda_version = "none"
        gpu_name = "none"
        try:
            import torch
            if torch.cuda.is_available():
                cuda_version = torch.version.cuda or "unknown"
                gpu_name = torch.cuda.get_device_name(0)
        except ImportError:
            pass

        m = DatasetVersionManifest(
            manifest_id=manifest_id,
            paper_id=paper_id,
            pipeline_name=pipeline_name,
            model_id=model_id,
            hf_repo_id=hf_repo_id,
            dataset_name=dataset_name,
            dataset_version=dataset_version,
            dataset_split=dataset_split,
            dataset_num_examples=dataset_num_examples,
            tokenizer_id=tokenizer_id,
            tokenizer_version=_safe_version("tokenizers"),
            tokenizer_revision=tokenizer_revision,
            random_seed=random_seed,
            python_version=sys.version.split()[0],
            transformers_version=_safe_version("transformers"),
            torch_version=_safe_version("torch"),
            numpy_version=_safe_version("numpy"),
            platform_info=platform.platform(),
            os_info=platform.system() + " " + platform.release(),
            cuda_version=cuda_version,
            gpu_name=gpu_name,
            git_commit_sha=sha,
            pipeline_version=pipeline_version,
            model_revision=model_revision,
        )
        self._manifests[manifest_id] = m
        return m

    def get_manifest(self, manifest_id: str) -> Optional[DatasetVersionManifest]:
        return self._manifests.get(manifest_id)

    def list_manifests(self) -> List[Dict[str, Any]]:
        return [asdict(m) for m in self._manifests.values()]
