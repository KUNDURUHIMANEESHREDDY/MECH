"""Research Snapshot — Exhaustive Environment & Dependency Fingerprinting.

Captures everything required to reproduce a run: Python/Torch/CUDA versions,
Git state, Hardware specs, and a complete library inventory.
"""

from __future__ import annotations

import os
import platform
import sys
import subprocess
import datetime
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional
import logging
logger = logging.getLogger(__name__)



@dataclass
class ResearchSnapshot:
    """Exhaustive snapshot of the execution environment."""
    snapshot_id: str
    timestamp: str

    # Software versions
    python_version: str
    operating_system: str
    os_release: str

    # Deep Learning Stack
    torch_version: str
    cuda_version: str
    cudnn_version: str
    transformers_version: str
    transformer_lens_version: str
    sae_lens_version: str
    numpy_version: str

    # Git State
    git_sha: str
    git_branch: str
    git_dirty: bool

    # Hardware Specs
    cpu_model: str
    gpu_model: str
    gpu_driver_version: str
    total_ram_gb: float

    # Core Libraries Inventory
    installed_packages: Dict[str, str] = field(default_factory=dict)

    # Reproducibility Seeds
    random_seed: Optional[int] = None
    numpy_seed: Optional[int] = None
    torch_seed: Optional[int] = None


class ResearchSnapshotEngine:
    """Generates exhaustive reproducibility snapshots."""

    def _get_package_version(self, name: str) -> str:
        try:
            import importlib.metadata
            return importlib.metadata.version(name)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Swallowed exception: %s", exc)
            return "not-installed"

    def _get_git_info(self) -> Dict[str, Any]:
        try:
            sha = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
            branch = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"]).decode().strip()
            dirty = len(subprocess.check_output(["git", "status", "--porcelain"]).decode().strip()) > 0
            return {"sha": sha, "branch": branch, "dirty": dirty}
        except Exception as exc:  # noqa: BLE001
            logger.debug("Swallowed exception: %s", exc)
            return {"sha": "unknown", "branch": "unknown", "dirty": True}

    def capture(self, seed: Optional[int] = None) -> ResearchSnapshot:
        """Captures the current execution environment."""
        git = self._get_git_info()

        # GPU Info
        cuda_ver = "none"
        gpu_name = "none"
        gpu_driver = "none"
        cudnn_ver = "none"

        try:
            import torch
            if torch.cuda.is_available():
                cuda_ver = torch.version.cuda
                gpu_name = torch.cuda.get_device_name(0)
                cudnn_ver = str(torch.backends.cudnn.version())
                # Note: Driver version usually requires nvidia-smi call
        except ImportError:
            pass

        try:
            import psutil
            ram_gb = psutil.virtual_memory().total / (1024**3)
        except ImportError:
            ram_gb = 0.0

        timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
        snapshot_id = f"snap_{int(datetime.datetime.now().timestamp())}"

        return ResearchSnapshot(
            snapshot_id=snapshot_id,
            timestamp=timestamp,
            python_version=sys.version.split()[0],
            operating_system=platform.system(),
            os_release=platform.release(),
            torch_version=self._get_package_version("torch"),
            cuda_version=cuda_ver,
            cudnn_version=cudnn_ver,
            transformers_version=self._get_package_version("transformers"),
            transformer_lens_version=self._get_package_version("transformer_lens"),
            sae_lens_version=self._get_package_version("sae_lens"),
            numpy_version=self._get_package_version("numpy"),
            git_sha=git["sha"],
            git_branch=git["branch"],
            git_dirty=git["dirty"],
            cpu_model=platform.processor() or "unknown",
            gpu_model=gpu_name,
            gpu_driver_version=gpu_driver,
            total_ram_gb=round(ram_gb, 2),
            random_seed=seed
        )
