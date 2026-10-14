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
import hashlib
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


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

    # Reproducibility Seeds & RNG States
    random_seed: Optional[int] = None
    numpy_seed: Optional[Dict[str, Any]] = None
    torch_cpu_seed: Optional[Any] = None
    torch_cuda_seed: Optional[Any] = None
    python_random_state: Optional[Any] = None


class ResearchSnapshotEngine:
    """Generates exhaustive reproducibility snapshots."""

    def _get_package_version(self, name: str) -> str:
        try:
            import importlib.metadata
            return importlib.metadata.version(name)
        except Exception:
            return "not-installed"

    def _get_installed_packages(self) -> Dict[str, str]:
        """Capture all installed packages and their versions."""
        try:
            import importlib.metadata
            packages = {}
            for dist in importlib.metadata.distributions():
                packages[dist.metadata["Name"]] = dist.version
            return packages
        except Exception:
            return {}

    def _get_git_info(self) -> Dict[str, Any]:
        try:
            sha = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
            branch = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"]).decode().strip()
            dirty = len(subprocess.check_output(["git", "status", "--porcelain"]).decode().strip()) > 0
            return {"sha": sha, "branch": branch, "dirty": dirty}
        except Exception:
            return {"sha": "unknown", "branch": "unknown", "dirty": True}

    def _get_gpu_driver_version(self) -> str:
        """Get GPU driver version via nvidia-smi if available."""
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception:
            pass
        return "unavailable"

    def _capture_rng_states(self, seed: Optional[int] = None) -> Dict[str, Any]:
        """Capture actual RNG states from Python, NumPy, and PyTorch."""
        states = {}

        # Python random state
        try:
            import random
            states["python_random_state"] = random.getstate()
        except Exception:
            states["python_random_state"] = "unavailable"

        # NumPy state
        try:
            import numpy as np
            state_tuple = np.random.get_state()
            states["numpy_random_state"] = {
                "type": str(state_tuple[0]),
                "keys": state_tuple[1].tolist() if hasattr(state_tuple[1], "tolist") else list(state_tuple[1]),
                "pos": state_tuple[2],
                "has_gauss": state_tuple[3],
                "cached_gaussian": state_tuple[4],
            }
        except Exception:
            states["numpy_random_state"] = "unavailable"

        # PyTorch state
        try:
            import torch
            if torch.cuda.is_available():
                states["torch_cuda_rng_state"] = torch.cuda.get_rng_state_all()
            states["torch_cpu_rng_state"] = torch.get_rng_state()
        except Exception:
            states["torch_cpu_rng_state"] = "unavailable"
            states["torch_cuda_rng_state"] = "unavailable"

        return states

    def _generate_snapshot_id(self, git_sha: str) -> str:
        """Generate a collision-resistant snapshot ID using SHA-256."""
        timestamp_micro = datetime.datetime.now(datetime.timezone.utc).isoformat()
        unique_suffix = uuid.uuid4().hex[:8]
        payload = f"{timestamp_micro}:{git_sha}:{unique_suffix}"
        return f"snap_{hashlib.sha256(payload.encode('utf-8')).hexdigest()[:16]}"

    def _get_gpu_driver_version(self) -> str:
        """Get GPU driver version via nvidia-smi if available."""
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception:
            pass
        return "unavailable"

    def _get_installed_packages(self) -> Dict[str, str]:
        """Capture all installed packages and their versions."""
        try:
            import importlib.metadata
            packages = {}
            for dist in importlib.metadata.distributions():
                packages[dist.metadata["Name"]] = dist.version
            return packages
        except Exception:
            return {}

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
        except ImportError:
            pass

        try:
            import psutil
            ram_gb = psutil.virtual_memory().total / (1024**3)
        except ImportError:
            ram_gb = 0.0

        timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()

        # Capture actual installed packages
        installed_packages = self._get_installed_packages()

        # Capture GPU driver version
        gpu_driver_version = self._get_gpu_driver_version()

        # Generate collision-resistant snapshot ID
        snapshot_id = self._generate_snapshot_id(git["sha"])

        # Capture RNG states
        rng_states = self._capture_rng_states(seed)

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
            gpu_driver_version=gpu_driver_version,
            total_ram_gb=round(ram_gb, 2),
            installed_packages=self._get_installed_packages(),
            random_seed=seed,
            numpy_seed=rng_states.get("numpy_random_state"),
            torch_cpu_seed=rng_states.get("torch_cpu_rng_state"),
            torch_cuda_seed=rng_states.get("torch_cuda_rng_state"),
            python_random_state=rng_states.get("python_random_state"),
        )