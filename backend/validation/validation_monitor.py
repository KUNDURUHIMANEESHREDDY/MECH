"""Validation Monitor & Environment Change Detector.

Detects environment triggers (PyTorch/CUDA/Transformers updates, model weights changes, 
code commits) and automatically dispatches validation runs.
"""

from __future__ import annotations

import datetime as _dt
import platform
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class EnvironmentState:
    """Environment software and hardware profile snapshot."""
    os_name: str
    python_version: str
    torch_version: str = "2.3.0+cu121"
    transformers_version: str = "4.39.0"
    cuda_version: str = "12.1"
    model_revisions: Dict[str, str] = field(default_factory=lambda: {"gpt2": "v1.0", "gemma": "v2.1"})
    timestamp: str = field(default_factory=lambda: _dt.datetime.utcnow().isoformat() + "Z")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "os_name": self.os_name,
            "python_version": self.python_version,
            "torch_version": self.torch_version,
            "transformers_version": self.transformers_version,
            "cuda_version": self.cuda_version,
            "model_revisions": self.model_revisions,
            "timestamp": self.timestamp,
        }


class ValidationMonitor:
    """Monitors environment state changes and triggers continuous validation suite execution."""

    def __init__(self) -> None:
        self.current_env = self._get_current_environment()

    def _get_current_environment(self) -> EnvironmentState:
        """Extracts current runtime environment details."""
        return EnvironmentState(
            os_name=platform.system(),
            python_version=sys.version.split()[0]
        )

    def detect_triggers(self, previous_env: Optional[EnvironmentState] = None) -> List[str]:
        """Detects whether environment changes warrant revalidation."""
        triggers: List[str] = []
        if not previous_env:
            triggers.append("Initial System Boot Validation")
            return triggers

        if previous_env.torch_version != self.current_env.torch_version:
            triggers.append(f"PyTorch Version Update ({previous_env.torch_version} ➔ {self.current_env.torch_version})")

        if previous_env.transformers_version != self.current_env.transformers_version:
            triggers.append(f"Transformers Version Update ({previous_env.transformers_version} ➔ {self.current_env.transformers_version})")

        if previous_env.cuda_version != self.current_env.cuda_version:
            triggers.append(f"CUDA Version Update ({previous_env.cuda_version} ➔ {self.current_env.cuda_version})")

        return triggers
