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
    """Environment software and hardware profile snapshot.

    The three version fields were dataclass *defaults* -- "2.3.0+cu121",
    "4.39.0", "12.1" -- and `_get_current_environment` only passed os_name and
    python_version. So every EnvironmentState carried those constants
    regardless of what was actually installed, and `detect_triggers` compared
    the constant against itself. The version-change triggers could never fire:
    a change detector that cannot detect.

    They are now Optional with no default and are read from the installed
    packages. "not installed" is a legitimate, common state and is reported as
    None rather than a plausible version string.
    """
    os_name: str
    python_version: str
    torch_version: Optional[str] = None
    transformers_version: Optional[str] = None
    cuda_version: Optional[str] = None
    model_revisions: Dict[str, str] = field(default_factory=dict)
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


def _installed_version(package: str) -> Optional[str]:
    """The version of an installed package, or None if it is not importable.

    Returns None rather than a guess. A hardcoded version here is worse than an
    absent one: it makes an environment report look identical to the reference
    environment regardless of what is actually on the machine.
    """
    try:
        import importlib.metadata as _metadata
        return _metadata.version(package)
    except Exception:
        pass
    try:
        module = __import__(package)
        return getattr(module, "__version__", None)
    except Exception:
        return None


def _cuda_version() -> Optional[str]:
    """The CUDA runtime torch is actually built against, if any."""
    try:
        import torch
    except Exception:
        return None
    try:
        # `torch.version.cuda` is the CUDA the build targets, not the driver.
        # A CPU-only build reports None here, which is the honest answer.
        return getattr(torch.version, "cuda", None)
    except Exception:
        return None


class ValidationMonitor:
    """Monitors environment state changes and triggers continuous validation suite execution."""

    def __init__(self) -> None:
        self.current_env = self._get_current_environment()

    def _get_current_environment(self) -> EnvironmentState:
        """Extracts the current runtime environment.

        Every field is read from the machine. Nothing here is asserted.
        """
        return EnvironmentState(
            os_name=platform.system(),
            python_version=sys.version.split()[0],
            torch_version=_installed_version("torch"),
            transformers_version=_installed_version("transformers"),
            cuda_version=_cuda_version(),
            model_revisions=self._installed_model_revisions(),
        )

    @staticmethod
    def _installed_model_revisions() -> Dict[str, str]:
        """Resolve the actual commit each model is pinned to, if resolvable.

        Previously defaulted to {"gpt2": "v1.0", "gemma": "v2.1"} -- invented
        revision identifiers for models whose real revisions were never looked
        up. A revision tag is exactly the kind of value that must not be
        invented: it is what a reproduction is keyed on.
        """
        try:
            from huggingface_hub import HfApi
        except Exception:
            return {}
        revisions: Dict[str, str] = {}
        for repo in ("gpt2", "google/gemma-2b"):
            try:
                info = HfApi().model_info(repo)
                revisions[repo] = info.sha
            except Exception:
                # Offline, unauthenticated, or the model is not cached. An
                # unresolvable revision stays absent.
                continue
        return revisions

    def detect_triggers(self, previous_env: Optional[EnvironmentState] = None) -> List[str]:
        """Detects whether environment changes warrant revalidation."""
        triggers: List[str] = []
        if not previous_env:
            triggers.append("Initial System Boot Validation")
            return triggers

        # Only compare fields that were actually resolved on both sides. If
        # torch is not importable on one side, "None -> 2.3.0" is not a version
        # upgrade, it is a detection failure, and calling it an upgrade would
        # schedule a revalidation for a change that may not exist.
        for field_name, label in (
            ("torch_version", "PyTorch"),
            ("transformers_version", "Transformers"),
            ("cuda_version", "CUDA"),
            ("python_version", "Python"),
        ):
            before = getattr(previous_env, field_name, None)
            after = getattr(self.current_env, field_name, None)
            if before is None or after is None:
                if before != after:
                    triggers.append(
                        f"{label} Availability Change "
                        f"({before or 'absent'} -> {after or 'absent'})"
                    )
                continue
            if before != after:
                triggers.append(
                    f"{label} Version Update ({before} -> {after})"
                )

        if previous_env.model_revisions != self.current_env.model_revisions:
            triggers.append("Model Revision Change")

        return triggers
