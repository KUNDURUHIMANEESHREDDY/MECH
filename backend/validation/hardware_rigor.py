import logging
import platform
import psutil
import subprocess
import os
from typing import Dict, Any, List, Optional

logger = logging.getLogger("MECH.validation.hardware_rigor")

class HardwarePassport:
    """
    Captures a comprehensive 'Hardware Passport' for the environment.
    Includes hardware specs, driver versions, and library state.
    """

    @staticmethod
    def capture_passport() -> Dict[str, Any]:
        passport = {
            "os": f"{platform.system()} {platform.release()}",
            "cpu": platform.processor(),
            "ram_gb": round(psutil.virtual_memory().total / (1024**3), 2),
            "python": platform.python_version(),
            "gpu": HardwarePassport._get_gpu_info(),
            "libraries": HardwarePassport._get_lib_versions(),
            "vcs": HardwarePassport._get_vcs_state()
        }
        return passport

    @staticmethod
    def _get_gpu_info() -> Dict[str, Any]:
        info = {"status": "None"}
        try:
            import torch
            if torch.cuda.is_available():
                info = {
                    "name": torch.cuda.get_device_name(0),
                    "driver": "unknown", # default
                    "cuda_version": torch.version.cuda,
                    "cudnn_version": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else "N/A"
                }
                # Try getting driver via nvidia-smi
                try:
                    smi = subprocess.check_output(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader,nounits"]).decode().strip()
                    info["driver"] = smi
                except subprocess.CalledProcessError as exc:
                    logger.debug("nvidia-smi unavailable: %s", exc)
            return info
        except ImportError as exc:
            logger.debug("torch unavailable: %s", exc)
            return info

    @staticmethod
    def _get_lib_versions() -> Dict[str, Any]:
        libs = {}
        for lib in ["torch", "transformers", "transformer_lens", "sae_lens"]:
            try:
                m = __import__(lib.replace("-", "_"))
                # Some packages don't expose __version__; fall back to importlib.metadata
                version = getattr(m, "__version__", None)
                if version is None:
                    try:
                        from importlib.metadata import version as _pkg_version
                        version = _pkg_version(lib)
                    except Exception as exc:
                        logger.debug("Could not resolve package version via importlib.metadata: %s", exc)
                libs[lib] = version
            except ImportError:
                libs[lib] = "not_installed"
        return libs

    @staticmethod
    def _get_vcs_state() -> Dict[str, Any]:
        state = {"git_sha": "unknown", "is_dirty": False}
        try:
            state["git_sha"] = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
            dirty = subprocess.check_output(["git", "status", "--porcelain"]).decode().strip()
            state["is_dirty"] = len(dirty) > 0
        except (subprocess.CalledProcessError, FileNotFoundError, OSError) as exc:
            logger.debug("Could not get git state: %s", exc)
        return state


class HardwareRigorTracker:
    """
    Tracks hardware passports across reproduction runs and flags
    hardware-related rigor concerns (e.g. differing devices, untracked code).
    """

    def __init__(self):
        self._baseline: Optional[Dict[str, Any]] = None
        self._history: List[Dict[str, Any]] = []

    def set_baseline(self, passport: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Capture (or accept) a passport and store it as the rigor baseline."""
        self._baseline = passport if passport is not None else HardwarePassport.capture_passport()
        return self._baseline

    def record_run(self, passport: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Record a reproduction run's passport and append it to history."""
        captured = passport if passport is not None else HardwarePassport.capture_passport()
        self._history.append(captured)
        return captured

    def assess(self, current: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Compare a passport (or the latest recorded run) against the baseline.
        Returns a report describing which rigor-relevant dimensions match.
        """
        passport = current if current is not None else (
            self._history[-1] if self._history else HardwarePassport.capture_passport()
        )
        baseline = self._baseline if self._baseline is not None else passport

        report: Dict[str, Any] = {"checks": {}, "overall_status": "PENDING"}

        # Hardware / device dimensions worth comparing
        dimensions = {
            "GPU": lambda p: p.get("gpu", {}).get("name", "None"),
            "CUDA Version": lambda p: p.get("gpu", {}).get("cuda_version", "N/A"),
            "CPU": lambda p: p.get("cpu", ""),
            "Python": lambda p: p.get("python", ""),
        }

        for label, getter in dimensions.items():
            report["checks"][label] = "MATCH" if getter(passport) == getter(baseline) else "DIFFERS"

        # Flag untracked / dirty code as a rigor risk
        vcs_dirty = passport.get("vcs", {}).get("is_dirty", False)
        report["checks"]["Code Tracked"] = "CLEAN" if not vcs_dirty else "DIRTY"

        # Version-pinned core libraries should be present
        libs = passport.get("libraries", {})
        report["checks"]["Core Libraries"] = (
            "PINNED" if all(v != "not_installed" for v in libs.values()) else "MISSING"
        )

        if all(v == "MATCH" for v in report["checks"].values() if v in ("MATCH", "CLEAN", "PINNED")) \
                and not any(v in ("DIFFERS", "DIRTY", "MISSING") for v in report["checks"].values()):
            report["overall_status"] = "RIGOROUS"

        return report
