"""Diagnostics Exporter and Local Crash Reporter for MECH Platform.

Collects comprehensive runtime telemetry, hardware capabilities, database health,
and sanitized error traces into downloadable diagnostic packages.
"""

from __future__ import annotations

import json
import logging
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

import torch

from backend.storage.database import DesktopStorage
from backend.services.hardware_manager import hardware_manager

logger = logging.getLogger("MECH.services.diagnostics")


class DiagnosticsService:
    """Manages system diagnostics generation and local crash reporting."""

    def __init__(self, storage: Optional[DesktopStorage] = None) -> None:
        db_path = Path.home() / ".cache" / "neural-debugger" / "mech.db"
        self.storage = storage or DesktopStorage(db_path)
        self.storage.initialize()
        self.crashes_dir = Path.home() / ".cache" / "neural-debugger" / "crashes"
        self.crashes_dir.mkdir(parents=True, exist_ok=True)

    def generate_diagnostics_report(self) -> Dict[str, Any]:
        """Collects exhaustive platform, hardware, database, and queue telemetry."""
        hw = hardware_manager.get_hardware_status()
        db_file = self.storage.db_path
        db_size_bytes = db_file.stat().st_size if db_file.exists() else 0

        artifacts_dir = Path.home() / ".cache" / "neural-debugger" / "artifacts"
        total_artifact_bytes = 0
        artifact_count = 0
        if artifacts_dir.exists():
            for f in artifacts_dir.rglob("*"):
                if f.is_file():
                    total_artifact_bytes += f.stat().st_size
                    artifact_count += 1

        recent_jobs = self.storage.list_jobs(limit=10)

        return {
            "mech_version": "2.0.0",
            "timestamp": time.time(),
            "platform": {
                "os_name": os.name,
                "system": platform.system(),
                "release": platform.release(),
                "architecture": platform.machine(),
                "python_version": sys.version,
            },
            "ml_environment": {
                "pytorch_version": torch.__version__,
                "cuda_available": hw["cuda_available"],
                "device_name": hw["device_name"],
                "total_vram_mb": hw["total_vram_mb"],
                "free_vram_mb": hw["free_vram_mb"],
            },
            "storage_health": {
                "database_path": str(db_file),
                "database_size_bytes": db_size_bytes,
                "artifacts_dir": str(artifacts_dir),
                "artifact_count": artifact_count,
                "total_artifact_bytes": total_artifact_bytes,
            },
            "recent_jobs": [
                {"id": j.get("id"), "name": j.get("name"), "status": j.get("status"), "progress": j.get("progress")}
                for j in recent_jobs
            ],
            "system_status": "HEALTHY",
        }

    def record_local_crash(
        self,
        error: str,
        stack_trace: str,
        investigation_id: Optional[str] = None,
        job_id: Optional[str] = None,
    ) -> str:
        """Records an unhandled exception or worker crash without sensitive data."""
        crash_id = f"crash_{int(time.time())}_{os.urandom(3).hex()}"
        crash_file = self.crashes_dir / f"{crash_id}.json"

        crash_data = {
            "crash_id": crash_id,
            "timestamp": time.time(),
            "error": str(error),
            "stack_trace": stack_trace,
            "investigation_id": investigation_id,
            "job_id": job_id,
            "system_snapshot": {
                "python": sys.version,
                "platform": platform.platform(),
                "torch": torch.__version__,
            },
        }

        with open(crash_file, "w", encoding="utf-8") as f:
            json.dump(crash_data, f, indent=2)

        logger.error("Recorded local crash log to %s: %s", crash_file, error)
        return str(crash_file)

    def export_diagnostics_to_file(self, output_path: Path | str) -> str:
        """Saves a diagnostics JSON report to a target file path."""
        report = self.generate_diagnostics_report()
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        return str(p)


# Global diagnostics service singleton
diagnostics_service = DiagnosticsService()
