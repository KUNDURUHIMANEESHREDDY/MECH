import json
import tempfile
from pathlib import Path
import pytest

from backend.storage.database import DesktopStorage
from backend.services.diagnostics_service import DiagnosticsService


def test_diagnostics_report_generation():
    """Validates structure and exhaustiveness of system telemetry report."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        storage = DesktopStorage(Path(tmpdir) / "diag.db")
        storage.initialize()
        svc = DiagnosticsService(storage=storage)

        report = svc.generate_diagnostics_report()
        assert report["mech_version"] == "2.0.0"
        assert "platform" in report
        assert "ml_environment" in report
        assert "storage_health" in report
        assert report["system_status"] == "HEALTHY"
        assert report["platform"]["python_version"] != ""


def test_diagnostics_export_to_file():
    """Verifies exporting diagnostics report to disk as JSON."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        storage = DesktopStorage(Path(tmpdir) / "diag.db")
        storage.initialize()
        svc = DiagnosticsService(storage=storage)

        out_file = Path(tmpdir) / "MECH_Diagnostics_Export.json"
        saved_path = svc.export_diagnostics_to_file(out_file)

        assert Path(saved_path).exists()
        with open(saved_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert data["mech_version"] == "2.0.0"
        assert data["storage_health"]["database_size_bytes"] >= 0


def test_local_crash_reporting():
    """Verifies that an unhandled crash writes a structured local crash log."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        storage = DesktopStorage(Path(tmpdir) / "diag.db")
        storage.initialize()
        svc = DiagnosticsService(storage=storage)
        svc.crashes_dir = Path(tmpdir) / "crashes"
        svc.crashes_dir.mkdir(parents=True, exist_ok=True)

        crash_file = svc.record_local_crash(
            error="CUDA Out of Memory in forward pass",
            stack_trace="Traceback: line 42 in runner.py",
            investigation_id="inv_crash_test",
        )

        assert Path(crash_file).exists()
        with open(crash_file, "r", encoding="utf-8") as f:
            crash_data = json.load(f)
        assert "CUDA Out of Memory" in crash_data["error"]
        assert crash_data["investigation_id"] == "inv_crash_test"
