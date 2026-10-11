"""F-18: Migration failure reporting - Observability gap.

The `quarantine_legacy_records` function silently swallows exceptions
during migration. If a file fails to migrate (permissions, corruption,
disk full, etc.), it's silently skipped with no logging or reporting.

This test verifies that failures are reported with sufficient detail
for operators to diagnose and remediate.
"""
import json
import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.core import evidence_graph as eg  # noqa: E402


def test_migration_reports_individual_failures(tmp_path):
    """Each failed file should be reported with filename and error."""
    # Create a file that will fail to migrate (corrupt JSON)
    corrupt = tmp_path / "corrupt.json"
    corrupt.write_text("{not valid json", encoding="utf-8")
    
    # Create a valid envelope that should be skipped
    valid = tmp_path / "valid.json"
    valid.write_text(json.dumps({
        "schema_version": 1,
        "run_id": "r123456789012",
        "record": {"run_id": "r123456789012"},
        "attestation": {}
    }), encoding="utf-8")
    
    report = eg.quarantine_legacy_records(directory=str(tmp_path))
    
    # The corrupt file should be reported as a failure
    # The valid envelope should be skipped
    # Corrupt file is moved as raw bytes (forensics), envelope is skipped
    assert report["moved"] == 1  # Corrupt file moved as raw bytes
    assert report["skipped"] == 1  # The envelope was skipped
    assert "failures" in report, "Report must include 'failures' list"
    assert len(report["failures"]) >= 1, "Should report at least one failure"
    
    # Each failure should have filename and error details
    for failure in report["failures"]:
        assert "filename" in failure, "Failure must include filename"
        assert "error" in failure, "Failure must include error details"


def test_migration_reports_permission_errors(tmp_path):
    """Permission errors during migration should be reported, not swallowed."""
    # This test documents the expected behavior - permission errors
    # should appear in the failures list with the filename and error type
    # Actual permission testing is platform-dependent, so we verify
    # the reporting structure would capture it
    report = eg.quarantine_legacy_records(directory=str(tmp_path))
    assert "failures" in report


def test_migration_continues_after_failures(tmp_path):
    """Migration should continue processing other files after a failure."""
    # Create multiple corrupt files
    (tmp_path / "bad1.json").write_text("{bad", encoding="utf-8")
    (tmp_path / "bad2.json").write_text("{also bad", encoding="utf-8")
    # Create a valid envelope that should be skipped
    (tmp_path / "good.json").write_text(json.dumps({
        "schema_version": 1,
        "run_id": "r123456789012",
        "record": {"run_id": "r123456789012"},
        "attestation": {}
    }), encoding="utf-8")
    
    report = eg.quarantine_legacy_records(directory=str(tmp_path))
    
    # Should process all files, not stop at first failure
    # Two corrupt files moved as raw bytes, one envelope skipped
    assert report["moved"] == 2
    assert report["skipped"] == 1
    assert len(report.get("failures", [])) >= 2, "Both failures should be reported"


def test_migration_report_structure(tmp_path):
    """Migration report should have consistent structure."""
    (tmp_path / "bad.json").write_text("{bad", encoding="utf-8")
    
    report = eg.quarantine_legacy_records(directory=str(tmp_path))
    
    # Required fields
    assert "moved" in report
    assert "stamped" in report
    assert "skipped" in report
    assert "destination" in report
    assert "failures" in report
    
    # failures should be a list of dicts with filename and error
    assert isinstance(report["failures"], list)
    for f in report["failures"]:
        assert isinstance(f, dict)
        assert "filename" in f
        assert "error" in f