import os
import tempfile
from pathlib import Path
import pytest

from backend.storage.database import DesktopStorage
from backend.storage.scientific_entities import (
    Investigation,
    Hypothesis,
    ExperimentRun,
    EvidenceRecord,
)


def test_packaged_electron_executable_integrity():
    """Validates the built MECH Platform.exe binary properties and structure."""
    repo_root = Path(__file__).resolve().parent.parent.parent
    exe_path = repo_root / "frontend" / "release" / "win-unpacked" / "MECH Platform.exe"
    
    assert exe_path.exists(), f"Packaged executable not found at {exe_path}"
    file_size_mb = exe_path.stat().st_size / (1024 * 1024)
    assert file_size_mb > 100.0, f"Executable size unexpectedly small: {file_size_mb:.1f} MB"

    # Verify resources directory and app asar
    resources_dir = exe_path.parent / "resources"
    assert resources_dir.exists(), "Resources directory missing from packaged distribution."


def test_installation_path_with_spaces_handling():
    """Validates database and storage operations when path contains spaces."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        spaced_dir = Path(tmpdir) / "MECH Platform Scientific Storage"
        spaced_dir.mkdir(parents=True, exist_ok=True)
        db_path = spaced_dir / "mech test with spaces.db"

        storage = DesktopStorage(db_path)
        storage.initialize()

        inv = Investigation(id="inv_space_test", title="Spaced Path Investigation", research_question="Testing spaces?")
        saved = storage.save_investigation(inv.model_dump())
        assert saved["id"] == "inv_space_test"

        fetched = storage.get_investigation("inv_space_test")
        assert fetched is not None
        assert fetched["title"] == "Spaced Path Investigation"


def test_cold_restart_persistence():
    """Simulates complete app close and cold reopen, ensuring 100% entity persistence."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        db_path = Path(tmpdir) / "persistent_mech.db"

        # Session 1: Create and populate
        storage_s1 = DesktopStorage(db_path)
        storage_s1.initialize()

        storage_s1.save_investigation(
            Investigation(id="inv_p1", title="Persistence Test", research_question="Does state survive?").model_dump()
        )
        storage_s1.save_hypothesis(
            Hypothesis(id="hyp_p1", investigation_id="inv_p1", title="Hypothesis 1", statement="Persistence works").model_dump()
        )
        storage_s1.save_run(
            ExperimentRun(id="run_p1", experiment_id="exp_1", investigation_id="inv_p1", delta_logit=1.85).model_dump()
        )
        storage_s1.save_evidence(
            EvidenceRecord(id="evi_p1", investigation_id="inv_p1", hypothesis_id="hyp_p1", claim="Verified", metric_name="delta_logit", metric_value=1.85).model_dump()
        )

        # Force close connections
        del storage_s1

        # Session 2: Cold reopen
        storage_s2 = DesktopStorage(db_path)
        storage_s2.initialize()

        assert storage_s2.get_investigation("inv_p1") is not None
        assert len(storage_s2.list_hypotheses("inv_p1")) == 1
        assert len(storage_s2.list_runs("inv_p1")) == 1
        assert len(storage_s2.list_evidence("inv_p1")) == 1
