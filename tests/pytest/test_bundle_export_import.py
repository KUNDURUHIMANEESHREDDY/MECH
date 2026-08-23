import tempfile
from pathlib import Path
import pytest

from backend.storage.database import DesktopStorage
from backend.science.bundle_manager import ResearchBundleManager, BundleIntegrityError
from backend.storage.scientific_entities import (
    Investigation,
    Hypothesis,
    ExperimentRun,
    EvidenceRecord,
)


def test_bundle_export_and_import_roundtrip():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        db_path = Path(tmpdir) / "source.db"
        storage = DesktopStorage(db_path)
        storage.initialize()

        # Seed data
        inv = Investigation(id="inv_test_bundle", title="Bundle Test Investigation", research_question="Testing export?")
        storage.save_investigation(inv.model_dump())

        hyp = Hypothesis(id="hyp_test_1", investigation_id="inv_test_bundle", title="Test Hypothesis", statement="L9H9 mediates X", target_component="L9H9")
        storage.save_hypothesis(hyp.model_dump())

        run = ExperimentRun(id="run_test_1", experiment_id="exp_test_1", investigation_id="inv_test_bundle", delta_logit=2.5)
        storage.save_run(run.model_dump())

        evi = EvidenceRecord(id="evi_test_1", investigation_id="inv_test_bundle", hypothesis_id="hyp_test_1", claim="L9H9 effect is 2.5", evidence_level="CAUSALLY_VERIFIED", metric_name="delta_logit", metric_value=2.5)
        storage.save_evidence(evi.model_dump())

        bundle_mgr = ResearchBundleManager(storage=storage)
        zip_output = Path(tmpdir) / "bundle_export.zip"

        # 1. Export bundle
        exported_path = bundle_mgr.export_bundle("inv_test_bundle", zip_output)
        assert Path(exported_path).exists()
        assert Path(exported_path).stat().st_size > 0

        # 2. Import bundle into a fresh database
        dest_db = Path(tmpdir) / "dest.db"
        dest_storage = DesktopStorage(dest_db)
        dest_storage.initialize()

        dest_bundle_mgr = ResearchBundleManager(storage=dest_storage)
        restored_inv = dest_bundle_mgr.import_bundle(exported_path)

        assert restored_inv["id"] == "inv_test_bundle"
        assert restored_inv["title"] == "Bundle Test Investigation"

        # Verify all entities were reconstructed
        dest_hypotheses = dest_storage.list_hypotheses("inv_test_bundle")
        assert len(dest_hypotheses) == 1
        assert dest_hypotheses[0]["title"] == "Test Hypothesis"

        dest_runs = dest_storage.list_runs("inv_test_bundle")
        assert len(dest_runs) == 1
        assert dest_runs[0]["delta_logit"] == 2.5

        dest_evidence = dest_storage.list_evidence("inv_test_bundle")
        assert len(dest_evidence) == 1
        assert dest_evidence[0]["metric_value"] == 2.5
