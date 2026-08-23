import json
import tempfile
import zipfile
from pathlib import Path
import pytest

from backend.storage.database import DesktopStorage
from backend.storage.scientific_entities import (
    Investigation,
    Hypothesis,
    ExperimentRun,
    EvidenceRecord,
)
from backend.science.bundle_manager import ResearchBundleManager, BundleIntegrityError


def _create_valid_bundle_zip(tmpdir: Path) -> Path:
    """Helper that creates a valid exported research bundle ZIP."""
    db_path = tmpdir / "source.db"
    storage = DesktopStorage(db_path)
    storage.initialize()

    storage.save_investigation(
        Investigation(id="inv_tamper_target", title="Original Title", research_question="Testing tamper?").model_dump()
    )
    storage.save_hypothesis(
        Hypothesis(id="hyp_tamper_target", investigation_id="inv_tamper_target", title="Original Hypothesis", statement="Claim").model_dump()
    )
    storage.save_run(
        ExperimentRun(id="run_tamper_target", experiment_id="exp_1", investigation_id="inv_tamper_target", delta_logit=2.0).model_dump()
    )

    bundle_mgr = ResearchBundleManager(storage=storage)
    zip_out = tmpdir / "valid_bundle.zip"
    bundle_mgr.export_bundle("inv_tamper_target", zip_out)
    return zip_out


def test_bundle_tampered_investigation_metadata_fails():
    """Verifies that tampering with investigation.json in ZIP triggers rejection."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        tmp_root = Path(tmpdir)
        valid_zip = _create_valid_bundle_zip(tmp_root)

        # Extract, modify investigation.json, repackage
        extract_dir = tmp_root / "extracted"
        with zipfile.ZipFile(valid_zip, "r") as zf:
            zf.extractall(extract_dir)

        inv_file = extract_dir / "investigation.json"
        with open(inv_file, "r", encoding="utf-8") as f:
            inv_data = json.load(f)
        inv_data["title"] = "TAMPERED_FRAUDULENT_TITLE"
        with open(inv_file, "w", encoding="utf-8") as f:
            json.dump(inv_data, f)

        tampered_zip = tmp_root / "tampered_bundle.zip"
        with zipfile.ZipFile(tampered_zip, "w", zipfile.ZIP_DEFLATED) as zf:
            for root, _, files in Path(extract_dir).walk():
                for file in files:
                    full_p = root / file
                    rel_p = str(full_p.relative_to(extract_dir)).replace("\\", "/")
                    zf.write(full_p, rel_p)

        # Attempt to import into fresh DB
        dest_storage = DesktopStorage(tmp_root / "dest.db")
        dest_storage.initialize()
        dest_mgr = ResearchBundleManager(storage=dest_storage)

        with pytest.raises(BundleIntegrityError, match="TAMPER DETECTED in bundle file investigation.json"):
            dest_mgr.import_bundle(tampered_zip)

        # Database must remain empty
        assert dest_storage.get_investigation("inv_tamper_target") is None


def test_bundle_missing_manifest_fails():
    """Verifies that a bundle missing manifest.json is rejected immediately."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        tmp_root = Path(tmpdir)
        valid_zip = _create_valid_bundle_zip(tmp_root)

        extract_dir = tmp_root / "extracted_no_manifest"
        with zipfile.ZipFile(valid_zip, "r") as zf:
            zf.extractall(extract_dir)

        # Delete manifest.json
        (extract_dir / "manifest.json").unlink()

        no_manifest_zip = tmp_root / "no_manifest_bundle.zip"
        with zipfile.ZipFile(no_manifest_zip, "w", zipfile.ZIP_DEFLATED) as zf:
            for root, _, files in Path(extract_dir).walk():
                for file in files:
                    full_p = root / file
                    rel_p = str(full_p.relative_to(extract_dir)).replace("\\", "/")
                    zf.write(full_p, rel_p)

        dest_storage = DesktopStorage(tmp_root / "dest2.db")
        dest_storage.initialize()
        dest_mgr = ResearchBundleManager(storage=dest_storage)

        with pytest.raises(BundleIntegrityError, match="missing manifest.json"):
            dest_mgr.import_bundle(no_manifest_zip)
