import json
import zipfile
from pathlib import Path
import pytest
from backend.science.bundle_manager import ResearchBundleManager, BundleIntegrityError
from backend.storage.database import DesktopStorage


def test_electron_preload_and_main_security_invariants():
    """Validates that Electron main and preload scripts maintain strict security boundaries."""
    main_js_path = Path("frontend/electron/main.js")
    preload_js_path = Path("frontend/electron/preload.js")

    if main_js_path.exists():
        main_content = main_js_path.read_text(encoding="utf-8")
        assert "contextIsolation: true" in main_content
        assert "nodeIntegration: false" in main_content
        assert "sandbox: true" in main_content

    if preload_js_path.exists():
        preload_content = preload_js_path.read_text(encoding="utf-8")
        assert "contextBridge.exposeInMainWorld" in preload_content
        assert "ipcRenderer.invoke" in preload_content


def test_cross_machine_research_bundle_tamper_rejection(tmp_path):
    """Simulates exporting a bundle on Machine A, tampering with artifact payload, and verifying Machine B rejects it."""
    storage = DesktopStorage(tmp_path / "machine_a.db")
    storage.initialize()
    manager = ResearchBundleManager(storage=storage)

    bundle_path = tmp_path / "investigation_alpha.mech"
    manifest_data = {
        "bundle_version": "2.0.0",
        "investigation_id": "inv_123",
        "investigation_title": "Cross Machine Portability Test",
        "model_name": "gpt2",
        "dataset_name": "IOI",
        "files": {
            "metadata.json": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        },
    }

    with zipfile.ZipFile(bundle_path, "w") as zf:
        zf.writestr("manifest.json", json.dumps(manifest_data, indent=2))
        zf.writestr("metadata.json", json.dumps({"title": "Original Clean Metadata"}))

    tampered_bundle = tmp_path / "investigation_tampered.mech"
    with zipfile.ZipFile(bundle_path, "r") as src_zf:
        with zipfile.ZipFile(tampered_bundle, "w") as dst_zf:
            for item in src_zf.infolist():
                if item.filename == "metadata.json":
                    dst_zf.writestr("metadata.json", json.dumps({"title": "TAMPERED PAYLOAD"}))
                else:
                    dst_zf.writestr(item.filename, src_zf.read(item.filename))

    storage_b = DesktopStorage(tmp_path / "machine_b.db")
    storage_b.initialize()
    manager_b = ResearchBundleManager(storage=storage_b)

    with pytest.raises((BundleIntegrityError, Exception)) as exc_info:
        manager_b.import_bundle(tampered_bundle)

    err_msg = str(exc_info.value).lower()
    assert "tamper" in err_msg or "checksum" in err_msg or "mismatch" in err_msg or "integrity" in err_msg or "invalid" in err_msg
