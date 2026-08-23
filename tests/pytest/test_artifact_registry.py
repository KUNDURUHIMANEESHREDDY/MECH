import tempfile
from pathlib import Path
import pytest
import torch

from backend.storage.database import DesktopStorage
from backend.science.artifact_registry import ArtifactRegistry, ArtifactStorageError


def test_artifact_two_phase_atomic_write_and_load():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        base_dir = Path(tmpdir) / "artifacts"
        db_path = Path(tmpdir) / "test.db"
        storage = DesktopStorage(db_path)
        storage.initialize()

        registry = ArtifactRegistry(base_dir=base_dir, storage=storage)

        # Create dummy tensor
        t = torch.randn(4, 12, 64)
        art = registry.store_tensor_artifact(
            tensor_data=t,
            name="Activation Layer 9",
            investigation_id="inv_test",
            experiment_run_id="run_test",
        )

        assert art.id.startswith("art_")
        assert Path(art.file_path).exists()
        assert not Path(art.file_path).name.endswith(".tmp")
        assert art.checksum_sha256 != ""
        assert art.size_bytes > 0

        # Load and verify equality
        loaded = registry.load_tensor_artifact(art.id, verify_checksum=True)
        assert torch.allclose(t, loaded)


def test_artifact_tamper_detection():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        base_dir = Path(tmpdir) / "artifacts"
        db_path = Path(tmpdir) / "test.db"
        storage = DesktopStorage(db_path)
        storage.initialize()

        registry = ArtifactRegistry(base_dir=base_dir, storage=storage)

        t = torch.randn(2, 2)
        art = registry.store_tensor_artifact(
            tensor_data=t,
            name="Tamper Test Tensor",
            investigation_id="inv_test",
        )

        # Deliberately tamper with the file on disk
        with open(art.file_path, "wb") as f:
            f.write(b"CORRUPTED_TAMPERED_BYTES")

        # Loading must raise ArtifactStorageError for checksum mismatch!
        with pytest.raises(ArtifactStorageError, match="TAMPER DETECTED"):
            registry.load_tensor_artifact(art.id, verify_checksum=True)


def test_artifact_prune_temporary_files():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        base_dir = Path(tmpdir) / "artifacts"
        db_path = Path(tmpdir) / "test.db"
        storage = DesktopStorage(db_path)
        storage.initialize()

        registry = ArtifactRegistry(base_dir=base_dir, storage=storage)

        # Create leftover .tmp file
        fake_tmp = registry.tensors_dir / "art_leftover.tmp"
        fake_tmp.write_bytes(b"garbage")
        assert fake_tmp.exists()

        pruned = registry.prune_temporary_files()
        assert pruned >= 1
        assert not fake_tmp.exists()
