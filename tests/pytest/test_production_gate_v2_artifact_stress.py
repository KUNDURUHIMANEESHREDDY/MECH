import tempfile
from pathlib import Path
import pytest
import torch

from backend.storage.database import DesktopStorage
from backend.science.artifact_registry import ArtifactRegistry, ArtifactStorageError


def test_large_tensor_atomic_persistence():
    """Validates atomic storage of large multi-megabyte activation tensors."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        base_dir = Path(tmpdir) / "artifacts"
        storage = DesktopStorage(Path(tmpdir) / "test.db")
        storage.initialize()
        registry = ArtifactRegistry(base_dir=base_dir, storage=storage)

        # 10 Million float32 elements (~40 MB tensor)
        large_tensor = torch.randn(100, 12, 768)
        art = registry.store_tensor_artifact(
            tensor_data=large_tensor,
            name="Large Multi-Head Activations (40MB)",
            investigation_id="inv_stress",
        )

        assert art.size_bytes > 3_000_000
        assert Path(art.file_path).exists()

        # Load and verify mathematical equality
        loaded = registry.load_tensor_artifact(art.id, verify_checksum=True)
        assert torch.allclose(large_tensor, loaded)


def test_corrupted_tensor_fail_closed():
    """Verifies that tampering with even a single byte on disk fails closed."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        base_dir = Path(tmpdir) / "artifacts"
        storage = DesktopStorage(Path(tmpdir) / "test.db")
        storage.initialize()
        registry = ArtifactRegistry(base_dir=base_dir, storage=storage)

        tensor = torch.ones(50, 50)
        art = registry.store_tensor_artifact(
            tensor_data=tensor,
            name="Tamper Target",
            investigation_id="inv_stress",
        )

        # Flip bytes in the saved tensor file
        with open(art.file_path, "r+b") as f:
            f.seek(10)
            f.write(b"\x00\xFF\xAA\x55")

        with pytest.raises(ArtifactStorageError, match="TAMPER DETECTED"):
            registry.load_tensor_artifact(art.id, verify_checksum=True)


def test_missing_tensor_file_safe_handling():
    """Verifies safe error handling when an artifact file is deleted externally."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        base_dir = Path(tmpdir) / "artifacts"
        storage = DesktopStorage(Path(tmpdir) / "test.db")
        storage.initialize()
        registry = ArtifactRegistry(base_dir=base_dir, storage=storage)

        tensor = torch.randn(10, 10)
        art = registry.store_tensor_artifact(
            tensor_data=tensor,
            name="Deletable Tensor",
            investigation_id="inv_stress",
        )

        # Remove file from filesystem
        Path(art.file_path).unlink()

        with pytest.raises(ArtifactStorageError, match="missing from disk"):
            registry.load_tensor_artifact(art.id)
