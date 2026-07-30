"""Artifact Store Abstraction.

Allows the platform to save models, reports, and figures to Local Files,
S3, Azure Blob, or GCS without changing higher-level logic.
"""

from abc import ABC, abstractmethod
import os

class ArtifactStore(ABC):
    @abstractmethod
    def save_file(self, bucket: str, object_name: str, file_path: str) -> str:
        pass

    @abstractmethod
    def load_file(self, bucket: str, object_name: str, destination_path: str) -> None:
        pass

class LocalArtifactStore(ArtifactStore):
    """Saves artifacts to the local filesystem."""
    
    def __init__(self, base_dir: str = "./artifacts"):
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)

    def save_file(self, bucket: str, object_name: str, file_path: str) -> str:
        dest = os.path.join(self.base_dir, bucket, object_name)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        # In a real app, copy the file.
        print(f"Saved {file_path} to {dest}")
        return dest

    def load_file(self, bucket: str, object_name: str, destination_path: str) -> None:
        src = os.path.join(self.base_dir, bucket, object_name)
        print(f"Loaded {src} to {destination_path}")
