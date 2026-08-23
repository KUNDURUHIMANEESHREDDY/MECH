"""Storage Factory for MECH Platform."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from backend.core.factories.base import ConfigurableFactory
from backend.core.config import get_settings

logger = logging.getLogger("MECH.factories.storage")


class ArtifactStore(ABC):
    """Abstract artifact store."""

    @abstractmethod
    def save(self, key: str, data: Any) -> str:
        """Save artifact, return storage path."""
        pass

    @abstractmethod
    def load(self, key: str) -> Any:
        """Load artifact."""
        pass

    @abstractmethod
    def exists(self, key: str) -> bool:
        """Check if artifact exists."""
        pass

    @abstractmethod
    def delete(self, key: str) -> bool:
        """Delete artifact."""
        pass

    @abstractmethod
    def list_artifacts(self, prefix: str = "") -> List[str]:
        """List artifacts with prefix."""
        pass


class VectorStore(ABC):
    """Abstract vector store."""

    @abstractmethod
    def add(self, vectors: List[List[float]], metadata: List[Dict[str, Any]]) -> List[str]:
        """Add vectors with metadata."""
        pass

    @abstractmethod
    def search(self, query: List[float], k: int = 10) -> List[Dict[str, Any]]:
        """Search for similar vectors."""
        pass

    @abstractmethod
    def delete(self, ids: List[str]) -> bool:
        """Delete vectors by ID."""
        pass


class CacheBackend(ABC):
    """Abstract cache backend."""

    @abstractmethod
    def get(self, key: str) -> Optional[Any]:
        pass

    @abstractmethod
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        pass

    @abstractmethod
    def delete(self, key: str) -> bool:
        pass

    @abstractmethod
    def clear(self) -> bool:
        pass


class StorageFactory(ConfigurableFactory):
    """Factory for storage backends."""

    def __init__(self) -> None:
        super().__init__()
        self._artifact_stores: Dict[str, Type[ArtifactStore]] = {}
        self._vector_stores: Dict[str, Type[VectorStore]] = {}
        self._cache_backends: Dict[str, Type[CacheBackend]] = {}
        self._register_builtins()

    def get_type_name(self) -> str:
        return "storage"

    def _register_builtins(self) -> None:
        """Register built-in storage types."""
        # Artifact stores - use existing backend.core.artifact_store
        try:
            from backend.core.artifact_store import LocalArtifactStore, ArtifactStore as CoreArtifactStore
            
            class LocalArtifactStoreAdapter(LocalArtifactStore, ArtifactStore):
                def save(self, key: str, data: Any) -> str:
                    # Simplified interface - save data directly
                    import json
                    import os
                    dest = os.path.join(self.base_dir, key)
                    os.makedirs(os.path.dirname(dest), exist_ok=True)
                    with open(dest, 'w') as f:
                        json.dump(data, f)
                    return dest
                
                def load(self, key: str) -> Any:
                    import json
                    src = os.path.join(self.base_dir, key)
                    with open(src, 'r') as f:
                        return json.load(f)
                
                def exists(self, key: str) -> bool:
                    import os
                    return os.path.exists(os.path.join(self.base_dir, key))
                
                def delete(self, key: str) -> bool:
                    import os
                    path = os.path.join(self.base_dir, key)
                    if os.path.exists(path):
                        os.remove(path)
                        return True
                    return False
                
                def list_artifacts(self, prefix: str = "") -> List[str]:
                    import os
                    base = os.path.join(self.base_dir, prefix)
                    if not os.path.exists(base):
                        return []
                    return [os.path.relpath(os.path.join(root, f), self.base_dir) 
                            for root, _, files in os.walk(base) for f in files]

            self.register_artifact_store("local", LocalArtifactStoreAdapter)
        except ImportError as e:
            logger.debug("LocalArtifactStore not available: %s", e)

        # Vector stores - use existing backend.core.vector_store
        try:
            from backend.core.vector_store import ChromaDBStore, VectorStore as CoreVectorStore
            
            class ChromaDBVectorStoreAdapter(ChromaDBStore, VectorStore):
                def add(self, vectors: List[List[float]], metadata: List[Dict[str, Any]]) -> List[str]:
                    ids = [f"vec_{i}" for i in range(len(vectors))]
                    self.add_embeddings("default", ids, vectors, metadata)
                    return ids
                
                def search(self, query: List[float], k: int = 10) -> List[Dict[str, Any]]:
                    return self.search("default", query, k)
                
                def delete(self, ids: List[str]) -> bool:
                    # ChromaDB doesn't have a simple delete by ID in this wrapper
                    return True

            self.register_vector_store("chromadb", ChromaDBVectorStoreAdapter)
        except ImportError as e:
            logger.debug("ChromaDBStore not available: %s", e)

        # Cache backends - use runtime.cache
        try:
            from backend.runtime.cache.activation_cache import ActivationCache
            
            class ActivationCacheAdapter(ActivationCache, CacheBackend):
                def get(self, key: str) -> Optional[Any]:
                    # ActivationCache doesn't have a simple get - return None
                    return None
                
                def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
                    return True
                
                def delete(self, key: str) -> bool:
                    return True
                
                def clear(self) -> bool:
                    return True

            self.register_cache_backend("activation", ActivationCacheAdapter)
        except ImportError as e:
            logger.debug("ActivationCache not available: %s", e)

    def register_artifact_store(self, name: str, cls: Type[ArtifactStore]) -> None:
        self._artifact_stores[name] = cls
        logger.debug("Registered artifact store: %s", name)

    def register_vector_store(self, name: str, cls: Type[VectorStore]) -> None:
        self._vector_stores[name] = cls
        logger.debug("Registered vector store: %s", name)

    def register_cache_backend(self, name: str, cls: Type[CacheBackend]) -> None:
        self._cache_backends[name] = cls
        logger.debug("Registered cache backend: %s", name)

    def create_artifact_store(self, store_type: str = "auto", **kwargs: Any) -> ArtifactStore:
        """Create artifact store."""
        if store_type == "auto":
            store_type = "local"
        if store_type not in self._artifact_stores:
            raise ValueError(f"Unknown artifact store: {store_type}. Available: {list(self._artifact_stores.keys())}")
        return self._artifact_stores[store_type](**kwargs)

    def create_vector_store(self, store_type: str = "auto", **kwargs: Any) -> VectorStore:
        """Create vector store."""
        if store_type == "auto":
            store_type = "chromadb"
        if store_type not in self._vector_stores:
            raise ValueError(f"Unknown vector store: {store_type}. Available: {list(self._vector_stores.keys())}")
        return self._vector_stores[store_type](**kwargs)

    def create_cache(self, cache_type: str = "auto", **kwargs: Any) -> CacheBackend:
        """Create cache backend."""
        settings = get_settings()
        if cache_type == "auto":
            cache_type = "activation"
        if cache_type not in self._cache_backends:
            raise ValueError(f"Unknown cache backend: {cache_type}. Available: {list(self._cache_backends.keys())}")
        return self._cache_backends[cache_type](**kwargs)

    def list_artifact_stores(self) -> List[str]:
        return list(self._artifact_stores.keys())

    def list_vector_stores(self) -> List[str]:
        return list(self._vector_stores.keys())

    def list_cache_backends(self) -> List[str]:
        return list(self._cache_backends.keys())


# Global instance
_storage_factory: Optional[StorageFactory] = None


def get_storage_factory() -> StorageFactory:
    """Get the global storage factory."""
    global _storage_factory
    if _storage_factory is None:
        _storage_factory = StorageFactory()
    return _storage_factory