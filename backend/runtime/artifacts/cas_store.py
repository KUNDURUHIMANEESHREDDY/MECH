"""Content-Addressed Storage (CAS) & Execution Artifact Store.

Provides a two-tier storage hierarchy:
- L1: Fast In-Memory LRU Cache with dynamic eviction
- L2: Persistent NVMe disk storage with zero-copy / safetensors / torch storage
- SQLite Catalog: Searchable index tracking lineage, hit counts, size, and metadata
"""

from __future__ import annotations

import collections
import hashlib
import json
import logging
import os
import sqlite3
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import torch

from .models import ArtifactMetadata, ExecutionArtifact, Provenance

logger = logging.getLogger("MECH.cas_store")

try:
    from backend.storage.migrations import migrate, MIGRATIONS, backup_db
except ImportError:
    migrate = MIGRATIONS = backup_db = None


def compute_artifact_key(
    parent_ids: List[str],
    operation: str,
    operation_params: Dict[str, Any],
    provenance_digest: str,
    layer: Optional[int] = None,
    component: str = "",
) -> str:
    """Computes a deterministic SHA-256 Content-Addressed Storage (CAS) key.
    
    The key encapsulates all dependencies, operations, hyperparameters,
    model checksums, and architectural positions.
    """
    payload = {
        "parent_ids": sorted(parent_ids),
        "operation": operation,
        "operation_params": operation_params,
        "provenance_digest": provenance_digest,
        "layer": layer,
        "component": component,
    }
    raw = json.dumps(payload, sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class L1MemoryCache:
    """In-memory LRU cache bounded by entry count and byte size."""

    def __init__(self, max_entries: int = 1000, max_bytes: int = 2 * 1024 * 1024 * 1024) -> None:  # 2GB default
        self.max_entries = max_entries
        self.max_bytes = max_bytes
        self._cache: collections.OrderedDict[str, ExecutionArtifact] = collections.OrderedDict()
        self._current_bytes: int = 0

    def get(self, artifact_id: str) -> Optional[ExecutionArtifact]:
        if artifact_id in self._cache:
            self._cache.move_to_end(artifact_id)
            return self._cache[artifact_id]
        return None

    def put(self, artifact: ExecutionArtifact) -> None:
        art_bytes = artifact.metadata.size_bytes
        # Evict until within limits
        while self._cache and (
            len(self._cache) >= self.max_entries or (self._current_bytes + art_bytes > self.max_bytes)
        ):
            _, evicted = self._cache.popitem(last=False)
            self._current_bytes -= evicted.metadata.size_bytes

        self._cache[artifact.artifact_id] = artifact
        self._current_bytes += art_bytes

    def remove(self, artifact_id: str) -> bool:
        if artifact_id in self._cache:
            art = self._cache.pop(artifact_id)
            self._current_bytes -= art.metadata.size_bytes
            return True
        return False

    def clear(self) -> None:
        self._cache.clear()
        self._current_bytes = 0

    def size(self) -> int:
        return len(self._cache)

    def total_bytes(self) -> int:
        return self._current_bytes


class ArtifactStore:
    """Two-tier Content-Addressed Execution Artifact Store with SQLite catalog."""

    def __init__(
        self,
        storage_dir: str = "backend/storage/artifacts",
        db_path: str = "backend/storage/artifacts.db",
        l1_max_entries: int = 1000,
        l1_max_bytes: int = 2 * 1024 * 1024 * 1024,
    ) -> None:
        self.storage_dir = os.path.abspath(storage_dir)
        self.db_path = os.path.abspath(db_path)
        os.makedirs(self.storage_dir, exist_ok=True)
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)

        self.l1 = L1MemoryCache(max_entries=l1_max_entries, max_bytes=l1_max_bytes)
        
        if migrate and MIGRATIONS and backup_db:
            backup_db(self.db_path)
            migrate(self.db_path, MIGRATIONS["artifacts.db"])
        
        self._init_db()

    def _init_db(self) -> None:
        """Initializes SQLite catalog for persistent indexing."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS artifacts (
                    artifact_id TEXT PRIMARY KEY,
                    name TEXT,
                    component TEXT,
                    layer INTEGER,
                    head INTEGER,
                    shape TEXT,
                    dtype TEXT,
                    device TEXT,
                    session_id TEXT,
                    prompt_id TEXT,
                    model_id TEXT,
                    operation TEXT,
                    size_bytes INTEGER,
                    storage_path TEXT,
                    parent_ids TEXT,
                    provenance_json TEXT,
                    metadata_json TEXT,
                    hit_count INTEGER DEFAULT 0,
                    created_at REAL,
                    last_accessed REAL
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_art_model ON artifacts(model_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_art_layer ON artifacts(layer)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_art_comp ON artifacts(component)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_art_session ON artifacts(session_id)")

    def put(self, artifact: ExecutionArtifact, persist_to_disk: bool = True) -> str:
        """Stores an execution artifact in L1 and optionally persists to L2 disk."""
        # Estimate byte size if not set
        if artifact.metadata.size_bytes == 0:
            if artifact.tensor is not None:
                artifact.metadata.size_bytes = artifact.tensor.element_size() * artifact.tensor.nelement()
                artifact.metadata.shape = tuple(artifact.tensor.shape)
                artifact.metadata.dtype = str(artifact.tensor.dtype)
                artifact.metadata.device = str(artifact.tensor.device)
            elif artifact.data is not None:
                artifact.metadata.size_bytes = len(json.dumps(artifact.data, default=str).encode("utf-8"))

        # Put in L1
        self.l1.put(artifact)

        # Put in L2 NVMe / Disk
        if persist_to_disk:
            file_ext = ".pt" if artifact.tensor is not None else ".json"
            file_path = os.path.join(self.storage_dir, f"{artifact.artifact_id}{file_ext}")
            artifact.storage_path = file_path

            if artifact.tensor is not None:
                # Save tensor CPU copy
                torch.save(artifact.tensor.detach().cpu(), file_path)
            elif artifact.data is not None:
                with open(file_path, "w", encoding="utf-8") as f:
                    json.dump(artifact.data, f)

        # Record in SQLite catalog
        now = time.time()
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO artifacts (
                    artifact_id, name, component, layer, head, shape, dtype, device,
                    session_id, prompt_id, model_id, operation, size_bytes, storage_path,
                    parent_ids, provenance_json, metadata_json, hit_count, created_at, last_accessed
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 
                    COALESCE((SELECT hit_count FROM artifacts WHERE artifact_id = ?), 0),
                    ?, ?)
            """, (
                artifact.artifact_id,
                artifact.metadata.name,
                artifact.metadata.component,
                artifact.metadata.layer,
                artifact.metadata.head,
                json.dumps(list(artifact.metadata.shape)),
                artifact.metadata.dtype,
                artifact.metadata.device,
                artifact.metadata.session_id,
                artifact.metadata.prompt_id,
                artifact.provenance.model_id,
                artifact.provenance.operation,
                artifact.metadata.size_bytes,
                artifact.storage_path,
                json.dumps(artifact.parent_ids),
                json.dumps(artifact.provenance.to_dict()),
                json.dumps(artifact.metadata.to_dict()),
                artifact.artifact_id,
                artifact.metadata.created_at,
                now,
            ))

        return artifact.artifact_id

    def get(self, artifact_id: str, load_tensor: bool = True) -> Optional[ExecutionArtifact]:
        """Retrieves an artifact by its CAS key. Checks L1 then falls back to L2."""
        # 1. Check L1
        art = self.l1.get(artifact_id)
        if art is not None:
            self._record_hit(artifact_id)
            return art

        # 2. Check SQLite / L2
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM artifacts WHERE artifact_id = ?", (artifact_id,))
            row = cursor.fetchone()

        if not row:
            return None

        (
            art_id, name, component, layer, head, shape_json, dtype, device,
            session_id, prompt_id, model_id, operation, size_bytes, storage_path,
            parent_ids_json, prov_json, meta_json, hit_count, created_at, last_accessed
        ) = row

        parent_ids = json.loads(parent_ids_json) if parent_ids_json else []
        prov_dict = json.loads(prov_json) if prov_json else {}
        meta_dict = json.loads(meta_json) if meta_json else {}

        provenance = Provenance(
            model_id=prov_dict.get("model_id", model_id),
            weights_digest=prov_dict.get("weights_digest", "unknown"),
            tokenizer_digest=prov_dict.get("tokenizer_digest", "unknown"),
            precision=prov_dict.get("precision", "float32"),
            device=prov_dict.get("device", device),
            operation=prov_dict.get("operation", operation),
            operation_params=prov_dict.get("operation_params", {}),
        )

        metadata = ArtifactMetadata(
            name=name,
            component=component,
            layer=layer,
            head=head,
            shape=tuple(json.loads(shape_json)) if shape_json else (),
            dtype=dtype,
            device=device,
            session_id=session_id,
            prompt_id=prompt_id,
            created_at=created_at,
            size_bytes=size_bytes,
            tags=meta_dict.get("tags", []),
            custom=meta_dict.get("custom", {}),
        )

        tensor = None
        data = None
        if load_tensor and storage_path and os.path.exists(storage_path):
            if storage_path.endswith(".pt"):
                try:
                    tensor = torch.load(storage_path, weights_only=True, map_location="cpu")
                except Exception as exc:
                    logger.warning("Failed to load tensor from %s: %s", storage_path, exc)
            elif storage_path.endswith(".json"):
                try:
                    with open(storage_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                except Exception as exc:
                    logger.warning("Failed to load data from %s: %s", storage_path, exc)

        loaded_artifact = ExecutionArtifact(
            artifact_id=art_id,
            metadata=metadata,
            provenance=provenance,
            parent_ids=parent_ids,
            tensor=tensor,
            data=data,
            storage_path=storage_path,
        )

        # Warm L1 cache
        self.l1.put(loaded_artifact)
        self._record_hit(artifact_id)
        return loaded_artifact

    def exists(self, artifact_id: str) -> bool:
        """Fast check if artifact exists in L1 or L2."""
        if self.l1.get(artifact_id) is not None:
            return True
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT 1 FROM artifacts WHERE artifact_id = ? LIMIT 1", (artifact_id,))
            return cursor.fetchone() is not None

    def search(
        self,
        model_id: Optional[str] = None,
        layer: Optional[int] = None,
        component: Optional[str] = None,
        session_id: Optional[str] = None,
        operation: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Queries the catalog with flexible filters."""
        query = "SELECT artifact_id, name, component, layer, shape, dtype, size_bytes, model_id, operation, hit_count, created_at FROM artifacts WHERE 1=1"
        params: List[Any] = []
        if model_id is not None:
            query += " AND model_id = ?"
            params.append(model_id)
        if layer is not None:
            query += " AND layer = ?"
            params.append(layer)
        if component is not None:
            query += " AND component = ?"
            params.append(component)
        if session_id is not None:
            query += " AND session_id = ?"
            params.append(session_id)
        if operation is not None:
            query += " AND operation = ?"
            params.append(operation)

        query += " ORDER BY created_at DESC"

        results = []
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            for row in cursor.fetchall():
                results.append({
                    "artifact_id": row[0],
                    "name": row[1],
                    "component": row[2],
                    "layer": row[3],
                    "shape": json.loads(row[4]) if row[4] else [],
                    "dtype": row[5],
                    "size_bytes": row[6],
                    "model_id": row[7],
                    "operation": row[8],
                    "hit_count": row[9],
                    "created_at": row[10],
                })
        return results

    def _record_hit(self, artifact_id: str) -> None:
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    "UPDATE artifacts SET hit_count = hit_count + 1, last_accessed = ? WHERE artifact_id = ?",
                    (time.time(), artifact_id)
                )
        except Exception as exc:  # noqa: BLE001
            logger.debug("Swallowed exception: %s", exc)

    def delete(self, artifact_id: str) -> bool:
        self.l1.remove(artifact_id)
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT storage_path FROM artifacts WHERE artifact_id = ?", (artifact_id,))
            row = cursor.fetchone()
            if row and row[0] and os.path.exists(row[0]):
                try:
                    os.remove(row[0])
                except OSError:
                    pass
            cursor.execute("DELETE FROM artifacts WHERE artifact_id = ?", (artifact_id,))
            return cursor.rowcount > 0

    def clear(self) -> None:
        self.l1.clear()
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM artifacts")
        if os.path.exists(self.storage_dir):
            for f in os.listdir(self.storage_dir):
                fp = os.path.join(self.storage_dir, f)
                if os.path.isfile(fp):
                    try:
                        os.remove(fp)
                    except OSError:
                        pass


_global_artifact_store: Optional[ArtifactStore] = None


def get_artifact_store() -> ArtifactStore:
    global _global_artifact_store
    if _global_artifact_store is None:
        _global_artifact_store = ArtifactStore()
    return _global_artifact_store
