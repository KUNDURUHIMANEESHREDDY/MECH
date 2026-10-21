"""Disk-backed artifact registry (spec 21.1, ``GET /artifacts/{id}``).

The rule this module exists to enforce
--------------------------------------
**An artifact's identity is the SHA-256 of the bytes on disk.** An id is not a
label someone typed, and a manifest entry is not a claim that the bytes still
match it. So:

* :meth:`ArtifactRegistry.register_bytes` hashes the payload it is about to
  write and refuses to reuse an existing object file whose bytes hash to
  something else. It never overwrites a mismatched object, because that would
  destroy the evidence the digest is supposed to point at.
* :meth:`ArtifactRegistry.read_content` re-hashes the bytes on every read and
  raises :class:`ArtifactIntegrityError` when they no longer match the manifest.
  A caller therefore cannot receive bytes carrying a digest they do not have.
* An unknown id is simply absent -- there is no placeholder record, no
  "unknown" row and no synthesized metadata.

Layout::

    <root>/manifest.json          # {"manifest_version": 1, "artifacts": {...}}
    <root>/objects/<sha256>       # the artifact's actual bytes

``<root>`` defaults to ``backend/storage/artifacts`` and is overridden with
``MECH_ARTIFACTS_DIR`` (tests point it at a temporary directory).
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Union

MANIFEST_VERSION = 1

#: Bounded read used by the HTTP layer unless a caller asks for less. Large
#: enough for a manifest, a figure or a JSON result, small enough that a request
#: cannot pull an arbitrary object into memory.
DEFAULT_MAX_CONTENT_BYTES = 1024 * 1024

#: Hard ceiling; a request for more than this is clamped. Nothing in the control
#: plane should be streaming unbounded bytes over one response.
MAX_CONTENT_BYTES = 8 * 1024 * 1024


class ArtifactError(Exception):
    """Base class for registry failures that the HTTP layer maps to a status."""

    code = "artifact_error"


class ArtifactNotFound(ArtifactError):
    code = "artifact_not_found"


class ArtifactContentMissing(ArtifactError):
    """The manifest knows this id but the object file is gone."""

    code = "artifact_content_missing"


class ArtifactTooLarge(ArtifactError):
    code = "artifact_too_large"


class ArtifactIntegrityError(ArtifactError):
    """Stored bytes no longer hash to the id/manifest digest."""

    code = "artifact_integrity_mismatch"


def default_root() -> Path:
    """Resolve the artifact directory from the environment, else the default."""
    configured = os.environ.get("MECH_ARTIFACTS_DIR")
    if configured:
        return Path(configured)
    return Path(__file__).resolve().parent.parent / "storage" / "artifacts"


def is_digest(value: Any) -> bool:
    """True when ``value`` has the shape of a full lowercase SHA-256 digest."""
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(char in "0123456789abcdef" for char in value)
    )


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_atomic(path: Path, payload: bytes) -> None:
    """Write ``payload`` to ``path`` via a sibling temp file and ``os.replace``.

    A half-written object file would hash to something other than its name, so
    the registry would carry an id whose bytes are corrupt. The temp file is in
    the destination directory so ``os.replace`` stays on one filesystem.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    except BaseException:
        try:
            os.unlink(temp_name)
        except OSError:
            pass
        raise


class ArtifactRegistry:
    """A manifest plus object files under one directory."""

    def __init__(self, root: Union[str, Path, None] = None) -> None:
        self.root = Path(root) if root is not None else default_root()
        self.manifest_path = self.root / "manifest.json"
        self.objects_dir = self.root / "objects"
        self._lock = threading.Lock()

    # ── manifest ──────────────────────────────────────────────────────────

    def _load_manifest(self) -> Dict[str, Any]:
        try:
            raw = self.manifest_path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return {"manifest_version": MANIFEST_VERSION, "artifacts": {}}
        except OSError as exc:
            raise ArtifactError(f"artifact manifest unreadable: {exc}") from exc
        try:
            data = json.loads(raw)
        except ValueError as exc:
            raise ArtifactError(
                f"artifact manifest is not valid JSON: {exc}"
            ) from exc
        if not isinstance(data, dict) or not isinstance(data.get("artifacts"), dict):
            raise ArtifactError("artifact manifest has no 'artifacts' mapping")
        return data

    def _save_manifest(self, manifest: Dict[str, Any]) -> None:
        encoded = json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8")
        _write_atomic(self.manifest_path, encoded)

    # ── write ─────────────────────────────────────────────────────────────

    def register_bytes(
        self,
        payload: Union[bytes, bytearray, memoryview],
        *,
        media_type: str = "application/octet-stream",
        label: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Write ``payload`` and record its real digest. Returns the record.

        Registering identical bytes twice is idempotent: the second call returns
        the original record and keeps its ``created_at``. That is the intended
        meaning of a content-addressed id -- the same bytes are the same
        artifact.
        """
        if not isinstance(payload, (bytes, bytearray, memoryview)):
            raise TypeError("artifact payload must be bytes")
        data = bytes(payload)
        digest = hashlib.sha256(data).hexdigest()
        record: Dict[str, Any] = {
            "id": digest,
            "sha256": digest,
            "bytes": len(data),
            "media_type": str(media_type or "application/octet-stream"),
            "label": label,
            "metadata": dict(metadata or {}),
            "created_at": _utc_now(),
            "stored_relpath": f"objects/{digest}",
        }
        try:
            json.dumps(record["metadata"])
        except (TypeError, ValueError) as exc:
            raise ValueError("artifact metadata must be JSON-serialisable") from exc

        with self._lock:
            self.objects_dir.mkdir(parents=True, exist_ok=True)
            target = self.objects_dir / digest
            if target.exists():
                existing = target.read_bytes()
                if hashlib.sha256(existing).hexdigest() != digest:
                    raise ArtifactIntegrityError(
                        f"object file for {digest} does not hash to its name; "
                        "refusing to overwrite it"
                    )
            else:
                _write_atomic(target, data)

            manifest = self._load_manifest()
            previous = manifest["artifacts"].get(digest)
            if isinstance(previous, dict) and previous.get("created_at"):
                record["created_at"] = previous["created_at"]
            manifest["artifacts"][digest] = record
            self._save_manifest(manifest)
        return dict(record)

    # ── read ──────────────────────────────────────────────────────────────

    def get(self, artifact_id: str) -> Optional[Dict[str, Any]]:
        """Return the record for ``artifact_id``, or ``None`` when unknown.

        ``content_available`` says whether the object file exists right now, so
        a caller can distinguish "registered, bytes present" from "registered,
        bytes gone" without reading the file.
        """
        if not is_digest(artifact_id):
            return None
        record = self._load_manifest()["artifacts"].get(artifact_id)
        if not isinstance(record, dict):
            return None
        result = dict(record)
        result["content_available"] = (self.objects_dir / artifact_id).is_file()
        return result

    def read_content(
        self,
        artifact_id: str,
        *,
        max_bytes: int = DEFAULT_MAX_CONTENT_BYTES,
    ) -> tuple[bytes, Dict[str, Any]]:
        """Read bounded bytes for ``artifact_id``, verifying the digest first.

        Raises :class:`ArtifactTooLarge` rather than truncating: a partial body
        with a full-file digest header would be bytes the caller cannot verify.
        """
        record = self.get(artifact_id)
        if record is None:
            raise ArtifactNotFound(f"no artifact with id {artifact_id!r}")
        try:
            limit = int(max_bytes)
        except (TypeError, ValueError):
            limit = DEFAULT_MAX_CONTENT_BYTES
        limit = max(1, min(limit, MAX_CONTENT_BYTES))

        path = self.objects_dir / artifact_id
        if not path.is_file():
            raise ArtifactContentMissing(
                f"artifact {artifact_id} is registered but its object file is "
                f"missing at {path}; no content was returned"
            )
        size = path.stat().st_size
        if size > limit:
            raise ArtifactTooLarge(
                f"artifact is {size} bytes, above the {limit}-byte read limit; "
                "no partial content was returned"
            )
        data = path.read_bytes()
        actual = hashlib.sha256(data).hexdigest()
        if actual != record.get("sha256"):
            raise ArtifactIntegrityError(
                f"artifact {artifact_id} hashes to {actual} but its manifest "
                "records a different digest; the stored bytes are not the "
                "artifact that was registered"
            )
        return data, record
