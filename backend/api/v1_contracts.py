"""Contract endpoints from spec §21.1 (datasets, artifacts, interventions, reports).

Why this module exists
----------------------
The frontend calls ``/api/*`` and the spec documents ``/api/v1/*``. Neither
prefix had these four families: datasets, artifacts, intervention preflight and
per-experiment reports. This router supplies them from the same stores the rest
of the platform already uses, and it never fills a gap with placeholder data.

Rules it enforces (AGENTS.md)
-----------------------------
* ``GET /datasets`` reads the real ``DatasetManager`` and the real
  ``dataset.json`` files, hashes them, and reports missing hashes as
  ``not_recorded`` rather than as a pass. An empty directory yields an empty
  list with an explicit status -- never an illustrative row.
* ``POST /datasets/preview`` refuses any path that resolves outside the
  workspace and any non-UTF-8 / non-row file, with a structured 4xx. It returns
  at most 20 rows read from the file's own bytes, with that file's digest.
* ``GET /artifacts/{id}`` is served by :mod:`backend.api.artifact_registry`; the
  digest is of bytes actually written, an unknown id is a 404, and content is
  only returned when the object file exists *and* re-hashes correctly.
* ``POST /interventions/validate`` is a preflight, not an execution. It reports
  ``executed: false``, validates indices against the loaded model's real config
  or the adapter catalogue, and refuses an unsupported kind with reasons
  instead of answering "ok".
* ``GET /experiments/{id}/report`` reproduces stored records only. A run whose
  ``primary_score`` is NULL is listed under ``not_measured``; it is never
  reported as a zero.

The Lead mounts this router at both ``/api`` and ``/api/v1``, so paths here are
declared plain.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, NoReturn, Optional, Tuple

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from backend.api.artifact_registry import (
    DEFAULT_MAX_CONTENT_BYTES,
    MAX_CONTENT_BYTES,
    ArtifactContentMissing,
    ArtifactError,
    ArtifactIntegrityError,
    ArtifactNotFound,
    ArtifactRegistry,
    ArtifactTooLarge,
    default_root as default_artifact_root,
)

router = APIRouter()

#: Rows a preview may return. Spec 21.1 caps this at 20; callers asking for more
#: are clamped and told so via ``truncated``.
MAX_PREVIEW_ROWS = 20

#: Columns a preview may return, so a wide JSON object cannot produce a
#: response larger than the row bound suggests.
MAX_PREVIEW_COLUMNS = 50

#: Bytes a preview may read. Above this the request is refused rather than
#: silently truncated, because a truncated file is not a preview of the file.
MAX_PREVIEW_BYTES = 8 * 1024 * 1024

_PREVIEW_SUFFIXES = {".json", ".jsonl", ".ndjson", ".csv", ".tsv"}

#: Intervention kinds that a real engine path implements, with the index fields
#: each one reads. Keys are canonical; ``aliases`` accept the dispatcher route
#: spellings so a client can preflight exactly what it is about to call.
_INTERVENTION_KINDS: Dict[str, Dict[str, Any]] = {
    "head_patch": {"requires": ("layer", "head"), "aliases": ("patch_head",)},
    "neuron_patch": {
        "requires": ("layer", "neuron"),
        "aliases": ("patch_neuron",),
    },
    "layer_ablation": {"requires": ("layer",), "aliases": ("ablate_layer",)},
    "steering": {"requires": ("layer",), "aliases": ("steer",)},
}

_KIND_ALIASES: Dict[str, str] = {
    alias: canonical
    for canonical, spec in _INTERVENTION_KINDS.items()
    for alias in spec["aliases"]
}


# ── shared helpers ────────────────────────────────────────────────────────


def _fail(status_code: int, code: str, reason: str, **extra: Any) -> NoReturn:
    """Raise a structured refusal that carries a reason and no data fields."""
    detail: Dict[str, Any] = {"code": code, "reason": reason}
    detail.update(extra)
    raise HTTPException(status_code=status_code, detail=detail)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ── storage / dataset roots ───────────────────────────────────────────────

_STORE = None
_MANAGERS: Dict[str, Any] = {}
_REGISTRIES: Dict[str, ArtifactRegistry] = {}


def get_store():
    """The SQLite store the rest of the control plane writes experiments to."""
    global _STORE
    if _STORE is None:
        from backend.storage.database import DesktopStorage, get_default_db_path

        store = DesktopStorage(get_default_db_path())
        store.initialize()
        _STORE = store
    return _STORE


def datasets_dir() -> Path:
    configured = os.environ.get("MECH_DATASETS_DIR")
    if configured:
        return Path(configured)
    return Path(__file__).resolve().parent.parent / "research_datasets"


def _dataset_manager(directory: Path):
    key = str(directory)
    manager = _MANAGERS.get(key)
    if manager is None:
        from backend.research_datasets.dataset_manager import DatasetManager

        manager = DatasetManager(str(directory))
        _MANAGERS[key] = manager
    return manager


def _registry() -> ArtifactRegistry:
    root = default_artifact_root()
    key = str(root)
    registry = _REGISTRIES.get(key)
    if registry is None:
        registry = ArtifactRegistry(root)
        _REGISTRIES[key] = registry
    return registry


def _workspace_root() -> Path:
    """The single directory dataset previews may read from.

    Configured root wins. The default is the repository root, not ``backend/``:
    datasets, prompt files and capture configs live beside the code, so a
    boundary that stopped at ``backend/`` would refuse the files the feature
    exists for. It is still one confined root -- every path is resolved and
    checked against it -- and an operator running from a different workspace
    sets ``MECH_WORKSPACE_ROOT``.
    """
    configured = os.environ.get("MECH_WORKSPACE_ROOT")
    if configured:
        return Path(configured).resolve()
    return Path(__file__).resolve().parents[2]


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


# ── GET /datasets ────────────────────────────────────────────────────────


def _columns(rows: List[Dict[str, Any]], limit: int = MAX_PREVIEW_COLUMNS):
    """First-seen union of row keys, bounded, preserving a stable order."""
    seen: List[str] = []
    for row in rows:
        for key in row:
            text = str(key)
            if text not in seen:
                seen.append(text)
                if len(seen) >= limit:
                    return seen
    return seen


def _integrity_summary(status: Dict[str, Any], registered: bool) -> Dict[str, Any]:
    """Describe verification honestly instead of calling every load 'verified'."""
    checks = status.get("checks") if isinstance(status, dict) else None
    if not registered:
        return {
            "registered": False,
            "integrity_verified": None,
            "checks": {},
            "note": (
                "This dataset is not in golden_manifest.json, so no manifest "
                "hash existed to verify. Nothing was verified; the file was "
                "read directly and nothing was invented."
            ),
        }
    states = list(checks.values()) if isinstance(checks, dict) else []
    if states and all(state == "not_recorded" for state in states):
        return {
            "registered": True,
            "integrity_verified": False,
            "checks": dict(checks),
            "note": (
                "No manifest hash was populated for this dataset (every check "
                "is 'not_recorded'), so the load verified nothing. That is a "
                "gap in the manifest, not a pass."
            ),
        }
    return {
        "registered": True,
        "integrity_verified": bool(status.get("integrity_verified")),
        "checks": dict(checks) if isinstance(checks, dict) else {},
        "bypassed": bool(status.get("bypassed")),
        "note": (
            "Integrity checks were bypassed by MECH_INTEGRITY_CHECKS; this "
            "data must not support a scientific claim."
            if status.get("bypassed")
            else None
        ),
    }


def _describe_dataset(directory: Path, dataset_json: Path) -> Dict[str, Any]:
    try:
        raw = dataset_json.read_bytes()
    except OSError as exc:
        return {
            "id": directory.name,
            "path": str(dataset_json),
            "status": "error",
            "reason": f"{type(exc).__name__}: {exc}",
            "prompt_count": None,
            "columns": [],
            "integrity": {"registered": None, "integrity_verified": None},
        }

    entry: Dict[str, Any] = {
        "id": directory.name,
        "path": str(dataset_json),
        "provenance": {
            "kind": "on-disk",
            "sha256": _sha256_bytes(raw),
            "bytes": len(raw),
            "read_at": _utc_now(),
        },
    }

    manager = _dataset_manager(directory.parent)
    try:
        prompts = manager.load(directory.name)
    except Exception as exc:  # reported, never hidden
        try:
            data = json.loads(raw.decode("utf-8-sig"))
            direct_rows = data.get("prompts") if isinstance(data, dict) else None
        except (UnicodeDecodeError, ValueError):
            direct_rows = None
        rows = direct_rows if isinstance(direct_rows, list) else []
        readable = isinstance(direct_rows, list)
        entry.update(
            {
                # "unregistered" is a distinct, honest state from "error": the
                # bytes parsed and the row count is real, but no manifest hash
                # exists, so nothing about this dataset was verified.
                "status": "unregistered" if readable else "error",
                "reason": f"{type(exc).__name__}: {exc}"[:500],
                "prompt_count": len(rows) if readable else None,
                "columns": (
                    _columns([r for r in rows if isinstance(r, dict)])
                    if readable
                    else []
                ),
                "integrity": _integrity_summary({}, registered=False),
                "note": (
                    "DatasetManager refused this dataset, so it is not usable "
                    "for evaluation. The counts above come from reading the "
                    "file directly and are marked unverified."
                ),
            }
        )
        return entry

    rows = [row for row in prompts if isinstance(row, dict)]
    entry.update(
        {
            "status": "ok",
            "prompt_count": len(prompts),
            "columns": _columns(rows),
            "integrity": _integrity_summary(
                getattr(manager, "last_integrity_status", {}) or {},
                registered=True,
            ),
        }
    )
    return entry


@router.get("/datasets")
def list_datasets() -> Dict[str, Any]:
    """List datasets that exist on disk, with their real digests and counts.

    Nothing is listed that is not a directory containing ``dataset.json``. An
    empty or absent directory returns an empty list and says why; it never
    yields a sample row.
    """
    directory = datasets_dir()
    if not directory.is_dir():
        return {
            "datasets": [],
            "count": 0,
            "source": str(directory),
            "provenance": "none",
            "status": "empty",
            "reason": (
                f"No dataset directory at {directory}. Nothing was listed and "
                "nothing was invented; create a dataset to see it here."
            ),
            "field_provenance": {"datasets": "live", "reason": "live"},
        }

    entries: List[Dict[str, Any]] = []
    for child in sorted(directory.iterdir(), key=lambda item: item.name.lower()):
        dataset_json = child / "dataset.json"
        if child.is_dir() and dataset_json.is_file():
            entries.append(_describe_dataset(child, dataset_json))

    if not entries:
        return {
            "datasets": [],
            "count": 0,
            "source": str(directory),
            "provenance": "none",
            "status": "empty",
            "reason": (
                f"{directory} exists but holds no <dataset>/dataset.json. "
                "Nothing was listed and nothing was invented."
            ),
            "field_provenance": {"datasets": "live", "reason": "live"},
        }

    return {
        "datasets": entries,
        "count": len(entries),
        "source": str(directory),
        "provenance": "on-disk",
        "status": "ok",
        "note": (
            "Each entry was read from its dataset.json; provenance.sha256 is "
            "the digest of the bytes actually read."
        ),
        "field_provenance": {
            "datasets": "live",
            "prompt_count": "live",
            "integrity": "live",
        },
    }


# ── POST /datasets/preview ───────────────────────────────────────────────


class DatasetPreviewRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    path: Optional[str] = Field(
        default=None, description="Workspace-relative or in-workspace absolute path"
    )
    dataset_id: Optional[str] = Field(default=None, description="Listed dataset id")
    limit: int = Field(
        default=10,
        ge=1,
        description=(
            "Rows to return. Values above the 20-row ceiling are clamped, and "
            "'truncated'/'limits' in the response say so."
        ),
    )
    max_bytes: int = Field(
        default=MAX_PREVIEW_BYTES, ge=1, le=MAX_PREVIEW_BYTES
    )


def _resolve_workspace_path(raw: str) -> Path:
    """Resolve a client path, refusing anything outside the workspace.

    ``resolve(strict=False)`` runs before existence is checked so a traversal
    attempt is rejected as out-of-scope (403) rather than leaking whether the
    target exists (404).
    """
    root = _workspace_root()
    if not raw or not raw.strip():
        _fail(400, "invalid_path", "path must be a non-empty string")
    candidate = Path(raw.strip())
    if not candidate.is_absolute():
        candidate = root / candidate
    try:
        resolved = candidate.resolve(strict=False)
    except (OSError, RuntimeError) as exc:
        _fail(400, "path_unresolvable", f"path could not be resolved: {exc}")

    if not resolved.is_relative_to(root):
        _fail(
            403,
            "outside_workspace",
            f"{resolved} is outside the workspace root {root}. The path was "
            "refused and no bytes were read.",
        )
    if not resolved.exists():
        _fail(404, "file_not_found", f"no file exists at {resolved}")
    if resolved.is_dir():
        _fail(400, "not_a_file", f"{resolved} is a directory, not a file")
    if not resolved.is_file():
        _fail(400, "not_a_regular_file", f"{resolved} is not a regular file")
    return resolved


def _parse_rows(raw: bytes, suffix: str) -> Tuple[List[Dict[str, Any]], int, str]:
    """Parse file bytes into row dicts. Returns (rows, total_rows, format)."""
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        _fail(415, "not_utf8", f"file is not valid UTF-8 ({exc}); no preview exists")

    if suffix == ".json":
        try:
            data = json.loads(text)
        except ValueError as exc:
            _fail(422, "invalid_json", f"file is not valid JSON: {exc}")
        rows: Optional[List[Any]] = None
        if isinstance(data, list):
            rows = data
        elif isinstance(data, dict):
            for key in ("prompts", "rows", "data"):
                if isinstance(data.get(key), list):
                    rows = data[key]
                    break
        if rows is None:
            _fail(
                415,
                "no_row_collection",
                "JSON was parsed but holds no row collection (a list, or an "
                "object with a 'prompts', 'rows' or 'data' list). No preview "
                "was produced from a guessed shape.",
            )
        format_name = "json"
    elif suffix in (".jsonl", ".ndjson"):
        rows = []
        for number, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except ValueError as exc:
                _fail(
                    422,
                    "invalid_jsonl",
                    f"line {number} is not valid JSON: {exc}",
                )
        format_name = "jsonl"
    else:
        delimiter = "\t" if suffix == ".tsv" else ","
        reader = csv.DictReader(io.StringIO(text), delimiter=delimiter)
        if not reader.fieldnames:
            _fail(415, "missing_header", "delimited file has no header row")
        rows = list(reader)
        format_name = "tsv" if suffix == ".tsv" else "csv"

    normalized: List[Dict[str, Any]] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            _fail(
                422,
                "row_not_object",
                f"row {index} is not an object; a row collection of scalars "
                "cannot be previewed as columns",
            )
        normalized.append({str(key): value for key, value in row.items()})
    return normalized, len(normalized), format_name


def _preview_file(
    path: Path,
    *,
    request: DatasetPreviewRequest,
    source_kind: str,
    source_request: str,
) -> Dict[str, Any]:
    max_bytes = int(request.max_bytes)
    try:
        size = path.stat().st_size
    except OSError as exc:
        _fail(400, "file_unreadable", f"could not stat {path}: {exc}")
    if size > max_bytes:
        _fail(
            413,
            "file_too_large",
            f"{path} is {size} bytes, above the {max_bytes}-byte preview "
            "limit. No bytes were returned rather than a truncated preview.",
        )
    try:
        raw = path.read_bytes()
    except OSError as exc:
        _fail(400, "file_unreadable", f"could not read {path}: {exc}")

    suffix = path.suffix.lower()
    if suffix not in _PREVIEW_SUFFIXES:
        _fail(
            415,
            "unsupported_format",
            f"{suffix or 'no extension'} is not a previewable format "
            f"({', '.join(sorted(_PREVIEW_SUFFIXES))})",
        )

    rows, total_rows, format_name = _parse_rows(raw, suffix)
    limit = max(1, min(int(request.limit), MAX_PREVIEW_ROWS))
    returned = rows[:limit]
    digest = _sha256_bytes(raw)

    return {
        "status": "previewed",
        "source": {
            "kind": source_kind,
            "request": source_request,
            "path": str(path),
            "format": format_name,
            "sha256": digest,
            "bytes": len(raw),
        },
        "columns": _columns(rows),
        "rows": returned,
        "returned_rows": len(returned),
        "total_rows": total_rows,
        "truncated": total_rows > len(returned),
        "limits": {"max_rows": MAX_PREVIEW_ROWS, "max_bytes": max_bytes},
        "provenance": {
            "kind": "file",
            "sha256": digest,
            "read_bytes": len(raw),
            "note": (
                "These rows are a bounded read of the file's own bytes; "
                "nothing was generated or interpolated."
            ),
        },
    }


@router.post("/datasets/preview")
def preview_dataset(request: DatasetPreviewRequest) -> Dict[str, Any]:
    """Preview real bytes from a workspace file or a listed dataset id."""
    if bool(request.path) == bool(request.dataset_id):
        _fail(
            400,
            "exactly_one_source",
            "supply exactly one of 'path' or 'dataset_id'; a request naming "
            "both or neither is ambiguous",
        )

    if request.dataset_id is not None:
        dataset_id = request.dataset_id.strip()
        if (
            not dataset_id
            or dataset_id in (".", "..")
            or "/" in dataset_id
            or "\\" in dataset_id
        ):
            _fail(
                400,
                "invalid_dataset_id",
                f"{request.dataset_id!r} is not a listed dataset id",
            )
        directory = datasets_dir()
        candidate = (directory / dataset_id / "dataset.json").resolve(strict=False)
        if not candidate.is_relative_to(directory.resolve()):
            _fail(
                403,
                "outside_datasets_dir",
                f"{candidate} is outside {directory}; the id was refused",
            )
        if not candidate.is_file():
            _fail(
                404,
                "dataset_not_found",
                f"no dataset {dataset_id!r} at {candidate}; nothing was previewed",
            )
        return _preview_file(
            candidate,
            request=request,
            source_kind="dataset_id",
            source_request=dataset_id,
        )

    resolved = _resolve_workspace_path(str(request.path))
    return _preview_file(
        resolved,
        request=request,
        source_kind="path",
        source_request=str(request.path),
    )


# ── artifacts ────────────────────────────────────────────────────────────


def _registry_error(exc: ArtifactError) -> NoReturn:
    status = {
        ArtifactNotFound: 404,
        ArtifactContentMissing: 404,
        ArtifactTooLarge: 413,
        ArtifactIntegrityError: 409,
    }.get(type(exc), 503)
    _fail(status, exc.code, str(exc))


@router.get("/artifacts/{artifact_id}")
def get_artifact(artifact_id: str) -> Dict[str, Any]:
    """Metadata for a registered artifact; 404 when the id is unknown."""
    try:
        record = _registry().get(artifact_id)
    except ArtifactError as exc:
        _registry_error(exc)
    if record is None:
        _fail(
            404,
            "artifact_not_found",
            f"no artifact with id {artifact_id!r}; the registry carries no "
            "placeholder record",
        )
    payload = dict(record)
    payload["status"] = "ok" if record.get("content_available") else "content_missing"
    if not record.get("content_available"):
        payload["reason"] = (
            "The manifest records this artifact but its object file is "
            "missing; no content can be served."
        )
    return payload


@router.get("/artifacts/{artifact_id}/content")
def get_artifact_content(
    artifact_id: str, max_bytes: int = DEFAULT_MAX_CONTENT_BYTES
) -> Any:
    """Bounded content for an existing artifact, digest-verified before send."""
    from fastapi import Response

    try:
        data, record = _registry().read_content(
            artifact_id, max_bytes=max_bytes
        )
    except ArtifactError as exc:
        _registry_error(exc)
    return Response(
        content=data,
        media_type=str(record.get("media_type") or "application/octet-stream"),
        headers={
            "X-MECH-Artifact-Id": str(record.get("id")),
            "X-MECH-Artifact-Sha256": str(record.get("sha256")),
            "Content-Disposition": (
                f'attachment; filename="{record.get("id")}.bin"'
            ),
        },
    )


# ── POST /interventions/validate ─────────────────────────────────────────


class InterventionValidateRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    kind: str = Field(description="Intervention kind, e.g. head_patch")
    model: Optional[str] = Field(
        default=None, description="Model id; the loaded model when omitted"
    )
    layer: Optional[int] = None
    head: Optional[int] = None
    neuron: Optional[int] = None
    position: Optional[int] = Field(
        default=None, description="Token position, checked against context length"
    )
    prompt: Optional[str] = None


def _shape_from_catalogue(model_name: str) -> Dict[str, Any]:
    from backend.science.models.adapter_registry import ModelAdapterRegistry

    registry = ModelAdapterRegistry()
    try:
        adapter = registry.get_adapter(model_name, mock_mode=True)
    except ValueError:
        available = []
        try:
            for row in registry.list_adapters():
                available.append(str(row.get("model_id")))
        except Exception:  # pragma: no cover - report what we have
            available = []
        _fail(
            400,
            "unknown_model",
            f"{model_name!r} is not in the adapter registry; no dimensions "
            "exist to validate against",
            available_models=sorted(available)[:50],
        )
    spec = adapter.spec
    return {
        "requested": model_name,
        "model_id": spec.model_id,
        "source": "configured_catalogue",
        "weights_loaded": False,
        "n_layers": int(spec.num_layers),
        "n_heads": int(spec.num_heads),
        "d_model": int(spec.d_model),
        "d_mlp": int(spec.d_mlp) if spec.d_mlp is not None else None,
        "n_positions": int(spec.context_length),
        "note": (
            "Dimensions come from the adapter catalogue. No weights were "
            "loaded for this model, so this is a configured-model check."
        ),
    }


def _resolve_model_shape(model_name: Optional[str]) -> Dict[str, Any]:
    """Dimensions for the loaded model, else the configured catalogue.

    ``gpt2_engine.info()`` is a cheap read that returns an error dict when no
    model is loaded; it never triggers a load. Only when a model *is* loaded and
    the request names that model do its real config dimensions get used.
    """
    requested = (model_name or "gpt2-small").strip() or "gpt2-small"
    loaded: Optional[Dict[str, Any]] = None
    loaded_query_error: Optional[str] = None
    try:
        from backend.services import gpt2_engine

        info = gpt2_engine.info()
    except Exception as exc:
        # A failure here is not evidence that no model is loaded. Reporting it
        # as "no model" would let a catalogue check pass itself off as a check
        # against real weights, so the failure travels with the response.
        info = {}
        loaded_query_error = f"{type(exc).__name__}: {exc}"
    if isinstance(info, dict) and info.get("status") == "loaded":
        loaded = {
            "requested": requested,
            "model_id": info.get("model_name") or "gpt2",
            "source": "loaded",
            "weights_loaded": True,
            "n_layers": int(info["n_layers"]),
            "n_heads": int(info["n_heads"]),
            "d_model": int(info["d_model"]),
            "d_mlp": int(info["d_mlp"]),
            "n_positions": int(info.get("n_positions") or 1024),
            "note": "Dimensions read from the loaded model's config.",
        }

    normalized = requested.lower()
    loaded_names = {"gpt2", "gpt2-small"}
    if loaded is not None and (
        normalized in loaded_names
        or normalized == str(loaded["model_id"]).lower()
    ):
        return loaded
    shape = _shape_from_catalogue(requested)
    if loaded_query_error:
        shape["loaded_model_query_error"] = loaded_query_error
        shape["note"] = (
            "Dimensions come from the adapter catalogue. The loaded-model query "
            f"failed, so no weights were consulted: {loaded_query_error}"
        )
    return shape


@router.post("/interventions/validate")
def validate_intervention(request: InterventionValidateRequest) -> Dict[str, Any]:
    """Preflight an intervention against the model that would execute it.

    Nothing runs here -- ``executed`` is always false and the response says so.
    An unsupported kind returns ``supported: false`` with reasons, never a bare
    ``ok``; an out-of-range index returns the valid range.
    """
    raw_kind = (request.kind or "").strip().lower()
    canonical = _KIND_ALIASES.get(raw_kind, raw_kind)
    if canonical not in _INTERVENTION_KINDS:
        _fail(
            400,
            "unsupported_intervention_kind",
            f"{request.kind!r} is not an intervention this engine implements. "
            "Nothing was validated and nothing will run.",
            supported=False,
            supported_kinds=sorted(_INTERVENTION_KINDS),
            executed=False,
        )

    shape = _resolve_model_shape(request.model)
    spec = _INTERVENTION_KINDS[canonical]
    checks: List[Dict[str, Any]] = []
    reasons: List[str] = []

    for field in spec["requires"]:
        value = getattr(request, field)
        if value is None:
            checks.append(
                {"field": field, "value": None, "valid": False,
                 "reason": f"'{field}' is required for a {canonical} preflight"}
            )
            reasons.append(f"missing required field '{field}'")
            continue

        if field == "layer":
            upper = shape["n_layers"]
            label = f"layer must be in 0..{upper - 1} ({upper} layers)"
        elif field == "head":
            upper = shape["n_heads"]
            label = f"head must be in 0..{upper - 1} ({upper} heads per layer)"
        else:  # neuron
            upper = shape.get("d_mlp")
            if upper is None:
                checks.append(
                    {"field": field, "value": value, "valid": False,
                     "reason": (
                         "the model catalogue records no MLP width for this "
                         "model, so a neuron index cannot be checked"
                     )}
                )
                reasons.append("neuron index uncheckable: no recorded d_mlp")
                continue
            label = f"neuron must be in 0..{upper - 1} ({upper} neurons)"

        if 0 <= value < upper:
            checks.append(
                {"field": field, "value": value, "valid": True, "reason": None}
            )
        else:
            checks.append(
                {"field": field, "value": value, "valid": False, "reason": label}
            )
            reasons.append(f"{field}={value} is outside this model: {label}")

    if request.position is not None:
        upper = shape["n_positions"]
        if 0 <= request.position < upper:
            checks.append(
                {"field": "position", "value": request.position, "valid": True,
                 "reason": None}
            )
        else:
            checks.append(
                {"field": "position", "value": request.position, "valid": False,
                 "reason": f"position must be in 0..{upper - 1}"}
            )
            reasons.append(
                f"position={request.position} is outside the model's context "
                f"length ({upper} positions)"
            )

    payload: Dict[str, Any] = {
        "status": "validated" if not reasons else "refused",
        "supported": True,
        "valid": not reasons,
        "kind": canonical,
        "requested_kind": request.kind,
        "model": shape,
        "checks": checks,
        "reasons": reasons,
        "executed": False,
        "note": (
            "Preflight only: this endpoint compares the request with the model's "
            "dimensions. It did not run an intervention or measure an effect."
        ),
    }

    if reasons:
        _fail(
            422,
            "invalid_intervention_parameters",
            "; ".join(reasons),
            **payload,
        )
    return payload


# ── GET /experiments/{experiment_id}/report ──────────────────────────────


def _experiment_payload(store, experiment_id: str) -> Optional[Dict[str, Any]]:
    for item in store.list_experiments():
        if isinstance(item, dict) and str(item.get("id")) == experiment_id:
            return item
    return None


def _explicit_run_ids(experiment: Dict[str, Any]) -> List[str]:
    """Run ids the stored experiment record itself names, in first-seen order."""
    found: List[str] = []
    values: List[Any] = []
    single = experiment.get("run_id")
    if isinstance(single, str):
        values.append(single)
    for key in ("run_ids", "runs"):
        value = experiment.get(key)
        if isinstance(value, list):
            values.extend(value)
    for value in values:
        candidate = value
        if isinstance(value, dict):
            candidate = value.get("run_id") or value.get("id")
        if isinstance(candidate, str) and candidate.strip():
            candidate = candidate.strip()
            if candidate not in found:
                found.append(candidate)
    return found


@router.get("/experiments/{experiment_id}/report")
def experiment_report(experiment_id: str) -> Dict[str, Any]:
    """Assemble a report from stored experiments and stored runs only.

    Every number in ``measurements`` was copied from a ``benchmark_runs`` row
    whose ``primary_score`` is not NULL. A run with no score is listed under
    ``not_measured``; it is not rendered as zero.
    """
    from backend.storage.database import StorageError

    store = get_store()
    experiment = _experiment_payload(store, experiment_id)
    if experiment is None:
        _fail(
            404,
            "experiment_not_found",
            f"no stored experiment with id {experiment_id!r}; no report was "
            "generated",
        )

    runs: List[Dict[str, Any]] = []
    unresolved: List[Dict[str, str]] = []
    seen: set[str] = set()

    named_run_ids = _explicit_run_ids(experiment)
    # Dropping run ids past the cap without saying so would let a partial report
    # read as a complete one, so the cut is recorded in ``runs_truncated``.
    linkage_truncated = len(named_run_ids) > 50
    for run_id in named_run_ids[:50]:
        try:
            row = store._get_run(run_id)
        except StorageError as exc:
            unresolved.append({"run_id": run_id, "reason": str(exc)})
            continue
        if row["run_id"] not in seen:
            runs.append(row)
            seen.add(row["run_id"])

    try:
        recent = store.history(limit=500)
    except StorageError as exc:
        recent = []
        unresolved.append(
            {"run_id": "*", "reason": f"run history unavailable: {exc}"}
        )
    for row in recent:
        detail = row.get("detail")
        if (
            isinstance(detail, dict)
            and detail.get("experiment_id") == experiment_id
            and row["run_id"] not in seen
        ):
            runs.append(row)
            seen.add(row["run_id"])

    if len(runs) > 50:
        runs = runs[:50]
        linkage_truncated = True

    measurements = [
        {
            "run_id": row["run_id"],
            "metric": row.get("primary_metric"),
            "score": row["primary_score"],
            "provenance": row.get("provenance"),
            "status": row.get("status"),
            "model_id": row.get("model_id"),
            "weights_fingerprint": row.get("weights_fingerprint"),
            "git_commit": row.get("git_commit"),
        }
        for row in runs
        if row.get("primary_score") is not None
    ]
    not_measured = [
        {
            "run_id": row["run_id"],
            "metric": row.get("primary_metric"),
            "reason": (
                "stored primary_score is NULL; the run recorded no measurement "
                "of this metric, so no value is reported"
            ),
        }
        for row in runs
        if row.get("primary_score") is None
    ]

    notes = [
        "This report reproduces stored rows; it computed no values.",
        "measurements contains only runs whose stored primary_score is not "
        "NULL. Absent scores are listed under not_measured, never as zeros.",
    ]
    if not runs:
        notes.append(
            "No run in the store references this experiment (by 'run_ids' on "
            "the experiment record or 'detail.experiment_id' on a run). The "
            "report is therefore empty rather than illustrative."
        )
    if len(recent) >= 500:
        # The linkage scan reads at most 500 recent runs. Saying only "no run"
        # when the window was full would present an unscanned store as an empty
        # one, so the ceiling is reported instead.
        notes.append(
            "The run-linkage scan is capped at the 500 most recent runs and "
            f"that cap was reached ({len(recent)} rows). A run older than the "
            "window may reference this experiment and would not appear above."
        )

    return {
        "experiment_id": experiment_id,
        "status": "ok" if runs else "no_runs_recorded",
        "provenance": {
            "kind": "stored_records",
            "store": str(getattr(store, "db_path", "")),
            "generated_at": _utc_now(),
            "experiment_source": "experiments table (payload reproduced verbatim)",
            "runs_source": "benchmark_runs table",
            "values_computed": False,
        },
        "experiment": experiment,
        "runs": runs,
        "run_count": len(runs),
        "measurements": measurements,
        "not_measured": not_measured,
        "unresolved_run_ids": unresolved,
        "runs_truncated": linkage_truncated,
        "notes": notes,
    }
