"""HTTP contract for the project export (spec §19.3, §20.1).

One route, ``POST /exports/{project_id}``, mounted by the Lead at both
``/api`` and ``/api/v1`` so the frontend's ``/api/*`` and the spec's
``/api/v1/*`` reach the same implementation.

What it does and does not accept
--------------------------------

* The destination is the server's export root -- ``MECH_EXPORT_ROOT``, else
  ``~/.mech/exports``. A client cannot choose one: the body model forbids
  unknown fields, so a request carrying ``dest_dir`` is rejected rather
  than silently ignored.
* The handler is a plain ``def``. It does blocking file IO, and FastAPI
  runs a sync handler in a threadpool instead of blocking the event loop.
* A project that no stored session names is unknown. It is answered
  ``404 project_not_found`` and no file is written: an export must never
  report success for a project that does not exist.
* An export estimated above ``MECH_EXPORT_CONFIRM_BYTES`` (default 100 MiB)
  is answered ``409 confirmation_required`` with the estimate and the
  threshold, and no file is written, until the caller confirms.
* The response carries no absolute server path. The archive is named
  relative to the export root.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, NoReturn, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict

from backend.api.project_export import (
    EXPORT_VERSION,
    InvalidProjectId,
    ProjectExporter,
    ProjectNotFound,
    estimate_export,
    export_root,
)

router = APIRouter()


class ExportRequest(BaseModel):
    """The body of an export request.

    ``extra="forbid"`` because the destination is the server's export root
    and nothing else: a request naming one is asking for a path the server
    must not take, and it is refused rather than quietly dropped.
    """

    model_config = ConfigDict(extra="forbid")

    confirm: bool = False


def _fail(status_code: int, code: str, reason: str, **extra: Any) -> NoReturn:
    """Raise a structured refusal that carries a reason and no data fields."""
    detail: Dict[str, Any] = {"code": code, "reason": reason}
    detail.update(extra)
    raise HTTPException(status_code=status_code, detail=detail)


@router.post("/exports/{project_id}")
def export_project_route(project_id: str,
                         request: Optional[ExportRequest] = None
                         ) -> Dict[str, Any]:
    """Export one project's sessions as a .interp-project archive."""
    confirm = bool(request.confirm) if request is not None else False

    # (1) The slug is validated before anything is read or created: a bad
    #     project id must not touch the storage authority and must not
    #     bring the export root into existence.
    try:
        estimate = estimate_export(project_id)
    except InvalidProjectId as exc:
        _fail(400, "invalid_project_id", str(exc))
    except ValueError as exc:
        _fail(400, "invalid_export_request", str(exc))

    # (2) The estimate is computed from the same serialized payload the
    #     archive would contain, so it cannot disagree with the result.
    #
    # (3) A project no stored session names does not exist. Nothing is
    #     written for it.
    if estimate["sessions_included"] == 0:
        _fail(
            404,
            "project_not_found",
            f"no stored session names project {project_id!r}; "
            f"nothing was exported",
        )

    # (4) A large export needs confirmation before it is written.
    threshold = estimate["threshold_bytes"]
    if estimate["estimated_bytes"] > threshold and not confirm:
        _fail(
            409,
            "confirmation_required",
            f"the export of {project_id!r} is estimated at "
            f"{estimate['estimated_bytes']} bytes, above the "
            f"{threshold}-byte confirmation threshold; nothing was written",
            estimate=estimate,
            threshold_bytes=threshold,
        )

    # (5) The archive is written atomically: to a temporary file in the
    #     destination directory, then renamed into place.
    try:
        result = ProjectExporter().export_project(project_id, dest_dir=None)
    except InvalidProjectId as exc:
        _fail(400, "invalid_project_id", str(exc))
    except ProjectNotFound as exc:
        _fail(404, "project_not_found", str(exc))
    except ValueError as exc:
        _fail(400, "invalid_export_request", str(exc))
    except OSError as exc:
        _fail(500, "export_write_failed",
              f"the archive could not be written: {exc}")

    # (6) The answer names the archive relative to the export root. An
    #     absolute server path is never returned to a client.
    archive_path = Path(result["archive_path"])
    try:
        relative = archive_path.relative_to(export_root()).as_posix()
    except ValueError:
        relative = archive_path.name

    return {
        "status": "ok",
        "project_id": result["project_id"],
        "archive_name": result["archive_name"],
        "archive_path_relative": relative,
        "bytes_written": result["bytes_written"],
        "sha256": result["sha256"],
        "estimated_bytes": result["estimated_bytes"],
        "estimate_basis": result["estimate_basis"],
        "threshold_bytes": result["threshold_bytes"],
        "sessions_included": result["sessions_included"],
        "sessions_omitted_unattributed":
            result["sessions_omitted_unattributed"],
        "sessions_omitted_other_projects":
            result["sessions_omitted_other_projects"],
        "exported_at": result["exported_at"],
        "export_version": EXPORT_VERSION,
    }
