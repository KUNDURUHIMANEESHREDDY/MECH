"""Contract tests for the spec §21.1 endpoint families (``backend/api/v1_contracts.py``).

These are not smoke tests. Each one checks a number that came from somewhere
real -- the digest of bytes actually on disk, the row count of a file that was
actually read, the dimension of the model that would actually run -- or checks
that MECH *refused* the request with a structured reason. AGENTS.md is explicit
that a 200 alone proves nothing, and that a negative test passes when the
operation is correctly rejected.

Every test runs against a temporary SQLite store, a temporary dataset root, a
temporary artifact root and a temporary workspace, so the developer's real
``mech.db`` and real ``backend/research_datasets`` are never touched.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import backend.api.v1_contracts as contracts  # noqa: E402
from backend.api.artifact_registry import (  # noqa: E402
    ArtifactRegistry,
    default_root as default_artifact_root,
)

PREFIX = "/api/v1"


@pytest.fixture()
def mech(tmp_path, monkeypatch):
    """A router wired to temporary storage roots, with module caches cleared.

    ``v1_contracts`` memoises the store, the dataset managers and the artifact
    registries at module level. Clearing them here is what makes each test
    independent of the order it ran in.
    """
    workspace = tmp_path / "workspace"
    datasets = tmp_path / "datasets"
    artifacts = tmp_path / "artifacts"
    for path in (workspace, datasets, artifacts):
        path.mkdir()

    store_db = tmp_path / "store.sqlite"
    monkeypatch.setenv("MECH_WORKSPACE_ROOT", str(workspace))
    monkeypatch.setenv("MECH_DATASETS_DIR", str(datasets))
    monkeypatch.setenv("MECH_ARTIFACTS_DIR", str(artifacts))
    monkeypatch.setenv("MECH_STORAGE_DB", str(store_db))
    # An integrity bypass inherited from the developer's shell would make a
    # dataset look verified when nothing was checked.
    monkeypatch.delenv("MECH_INTEGRITY_CHECKS", raising=False)

    contracts._STORE = None
    contracts._MANAGERS.clear()
    contracts._REGISTRIES.clear()

    app = FastAPI()
    app.include_router(contracts.router, prefix=PREFIX)

    yield SimpleNamespace(
        client=TestClient(app),
        workspace=workspace,
        datasets=datasets,
        artifacts=artifacts,
        store_db=store_db,
        tmp_path=tmp_path,
    )

    contracts._STORE = None
    contracts._MANAGERS.clear()
    contracts._REGISTRIES.clear()


def _detail(response) -> dict:
    """The structured refusal body, asserting it exists and no data rode along."""
    body = response.json()
    assert isinstance(body.get("detail"), dict), f"no structured detail: {body}"
    return body["detail"]


def _write_dataset(root: Path, dataset_id: str, payload: dict) -> bytes:
    raw = json.dumps(payload).encode("utf-8")
    target = root / dataset_id
    target.mkdir(parents=True, exist_ok=True)
    (target / "dataset.json").write_bytes(raw)
    return raw


# ── GET /datasets ────────────────────────────────────────────────────────


def test_dataset_listing_is_empty_and_says_so(mech):
    response = mech.client.get(f"{PREFIX}/datasets")

    assert response.status_code == 200
    body = response.json()
    assert body["datasets"] == []
    assert body["count"] == 0
    # Nothing was invented: the response carries a status and a reason rather
    # than an illustrative row.
    assert body["status"] == "empty"
    assert body["provenance"] == "none"
    assert "nothing was invented" in body["reason"].lower()
    assert body["source"] == str(mech.datasets)


def test_dataset_listing_reports_real_counts_columns_and_digest(mech):
    prompts = [
        {"clean": "a", "corrupt": "b"},
        {"clean": "c", "corrupt": "d"},
        {"clean": "e", "corrupt": "f"},
    ]
    raw = _write_dataset(mech.datasets, "demo", {"prompts": prompts})

    body = mech.client.get(f"{PREFIX}/datasets").json()

    assert body["count"] == 1
    entry = body["datasets"][0]
    assert entry["id"] == "demo"
    assert entry["prompt_count"] == 3 == len(prompts)
    assert entry["columns"] == ["clean", "corrupt"]
    # The digest is of the bytes on disk, using the bytes the file itself
    # declares -- not a string typed into the endpoint.
    assert entry["provenance"]["sha256"] == hashlib.sha256(raw).hexdigest()
    assert entry["provenance"]["bytes"] == len(raw)
    # This dataset is not in golden_manifest.json, so nothing was verified and
    # the entry says exactly that instead of implying a pass.
    assert entry["status"] == "unregistered"
    assert entry["integrity"]["registered"] is False
    assert entry["integrity"]["integrity_verified"] is None

    # The same file previewed by id must return the rows it actually holds.
    preview = mech.client.post(
        f"{PREFIX}/datasets/preview", json={"dataset_id": "demo"}
    ).json()
    assert preview["source"]["kind"] == "dataset_id"
    assert preview["source"]["sha256"] == hashlib.sha256(raw).hexdigest()
    assert preview["rows"] == prompts
    assert preview["total_rows"] == 3


def test_registered_dataset_with_placeholder_hash_is_not_called_verified(mech):
    prompts = [{"clean": "a", "corrupt": "b"}, {"clean": "c", "corrupt": "d"}]
    _write_dataset(mech.datasets, "demo", {"prompts": prompts})
    # Placeholder hashes: the manifest names the dataset but records no digest,
    # which is "nothing to check against", not "checked and passed".
    (mech.datasets / "golden_manifest.json").write_text(
        json.dumps(
            {
                "datasets": {
                    "demo": {
                        "dataset_id": "demo",
                        "version": "1.0.0",
                        "hashes": {"bundle_hash": "", "prompt_hash": ""},
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    entry = mech.client.get(f"{PREFIX}/datasets").json()["datasets"][0]

    assert entry["status"] == "ok"
    assert entry["prompt_count"] == 2
    assert entry["integrity"]["registered"] is True
    assert entry["integrity"]["integrity_verified"] is False
    assert entry["integrity"]["checks"] == {
        "bundle_hash": "not_recorded",
        "prompt_hash": "not_recorded",
    }
    assert "not a pass" in entry["integrity"]["note"]


# ── POST /datasets/preview ───────────────────────────────────────────────


def test_dataset_preview_reads_real_bytes_and_reports_digest(mech):
    rows = [{"x": 0, "label": "row-0"}, {"x": 1, "label": "row-1"}, {"x": 2, "label": "row-2"}]
    path = mech.workspace / "ok.json"
    path.write_text(json.dumps({"prompts": rows}), encoding="utf-8")

    body = mech.client.post(
        f"{PREFIX}/datasets/preview", json={"path": "ok.json", "limit": 10}
    ).json()

    assert body["status"] == "previewed"
    assert body["rows"] == rows
    assert body["total_rows"] == len(rows)
    assert body["columns"] == ["x", "label"]
    assert body["truncated"] is False
    assert body["source"]["format"] == "json"
    assert body["source"]["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert body["source"]["bytes"] == path.stat().st_size
    assert body["provenance"]["read_bytes"] == path.stat().st_size


def test_dataset_preview_is_bounded_and_flags_truncation(mech):
    rows = [{"i": index} for index in range(25)]
    (mech.workspace / "wide.json").write_text(
        json.dumps({"rows": rows}), encoding="utf-8"
    )

    # Above the 20-row ceiling the response is clamped and says so.
    capped = mech.client.post(
        f"{PREFIX}/datasets/preview", json={"path": "wide.json", "limit": 25}
    ).json()
    assert capped["returned_rows"] == 20 == contracts.MAX_PREVIEW_ROWS
    assert capped["total_rows"] == 25
    assert capped["truncated"] is True
    assert capped["rows"] == rows[:20]
    assert capped["limits"] == {"max_rows": 20, "max_bytes": contracts.MAX_PREVIEW_BYTES}

    # A caller-supplied lower limit truncates against the real total.
    asked = mech.client.post(
        f"{PREFIX}/datasets/preview", json={"path": "wide.json", "limit": 2}
    ).json()
    assert asked["returned_rows"] == 2
    assert asked["total_rows"] == 25
    assert asked["truncated"] is True
    assert asked["rows"] == rows[:2]


def test_dataset_preview_parses_csv_columns_from_header(mech):
    (mech.workspace / "table.csv").write_text("a,b\n1,2\n3,4\n", encoding="utf-8")

    body = mech.client.post(
        f"{PREFIX}/datasets/preview", json={"path": "table.csv"}
    ).json()

    assert body["source"]["format"] == "csv"
    assert body["columns"] == ["a", "b"]
    assert body["total_rows"] == 2
    assert body["rows"] == [{"a": "1", "b": "2"}, {"a": "3", "b": "4"}]


def test_dataset_preview_refuses_path_outside_workspace(mech):
    outside = mech.tmp_path / "outside.csv"
    outside.write_text("a\n1\n", encoding="utf-8")

    for request in (
        {"path": str(outside)},          # absolute, outside the root
        {"path": "../outside.csv"},      # traversal, outside the root
        {"path": "../../outside.csv"},   # traversal, further out
    ):
        response = mech.client.post(f"{PREFIX}/datasets/preview", json=request)
        assert response.status_code == 403, request
        detail = _detail(response)
        assert detail["code"] == "outside_workspace", request
        # A refusal carries no preview.
        assert "rows" not in detail
        assert "total_rows" not in detail


def test_dataset_preview_refuses_ambiguous_or_missing_source(mech):
    ambiguous = mech.client.post(
        f"{PREFIX}/datasets/preview",
        json={"path": "ok.json", "dataset_id": "demo"},
    )
    assert ambiguous.status_code == 400
    assert _detail(ambiguous)["code"] == "exactly_one_source"

    neither = mech.client.post(f"{PREFIX}/datasets/preview", json={})
    assert neither.status_code == 400
    assert _detail(neither)["code"] == "exactly_one_source"

    missing = mech.client.post(
        f"{PREFIX}/datasets/preview", json={"path": "nope.csv"}
    )
    assert missing.status_code == 404
    assert _detail(missing)["code"] == "file_not_found"

    traversal_id = mech.client.post(
        f"{PREFIX}/datasets/preview", json={"dataset_id": "../secret"}
    )
    assert traversal_id.status_code == 400
    assert _detail(traversal_id)["code"] == "invalid_dataset_id"

    unknown_id = mech.client.post(
        f"{PREFIX}/datasets/preview", json={"dataset_id": "ghost"}
    )
    assert unknown_id.status_code == 404
    assert _detail(unknown_id)["code"] == "dataset_not_found"


def test_dataset_preview_refuses_non_previewable_input(mech):
    (mech.workspace / "notes.txt").write_text("hello", encoding="utf-8")
    unsupported = mech.client.post(
        f"{PREFIX}/datasets/preview", json={"path": "notes.txt"}
    )
    assert unsupported.status_code == 415
    assert _detail(unsupported)["code"] == "unsupported_format"

    (mech.workspace / "bad.csv").write_bytes(b"\xff\xfe\x00\x00")
    not_utf8 = mech.client.post(
        f"{PREFIX}/datasets/preview", json={"path": "bad.csv"}
    )
    assert not_utf8.status_code == 415
    assert _detail(not_utf8)["code"] == "not_utf8"

    (mech.workspace / "shape.json").write_text(json.dumps({"a": 1}), encoding="utf-8")
    no_rows = mech.client.post(
        f"{PREFIX}/datasets/preview", json={"path": "shape.json"}
    )
    assert no_rows.status_code == 415
    assert _detail(no_rows)["code"] == "no_row_collection"

    (mech.workspace / "big.json").write_text(
        json.dumps({"rows": [{"i": index} for index in range(50)]}), encoding="utf-8"
    )
    too_large = mech.client.post(
        f"{PREFIX}/datasets/preview", json={"path": "big.json", "max_bytes": 8}
    )
    assert too_large.status_code == 413
    detail = _detail(too_large)
    assert detail["code"] == "file_too_large"
    assert "rows" not in detail


# ── artifacts ────────────────────────────────────────────────────────────


def _seed_artifact(payload: bytes) -> tuple[dict, ArtifactRegistry]:
    registry = ArtifactRegistry(default_artifact_root())
    record = registry.register_bytes(payload, media_type="application/json")
    return record, registry


def test_artifact_unknown_id_is_404(mech):
    unknown = "a" * 64
    response = mech.client.get(f"{PREFIX}/artifacts/{unknown}")

    assert response.status_code == 404
    detail = _detail(response)
    assert detail["code"] == "artifact_not_found"
    # No placeholder record, no fabricated metadata.
    assert "sha256" not in detail
    assert "bytes" not in detail


def test_artifact_metadata_and_content_carry_the_real_digest(mech):
    payload = b'{"artifact": "real-bytes"}'
    record, _ = _seed_artifact(payload)
    assert record["id"] == hashlib.sha256(payload).hexdigest()

    body = mech.client.get(f"{PREFIX}/artifacts/{record['id']}").json()
    assert body["id"] == record["id"]
    assert body["sha256"] == hashlib.sha256(payload).hexdigest()
    assert body["bytes"] == len(payload)
    assert body["content_available"] is True
    assert body["status"] == "ok"
    assert body["media_type"] == "application/json"

    content = mech.client.get(f"{PREFIX}/artifacts/{record['id']}/content")
    assert content.status_code == 200
    assert content.content == payload
    assert content.headers["x-mech-artifact-sha256"] == hashlib.sha256(payload).hexdigest()
    assert content.headers["x-mech-artifact-id"] == record["id"]


def test_artifact_missing_content_is_404_not_500(mech):
    """Regression: the error mapper referenced an unimported class name.

    An ``ArtifactContentMissing`` therefore escaped as a ``NameError`` inside
    the handler -- a 500 with a traceback instead of a structured 404. This
    test fails loudly if that mapping regresses.
    """
    payload = b"gone-soon"
    record, _ = _seed_artifact(payload)
    (mech.artifacts / "objects" / record["id"]).unlink()

    metadata = mech.client.get(f"{PREFIX}/artifacts/{record['id']}").json()
    assert metadata["content_available"] is False
    assert metadata["status"] == "content_missing"
    assert "object file is missing" in metadata["reason"]

    response = mech.client.get(f"{PREFIX}/artifacts/{record['id']}/content")
    assert response.status_code == 404
    detail = _detail(response)
    assert detail["code"] == "artifact_content_missing"
    assert "no content was returned" in detail["reason"]


def test_artifact_oversized_content_is_refused(mech):
    record, _ = _seed_artifact(b"x" * 4096)

    response = mech.client.get(
        f"{PREFIX}/artifacts/{record['id']}/content", params={"max_bytes": 1}
    )

    assert response.status_code == 413
    detail = _detail(response)
    assert detail["code"] == "artifact_too_large"
    # The alternative -- truncating and sending bytes with a full-file digest
    # header -- would hand the caller content they cannot verify.
    assert "no partial content" in detail["reason"]
    assert response.content != b"x"


# ── POST /interventions/validate ─────────────────────────────────────────


def _gpt2_medium_spec():
    from backend.science.models.adapter_registry import ModelAdapterRegistry

    return ModelAdapterRegistry().get_adapter("gpt2-medium", mock_mode=True).spec


def test_intervention_unsupported_kind_is_refused_not_ok(mech):
    response = mech.client.post(
        f"{PREFIX}/interventions/validate", json={"kind": "teleport", "layer": 0}
    )

    assert response.status_code == 400
    detail = _detail(response)
    assert detail["code"] == "unsupported_intervention_kind"
    assert detail["supported"] is False
    assert detail["executed"] is False
    assert "head_patch" in detail["supported_kinds"]
    # The refusal is not dressed up as a successful preflight.
    assert detail.get("valid") is not True
    assert detail.get("status") != "ok"
    assert "checks" not in detail


def test_intervention_in_range_indices_match_catalogue_dims(mech):
    spec = _gpt2_medium_spec()

    body = mech.client.post(
        f"{PREFIX}/interventions/validate",
        json={"kind": "head_patch", "model": "gpt2-medium", "layer": 3, "head": 5},
    ).json()

    assert body["status"] == "validated"
    assert body["valid"] is True
    assert body["reasons"] == []
    assert body["kind"] == "head_patch"
    # The dimensions are the catalogue's, and the response says they were not
    # read off loaded weights.
    assert body["model"]["source"] == "configured_catalogue"
    assert body["model"]["weights_loaded"] is False
    assert body["model"]["n_layers"] == spec.num_layers
    assert body["model"]["n_heads"] == spec.num_heads
    assert body["model"]["n_positions"] == spec.context_length
    # A preflight is not an execution.
    assert body["executed"] is False


def test_intervention_out_of_range_layer_reports_valid_range(mech):
    spec = _gpt2_medium_spec()

    response = mech.client.post(
        f"{PREFIX}/interventions/validate",
        json={"kind": "layer_ablation", "model": "gpt2-medium", "layer": spec.num_layers},
    )

    assert response.status_code == 422
    detail = _detail(response)
    assert detail["code"] == "invalid_intervention_parameters"
    assert detail["valid"] is False
    assert detail["status"] == "refused"
    assert f"0..{spec.num_layers - 1}" in detail["reason"]
    bad = [check for check in detail["checks"] if check["field"] == "layer"]
    assert bad and bad[0]["valid"] is False


def test_intervention_missing_required_field_is_refused(mech):
    response = mech.client.post(
        f"{PREFIX}/interventions/validate",
        json={"kind": "head_patch", "model": "gpt2-medium", "layer": 1},
    )

    assert response.status_code == 422
    detail = _detail(response)
    assert detail["code"] == "invalid_intervention_parameters"
    assert "'head'" in detail["reason"]
    head_check = [c for c in detail["checks"] if c["field"] == "head"]
    assert head_check and head_check[0]["value"] is None
    assert head_check[0]["valid"] is False


def test_intervention_unknown_model_is_refused(mech):
    response = mech.client.post(
        f"{PREFIX}/interventions/validate",
        json={"kind": "steering", "model": "definitely-not-a-model", "layer": 0},
    )

    assert response.status_code == 400
    detail = _detail(response)
    assert detail["code"] == "unknown_model"
    assert detail["available_models"], "the refusal must name what does exist"


# ── GET /experiments/{id}/report ─────────────────────────────────────────


def test_report_unknown_experiment_is_404(mech):
    response = mech.client.get(f"{PREFIX}/experiments/ghost/report")

    assert response.status_code == 404
    detail = _detail(response)
    assert detail["code"] == "experiment_not_found"
    assert "measurements" not in detail


def test_report_separates_measured_from_unmeasured(mech):
    store = contracts.get_store()
    store.add_experiment({"id": "exp-1", "name": "demo", "run_ids": ["r-measured"]})
    store.record_run(
        run_id="r-measured",
        benchmark="ioi",
        primary_metric="accuracy",
        primary_score=0.5,
        status="ok",
        provenance="real-run",
        model_id="gpt2-small",
        detail={"experiment_id": "exp-1"},
    )
    # A NULL score is "not measured", and must never be reported as 0.0.
    store.record_run(
        run_id="r-unmeasured",
        benchmark="ioi",
        primary_metric="accuracy",
        primary_score=None,
        status="not_executed",
        provenance="real-run",
        detail={"experiment_id": "exp-1"},
    )
    # A run belonging to another experiment must not leak into this report.
    store.record_run(
        run_id="r-other",
        benchmark="ioi",
        primary_metric="accuracy",
        primary_score=0.9,
        status="ok",
        provenance="real-run",
        detail={"experiment_id": "exp-2"},
    )

    body = mech.client.get(f"{PREFIX}/experiments/exp-1/report").json()

    assert body["experiment_id"] == "exp-1"
    assert body["status"] == "ok"
    assert body["run_count"] == 2
    run_ids = {row["run_id"] for row in body["runs"]}
    assert run_ids == {"r-measured", "r-unmeasured"}

    assert [row["run_id"] for row in body["measurements"]] == ["r-measured"]
    assert body["measurements"][0]["score"] == 0.5
    assert body["measurements"][0]["metric"] == "accuracy"
    assert body["measurements"][0]["model_id"] == "gpt2-small"

    unmeasured = body["not_measured"]
    assert [row["run_id"] for row in unmeasured] == ["r-unmeasured"]
    assert "score" not in unmeasured[0]
    serialized = json.dumps(body)
    assert '"score": 0.0' not in serialized
    assert "0.0" not in json.dumps(unmeasured)

    # The report reproduces rows; it does not compute values.
    assert body["provenance"]["values_computed"] is False
    assert body["experiment"]["id"] == "exp-1"


def test_report_without_runs_is_empty_not_illustrative(mech):
    contracts.get_store().add_experiment({"id": "exp-lonely", "name": "unstarted"})

    body = mech.client.get(f"{PREFIX}/experiments/exp-lonely/report").json()

    assert body["status"] == "no_runs_recorded"
    assert body["run_count"] == 0
    assert body["runs"] == []
    assert body["measurements"] == []
    assert body["not_measured"] == []
    assert any("empty rather than illustrative" in note for note in body["notes"])
    assert body["provenance"]["values_computed"] is False


def test_report_flags_a_truncated_run_list(mech):
    """A partial report must not read as a complete one."""
    store = contracts.get_store()
    run_ids = [f"r-{index:02d}" for index in range(51)]
    store.add_experiment({"id": "exp-many", "name": "wide", "run_ids": run_ids})
    for run_id in run_ids:
        store.record_run(
            run_id=run_id,
            benchmark="ioi",
            primary_metric="accuracy",
            primary_score=0.1,
            status="ok",
            provenance="real-run",
            detail={"experiment_id": "exp-many"},
        )

    body = mech.client.get(f"{PREFIX}/experiments/exp-many/report").json()

    assert body["run_count"] == 50
    # The 51st named run was dropped, and the response says so instead of
    # presenting fifty of fifty-one as the whole experiment.
    assert body["runs_truncated"] is True


def test_intervention_query_failure_is_reported_not_assumed(mech, monkeypatch):
    """``gpt2_engine.info()`` raising must not be read as 'no model loaded'.

    Treating the exception as \"nothing is loaded\" would let a catalogue check
    be mistaken for a check against the weights in memory.
    """
    import backend.services.gpt2_engine as gpt2_engine

    def boom():
        raise RuntimeError("engine unavailable")

    monkeypatch.setattr(gpt2_engine, "info", boom)

    response = mech.client.post(
        f"{PREFIX}/interventions/validate",
        json={"kind": "head_patch", "model": "gpt2-medium", "layer": 3, "head": 5},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["model"]["source"] == "configured_catalogue"
    assert body["model"]["weights_loaded"] is False
    assert "engine unavailable" in body["model"]["loaded_model_query_error"]
    assert "loaded-model query failed" in body["model"]["note"]
    assert body["executed"] is False
