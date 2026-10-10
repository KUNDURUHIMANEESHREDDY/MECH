"""SAE API surface: live where measured, empty where not.

P0 invariants, in priority order:

1. The training integration test requires ``phase == "completed"``.
   Accepting "unavailable"/"error" as green lets a broken trainer pass.
2. No import-time availability caching: each live test checks weights at
   run time and skips explicitly, so a skip is visible, never silent.
3. Train -> checkpoint file -> inspect is one verified chain, not three
   endpoints that merely share a name.
4. Unavailable responses carry no feature/metric payloads: status and
   reason are not the payload, and the payload must be absent or empty.
5. Checkpoint loading is confined (no traversal/absolute/UNC/`env:`
   identifiers from requests) and deserialised with `weights_only=True`
   (a pickle bomb must not execute).
"""

from __future__ import annotations

import hashlib
import os
import pickle
import time

import pytest


@pytest.fixture(autouse=True)
def _clean_sae_env(monkeypatch, tmp_path):
    """Deterministic environment: no ambient checkpoint config leaks in,
    and every test trains into its own directory."""
    monkeypatch.delenv("MECH_SAE_CHECKPOINT", raising=False)
    monkeypatch.delenv("MECH_SAE_SOURCE", raising=False)
    monkeypatch.setenv("MECH_SAE_DIR", str(tmp_path / "sae"))
    return tmp_path


def _weights_available() -> bool:
    try:
        from backend.services import gpt2_engine as eng
        return bool(eng.is_available())
    except Exception:
        return False


def _require_torch() -> None:
    try:
        import torch  # noqa: F401
    except Exception:
        pytest.skip("torch is not installed")


def _require_weights() -> None:
    if not _weights_available():
        pytest.skip("GPT-2 weights are not loaded; "
                    "SAE integration NOT RUN (not passed)")


def _client():
    from fastapi.testclient import TestClient
    from backend.core import auth as auth_mod
    from backend.main import app

    return TestClient(app, raise_server_exceptions=False,
                      headers=auth_mod.auth_headers())


def _no_feature_payload(body: dict) -> None:
    """Fail-closed means no feature/metric payload, not just a sad status."""
    assert body.get("status") == "unavailable"
    assert body.get("provenance") == "unavailable"
    assert body.get("reason"), "unavailable without a reason is a shrug"
    for key in ("feature_indices", "activations", "features",
                "top_activating_tokens", "observed_metrics", "result"):
        value = body.get(key)
        assert value in (None, [], {}), (
            f"unavailable response smuggles payload in {key!r}: {value!r}")


# ── status probe ──────────────────────────────────────────────────────

def test_status_probe_names_capabilities():
    from backend.contracts import models as C

    resp = _client().get("/api/sae/status")
    assert resp.status_code == 200
    obj = C.SAEStatusResponse.model_validate(resp.json())
    assert obj.status == "ok"
    assert obj.provenance == "live"
    assert isinstance(obj.torch_available, bool)
    assert isinstance(obj.gpt2_available, bool)
    assert isinstance(obj.pipeline_importable, bool)
    assert obj.checkpoint_configured is False
    assert obj.training_running is False


# ── inspect fail-closed ───────────────────────────────────────────────

def test_inspect_without_checkpoint_is_unavailable_and_empty():
    from backend.contracts import models as C

    resp = _client().post("/api/sae/inspect", json={"prompt": "Hello"})
    assert resp.status_code == 200
    body = resp.json()
    C.SAEInspectResponse.model_validate(body)
    _no_feature_payload(body)
    assert "MECH_SAE_CHECKPOINT" in body["reason"]


def test_inspect_with_missing_bare_name_is_unavailable():
    resp = _client().post("/api/sae/inspect", json={
        "prompt": "Hello", "source": "local", "identifier": "nope.pt",
    })
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "unavailable"
    assert body["provenance"] == "unavailable"
    assert body.get("feature_indices") in (None, [])


@pytest.mark.parametrize("identifier", [
    "../escape.pt",
    "../../etc/passwd",
    "/etc/passwd",
    "/nonexistent/sae.pt",
    "C:\\Windows\\sae.pt",
    "\\\\server\\share\\sae.pt",
    ".hidden.pt",
    ".",
    "..",
    "subdir/sae.pt",
    "env:MECH_SAE_CHECKPOINT",
    "env:anything",
])
def test_malicious_identifiers_are_rejected_before_loading(identifier):
    resp = _client().post("/api/sae/inspect", json={
        "prompt": "Hello", "source": "local", "identifier": identifier,
    })
    assert resp.status_code == 400, identifier


def test_corrupt_checkpoint_is_unavailable(tmp_path):
    d = tmp_path / "sae"
    d.mkdir()
    (d / "corrupt.pt").write_bytes(os.urandom(64))
    resp = _client().post("/api/sae/inspect", json={
        "prompt": "Hello", "source": "local", "identifier": "corrupt.pt",
    })
    assert resp.status_code == 200
    _no_feature_payload(resp.json())


def test_pickle_bomb_does_not_execute(tmp_path):
    """weights_only=True must refuse a hostile pickle without running it."""
    import torch

    d = tmp_path / "sae"
    d.mkdir()
    canary = d / "pwned.txt"

    class _Evil:
        def __reduce__(self):
            return (_touch_canary, (str(canary),))

    (d / "evil.pt").write_bytes(pickle.dumps(_Evil()))
    try:
        torch.load(str(d / "evil.pt"), map_location="cpu", weights_only=True)
        refused = False
    except Exception:
        refused = True
    assert refused, "hostile pickle was accepted for loading"
    assert not canary.exists(), "pickle payload executed during load"

    resp = _client().post("/api/sae/inspect", json={
        "prompt": "Hello", "source": "local", "identifier": "evil.pt",
    })
    assert resp.status_code == 200
    _no_feature_payload(resp.json())
    assert not canary.exists()


def _touch_canary(path: str) -> None:
    with open(path, "w") as handle:
        handle.write("executed")


def test_dimension_mismatch_is_unavailable(tmp_path):
    """A 128-wide encoder must never be applied to a 768/3072-wide state."""
    _require_torch()
    import torch

    d = tmp_path / "sae"
    d.mkdir()
    torch.save({
        "W_enc": torch.randn(128, 40) * 0.02,
        "b_enc": torch.zeros(40),
        "b_pre": torch.zeros(128),
        "W_dec": torch.randn(40, 128) * 0.02,
        "b_dec": torch.zeros(128),
    }, str(d / "narrow.pt"))
    resp = _client().post("/api/sae/inspect", json={
        "prompt": "Hello", "source": "local", "identifier": "narrow.pt",
    })
    assert resp.status_code == 200
    body = resp.json()
    _no_feature_payload(body)
    assert "mismatch" in body["reason"].lower() or "d_in" in body["reason"]


# ── train validation (no job started) ─────────────────────────────────

@pytest.mark.parametrize("corpus", ["", "   ", "\n\t", " \n  \n "])
def test_empty_corpus_spellings_are_unavailable(corpus):
    from backend.contracts import models as C

    resp = _client().post("/api/sae/train", json={"corpus": corpus})
    assert resp.status_code == 200
    body = resp.json()
    obj = C.SAETrainStartResponse.model_validate(body)
    assert obj.status == "unavailable"
    assert body.get("run_id") is None, "no run may start on an empty corpus"


def test_wrong_typed_corpus_is_a_client_error():
    resp = _client().post("/api/sae/train", json={"corpus": 123})
    assert resp.status_code == 400


def test_oversize_corpus_is_rejected_before_tokenisation():
    resp = _client().post("/api/sae/train",
                          json={"corpus": "x" * (200_000 + 1)})
    assert resp.status_code == 400


def test_bounds_are_clamped_before_allocation():
    """Caps are a pure function of the payload: no job is started."""
    import backend.api.dispatcher as dispatcher

    assert dispatcher._sae_clamp_params({}) == {
        "n_features": 64, "n_tokens": 1024, "steps": 100}
    assert dispatcher._sae_clamp_params({
        "n_features": 10 ** 9, "n_tokens": 10 ** 9, "steps": 10 ** 9,
    }) == {"n_features": 512, "n_tokens": 16384, "steps": 2000}
    assert dispatcher._sae_clamp_params({
        "n_features": 0, "n_tokens": 0, "steps": 0,
    }) == {"n_features": 2, "n_tokens": 64, "steps": 1}


def test_second_start_while_slot_held_is_busy():
    import backend.api.dispatcher as dispatcher

    assert dispatcher._sae_train_slot.acquire(blocking=False)
    try:
        resp = _client().post("/api/sae/train", json={"steps": 1})
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "busy"
        assert body.get("run_id") is None
    finally:
        dispatcher._sae_train_slot.release()


def test_unknown_run_id_is_unavailable():
    from backend.contracts import models as C

    resp = _client().get("/api/sae/runs/does_not_exist")
    assert resp.status_code == 200
    obj = C.SAERunStatusResponse.model_validate(resp.json())
    assert obj.status == "unavailable"


# ── live chain: train -> file -> inspect ──────────────────────────────

def _start_tiny_train(client, **overrides):
    params = {"n_features": 4, "seed": 7, "n_tokens": 64, "steps": 2}
    params.update(overrides)
    resp = client.post("/api/sae/train", json=params)
    assert resp.status_code == 200, resp.text
    return resp.json(), params


def _poll_to_terminal(client, run_id, timeout=600):
    from backend.contracts import models as C

    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        resp = client.get(f"/api/sae/runs/{run_id}")
        assert resp.status_code == 200
        last = C.SAERunStatusResponse.model_validate(resp.json())
        assert last.run_id == run_id
        if last.phase != "running":
            return last
        time.sleep(5)
    raise AssertionError(f"run {run_id} did not finish in {timeout}s")


def test_training_run_must_complete():
    """The integration test requires completion: error/unavailable fails."""
    _require_weights()
    client = _client()
    started, params = _start_tiny_train(client)
    assert started["status"] == "started", started
    assert started["run_id"].startswith("sae_")

    final = _poll_to_terminal(client, started["run_id"])
    assert final.phase == "completed", final.result
    assert final.result is not None
    # The returned run is the requested run.
    assert final.result["seed"] == params["seed"]
    assert final.result["steps"] == params["steps"]
    metrics = final.result["observed_metrics"]
    assert isinstance(metrics["reconstruction_useful"], bool)
    nmse = metrics["normalized_mse"]
    if nmse is not None:
        assert metrics["reconstruction_useful"] == (nmse < 1.0)
    assert metrics["monosemanticity_scored"] is False
    # Completed runs stay retrievable.
    again = _client().get(f"/api/sae/runs/{started['run_id']}").json()
    assert again["result"] == final.result


def test_train_inspect_chain_uses_the_trained_checkpoint():
    """Train -> checkpoint file -> inspect: one connected chain."""
    _require_weights()

    client = _client()
    started, params = _start_tiny_train(client, seed=21)
    assert started["status"] == "started", started
    final = _poll_to_terminal(client, started["run_id"])
    assert final.phase == "completed", final.result

    checkpoint = final.result.get("checkpoint", {})
    assert checkpoint.get("checkpoint_saved") is True
    path = checkpoint.get("checkpoint_path")
    assert path and os.path.isfile(path)
    with open(path, "rb") as handle:
        assert ("sha256:" + hashlib.sha256(handle.read()).hexdigest()
                == checkpoint.get("checkpoint_sha256"))
    assert checkpoint.get("checkpoint_d_sae") == params["n_features"]

    name = os.path.basename(path)
    resp = client.post("/api/sae/inspect", json={
        "prompt": "The capital of France is",
        "source": "local",
        "identifier": name,
    })
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "completed", body
    assert body["provenance"] == "live"
    assert body["d_sae"] == params["n_features"], (
        "inspected dictionary width differs from the trained one: "
        "the chain is not connected")
    assert body["weights_sha256"], "encoding unattributed to weights"
    assert body.get("feature_indices"), "no features decoded"
    assert all(isinstance(i, int) for i in body["feature_indices"])
    assert body["reconstruction_measured"] is False
    assert body["evidence_level"] == "OBSERVATIONAL"


def test_same_seed_reproduces_checkpoint_bytes():
    """Determinism: same seed + params + corpus pins the learned weights.

    The file digest is artifact identity, not reproducibility evidence:
    `torch.save` container bytes vary call to call for identical tensors
    (measured: equal shapes, equal values, equal sizes, different bytes).
    The content digest covers dtype + shape + raw tensor bytes, so it pins
    the science. Both are asserted: paths must differ (no overwrite) while
    content digests must match.
    """
    _require_weights()

    client = _client()
    first, _ = _start_tiny_train(client, seed=99)
    assert first["status"] == "started", first
    done = _poll_to_terminal(client, first["run_id"])
    assert done.phase == "completed", done.result
    content1 = done.result["checkpoint"]["checkpoint_content_sha256"]

    second, _ = _start_tiny_train(client, seed=99)
    assert second["status"] == "started", second
    assert second["run_id"] != first["run_id"], "run ids must be unique"
    done2 = _poll_to_terminal(client, second["run_id"])
    assert done2.phase == "completed", done2.result
    assert done2.result["checkpoint"]["checkpoint_path"] != \
        done.result["checkpoint"]["checkpoint_path"], \
        "runs must not overwrite each other's checkpoints"
    assert done2.result["checkpoint"]["checkpoint_content_sha256"] == content1, \
        "same seed/params/corpus produced different weights"
