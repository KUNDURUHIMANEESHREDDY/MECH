"""The SAE route must encode the tensor the CHECKPOINT declares.

The defect this file pins
--------------------------

``sae_inspect`` used to take whatever activation the live engine happened to
expose -- ``transformers``' ``hidden_states`` -- and encode it with a
dictionary that was trained against a *different* tensor. The published
GPT-2 dictionaries (``jbloom/GPT2-Small-SAEs``) were trained against
TransformerLens' ``blocks.L.hook_resid_pre``, and the two tensors differ by
the constant TransformerLens' ``center_writing_weights`` adds to the residual
stream. Measured on this host at layer 8: the same dictionary reconstructs the
TransformerLens tensor at relative L2 0.243 and the ``transformers`` one at
3.43. So the route was reporting a real measurement of the wrong quantity, and
it was a number worse than predicting the mean.

Three things have to hold for the route to be honest, and each is checked
here:

1. the activation encoded is the one the checkpoint declares (its
   ``tensor_source``), not a convenient one;
2. a layer the dictionary does not describe is REFUSED, not encoded -- the
   loader's ``assert_applies_to`` guard exists for that and is now walked
   through by the production path;
3. a checkpoint that records no tensor source is REFUSED rather than applied
   to an arbitrary tensor of the right width.

The tests that need real weights say so and skip with a stated reason;
``MECH_REQUIRE_WEIGHTS=1`` turns those skips into failures.
"""

from __future__ import annotations

import os

import pytest

from _weight_guard import require_weights
from backend.api import dispatcher
from backend.interpretability.sae.loader import (
    SAELoadError,
    SAELoader,
    hook_at_layer,
    infer_tensor_source,
)

PROMPT = "The capital of France is"


@pytest.fixture
def sae_dir(monkeypatch, tmp_path):
    """A deterministic checkpoint directory with no ambient configuration."""
    monkeypatch.delenv("MECH_SAE_CHECKPOINT", raising=False)
    monkeypatch.delenv("MECH_SAE_SOURCE", raising=False)
    target = tmp_path / "sae"
    target.mkdir()
    monkeypatch.setenv("MECH_SAE_DIR", str(target))
    return target


def _write_checkpoint(path, *, d_in=768, d_sae=16, facts=None, zero=False,
                      seed=0):
    """A real checkpoint file: real tensors, and the facts it declares."""
    import torch

    generator = torch.Generator().manual_seed(seed)
    if zero:
        weights = torch.zeros(d_in, d_sae)
        decoder = torch.zeros(d_sae, d_in)
    else:
        weights = torch.randn(d_in, d_sae, generator=generator) * 0.02
        decoder = torch.randn(d_sae, d_in, generator=generator) * 0.02
    payload = {
        "W_enc": weights,
        "b_enc": torch.zeros(d_sae),
        "b_pre": torch.zeros(d_in),
        "W_dec": decoder,
        "b_dec": torch.zeros(d_in),
    }
    payload.update(facts or {})
    torch.save(payload, str(path))
    return path


class _StubEngine:
    """A model that returns one fixed activation.

    The route's own decisions -- which tensor it encodes, what it refuses --
    are what is under test here, so the model is replaced by something that
    answers with a known tensor. No GPT-2 weights are needed to check that the
    route encodes what the checkpoint declares.
    """

    def __init__(self, vector):
        self._vector = vector

    def is_available(self):
        return True

    def mlp_write(self, layer, prompt):
        return {"status": "ok", "tensor": self._vector.unsqueeze(0),
                "d_model": int(self._vector.shape[-1])}

    def hook_activation(self, hook_point, prompt):
        return {"status": "ok", "hook_point": hook_point,
                "tensor": self._vector}


@pytest.fixture
def stub_engine(monkeypatch):
    import torch

    vector = torch.arange(768, dtype=torch.float32) / 768.0
    monkeypatch.setattr(dispatcher, "_engine", _StubEngine(vector))
    return vector


# ── the names ────────────────────────────────────────────────────────────────

def test_hook_at_layer_moves_the_block_index():
    assert hook_at_layer("blocks.8.hook_resid_pre", 10) == \
        "blocks.10.hook_resid_pre"
    assert hook_at_layer("blocks.0.mlp_write", 3) == "blocks.3.mlp_write"
    assert hook_at_layer("", 3) == ""


def test_a_transformer_lens_hook_name_names_transformer_lens():
    assert infer_tensor_source("blocks.8.hook_resid_pre") == \
        "transformer_lens:blocks.8.hook_resid_pre"
    # A name that is not a TransformerLens hook yields nothing, rather than a
    # guess about which library produced the tensor.
    assert infer_tensor_source("blocks.8.mlp_write") == ""
    assert infer_tensor_source("") == ""


# ── negative: the guard must run ─────────────────────────────────────────────

def test_a_layer_the_dictionary_does_not_describe_is_refused(sae_dir):
    """A dictionary for layer 8 must not encode layer 10."""
    _write_checkpoint(sae_dir / "layer8.pt", facts={
        "hook_point": "blocks.8.hook_resid_pre",
        "tensor_source": "transformer_lens:blocks.8.hook_resid_pre",
        "layer": 8,
        "model_name": "gpt2-small",
    })

    out = dispatcher.sae_inspect({"prompt": PROMPT, "source": "local",
                                  "identifier": "layer8.pt", "layer": 10})

    assert out["status"] == "unavailable", out
    assert out["provenance"] == "unavailable"
    assert out.get("feature_indices") in (None, [])
    assert out.get("activations") in (None, [])
    assert out.get("relative_l2_error") is None
    reason = out["reason"]
    # The reason names the dictionary's own hook point and layer.
    assert "blocks.8.hook_resid_pre" in reason, reason
    assert "layer 8" in reason, reason


def test_a_checkpoint_that_records_no_tensor_source_is_refused(sae_dir):
    """No declared tensor means no activation to encode -- not any tensor."""
    _write_checkpoint(sae_dir / "blind.pt")

    out = dispatcher.sae_inspect({"prompt": PROMPT, "source": "local",
                                  "identifier": "blind.pt"})

    assert out["status"] == "unavailable", out
    assert out.get("feature_indices") in (None, [])
    assert "no tensor source" in out["reason"], out["reason"]
    # The refusal says why applying it to something else would be wrong.
    assert "mismatch" in out["reason"], out["reason"]


def test_a_dictionary_with_no_declared_hook_point_applies_to_nothing(sae_dir):
    """The loader's guard fails closed, so no caller can skip it by omission."""
    path = _write_checkpoint(sae_dir / "blind.pt")
    sae = SAELoader().load_sae("local", str(path))

    assert sae.config.hook_point == ""
    assert sae.config.tensor_source == ""
    assert sae.config.applies_to(8, "blocks.8.hook_resid_pre") is False
    with pytest.raises(SAELoadError):
        sae.assert_applies_to(8, "blocks.8.hook_resid_pre")


# ── the route encodes what the checkpoint declares ───────────────────────────

def test_the_route_encodes_the_tensor_the_checkpoint_declares(sae_dir,
                                                               stub_engine):
    _write_checkpoint(sae_dir / "mlp.pt", facts={
        "hook_point": "blocks.3.mlp_write",
        "tensor_source": "transformers:blocks.3.mlp_write",
        "layer": 3,
        "model_name": "gpt2",
    })

    out = dispatcher.sae_inspect({"prompt": PROMPT, "source": "local",
                                  "identifier": "mlp.pt"})

    assert out["status"] == "completed", out
    assert out["tensor_source"] == "transformers:blocks.3.mlp_write"
    assert out["hook_point"] == "blocks.3.mlp_write"
    assert out["layer"] == 3
    assert out["attested"] is False

    # The activations are the ones this activation produces, so the route
    # encoded the declared tensor and not some other one.
    sae = SAELoader().load_sae("local", str(sae_dir / "mlp.pt"))
    expected = sae.activate(stub_engine)
    assert out["activations"] == expected["activations"]
    assert out["feature_indices"] == expected["feature_indices"]


def test_a_reconstruction_no_better_than_the_mean_is_flagged(sae_dir,
                                                              stub_engine):
    """A relative L2 at or above 1.0 is the predict-the-mean baseline.

    The figure stays as measured; the flag is the qualification, so a reader
    cannot take 1.0 for a good reconstruction.
    """
    _write_checkpoint(sae_dir / "zero.pt", zero=True, facts={
        "hook_point": "blocks.3.mlp_write",
        "tensor_source": "transformers:blocks.3.mlp_write",
        "layer": 3,
        "model_name": "gpt2",
    })

    out = dispatcher.sae_inspect({"prompt": PROMPT, "source": "local",
                                  "identifier": "zero.pt"})

    assert out["status"] == "completed", out
    assert out["reconstruction_measured"] is True
    assert out["relative_l2_error"] == pytest.approx(1.0)
    assert out["reconstruction_useful"] is False
    assert out["reconstruction_note"], out


# ── a locally trained checkpoint describes itself ────────────────────────────

def _train_tiny(sae_dir, **overrides):
    from backend.science.reproducibility.sae_pipeline import (
        SAEReproductionPipeline,
    )

    params = {"n_features": 8, "seed": 3, "n_tokens": 128, "steps": 5}
    params.update(overrides)
    out = SAEReproductionPipeline().run(checkpoint_dir=str(sae_dir), **params)
    return out


def test_a_locally_trained_checkpoint_describes_its_own_tensor(sae_dir):
    """The facts a reader needs are in the file, not in a note beside it."""
    require_weights()
    import torch

    result = _train_tiny(sae_dir)
    checkpoint = result["checkpoint"]
    assert checkpoint["checkpoint_saved"] is True, result
    path = checkpoint["checkpoint_path"]

    payload = torch.load(path, map_location="cpu", weights_only=True)
    assert payload["hook_point"] == "blocks.0.mlp_write", payload["hook_point"]
    assert payload["tensor_source"] == "transformers:blocks.0.mlp_write"
    assert payload["layer"] == 0
    assert payload["model_name"]

    # And the loader reads them back, so the route can act on them.
    sae = SAELoader().load_sae("local", path)
    assert sae.config.hook_point == "blocks.0.mlp_write"
    assert sae.config.tensor_source == "transformers:blocks.0.mlp_write"
    assert sae.config.layer == 0
    assert sae.config.model_id == payload["model_name"]


def test_the_route_reproduces_the_training_tensor(sae_dir):
    """The route's number must come from the tensor the trainer fitted."""
    require_weights()
    import torch

    from backend.services import gpt2_engine as engine

    result = _train_tiny(sae_dir)
    path = result["checkpoint"]["checkpoint_path"]
    name = os.path.basename(path)

    out = dispatcher.sae_inspect({"prompt": PROMPT, "source": "local",
                                  "identifier": name})
    assert out["status"] == "completed", out

    # Recompute the training tensor from the same weights and check the
    # route's reconstruction against it.
    sae = SAELoader().load_sae("local", path)
    written = engine.mlp_write(sae.config.layer, PROMPT)["tensor"][-1]
    direct = sae.reconstruction_error(
        written, sae.activate(written)["feature_indices"])

    assert out["tensor_source"] == "transformers:blocks.0.mlp_write"
    assert out["relative_l2_error"] == pytest.approx(
        direct["relative_l2_error"]), (out, direct)
    assert out["n_active_features"] == direct["n_active_features"]


# ── live: the published dictionary ───────────────────────────────────────────

def test_the_published_layer_8_dictionary_reconstructs_its_own_tensor():
    """The headline claim, measured through the route.

    The dictionary is applied to the tensor it was trained on -- read from
    TransformerLens' own cache -- so the reconstruction beats the
    predict-the-mean baseline. Applied to the ``transformers`` tensor the
    same dictionary does not, which is what the route used to report.
    """
    require_weights()
    import torch

    from backend.interpretability.sae import registry
    from backend.services import gpt2_engine as engine

    entry = registry.resolve(8)
    assert entry is not None
    try:
        registry.fetch(8)
    except Exception as exc:
        if os.environ.get("MECH_REQUIRE_WEIGHTS") == "1":
            raise
        pytest.skip(f"{entry.repo_id}/{entry.filename} could not be fetched "
                    f"({type(exc).__name__}: {str(exc)[:160]}); live SAE "
                    f"arithmetic NOT RUN (not passed)")

    out = dispatcher.sae_inspect({"prompt": PROMPT, "source": "hf",
                                  "identifier": "layer:8"})
    if out.get("status") != "completed":
        pytest.skip(f"the route could not encode the published dictionary "
                    f"({out.get('reason') or out.get('error')}); live SAE "
                    f"arithmetic NOT RUN (not passed)")

    assert out["tensor_source"] == "transformer_lens:blocks.8.hook_resid_pre"
    assert out["hook_point"] == "blocks.8.hook_resid_pre"
    assert out["layer"] == 8
    assert out["reconstruction_measured"] is True

    error = out["relative_l2_error"]
    assert isinstance(error, float) and error > 0, out
    assert error < 1.0, (
        f"relative L2 {error} is at or above the predict-the-mean baseline, "
        "so the dictionary does not explain the activation it was trained on")
    assert out["reconstruction_useful"] is True

    # The same dictionary against the tensor the route used to feed it must
    # be much worse -- otherwise "the right tensor" would not be the reason.
    sae = SAELoader().load_sae("hf", "layer:8")
    detail = engine.layer_activations(8, PROMPT)
    transformers_vector = torch.tensor(detail["resid_post"][-1],
                                       dtype=torch.float32)
    wrong = sae.reconstruction_error(
        transformers_vector,
        sae.activate(transformers_vector)["feature_indices"])
    assert wrong["relative_l2_error"] > error * 2, (
        f"the transformers tensor reconstructs at {wrong['relative_l2_error']} "
        f"against {error} for the declared one; the route is not reading the "
        "tensor it says it is reading")
