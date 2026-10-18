"""Live evidence for Step 1: the real GPT-2 layer-8 and layer-10 SAE checkpoints.

What is being proved, and why a route test is not enough
--------------------------------------------------------
`test_sae_routes.py` exercises the HTTP surface. It would stay green if the
registry pointed at a file that does not exist, because an `unavailable` response
is a correct response. Step 1 claims something stronger -- that *these* two
checkpoints are on disk, load through `backend/interpretability/sae/loader.py`
into real tensors, and describe the hook points the registry says they do -- and
that claim can only be checked by loading them.

So every assertion below is about a tensor that was materialised: the encoder and
decoder shapes, the dictionary width, the bias vectors, the hook point read out
of the checkpoint's own `cfg`, and a digest computed over the loaded weight
values. The file's byte size is reconciled against the sum of the tensors' own
`numel() * element_size()` so the on-disk artefact and the loaded tensors are
provably the same object -- a truncated or substituted file cannot satisfy both.

The `hook_resid_pre` link
-------------------------
An SAE describes one activation at one hook point on one model. The registry
carries `blocks.{L}.hook_resid_pre`, and the checkpoint's own config carries the
same string; both are asserted against each other and against the layer number.
A dictionary applied to the wrong activation returns feature indices that mean
nothing, so the wrong-layer application is asserted to *raise* rather than be
silently permitted.

The arithmetic test uses a real GPT-2 residual-stream activation
---------------------------------------------------------------
`test_the_dictionaries_encode_a_real_gpt2_residual_activation` runs a genuine
GPT-2 forward pass and takes the residual stream from TransformerLens' own
`blocks.L.hook_resid_pre` -- the hook point these dictionaries were trained
against. That part is load-bearing rather than stylistic: `transformers`'
`hidden_states[L]` is a *different* tensor, because TransformerLens'
`center_writing_weights` shifts the residual stream by a constant. Measured on
this host at layer 8, the two differ by 20.7 in L2 norm, and the same dictionary
reconstructs the TransformerLens tensor at relative L2 0.243 against 3.43 for the
`transformers` one. The hook point names a specific tensor, so the test reads the
one the registry names.

No synthetic activation is used anywhere in this module: every input to
`activate()` is a tensor produced by real GPT-2 weights, so a passing arithmetic
assertion is a joint property of the checkpoint and the model.

What is deliberately not claimed: no feature is interpreted, no name is attached
to a feature index, and `monosemanticity` is not scored. The reconstruction error
is reported as a measured number, not compared against a threshold chosen to make
it look good.

Skip convention: this is the suite's existing one. A missing checkpoint or model
skips with an explicit, visible reason (never a silent pass), and
``MECH_REQUIRE_WEIGHTS=1`` turns that skip into a failure so the live lane cannot
silently lose this coverage.
"""

from __future__ import annotations

import functools
import math
import os

import pytest

from _weight_guard import require_weights

from backend.interpretability.sae import registry
from backend.interpretability.sae.loader import SAELoadError, SAELoader

LAYERS = (8, 10)

#: The dictionary width every entry in the registry claims. Asserted as a
#: literal because it is the property under test, not read from the registry --
#: a registry that renamed the number would otherwise take the test with it.
D_SAE = 24576
D_IN = 768


def _checkpoint_path(entry) -> str:
    """The real checkpoint on disk, fetched if the local cache lacks it.

    A fetch failure is NOT RUN, stated plainly: it is not a pass, and it is not
    a silent skip of the module either. `MECH_REQUIRE_WEIGHTS=1` (the live lane)
    turns it into a failure.
    """
    try:
        return registry.fetch(entry.layer)
    except Exception as exc:
        if os.environ.get("MECH_REQUIRE_WEIGHTS") == "1":
            raise
        pytest.skip(
            f"{entry.repo_id}/{entry.filename} could not be fetched "
            f"({type(exc).__name__}: {str(exc)[:160]}); live SAE checkpoint "
            f"load NOT RUN (not passed)")


@pytest.fixture(scope="module")
def layer_saes():
    """Both real checkpoints, loaded once through the project's own loader."""
    loader = SAELoader()
    loaded = {}
    for layer in LAYERS:
        entry = registry.resolve(layer)
        assert entry is not None, f"no registry entry for layer {layer}"
        _checkpoint_path(entry)  # fetch (or verify presence) before loading
        loaded[layer] = loader.load_sae("hf", f"layer:{layer}")
    return loaded


def test_the_registry_names_two_distinct_checkpoints():
    """A duplicate row would make two layers share one dictionary."""
    entries = [registry.resolve(layer) for layer in LAYERS]
    assert all(entry is not None for entry in entries)
    files = {entry.filename for entry in entries}
    hooks = {entry.hook_point for entry in entries}
    assert len(files) == len(LAYERS), f"layers share a filename: {files}"
    assert hooks == {f"blocks.{layer}.hook_resid_pre" for layer in LAYERS}
    assert {entry.repo_id for entry in entries} == {registry.REPO_ID}


@pytest.mark.parametrize("layer", LAYERS)
def test_a_layer_checkpoint_loads_into_real_tensors(layer, layer_saes):
    """The encoder and decoder are real, and they are the size they claim.

    Nothing here is read from the registry's metadata: the shapes come off the
    tensors that `torch.load` returned, and the byte reconciliation ties them to
    the bytes on disk.
    """
    entry = registry.resolve(layer)
    path = _checkpoint_path(entry)
    file_size = os.path.getsize(path)
    sae = layer_saes[layer]

    assert sae.loaded is True, "the loader returned an SAE with no weights"
    assert sae.weights is not None

    # -- the tensors, not the metadata --
    import torch

    assert tuple(sae.weights.shape) == (D_IN, D_SAE), (
        f"layer {layer}: encoder is {tuple(sae.weights.shape)}, "
        f"expected {(D_IN, D_SAE)}")
    assert sae.weights.dim() == 2
    assert int(sae.weights.shape[1]) == D_SAE, "feature count is not 24576"
    assert sae.decoder is not None, "no decoder (W_dec) was loaded"
    assert tuple(sae.decoder.shape) == (D_SAE, D_IN), (
        f"layer {layer}: decoder is {tuple(sae.decoder.shape)}, "
        f"expected {(D_SAE, D_IN)}")
    assert sae.biases.get("b_enc") is not None
    assert tuple(sae.biases["b_enc"].shape) == (D_SAE,)
    assert sae.bias_dec is not None
    assert tuple(sae.bias_dec.shape) == (D_IN,)
    # This format carries no dedicated `b_pre`: the encoder is centered on
    # `b_dec`, the data mean. The loader records which vector it used, and the
    # assertion is that the two are one tensor -- dropping `b_dec` from the
    # encoder input changes the decomposition wholesale (measured here: 51
    # active features at relative L2 0.243 centered, 1558 at 5.56 uncentered).
    assert sae.config.pre_encoder_center == "b_dec", (
        f"the encoder is centered on {sae.config.pre_encoder_center!r}")
    assert sae.biases.get("b_pre") is not None, (
        "no pre-encoder centering vector was read, so the encoder input is "
        "uncentered")
    assert torch.equal(sae.biases["b_pre"], sae.bias_dec), (
        "for this checkpoint format b_pre must be the checkpoint's own b_dec")

    assert bool(_finite(sae.weights)), "encoder weights contain non-finite values"
    assert float(sae.weights.std()) > 0, "encoder weights are constant"
    assert float(sae.decoder.std()) > 0, "decoder weights are constant"

    # -- the file and the tensors are the same artefact --
    tensors = [sae.weights, sae.decoder, sae.biases["b_enc"], sae.bias_dec]
    tensor_bytes = sum(int(t.numel()) * int(t.element_size()) for t in tensors)
    header_bytes = file_size - tensor_bytes
    assert 0 <= header_bytes < 200_000, (
        f"layer {layer}: file is {file_size} bytes but its tensors total "
        f"{tensor_bytes} ({header_bytes} bytes unaccounted for). The loaded "
        f"tensors are not the file's contents.")
    assert file_size > 100 * 1024 * 1024, (
        f"layer {layer}: checkpoint is only {file_size} bytes, which is not a "
        f"24576-feature dictionary")


@pytest.mark.parametrize("layer", LAYERS)
def test_the_hook_point_is_the_one_the_registry_claims(layer, layer_saes):
    """`blocks.{L}.hook_resid_pre`, from the checkpoint's own config."""
    entry = registry.resolve(layer)
    sae = layer_saes[layer]
    expected = f"blocks.{layer}.hook_resid_pre"

    assert sae.config.hook_point == expected, (
        f"layer {layer}: checkpoint says {sae.config.hook_point!r}, registry "
        f"says {entry.hook_point!r}")
    assert entry.hook_point == expected
    assert sae.config.layer == layer, (
        f"layer {layer}: the loader parsed layer {sae.config.layer} out of the "
        f"hook point")
    assert sae.config.d_in == D_IN and sae.config.d_sae == D_SAE
    assert sae.config.repo_id == registry.REPO_ID
    assert sae.config.model_id == "gpt2-small"
    assert sae.config.dataset == "Skylion007/openwebtext"
    assert sae.config.l1_coefficient == pytest.approx(8e-05)
    assert sae.config.revision == entry.revision

    # The dictionary refuses an activation it does not describe.
    sae.assert_applies_to(layer, expected)
    other = [candidate for candidate in LAYERS if candidate != layer][0]
    with pytest.raises(SAELoadError):
        sae.assert_applies_to(other, f"blocks.{other}.hook_resid_pre")


def test_the_two_layers_are_different_checkpoints(layer_saes):
    """Not one file loaded twice under two layer numbers."""
    eight = layer_saes[8].weights_sha256()
    ten = layer_saes[10].weights_sha256()

    assert eight.startswith("sha256:") and len(eight) == len("sha256:") + 64
    assert ten.startswith("sha256:")
    assert eight != ten, (
        "layers 8 and 10 have identical encoder weights, so they are the same "
        "checkpoint filed twice")

    # The digest is over real values, so verifying it is a real check.
    assert layer_saes[8].verify_integrity(eight) is True
    assert layer_saes[8].verify_integrity(ten) is False


def _finite(tensor) -> bool:
    import torch

    return bool(torch.isfinite(tensor).all())


@functools.lru_cache(maxsize=1)
def _tl_model():
    """GPT-2 through TransformerLens -- the library the dictionaries were
    trained with, so its hook points are the activations they describe.

    A load failure is a visible skip (or a failure under
    ``MECH_REQUIRE_WEIGHTS=1``), never a silent pass.
    """
    try:
        from transformer_lens import HookedTransformer

        return HookedTransformer.from_pretrained("gpt2", device="cpu")
    except Exception as exc:
        if os.environ.get("MECH_REQUIRE_WEIGHTS") == "1":
            raise
        pytest.skip(
            f"TransformerLens could not load gpt2 ({type(exc).__name__}: "
            f"{str(exc)[:160]}); live SAE arithmetic NOT RUN (not passed)")


def _gpt2_residual(prompt: str, layers=LAYERS):
    """Real GPT-2 residual-stream activations at `blocks.L.hook_resid_pre`.

    Read from TransformerLens' own hook, which is the tensor these dictionaries
    were trained against -- not `transformers`' `hidden_states[L]`, which differs
    by the constant `center_writing_weights` adds to the residual stream.

    Returns ``{layer: tensor(D_IN,)}`` at the final position.
    """
    import torch

    model = _tl_model()
    with torch.no_grad():
        _, cache = model.run_with_cache(prompt)

    values = {}
    for layer in layers:
        key = f"blocks.{layer}.hook_resid_pre"
        assert key in cache, f"TransformerLens returned no {key}"
        values[layer] = cache[key][0, -1, :].detach().float().cpu()
    return values


def test_the_dictionaries_encode_a_real_gpt2_residual_activation(layer_saes):
    """Real weights -> real activation -> real decomposition.

    The input to `activate()` is GPT-2's own residual stream at the dictionary's
    own hook point. What is asserted is the arithmetic of the loaded checkpoint
    on that tensor: the width, the number of active features, their ordering, and
    a reconstruction that decoded back through the real `W_dec`. No feature is
    named and no score is assigned.
    """
    import torch

    pytest.importorskip("torch")
    require_weights()

    prompt = "The capital of France is"
    activations = _gpt2_residual(prompt)

    for layer in LAYERS:
        vector = activations[layer]
        assert tuple(vector.shape) == (D_IN,), (
            f"layer {layer}: residual activation is {tuple(vector.shape)}")
        assert bool(torch.isfinite(vector).all())
        assert float(vector.std()) > 0, (
            f"layer {layer}: the activation is constant, so the encode below "
            f"would say nothing")

        sae = layer_saes[layer]
        result = sae.activate(vector)

        assert result["status"] == "completed", result.get("reason")
        assert result["provenance"] == "live"
        assert result["measured"] is True
        assert result["d_sae"] == D_SAE
        assert len(result["feature_indices"]) == 32
        assert len(set(result["feature_indices"])) == 32, "duplicate features"
        assert all(0 <= index < D_SAE for index in result["feature_indices"])
        assert all(value >= 0 for value in result["activations"]), (
            "a ReLU produced a negative activation")
        assert result["activations"] == sorted(result["activations"], reverse=True)
        assert result["sparsity"] == pytest.approx(32 / D_SAE)

        recon = sae.reconstruction_error(vector, result["feature_indices"])
        assert recon["status"] == "completed", recon.get("reason")
        error = recon["relative_l2_error"]
        assert isinstance(error, float) and math.isfinite(error) and error > 0, (
            f"layer {layer}: reconstruction error {error!r} is not a "
            f"measurement")
        assert recon["n_active_features"] > 0
        assert recon["d_sae"] == D_SAE
        assert recon["hook_point"] == f"blocks.{layer}.hook_resid_pre"
        assert recon["model_id"] == "gpt2-small"

        # -- and it is a decomposition, not merely a finite number --
        #
        # A dictionary explains an activation only if its reconstruction beats
        # predicting the mean. `mean_baseline` is that floor, computed from the
        # checkpoint's own b_dec. Relative L2 < 1 is the weaker statement
        # (beating a prediction of zero), so both are asserted. Measured on this
        # host across three prompts: `error` 0.235-0.295, mean baseline
        # 0.59-0.70, active features 36-94 of 24576.
        mean_baseline = float(
            (vector - sae.bias_dec.float()).norm() / vector.norm())
        assert error < 1.0, (
            f"layer {layer}: relative L2 {error} means the reconstruction is "
            f"worse than predicting zero")
        assert error < mean_baseline, (
            f"layer {layer}: relative L2 {error} is above the "
            f"predict-the-mean baseline {mean_baseline}, so this dictionary "
            f"does not explain this activation")
        assert recon["n_active_features"] < int(D_SAE / 16), (
            f"layer {layer}: {recon['n_active_features']} of {D_SAE} features "
            f"are active, which is not a sparse decomposition")

        # The negative control for the centering this test depends on, computed
        # here rather than trusted: the same dictionary, same activation, with
        # b_dec *not* subtracted from the encoder input. If that ever stops
        # being much worse, the centering assertions above have stopped testing
        # anything.
        uncentered_latent = torch.relu(
            vector @ sae.weights.float() + sae.biases["b_enc"].float())
        uncentered_recon = (uncentered_latent @ sae.decoder.float()
                            + sae.bias_dec.float())
        uncentered_error = float(
            (vector - uncentered_recon).norm() / vector.norm())
        assert uncentered_error > error * 4, (
            f"layer {layer}: centered relative L2 {error} against "
            f"{uncentered_error} uncentered, so centering on b_dec is not the "
            f"load-bearing step this test assumes")

    # The two dictionaries are different, and they are reading different
    # activations: identical feature sets would mean the layer is decorative.
    eight = layer_saes[8].activate(activations[8])
    ten = layer_saes[10].activate(activations[10])
    assert eight["feature_indices"] != ten["feature_indices"]
    assert not torch.equal(activations[8], activations[10]), (
        "GPT-2's residual stream is identical at layers 8 and 10")
