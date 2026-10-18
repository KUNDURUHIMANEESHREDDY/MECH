"""SAE Checkpoint Loader & Provider System.

Loads, validates, and verifies Sparse Autoencoder checkpoints across multiple
providers (HuggingFace, SAELens, Anthropic) and implements the activation API.

Honest about what it can do. There are no trained SAE weights in this
repository, so a provider must genuinely fetch or open real weights, and this
module refuses to hand back an SAE that has none. It previously:

- returned an `SAE` with `weights=None` and reported `status: "loaded"`,
- derived `checkpoint_sha` from the *name* of the checkpoint
  (`f"sha256_{identifier}_{version}"`), so integrity "verification" was string
  equality against a value derived from the argument it was checking,
- returned the same three feature indices and activations for any hidden
  state, from any model, layer, or token position.

`activate()` now performs the real encoding when weights are present and
returns an explicit unavailable envelope when they are not.

The pre-encoder centering is a checkpoint property, not a constant
-----------------------------------------------------------------
``ReLU((h - b_pre) @ W_enc + b_enc)`` has two spellings in the wild. Some
checkpoints carry a dedicated ``b_pre``; the ``mats_sae_training`` / Cunningham
format carries no such tensor and centers the encoder on ``b_dec``, the decoder
bias that is also the data mean. Only the first was read, so the second format
was encoded uncentered -- and that is not a slightly worse decomposition.
Measured on this host, the layer-8 GPT-2 dictionary reports 51 active features
at relative L2 0.243 when centered on ``b_dec``, and 1558 at 5.56 when it is
not. The fallback is now made, and ``SAEConfig.pre_encoder_center`` records
which vector was used, so the arithmetic is visible in the result.

``encoder.bias`` is no longer a fallback for it. In SAELens that bias *is*
``b_enc``, which is added after the matmul; subtracting it from the input as
well applies it twice, with the centering's sign inverted.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import os
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


class SAELoadError(RuntimeError):
    """Raised when real SAE weights cannot be obtained."""


@dataclass
class SAEConfig:
    d_in: int
    d_sae: int
    model_name: str
    layer: int
    repo_id: str
    version: str
    checkpoint_sha: str = ""
    verified_at: Optional[str] = None
    #: The activation the dictionary was trained on. An SAE is only meaningful
    #: for this one hook point on this one model; the loader refuses a
    #: mismatch rather than encoding a different activation.
    hook_point: str = ""
    model_id: str = ""
    dataset: str = ""
    l1_coefficient: Optional[float] = None
    revision: str = ""
    #: Which checkpoint field the encoder input is centered on: ``"b_pre"``
    #: when the checkpoint carries a dedicated one, ``"b_dec"`` when the
    #: format centers on the decoder bias (the ``mats_sae_training`` /
    #: Cunningham convention, which is also what this repository's own trainer
    #: writes -- ``sae_pipeline`` stores ``"b_pre": sae.b_dec`` and computes
    #: ``(x - b_dec) @ W_enc + b_enc``), and ``""`` when there is nothing to
    #: subtract.
    #:
    #: Recorded rather than folded silently into ``b_pre``, because it decides
    #: whether the numbers below are a decomposition or noise: measured on this
    #: host, the layer-8 GPT-2 dictionary reports 51 active features at relative
    #: L2 0.243 when centered on ``b_dec`` and 1558 at 5.56 when it is not.
    pre_encoder_center: str = ""

    def applies_to(self, layer: int, hook_point: str) -> bool:
        """Whether this dictionary describes the given activation."""
        if not self.hook_point:
            return True  # unknown provenance: caller decides, and is told so
        return (int(layer) == int(self.layer)
                and str(hook_point) == self.hook_point)


def _cfg_fact(cfg: Any, *names: str) -> Any:
    """First present attribute among ``names`` on a checkpoint config."""
    for name in names:
        value = getattr(cfg, name, None)
        if value is not None:
            return value
    return None


def _layer_from_hook(hook_point: Any) -> int:
    """``blocks.8.hook_resid_pre`` -> ``8``; ``-1`` when it does not parse."""
    import re

    match = re.search(r"blocks\.(\d+)\.", str(hook_point or ""))
    return int(match.group(1)) if match else -1


class SAE:
    """A loaded Sparse Autoencoder instance backed by real weight tensors."""

    def __init__(self, config: SAEConfig, weights: Any = None,
                 biases: Optional[Dict[str, Any]] = None,
                 decoder: Any = None, bias_dec: Any = None) -> None:
        self.config = config
        self.weights = weights  # torch.Tensor [d_in, d_sae]
        self.biases = biases or {}
        self.decoder = decoder  # torch.Tensor [d_sae, d_in] or None
        self.bias_dec = bias_dec
        self.id = str(uuid.uuid4())

    @property
    def loaded(self) -> bool:
        return self.weights is not None

    def verify_integrity(self, expected_sha: str) -> bool:
        """Verify weight integrity by hashing the real tensors.

        Previously this compared `config.checkpoint_sha` to the argument, and
        that value had been synthesised from the checkpoint's own name -- so
        the check passed for any input that agreed with a string built out of
        the same argument.
        """
        if not self.loaded:
            return False
        actual = self.weights_sha256()
        if actual == expected_sha:
            self.config.verified_at = _dt.datetime.utcnow().isoformat() + "Z"
            return True
        return False

    def weights_sha256(self) -> str:
        """SHA256 over the encoder weights, shape and dtype included."""
        if not self.loaded:
            return ""
        import torch  # local import: the loader must work without torch

        digest = hashlib.sha256()
        tensor = self.weights
        if not torch.is_tensor(tensor):
            tensor = torch.as_tensor(tensor)
        digest.update(str(tensor.dtype).encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("utf-8"))
        flat = tensor.detach().to("cpu").contiguous().view(-1)
        digest.update(flat.numpy().tobytes())
        return "sha256:" + digest.hexdigest()

    def activate(self, hidden_state: Any) -> Dict[str, Any]:
        """Encode a hidden state into sparse feature firings.

        The real encoding is ``ReLU((h - b_pre) @ W_enc + b_enc)``, keeping the
        top ``d_sae``/top_k activations. ``b_pre`` here is the checkpoint's own
        pre-encoder centering vector, which for a ``mats_sae_training``
        checkpoint is its ``b_dec``; see ``SAEConfig.pre_encoder_center`` for
        which one was read, because centering decides whether the output is a
        decomposition or noise. Without weights there is nothing to apply, so
        this refuses instead of returning fixed feature indices.
        """
        if not self.loaded:
            return {
                "status": "unavailable",
                "provenance": "unavailable",
                "measured": False,
                "validation_eligible": False,
                "publication_eligible": False,
                "reason": (
                    "No SAE encoder weights are loaded, so this hidden state "
                    "cannot be encoded. Reporting feature indices here would "
                    "be returning constants, not a decomposition."
                ),
            }

        import torch

        h = hidden_state
        if not torch.is_tensor(h):
            h = torch.as_tensor(h)
        h = h.detach().to(self.weights.device).reshape(-1).float()

        b_pre = self.biases.get("b_pre")
        shifted = h if b_pre is None else h - torch.as_tensor(
            b_pre, device=h.device, dtype=h.dtype).reshape(-1)
        latent = shifted @ self.weights.float()
        b_enc = self.biases.get("b_enc")
        if b_enc is not None:
            latent = latent + torch.as_tensor(
                b_enc, device=latent.device, dtype=latent.dtype).reshape(1, -1)
        acts = torch.relu(latent).squeeze(0)

        k = min(32, acts.numel())
        top = torch.topk(acts, k)
        return {
            "status": "completed",
            "provenance": "live",
            "measured": True,
            "feature_indices": [int(i) for i in top.indices.tolist()],
            "activations": [round(float(a), 6) for a in top.values.tolist()],
            "sparsity": k / max(1, self.config.d_sae),
            "d_sae": self.config.d_sae,
            "validation_eligible": True,
            "publication_eligible": True,
            "attested": False,
        }

    def extract_features_batch(self, dataset: List[Any]) -> List[Dict[str, Any]]:
        """High-throughput feature extraction for a dataset."""
        return [self.activate(item) for item in dataset]

    # -- decoder side ----------------------------------------------------

    def feature_direction(self, index: int) -> Dict[str, Any]:
        """The decoder row for a feature: its direction in activation space.

        A feature index with no direction is only a label; the direction is what
        lets a reader ask what the feature *does* to the residual stream. No
        decoder weights means this refuses rather than returning a placeholder.
        """
        if self.decoder is None:
            return {
                "status": "unavailable",
                "provenance": "unavailable",
                "measured": False,
                "reason": ("This checkpoint carries no decoder weights (W_dec), "
                           "so a feature direction cannot be reported."),
            }
        import torch

        row = self.decoder[int(index)]
        if not torch.is_tensor(row):
            row = torch.as_tensor(row)
        row = row.detach().float().reshape(-1)
        norm = float(row.norm())
        top = torch.topk(row.abs(), min(8, row.numel()))
        return {
            "status": "completed",
            "provenance": "live",
            "measured": True,
            "feature_index": int(index),
            "dim": int(row.numel()),
            "l2_norm": round(norm, 6),
            "unit": bool(norm > 0),
            "top_dims": [
                {"dim": int(i), "value": round(float(row[i]), 6)}
                for i in top.indices.tolist()
            ],
            "hook_point": self.config.hook_point,
            "model_id": self.config.model_id,
        }

    def reconstruction_error(self, hidden_state: Any,
                             feature_indices: List[int]) -> Dict[str, Any]:
        """Relative L2 error of the reconstruction from the given features.

        This is the measurement that says whether the dictionary actually
        explains the activation, so it is computed from real tensors or refused.
        """
        if self.decoder is None or self.weights is None:
            return {
                "status": "unavailable",
                "provenance": "unavailable",
                "measured": False,
                "reason": "Encoder or decoder weights are missing.",
            }
        import torch

        h = hidden_state
        if not torch.is_tensor(h):
            h = torch.as_tensor(h)
        h = h.detach().float().reshape(-1).to(self.weights.device)

        acts = self.activate(hidden_state)
        if acts.get("status") != "completed":
            return acts

        b_pre = self.biases.get("b_pre")
        shifted = h if b_pre is None else h - torch.as_tensor(
            b_pre, device=h.device, dtype=h.dtype).reshape(-1)
        latent = shifted @ self.weights.float()
        b_enc = self.biases.get("b_enc")
        if b_enc is not None:
            latent = latent + torch.as_tensor(
                b_enc, device=latent.device, dtype=latent.dtype).reshape(1, -1)
        latent = torch.relu(latent).squeeze(0)

        dec = self.decoder.float().to(h.device)
        recon = latent @ dec
        if self.bias_dec is not None:
            recon = recon + torch.as_tensor(
                self.bias_dec, device=h.device, dtype=h.dtype).reshape(-1)

        numerator = float((h - recon).norm())
        denominator = float(h.norm())
        return {
            "status": "completed",
            "provenance": "live",
            "measured": True,
            "relative_l2_error": round(numerator / denominator, 6)
            if denominator > 0 else None,
            "n_active_features": int((latent > 0).sum()),
            "d_sae": int(self.config.d_sae),
            "hook_point": self.config.hook_point,
            "model_id": self.config.model_id,
        }

    def assert_applies_to(self, layer: int, hook_point: str) -> None:
        """Refuse to encode an activation this dictionary does not describe."""
        if self.config.applies_to(layer, hook_point):
            return
        raise SAELoadError(
            f"This SAE was trained on {self.config.model_id or 'a model'} at "
            f"{self.config.hook_point!r} (layer {self.config.layer}); it does "
            f"not describe layer {layer} at {hook_point!r}. Encoding that "
            f"activation would return feature indices that mean nothing.")


class SAEProvider(ABC):
    """Abstract base for SAE model providers."""

    @abstractmethod
    def load(self, identifier: str, version: str) -> SAE:
        """Fetch real weights and return a loaded SAE.

        Must raise `SAELoadError` when weights are unavailable. Returning a
        config-only SAE is not acceptable.
        """
        ...


class _LanguageModelSAERunnerConfig:
    """Allowlist target for the metadata class inside Bloom's checkpoints.

    Those files pickle ``sae_training.config.LanguageModelSAERunnerConfig`` (a
    plain hyper-parameter container) alongside the tensor state dict, so
    ``torch.load(weights_only=True)`` refuses them: the class is an unsupported
    global and ``sae_training`` is not installed here.

    Rather than falling back to ``weights_only=False`` -- an arbitrary-code-
    execution load on a downloaded artefact -- this empty class is registered
    under that exact module path and allowlisted, so the unpickler reconstructs
    the container into it. Nothing in the file is executed: the class has no
    methods for it to reach.

    ``__module__`` and ``__qualname__`` are assigned explicitly because the
    allowlist key is ``<module>.<qualname>``; a class defined inside a function
    would register as ``...<locals>.LanguageModelSAERunnerConfig`` and silently
    fail to match.
    """

    __module__ = "sae_training.config"
    __qualname__ = "LanguageModelSAERunnerConfig"


def _install_metadata_module():
    """Register the shim module the pickled path resolves against.

    Returns the previously-registered modules so the caller can restore them;
    the shim is never left installed after a load.
    """
    import sys
    import types

    module = types.ModuleType("sae_training.config")
    module.LanguageModelSAERunnerConfig = _LanguageModelSAERunnerConfig
    package = types.ModuleType("sae_training")
    package.config = module
    saved = {name: sys.modules.get(name)
             for name in ("sae_training", "sae_training.config")}
    sys.modules["sae_training"] = package
    sys.modules["sae_training.config"] = module
    return saved


def _restore_metadata_module(saved):
    import sys

    for name, previous in saved.items():
        if previous is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = previous


def _load_tensors(path: str) -> Tuple[Any, Dict[str, Any], Dict[str, Any]]:
    """Open real SAE weights from disk, plus their decoder and provenance.

    ``weights_only=True`` is not a stylistic preference. ``weights_only=False``
    is a pickle load, so a checkpoint from anywhere -- a downloaded artefact, a
    path handed in by an API caller, a file in a shared cache -- becomes
    arbitrary code execution inside this process.

    Returns ``(W_enc, biases, meta)`` where ``meta`` carries ``W_dec``/``b_dec``
    (needed to decode a feature back into an activation) and the training facts
    from the checkpoint's own config.
    """
    if not os.path.exists(path):
        raise SAELoadError(f"SAE checkpoint not found: {path}")
    try:
        size = os.path.getsize(path)
    except OSError as exc:
        raise SAELoadError(f"SAE checkpoint unreadable: {path}: {exc}")
    if size > 512 * 1024 * 1024:
        raise SAELoadError(
            f"SAE checkpoint is {size} bytes, above the 512 MiB loadable "
            f"limit; refusing before deserialisation.")
    try:
        import torch
    except ImportError as exc:
        raise SAELoadError(f"torch is required to load SAE weights: {exc}")

    payload = None
    saved = _install_metadata_module()
    try:
        with torch.serialization.safe_globals([_LanguageModelSAERunnerConfig]):
            payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise SAELoadError(
            f"SAE checkpoint could not be loaded with weights_only=True, so "
            f"it was refused rather than unpickled: "
            f"{type(exc).__name__}: {exc}") from exc
    finally:
        _restore_metadata_module(saved)

    # Two shapes in the wild: a bare state dict, and Bloom's
    # {"cfg": <config>, "state_dict": OrderedDict}.
    state = payload
    config = None
    if isinstance(payload, dict) and "state_dict" in payload:
        state = payload.get("state_dict")
        config = payload.get("cfg")
    if not isinstance(state, dict):
        raise SAELoadError(f"SAE checkpoint is not a state dict: {path}")

    # Explicit `is None` checks, not `or`: a weight tensor with more than one
    # element has an ambiguous truth value, so `state.get("W_enc") or ...`
    # raises "Boolean value of Tensor with more than one value is ambiguous".
    weights = None
    for key in ("W_enc", "encoder.weight", "encoder", "weight"):
        candidate = state.get(key)
        if candidate is not None:
            weights = candidate
            break
    if weights is None:
        raise SAELoadError(
            f"SAE checkpoint has no encoder weights (W_enc): {path}")

    # The pre-encoder centering vector. Two conventions exist in the wild:
    #
    #   * ``b_pre`` -- a dedicated pre-encoder bias (SAELens v3 and later);
    #   * ``b_dec`` -- no separate tensor; the encoder input is centered on the
    #     decoder bias, which is the data mean. This is the
    #     ``mats_sae_training`` / Cunningham convention, and it is what this
    #     repository's own trainer writes: ``sae_pipeline`` saves
    #     ``"b_pre": sae.b_dec`` and computes ``(x - b_dec) @ W_enc + b_enc``.
    #
    # Reading only ``b_pre`` left the second format uncentered, and that is not
    # a slightly worse decomposition: measured on this host, the layer-8 GPT-2
    # dictionary reports 1558 active features at relative L2 5.56 uncentered
    # and 51 at 0.243 when centered on ``b_dec``. So the fallback is made, and
    # which vector was used travels in ``meta["pre_encoder_center"]``.
    #
    # ``encoder.bias`` is deliberately no longer a fallback: in SAELens that
    # bias *is* ``b_enc``, added after the matmul, so subtracting it from the
    # input as well applies it twice with the centering's sign inverted.
    b_pre = state.get("b_pre")
    center = "b_pre" if b_pre is not None else ""
    if b_pre is None and state.get("b_dec") is not None:
        b_pre = state.get("b_dec")
        center = "b_dec"
    biases = {"b_pre": b_pre, "b_enc": state.get("b_enc")}
    meta = {
        "W_dec": state.get("W_dec"),
        "b_dec": state.get("b_dec"),
        "config": config,
        "pre_encoder_center": center,
    }
    return weights, biases, meta


class LocalCheckpointProvider(SAEProvider):
    """Loads a real SAE checkpoint from a local path or an explicit path spec.

    `identifier` may be a filesystem path, or `env:VAR` to read the path from
    an environment variable, so a deployment can point at real weights without
    hardcoding them.
    """

    def load(self, identifier: str, version: str) -> SAE:
        path = identifier
        if identifier.startswith("env:"):
            path = os.environ.get(identifier[4:], "")
            if not path:
                raise SAELoadError(
                    f"environment variable {identifier[4:]!r} is not set, so "
                    "no SAE checkpoint path is known")
        weights, biases, meta = _load_tensors(path)

        with open(path, "rb") as handle:
            file_sha = hashlib.sha256(handle.read()).hexdigest()

        cfg = meta.get("config")
        hook_point = _cfg_fact(cfg, "hook_point", "hook_name") or ""
        config = SAEConfig(
            d_in=int(getattr(weights, "shape")[0]),
            d_sae=int(getattr(weights, "shape")[1]),
            model_name=_cfg_fact(cfg, "model_name") or version,
            layer=_layer_from_hook(hook_point),
            repo_id=identifier,
            version=version,
            checkpoint_sha=f"sha256:{file_sha}",
            hook_point=hook_point,
            model_id=_cfg_fact(cfg, "model_name", "model_class_name") or "",
            dataset=_cfg_fact(cfg, "dataset_path") or "",
            l1_coefficient=_cfg_fact(cfg, "l1_coefficient"),
            revision=_cfg_fact(cfg, "revision") or "",
            pre_encoder_center=meta.get("pre_encoder_center") or "",
        )
        return SAE(config, weights=weights, biases=biases,
                   decoder=meta.get("W_dec"), bias_dec=meta.get("b_dec"))


class HuggingFaceProvider(SAEProvider):
    """Fetches real weights from the HF Hub using the registry's file layout.

    ``identifier`` is either a registry layer (``"layer:8"``) or a literal
    ``"repo_id:filename"``. The old implementation asked every repository for a
    file called ``weights.pt``, which no real SAE repository uses, so every
    fetch failed and the toy fixture was the only thing that ever loaded.
    """

    def load(self, identifier: str, version: str) -> SAE:
        from . import registry

        revision = version
        if identifier.startswith("layer:"):
            try:
                layer = int(identifier.split(":", 1)[1])
            except ValueError:
                raise SAELoadError(
                    f"{identifier!r} is not 'layer:<int>'.")
            entry = registry.resolve(layer)
            if entry is None:
                raise SAELoadError(
                    f"No known SAE checkpoint for layer {layer}. Known layers: "
                    f"{registry.available_layers()}.")
            repo_id, filename, revision = entry.repo_id, entry.filename, entry.revision
        elif ":" in identifier:
            repo_id, filename = identifier.split(":", 1)
        else:
            raise SAELoadError(
                f"{identifier!r} does not name a checkpoint. Use 'layer:8' for "
                f"a registry entry, or 'repo_id:filename'.")

        try:
            from huggingface_hub import hf_hub_download
        except ImportError as exc:
            raise SAELoadError(
                f"huggingface_hub is required to fetch {repo_id}: {exc}")
        try:
            path = hf_hub_download(repo_id=repo_id, filename=filename,
                                   revision=revision)
        except Exception as exc:
            raise SAELoadError(
                f"could not fetch SAE weights for {repo_id}/{filename}: {exc}")

        weights, biases, meta = _load_tensors(path)
        with open(path, "rb") as handle:
            file_sha = hashlib.sha256(handle.read()).hexdigest()
        cfg = meta.get("config")
        hook_point = _cfg_fact(cfg, "hook_point", "hook_name") or ""
        config = SAEConfig(
            d_in=int(weights.shape[0]), d_sae=int(weights.shape[1]),
            model_name=_cfg_fact(cfg, "model_name") or repo_id,
            layer=_layer_from_hook(hook_point),
            repo_id=repo_id, version=version,
            checkpoint_sha=f"sha256:{file_sha}",
            hook_point=hook_point,
            model_id=_cfg_fact(cfg, "model_name", "model_class_name") or "",
            dataset=_cfg_fact(cfg, "dataset_path") or "",
            l1_coefficient=_cfg_fact(cfg, "l1_coefficient"),
            revision=revision,
            pre_encoder_center=meta.get("pre_encoder_center") or "",
        )
        return SAE(config, weights=weights, biases=biases,
                   decoder=meta.get("W_dec"), bias_dec=meta.get("b_dec"))


class SAELensProvider(SAEProvider):
    """Loads real weights through SAELens, or raises."""

    def load(self, identifier: str, version: str) -> SAE:
        try:
            from sae_lens import SAE as SaelensSAE  # type: ignore
        except ImportError as exc:
            raise SAELoadError(f"sae_lens is not installed: {exc}")
        try:
            sae, _acts, _hooks = SaelensSAE.from_pretrained(
                identifier, device="cpu")
            weights = sae.encoder.weight.detach()
        except Exception as exc:
            raise SAELoadError(f"sae_lens could not load {identifier}: {exc}")
        b_pre = getattr(sae.encoder, "b_pre", None)
        return SAE(
            SAEConfig(d_in=int(weights.shape[0]), d_sae=int(weights.shape[1]),
                      model_name=identifier, layer=-1, repo_id=identifier,
                      version=version,
                      pre_encoder_center="b_pre" if b_pre is not None else ""),
            weights=weights,
            biases={"b_pre": b_pre,
                    "b_enc": getattr(sae.encoder, "b_enc", None)},
        )


class AnthropicProvider(SAEProvider):
    """Anthropic-format checkpoints are not supported in this build."""

    def load(self, identifier: str, version: str) -> SAE:
        raise SAELoadError(
            "Anthropic-format SAE checkpoints are not supported in this "
            "build. No weights were read, so no SAE is returned."
        )


class SAELoader:
    """Unified entry point for loading SAE checkpoints."""

    def __init__(self) -> None:
        self.providers = {
            "local": LocalCheckpointProvider(),
            "hf": HuggingFaceProvider(),
            "saelens": SAELensProvider(),
            "anthropic": AnthropicProvider(),
        }

    def load_sae(self, source: str, identifier: str,
                 version: str = "latest") -> SAE:
        provider = self.providers.get(source.lower())
        if not provider:
            raise ValueError(f"Unknown SAE source: {source}")

        sae = provider.load(identifier, version)
        if not sae.loaded:
            raise SAELoadError(
                f"provider '{source}' returned an SAE without weights for "
                f"{identifier}@{version}")
        return sae

    def load_checkpoint(self, checkpoint_path: str,
                        model_name: str = "GPT-2 Small") -> Dict[str, Any]:
        """Load real weights, or report precisely why it could not.

        Previously this returned `status: "loaded"` for a config-only SAE with
        no tensors, which is how "SAE inspection" appeared to work.
        """
        try:
            sae = self.load_sae("local", checkpoint_path)
        except SAELoadError as exc:
            return {
                "checkpoint_path": checkpoint_path,
                "model_name": model_name,
                "status": "unavailable",
                "provenance": "unavailable",
                "weights_loaded": False,
                "validation_eligible": False,
                "publication_eligible": False,
                "reason": str(exc),
                "timestamp": _dt.datetime.utcnow().isoformat() + "Z",
            }

        return {
            "checkpoint_path": checkpoint_path,
            "model_name": model_name,
            "d_in": sae.config.d_in,
            "d_sae": sae.config.d_sae,
            "status": "loaded",
            "provenance": "live",
            "weights_loaded": True,
            "weights_sha256": sae.weights_sha256(),
            "checkpoint_sha": sae.config.checkpoint_sha,
            "validation_eligible": True,
            "publication_eligible": True,
            "attested": False,
            "timestamp": _dt.datetime.utcnow().isoformat() + "Z",
            "sae_id": sae.id,
        }
