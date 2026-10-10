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


class SAE:
    """A loaded Sparse Autoencoder instance backed by real weight tensors."""

    def __init__(self, config: SAEConfig, weights: Any = None,
                 biases: Optional[Dict[str, Any]] = None) -> None:
        self.config = config
        self.weights = weights  # torch.Tensor [d_in, d_sae]
        self.biases = biases or {}
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
        top ``d_sae``/top_k activations. Without weights there is nothing to
        apply, so this refuses instead of returning fixed feature indices.
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


class SAEProvider(ABC):
    """Abstract base for SAE model providers."""

    @abstractmethod
    def load(self, identifier: str, version: str) -> SAE:
        """Fetch real weights and return a loaded SAE.

        Must raise `SAELoadError` when weights are unavailable. Returning a
        config-only SAE is not acceptable.
        """
        ...


def _load_tensors(path: str) -> Tuple[Any, Dict[str, Any]]:
    """Open real SAE weights from disk and hash the file.

    `weights_only=True` is not a stylistic preference here. `weights_only=False`
    is a pickle load, so a checkpoint from anywhere -- a downloaded artefact, a
    path handed in by an API caller, a file in a shared cache -- becomes
    arbitrary code execution inside this process. The previous value was False,
    and it was False with no comment saying why, so it read as an oversight that
    a later reader might "fix" the wrong way.

    `weights_only=True` refuses to reconstruct anything that is not a tensor,
    dict, or a small set of primitives. A checkpoint that genuinely needs more
    than that is a checkpoint that has to be converted first, which is the
    correct place to make that decision deliberately.
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

    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        # A checkpoint that cannot be loaded safely is refused, not retried with
        # a permissive loader: retrying is how `weights_only=False` gets
        # reintroduced two releases later.
        raise SAELoadError(
            f"SAE checkpoint could not be loaded with weights_only=True, so it "
            f"was refused rather than unpickled: {type(exc).__name__}: {exc}"
        ) from exc
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

    biases = {
        "b_pre": state.get("b_pre") if state.get("b_pre") is not None
        else state.get("encoder.bias"),
        "b_enc": state.get("b_enc"),
    }
    return weights, biases


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
        weights, biases = _load_tensors(path)

        import torch
        with open(path, "rb") as handle:
            file_sha = hashlib.sha256(handle.read()).hexdigest()

        config = SAEConfig(
            d_in=int(getattr(weights, "shape")[0]),
            d_sae=int(getattr(weights, "shape")[1]),
            model_name=version,
            layer=-1,
            repo_id=identifier,
            version=version,
            checkpoint_sha=f"sha256:{file_sha}",
        )
        return SAE(config, weights=weights, biases=biases)


class HuggingFaceProvider(SAEProvider):
    """Fetches real weights from the HF Hub, or raises."""

    def load(self, identifier: str, version: str) -> SAE:
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as exc:
            raise SAELoadError(
                f"huggingface_hub is required to fetch {identifier}: {exc}")
        try:
            path = hf_hub_download(repo_id=identifier,
                                   filename="weights.pt",
                                   revision=version)
        except Exception as exc:
            raise SAELoadError(
                f"could not fetch SAE weights for {identifier}@{version}: {exc}")

        weights, biases = _load_tensors(path)
        import torch
        with open(path, "rb") as handle:
            file_sha = hashlib.sha256(handle.read()).hexdigest()
        config = SAEConfig(
            d_in=int(weights.shape[0]), d_sae=int(weights.shape[1]),
            model_name=identifier, layer=-1, repo_id=identifier, version=version,
            checkpoint_sha=f"sha256:{file_sha}",
        )
        return SAE(config, weights=weights, biases=biases)


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
        return SAE(
            SAEConfig(d_in=int(weights.shape[0]), d_sae=int(weights.shape[1]),
                      model_name=identifier, layer=-1, repo_id=identifier,
                      version=version),
            weights=weights,
            biases={"b_pre": getattr(sae.encoder, "b_pre", None),
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
