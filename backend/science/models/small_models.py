"""Real small models for consumer hardware.

Why this module exists
----------------------
`model_adapters.py` names five families -- Gemma-2B, Llama-3.2-3B/2-7B,
Qwen2.5-0.5B/1.5B, Mistral-7B, DeepSeek-1.5B -- whose published sizes reach
13.5 GB. On a 6 GB card most of them cannot be loaded at all, and the ones that
can are the exception rather than the rule. A registry whose entries cannot run
is a list of names, not a capability.

This module is the small half of that: models that fit in a few hundred MB to a
few GB of 16-bit weights, whose architecture numbers were read from the
checkpoint's own `config.json` rather than typed from memory, and whose weight
byte counts were read from the Hub's file sizes. `HFAdapterMixin` supplies the
five measured methods unchanged, so a measurement from a 135M SmolLM2 and a
measurement from GPT-2 are produced by the same code.

Loading policy: local first, download only on request
----------------------------------------------------
`load_small_model` reads the local cache first (`local_files_only=True`). It
touches the network only when `MECH_ALLOW_MODEL_DOWNLOAD=1` is set, or when
`allow_download=True` is passed explicitly. An offline box therefore fails with
a reason that names the repo and the switch to set, instead of appearing to hang
while a fetch stalls.

Gated repos are refused, not attempted
--------------------------------------
`meta-llama/Llama-3.2-1B` is a gated repo. Measured on this host:

    OSError: You are trying to access a gated repo. ... 401 Client Error.

Attempting it produces a download-failure message that reads like a network
fault. The gate is the reason, so it is reported as the reason: a gated entry
without a Hub token raises ``"gated: needs HF token"`` before any request is
made.

Nothing here fabricates
-----------------------
A model that cannot be loaded raises :class:`SmallModelUnavailable` (a
``LiveUnavailable``) carrying the reason. It is never silently replaced by a
simulating adapter: the caller either holds weights and measures, or holds a
refusal. `provenance` on every measured result is ``"live"``, and no path in
this module can produce that string without a completed forward pass.

Measured on this host (2026-10-17, 6 GB RTX 3050, CPython 3.11.9)
----------------------------------------------------------------
Loaded and measured end to end: ``qwen2.5-0.5b`` (cached), ``smollm2-135m``
(downloaded with the opt-in switch), ``pythia-70m`` (downloaded, and the only
entry whose blocks are laid out as ``gpt_neox.layers``). Registered but not
downloaded on this host: ``smollm2-360m``, ``qwen2.5-1.5b`` (weights present,
not driven), ``pythia-160m``. Registered and unavailable: ``llama-3.2-1b``
(gated).
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

from .adapter_base import LiveUnavailable, ModelAdapter, ModelSpec
from .hf_adapter import HFAdapterMixin

#: Set to ``1`` to permit a Hub download for a model that is not cached.
DOWNLOAD_ENV = "MECH_ALLOW_MODEL_DOWNLOAD"

#: Device the loader places weights on unless told otherwise. CPU is the
#: default because this box is shared: another process may be holding the GPU,
#: and a small model on CPU is still a measurement.
DEVICE_ENV = "MECH_SMALL_MODEL_DEVICE"

#: Environment variables huggingface_hub reads for an access token, in the
#: order it consults them.
TOKEN_ENVS: Tuple[str, ...] = (
    "HF_TOKEN",
    "HUGGING_FACE_HUB_TOKEN",
    "HUGGINGFACE_HUB_TOKEN",
)


@dataclass(frozen=True)
class SmallModelSpec:
    """One small model, with the numbers that decide whether it can run here.

    ``params_millions`` and ``approx_fp16_vram_gb`` are **measured**: the Hub's
    per-file byte counts for the weight shards, divided by two (every entry here
    ships 16-bit weights) for the parameter count, and the raw byte total for the
    memory figure. They are not the numbers in the model's name. The two differ
    for the Pythia entries, whose untied embedding is not counted in the name:
    ``pythia-70m`` weighs 332,078,951 bytes, i.e. ~166M parameters.

    ``n_kv_heads`` is ``None`` where the checkpoint declares none, which means
    multi-head attention and therefore ``n_kv_heads == n_heads``. ``None`` is
    kept rather than filled in, so "the config did not say" stays distinguishable
    from "the config said equal".
    """

    model_id: str
    hf_repo_id: str
    arch: str
    params_millions: float
    d_model: int
    n_layers: int
    n_heads: int
    n_kv_heads: Optional[int]
    gated: bool
    approx_fp16_vram_gb: float
    licence: str
    vocab_size: int
    context_length: int
    d_mlp: Optional[int] = None
    #: True when ``arch``/``d_model``/``n_layers``/``n_heads``/``n_kv_heads``
    #: were read from the checkpoint's ``config.json`` rather than published
    #: documentation. Every non-gated entry here is True; the gated entry could
    #: not be read (401) so its shape is documentation-sourced.
    shape_verified: bool = False


#: The registry. Sizes and shapes verified against each checkpoint's own
#: ``config.json`` and the Hub's file metadata; see ``_verify_registry``.
SMALL_MODELS: Dict[str, SmallModelSpec] = {
    "smollm2-135m": SmallModelSpec(
        model_id="smollm2-135m",
        hf_repo_id="HuggingFaceTB/SmolLM2-135M",
        arch="LlamaForCausalLM",
        params_millions=134.5,
        d_model=576, n_layers=30, n_heads=9, n_kv_heads=3,
        gated=False, approx_fp16_vram_gb=0.27, licence="apache-2.0",
        vocab_size=49152, context_length=8192, d_mlp=1536,
        shape_verified=True,
    ),
    "smollm2-360m": SmallModelSpec(
        model_id="smollm2-360m",
        hf_repo_id="HuggingFaceTB/SmolLM2-360M",
        arch="LlamaForCausalLM",
        params_millions=361.8,
        d_model=960, n_layers=32, n_heads=15, n_kv_heads=5,
        gated=False, approx_fp16_vram_gb=0.72, licence="apache-2.0",
        vocab_size=49152, context_length=8192, d_mlp=2560,
        shape_verified=True,
    ),
    "qwen2.5-0.5b": SmallModelSpec(
        model_id="qwen2.5-0.5b",
        hf_repo_id="Qwen/Qwen2.5-0.5B",
        arch="Qwen2ForCausalLM",
        params_millions=494.0,
        d_model=896, n_layers=24, n_heads=14, n_kv_heads=2,
        gated=False, approx_fp16_vram_gb=0.99, licence="apache-2.0",
        vocab_size=151936, context_length=32768, d_mlp=4864,
        shape_verified=True,
    ),
    "qwen2.5-1.5b": SmallModelSpec(
        model_id="qwen2.5-1.5b",
        hf_repo_id="Qwen/Qwen2.5-1.5B",
        arch="Qwen2ForCausalLM",
        params_millions=1543.7,
        d_model=1536, n_layers=28, n_heads=12, n_kv_heads=2,
        gated=False, approx_fp16_vram_gb=3.09, licence="apache-2.0",
        vocab_size=151936, context_length=131072, d_mlp=8960,
        shape_verified=True,
    ),
    "llama-3.2-1b": SmallModelSpec(
        model_id="llama-3.2-1b",
        hf_repo_id="meta-llama/Llama-3.2-1B",
        arch="LlamaForCausalLM",
        params_millions=1235.8,
        d_model=2048, n_layers=16, n_heads=32, n_kv_heads=8,
        gated=True, approx_fp16_vram_gb=2.47, licence="llama3.2",
        vocab_size=128256, context_length=131072, d_mlp=8192,
        # The repo returns 401 without a token, so these are the published
        # config values, not values read off the checkpoint on this host.
        shape_verified=False,
    ),
    "pythia-70m": SmallModelSpec(
        model_id="pythia-70m",
        hf_repo_id="EleutherAI/pythia-70m",
        arch="GPTNeoXForCausalLM",
        params_millions=166.0,
        d_model=512, n_layers=6, n_heads=8, n_kv_heads=None,
        gated=False, approx_fp16_vram_gb=0.33, licence="apache-2.0",
        vocab_size=50304, context_length=2048, d_mlp=2048,
        shape_verified=True,
    ),
    "pythia-160m": SmallModelSpec(
        model_id="pythia-160m",
        hf_repo_id="EleutherAI/pythia-160m",
        arch="GPTNeoXForCausalLM",
        params_millions=375.0,
        d_model=768, n_layers=12, n_heads=12, n_kv_heads=None,
        gated=False, approx_fp16_vram_gb=0.75, licence="apache-2.0",
        vocab_size=50304, context_length=2048, d_mlp=3072,
        shape_verified=True,
    ),
}

#: Aliases that resolve to the same checkpoint. Only exact synonyms: a repo id
#: and its short id are the same weights, so pointing one at the other cannot
#: hand back a different model's numbers under a familiar name.
_ALIASES: Dict[str, str] = {
    "smollm2-135m-instruct": "smollm2-135m",
    "smollm2-360m-instruct": "smollm2-360m",
    "qwen2.5-0.5b-instruct": "qwen2.5-0.5b",
    "qwen2.5-1.5b-instruct": "qwen2.5-1.5b",
    "pythia-70m-deduped": "pythia-70m",
    "pythia-160m-deduped": "pythia-160m",
}

#: Not aliases, deliberately: `HuggingFaceTB/SmolLM2-135M` and
#: `...-135M-Instruct` are different fine-tunes of one base. They are not in
#: `SMALL_MODELS` because their weights are different files; adding one means
#: adding a spec, not an alias.


class SmallModelUnavailable(LiveUnavailable):
    """No weights for this small model are available, with the reason.

    ``LiveUnavailable`` is the project's NOT_RUN marker: a statement about what
    could not be measured, distinct from a measurement that broke. Raised in
    place of returning a simulating adapter, so a caller cannot mistake a
    refusal for fixtures.
    """

    def __init__(self, model_id: str, reason: str, *,
                 gated: bool = False, hint: Optional[str] = None) -> None:
        self.model_id = model_id
        self.reason = reason
        self.gated = gated
        self.hint = hint
        super().__init__(reason)


def _hf_token_present() -> bool:
    return any(os.environ.get(name) for name in TOKEN_ENVS)


def downloads_allowed(allow_download: Optional[bool] = None) -> bool:
    """Whether a Hub fetch is permitted for an uncached model.

    Explicit argument wins; otherwise ``MECH_ALLOW_MODEL_DOWNLOAD=1``. Anything
    else -- unset, ``0``, ``false`` -- means local-only, because the failure mode
    of an accidental download is minutes of stalled I/O on a box that may be
    offline.
    """
    if allow_download is not None:
        return bool(allow_download)
    return os.environ.get(DOWNLOAD_ENV, "").strip().lower() in {"1", "true", "yes"}


def resolve(name: str) -> SmallModelSpec:
    """Registry entry for ``name``, following exact-synonym aliases."""
    key = (name or "").strip().lower()
    key = _ALIASES.get(key, key)
    entry = SMALL_MODELS.get(key)
    if entry is None:
        raise KeyError(
            f"{name!r} is not a registered small model. Available: "
            f"{sorted(SMALL_MODELS)}")
    return entry


def _cached_snapshot_dir(repo_id: str) -> Optional[str]:
    """The local snapshot directory for ``repo_id``, or ``None``.

    Read from ``huggingface_hub``'s own cache root rather than a hand-built
    path, so an ``HF_HOME``/``HF_HUB_CACHE`` override is honoured.
    """
    try:
        from huggingface_hub.constants import HF_HUB_CACHE
    except Exception:
        return None
    folder = os.path.join(HF_HUB_CACHE, "models--" + repo_id.replace("/", "--"),
                          "snapshots")
    if not os.path.isdir(folder):
        return None
    candidates = sorted(
        (os.path.join(folder, name) for name in os.listdir(folder)),
        key=os.path.getmtime)
    return candidates[-1] if candidates else None


#: Weight-file suffixes a `transformers` checkpoint can ship.
_WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pt", ".pth")


def local_weights_present(entry: SmallModelSpec) -> Tuple[bool, str]:
    """Whether this checkpoint's **weights** are in the local HF cache.

    Deliberately not a config check. A config-only cache entry is created by any
    metadata read -- including the ``AutoConfig.from_pretrained`` this module's
    own registry verification performs -- and reporting that as "cached" would
    say a model is loadable when nothing but its ``config.json`` is on disk.
    That distinction is the whole point of the status column, so it is measured
    from the presence of a weight file.
    """
    snapshot = _cached_snapshot_dir(entry.hf_repo_id)
    if snapshot is None:
        return False, "no snapshot directory in the local cache"
    try:
        names = os.listdir(snapshot)
    except OSError as exc:
        return False, f"snapshot directory unreadable ({exc})"
    weights = [n for n in names
               if n.endswith(_WEIGHT_SUFFIXES) and "optimizer" not in n]
    if not weights:
        return False, ("config-only cache entry: no weight file "
                       "(*.safetensors, *.bin, *.pt) present")
    return True, f"{len(weights)} weight file(s) cached"


def local_cache_present(entry: SmallModelSpec) -> Tuple[bool, str]:
    """Alias of :func:`local_weights_present`, kept for call-site clarity."""
    return local_weights_present(entry)


def available_small_models() -> List[Dict[str, Any]]:
    """The registry with a per-entry honest status, for a UI or a report.

    ``loadable_now`` means the weights are in the local cache. It does not mean a
    forward pass was run; nothing in this function executes a model.
    """
    rows: List[Dict[str, Any]] = []
    for key in sorted(SMALL_MODELS):
        entry = SMALL_MODELS[key]
        cached, cache_reason = local_weights_present(entry)
        if entry.gated and not _hf_token_present():
            status = "gated: needs HF token"
        elif cached:
            status = "cached"
        else:
            status = "not downloaded"
        rows.append({
            **asdict(entry),
            "registered": True,
            "locally_cached": cached,
            "cache_reason": cache_reason,
            "loadable_now": bool(cached and not entry.gated),
            "status": status,
        })
    return rows


class SmallModelAdapter(HFAdapterMixin, ModelAdapter):
    """A loaded small model, with the five measured methods from ``HFAdapterMixin``.

    Construction is deliberately not `ModelAdapter.__init__`: that constructor
    calls ``self._load_model()`` whenever ``spec.mock_mode`` is false, which
    would attempt a load with the network-permitted path before this module's
    local-first policy had a chance to run. The attributes the mixin expects are
    set here instead, and :meth:`load` is the single place weights are fetched.

    There is no mock branch. A caller either holds a loaded model or holds a
    :class:`SmallModelUnavailable`; a simulated small model would be a fixture
    wearing a real repo id, which is the failure this repository has already
    removed once.
    """

    def __init__(self, entry: SmallModelSpec, *, model: Any, tokenizer: Any,
                 device: str, precision: str, allow_download: bool,
                 spec: ModelSpec) -> None:
        self.entry = entry
        self.spec = spec
        self._model = model
        self._tokenizer = tokenizer
        self._cache = None
        self._manager = None
        self._load_failure = None
        self.simulated = False
        self.simulation_reason = ""
        self.mock_mode = False
        self.device = device
        self.precision = precision
        self.allow_download = allow_download
        self.loaded = True

    #: The mixin reads `spec.mock_mode`; this instance is never mock.
    def _load_model(self) -> None:  # pragma: no cover - load() is the entry point
        raise RuntimeError(
            "SmallModelAdapter loads through load_small_model(), not by "
            "re-entering _load_model(). Nothing was loaded.")

    def describe(self) -> Dict[str, Any]:
        """What is loaded, and what it was loaded from."""
        return {
            "model_id": self.entry.model_id,
            "hf_repo_id": self.entry.hf_repo_id,
            "arch": type(self._model).__name__ if self._model is not None else None,
            "device": self.device,
            "precision": self.precision,
            "params_millions": self.entry.params_millions,
            "approx_fp16_vram_gb": self.entry.approx_fp16_vram_gb,
            "licence": self.entry.licence,
            "gated": self.entry.gated,
            "provenance": "live",
            "simulated": False,
            "mock_mode": False,
            "shape": self.get_model_spec(),
        }

    def close(self) -> None:
        """Drop the weights. Idempotent."""
        self._model = None
        self._tokenizer = None
        self._cache = None
        self.loaded = False


def _pick_device(requested: Optional[str]) -> str:
    if requested:
        return requested
    env = os.environ.get(DEVICE_ENV)
    if env:
        return env
    return "cpu"


def _pick_precision(device: str, requested: Optional[str]) -> Tuple[str, Any]:
    """``(name, torch dtype)``. 16-bit on an accelerator, 32-bit on CPU.

    The dtype is pinned and recorded rather than inherited from the checkpoint,
    for the same reason `hf_adapter` pins it: a number measured in bf16 and
    reported beside fp32 baselines is a difference nobody can interpret.
    """
    import torch

    if requested:
        normalised = requested.lower()
        mapping = {
            "float32": torch.float32, "fp32": torch.float32,
            "float16": torch.float16, "fp16": torch.float16,
            "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
        }
        if normalised not in mapping:
            raise ValueError(
                f"Unknown dtype {requested!r}. Use one of "
                f"{sorted(set(mapping))}.")
        return normalised, mapping[normalised]
    if device.startswith("cuda"):
        return "float16", torch.float16
    return "float32", torch.float32


def load_small_model(
    name: str,
    *,
    allow_download: Optional[bool] = None,
    device: Optional[str] = None,
    dtype: Optional[str] = None,
) -> SmallModelAdapter:
    """Load a registered small model, local cache first.

    Raises :class:`SmallModelUnavailable` with the reason when the weights are
    not here and a download is not permitted, and with ``"gated: needs HF
    token"`` when the repo is gated and no token is set. Raises ``KeyError`` for
    an unregistered name.
    """
    entry = resolve(name)
    allow = downloads_allowed(allow_download)

    if entry.gated and not _hf_token_present():
        raise SmallModelUnavailable(
            entry.model_id,
            f"gated: needs HF token. {entry.hf_repo_id} is a gated repository, "
            f"so no request was made. Accept the licence at "
            f"https://huggingface.co/{entry.hf_repo_id} and set one of "
            f"{list(TOKEN_ENVS)}. Measured without a token, the Hub answers "
            f"401 Client Error: 'You are trying to access a gated repo.'",
            gated=True,
            hint="set HF_TOKEN after accepting the licence",
        )

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as exc:  # pragma: no cover
        raise SmallModelUnavailable(
            entry.model_id,
            f"transformers is not importable ({exc}), so no weights for "
            f"{entry.hf_repo_id} could be loaded.",
        ) from exc

    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "20")

    model, tokenizer, used_download = _load_with_policy(
        AutoModelForCausalLM, AutoTokenizer, entry, allow, dtype, device)

    loaded_shape = _shape_from_config(getattr(model, "config", None))
    shape, disagreements = _reconcile_shape(entry, loaded_shape)

    spec = ModelSpec(
        model_id=entry.model_id,
        family=_family_for(entry),
        num_layers=shape["n_layers"],
        num_heads=shape["n_heads"],
        d_model=shape["d_model"],
        d_mlp=shape["d_mlp"],
        vocab_size=shape["vocab_size"],
        context_length=shape["context_length"],
        hf_repo_id=entry.hf_repo_id,
        supports_sae=False,
        mock_mode=False,
    )

    device_used = _pick_device(device)
    precision_used = _precision_name(dtype, device_used)
    model.to(device_used)
    model.eval()

    adapter = SmallModelAdapter(
        entry, model=model, tokenizer=tokenizer, device=device_used,
        precision=precision_used, allow_download=allow, spec=spec)
    adapter.downloaded = used_download
    adapter.shape_disagreements = disagreements
    return adapter


def _family_for(entry: SmallModelSpec) -> str:
    repo = entry.hf_repo_id.lower()
    if "smollm" in repo or "llama" in repo:
        return "llama"
    if "qwen" in repo:
        return "qwen"
    if "pythia" in repo:
        return "pythia"
    return "unknown"


def _precision_name(dtype: Optional[str], device: str) -> str:
    if dtype:
        return dtype.lower()
    return "float16" if device.startswith("cuda") else "float32"


def _load_model_local_first(auto_model, repo_id: str, torch_dtype, *,
                            local_only: bool):
    """``from_pretrained`` with ``attn_implementation="eager"``, or without it.

    Eager attention is requested deliberately: `sdpa`, the default for these
    configs, ignores `output_attentions=True` and returns `attentions=None`, so
    `get_attention_patterns` -- one of the five measured methods -- could never
    produce a number. Measured on this host with `qwen2.5-0.5b`: under `sdpa`
    the call reports "The model returned no attention weights"; under eager it
    returns a real pattern. Where `transformers` does not accept the argument it
    is dropped rather than failing the load, and the adapter reports the same
    honest refusal instead.
    """
    try:
        return auto_model.from_pretrained(
            repo_id, local_files_only=local_only, torch_dtype=torch_dtype,
            attn_implementation="eager")
    except TypeError:
        return auto_model.from_pretrained(
            repo_id, local_files_only=local_only, torch_dtype=torch_dtype)


def _load_with_policy(auto_model, auto_tokenizer, entry: SmallModelSpec,
                      allow: bool, dtype: Optional[str], device: Optional[str]):
    """Local cache first; the Hub only when explicitly allowed.

    Returns ``(model, tokenizer, downloaded)``. Every failure is converted into a
    ``SmallModelUnavailable`` that names the repo and the switch, so an offline
    box produces a reason rather than a stall.
    """
    torch_dtype = _torch_dtype(dtype, _pick_device(device))
    local_error: Optional[BaseException] = None

    try:
        tokenizer = auto_tokenizer.from_pretrained(
            entry.hf_repo_id, local_files_only=True)
        model = _load_model_local_first(
            auto_model, entry.hf_repo_id, torch_dtype, local_only=True)
        return model, tokenizer, False
    except Exception as exc:
        local_error = exc

    if not allow:
        raise SmallModelUnavailable(
            entry.model_id,
            f"the weights for {entry.hf_repo_id} are not in the local cache "
            f"and downloads are off ({type(local_error).__name__}), so nothing "
            f"was loaded. Set {DOWNLOAD_ENV}=1 to fetch them, or pre-fetch with "
            f"`huggingface-cli download {entry.hf_repo_id}`.",
            hint=f"{DOWNLOAD_ENV}=1",
        )

    try:
        tokenizer = auto_tokenizer.from_pretrained(entry.hf_repo_id)
        model = _load_model_local_first(
            auto_model, entry.hf_repo_id, torch_dtype, local_only=False)
        return model, tokenizer, True
    except Exception as exc:
        raise SmallModelUnavailable(
            entry.model_id,
            f"{entry.hf_repo_id} could not be fetched "
            f"({type(exc).__name__}: {str(exc)[:200]}). Local read first "
            f"failed with {type(local_error).__name__}.",
        ) from exc


def _torch_dtype(dtype: Optional[str], device: str):
    import torch

    if dtype:
        naming = {
            "float32": torch.float32, "fp32": torch.float32,
            "float16": torch.float16, "fp16": torch.float16,
            "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
        }
        normalised = dtype.lower()
        if normalised not in naming:
            raise ValueError(
                f"Unknown dtype {dtype!r}. Use one of {sorted(set(naming))}.")
        return naming[normalised]
    return torch.float16 if device.startswith("cuda") else torch.float32


def _shape_from_config(config: Any) -> Dict[str, Any]:
    if config is None:
        return {}
    return {
        "n_layers": getattr(config, "num_hidden_layers", None),
        "n_heads": getattr(config, "num_attention_heads", None),
        "n_kv_heads": getattr(config, "num_key_value_heads", None),
        "d_model": getattr(config, "hidden_size", None),
        "d_mlp": getattr(config, "intermediate_size", None),
        "vocab_size": getattr(config, "vocab_size", None),
        "context_length": getattr(config, "max_position_embeddings", None),
    }


def _reconcile_shape(entry: SmallModelSpec,
                     loaded: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
    """The loaded config wins; the registry's disagreement is reported.

    The registry table is a claim about a checkpoint. Where it and the loaded
    config differ, the config is what was measured, so it is what the adapter
    reports -- and the difference is carried on the adapter rather than
    discarded, so a stale table cannot quietly relabel a model's numbers.
    """
    declared = {
        "n_layers": entry.n_layers,
        "n_heads": entry.n_heads,
        "n_kv_heads": entry.n_kv_heads,
        "d_model": entry.d_model,
        "d_mlp": entry.d_mlp,
        "vocab_size": entry.vocab_size,
        "context_length": entry.context_length,
    }
    disagreements: List[str] = []
    shape: Dict[str, Any] = {}
    for field, fallback in declared.items():
        actual = loaded.get(field)
        if actual is None:
            shape[field] = fallback
            continue
        if actual != fallback:
            disagreements.append(f"{field}: registry={fallback!r} config={actual!r}")
        shape[field] = actual
    return shape, disagreements


__all__ = [
    "SMALL_MODELS",
    "SmallModelSpec",
    "SmallModelAdapter",
    "SmallModelUnavailable",
    "DOWNLOAD_ENV",
    "DEVICE_ENV",
    "TOKEN_ENVS",
    "resolve",
    "downloads_allowed",
    "local_cache_present",
    "available_small_models",
    "load_small_model",
]
