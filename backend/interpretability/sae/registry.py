"""Known real, public, open-weights SAE checkpoints for GPT-2 small.

An SAE describes the activations of ONE hook point on ONE model. Applying a
layer-8 `hook_resid_pre` dictionary to layer-4 activations yields feature indices
that describe nothing, so the registry carries the layer and hook point as
facts, and the loader refuses a mismatch rather than silently encoding the
wrong activation.

Verified on 2026-10-17 by downloading both checkpoints and reading their `cfg`:
`d_in` 768, `d_sae` 24576, `model_name` 'gpt2-small', `hook_point`
'blocks.{L}.hook_resid_pre', `dataset_path` 'Skylion007/openwebtext',
`l1_coefficient` 8e-05.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List, Optional

REPO_ID = "jbloom/GPT2-Small-SAEs"
#: Snapshot revision the metadata below was read from.
REVISION = "3a66e1866bbc104b202fb2a6ba72e21cc9ee6014"


@dataclass(frozen=True)
class SAERegistryEntry:
    """Everything needed to interpret a dictionary's output correctly."""

    layer: int
    hook_point: str
    repo_id: str
    filename: str
    d_in: int
    d_sae: int
    model_id: str
    dataset: str
    l1_coefficient: float
    revision: str

    def as_dict(self) -> Dict[str, object]:
        return asdict(self)


def _entry(layer: int) -> SAERegistryEntry:
    hook = f"blocks.{layer}.hook_resid_pre"
    return SAERegistryEntry(
        layer=layer,
        hook_point=hook,
        repo_id=REPO_ID,
        filename=f"final_sparse_autoencoder_gpt2-small_{hook}_24576.pt",
        d_in=768,
        d_sae=24576,
        model_id="gpt2-small",
        dataset="Skylion007/openwebtext",
        l1_coefficient=8e-05,
        revision=REVISION,
    )


#: Layers with a checkpoint known to exist and to have been inspected.
KNOWN_SAES: Dict[int, SAERegistryEntry] = {layer: _entry(layer) for layer in (8, 10)}


def resolve(layer: int) -> Optional[SAERegistryEntry]:
    """The registry entry for ``layer``, or ``None`` when none is known."""
    try:
        return KNOWN_SAES.get(int(layer))
    except (TypeError, ValueError):
        return None


def available_layers() -> List[int]:
    return sorted(KNOWN_SAES)


def fetch(layer: int) -> str:
    """Download or locate the cached checkpoint for ``layer`` from HuggingFace Hub."""
    entry = resolve(layer)
    if entry is None:
        raise ValueError(
            f"No known SAE checkpoint for layer {layer}. Known layers: {available_layers()}."
        )
    from huggingface_hub import hf_hub_download

    return hf_hub_download(
        repo_id=entry.repo_id,
        filename=entry.filename,
        revision=entry.revision,
    )


def describe() -> List[Dict[str, object]]:
    """Machine-readable inventory for the UI / diagnostics."""
    return [KNOWN_SAES[layer].as_dict() for layer in available_layers()]
