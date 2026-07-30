"""Model Registry — Adapters for real model families.

Supports GPT-2 Small/Medium, Gemma, Llama, Qwen via HuggingFace /
TransformerLens when available, with graceful Ollama fallback for
environments without GPU or API keys.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class ModelFamily(str, Enum):
    GPT2_SMALL  = "gpt2"
    GPT2_MEDIUM = "gpt2-medium"
    GEMMA       = "gemma-2b"
    LLAMA       = "llama-3.2-1b"
    QWEN        = "qwen2.5-0.5b"


@dataclass
class ModelSpec:
    model_id: str
    family: ModelFamily
    n_layers: int
    n_heads: int
    d_model: int
    d_mlp: int
    n_params_b: float          # billions
    context_length: int
    hf_name: str               # HuggingFace model card name
    tl_name: Optional[str]     # TransformerLens model name (if supported)
    ollama_tag: Optional[str]  # Ollama model tag for fallback
    supports_sae: bool = False
    supports_tl: bool = False


# ------------------------------------------------------------------ #
# Model catalogue                                                      #
# ------------------------------------------------------------------ #

MODEL_CATALOGUE: Dict[ModelFamily, ModelSpec] = {
    ModelFamily.GPT2_SMALL: ModelSpec(
        model_id="gpt2_small",
        family=ModelFamily.GPT2_SMALL,
        n_layers=12, n_heads=12, d_model=768, d_mlp=3072,
        n_params_b=0.117,
        context_length=1024,
        hf_name="openai-community/gpt2",
        tl_name="gpt2",
        ollama_tag=None,
        supports_sae=True,
        supports_tl=True,
    ),
    ModelFamily.GPT2_MEDIUM: ModelSpec(
        model_id="gpt2_medium",
        family=ModelFamily.GPT2_MEDIUM,
        n_layers=24, n_heads=16, d_model=1024, d_mlp=4096,
        n_params_b=0.345,
        context_length=1024,
        hf_name="openai-community/gpt2-medium",
        tl_name="gpt2-medium",
        ollama_tag=None,
        supports_sae=True,
        supports_tl=True,
    ),
    ModelFamily.GEMMA: ModelSpec(
        model_id="gemma_2b",
        family=ModelFamily.GEMMA,
        n_layers=18, n_heads=8, d_model=2048, d_mlp=16384,
        n_params_b=2.0,
        context_length=8192,
        hf_name="google/gemma-2b",
        tl_name="gemma-2b",
        ollama_tag="gemma:2b",
        supports_sae=False,
        supports_tl=True,
    ),
    ModelFamily.LLAMA: ModelSpec(
        model_id="llama_3b",
        family=ModelFamily.LLAMA,
        n_layers=16, n_heads=8, d_model=2048, d_mlp=8192,
        n_params_b=1.0,
        context_length=131072,
        hf_name="meta-llama/Llama-3.2-1B",
        tl_name=None,
        ollama_tag="llama3.2:1b",
        supports_sae=False,
        supports_tl=False,
    ),
    ModelFamily.QWEN: ModelSpec(
        model_id="qwen2_5_05b",
        family=ModelFamily.QWEN,
        n_layers=24, n_heads=14, d_model=896, d_mlp=4864,
        n_params_b=0.5,
        context_length=32768,
        hf_name="Qwen/Qwen2.5-0.5B",
        tl_name=None,
        ollama_tag="qwen2.5:0.5b",
        supports_sae=False,
        supports_tl=False,
    ),
}


@dataclass
class ModelAvailability:
    """Result of probing whether a model can actually run in this environment."""
    spec: ModelSpec
    available: bool
    backend: str           # "transformerlens" | "huggingface" | "ollama" | "stub"
    error: Optional[str] = None
    probe_latency_ms: float = 0.0


class ModelRegistry:
    """
    Probes which models are actually runnable in the current environment.

    Priority order:
      1. TransformerLens (fastest for GPT-2 family, required for hook-level analysis)
      2. HuggingFace (direct model load)
      3. Ollama (local inference server, no GPU required)
      4. Stub (offline fallback — returns deterministic synthetic activations)
    """

    def __init__(self) -> None:
        self._availability: Dict[ModelFamily, ModelAvailability] = {}

    def probe_all(self) -> Dict[ModelFamily, ModelAvailability]:
        """Probe every model in the catalogue. Caches result."""
        for family, spec in MODEL_CATALOGUE.items():
            self._availability[family] = self._probe(spec)
        return self._availability

    def probe(self, family: ModelFamily) -> ModelAvailability:
        if family not in self._availability:
            self._availability[family] = self._probe(MODEL_CATALOGUE[family])
        return self._availability[family]

    def available_models(self) -> List[ModelSpec]:
        if not self._availability:
            self.probe_all()
        return [a.spec for a in self._availability.values() if a.available]

    def get_spec(self, family: ModelFamily) -> ModelSpec:
        return MODEL_CATALOGUE[family]

    # ------------------------------------------------------------------ #
    # Internal probing logic                                               #
    # ------------------------------------------------------------------ #

    def _probe(self, spec: ModelSpec) -> ModelAvailability:
        t0 = time.perf_counter()

        # 1. TransformerLens
        if spec.supports_tl and spec.tl_name:
            result = self._try_tl(spec)
            if result.available:
                result.probe_latency_ms = (time.perf_counter() - t0) * 1000
                return result

        # 2. HuggingFace
        result = self._try_hf(spec)
        if result.available:
            result.probe_latency_ms = (time.perf_counter() - t0) * 1000
            return result

        # 3. Ollama
        if spec.ollama_tag:
            result = self._try_ollama(spec)
            if result.available:
                result.probe_latency_ms = (time.perf_counter() - t0) * 1000
                return result

        # 4. Stub fallback — always available, clearly labelled
        return ModelAvailability(
            spec=spec,
            available=True,
            backend="stub",
            probe_latency_ms=(time.perf_counter() - t0) * 1000,
        )

    @staticmethod
    def _try_tl(spec: ModelSpec) -> ModelAvailability:
        try:
            import transformer_lens  # noqa: F401
            return ModelAvailability(spec=spec, available=True, backend="transformerlens")
        except ImportError as e:
            return ModelAvailability(spec=spec, available=False, backend="transformerlens", error=str(e))

    @staticmethod
    def _try_hf(spec: ModelSpec) -> ModelAvailability:
        try:
            import transformers  # noqa: F401
            return ModelAvailability(spec=spec, available=True, backend="huggingface")
        except ImportError as e:
            return ModelAvailability(spec=spec, available=False, backend="huggingface", error=str(e))

    @staticmethod
    def _try_ollama(spec: ModelSpec) -> ModelAvailability:
        try:
            import urllib.request
            with urllib.request.urlopen("http://localhost:11434/api/tags", timeout=1) as resp:
                if resp.status == 200:
                    return ModelAvailability(spec=spec, available=True, backend="ollama")
        except Exception as e:  # noqa: BLE001
            return ModelAvailability(spec=spec, available=False, backend="ollama", error=str(e))
        return ModelAvailability(spec=spec, available=False, backend="ollama")
