"""MECH Real Model Zoo Registry & Taxonomy.

Categorizes validated open checkpoints across scale tiers (Nano <100M -> 7B+ Scale)
with complete architectural metadata, parameter counts, and norm/attention specifications.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Dict, List, Optional


class ScaleTier(str, Enum):
    TIER_0_NANO = "TIER_0_NANO"       # < 100M (pythia-70m, distilgpt2)
    TIER_1_SMALL = "TIER_1_SMALL"     # 100M - 300M (gpt2-124M, opt-125m, gpt-neo-125m, pythia-160m)
    TIER_2_MEDIUM = "TIER_2_MEDIUM"   # 300M - 1.5B (qwen2.5-0.5b, tinyllama-1.1b, gpt2-medium-355m, pythia-410m)
    TIER_3_LARGE = "TIER_3_LARGE"     # 3B - 7B (qwen2.5-3b, llama-3.2-3b, mistral-7b, llama-2-7b)
    TIER_4_MASSIVE = "TIER_4_MASSIVE" # 13B - 70B+ (llama-2-13b, llama-3-70b, qwen2.5-72b)


@dataclass(frozen=True)
class ModelZooEntry:
    """Metadata and architectural descriptor for an open model checkpoint."""
    model_id: str
    display_name: str
    family: str
    scale_tier: ScaleTier
    parameter_count: int
    num_layers: int
    hidden_dim: int
    num_attention_heads: int
    vocab_size: int
    norm_type: str                     # "layernorm" | "rmsnorm"
    rotary_embeddings: bool
    gated_mlp: bool
    context_window: int
    recommended_min_vram_mb_fp16: float
    recommended_min_ram_mb_ooc: float
    hf_hub_url: str
    open_access: bool = True

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["scale_tier"] = self.scale_tier.value
        return d


MODEL_ZOO_REGISTRY: Dict[str, ModelZooEntry] = {
    # ── Tier 0: Nano (< 100M) ───────────────────────────────────────
    "EleutherAI/pythia-70m": ModelZooEntry(
        model_id="EleutherAI/pythia-70m",
        display_name="Pythia 70M",
        family="gpt_neox",
        scale_tier=ScaleTier.TIER_0_NANO,
        parameter_count=70_426_624,
        num_layers=6,
        hidden_dim=512,
        num_attention_heads=8,
        vocab_size=50304,
        norm_type="layernorm",
        rotary_embeddings=True,
        gated_mlp=False,
        context_window=2048,
        recommended_min_vram_mb_fp16=256.0,
        recommended_min_ram_mb_ooc=512.0,
        hf_hub_url="https://huggingface.co/EleutherAI/pythia-70m",
    ),
    "distilgpt2": ModelZooEntry(
        model_id="distilgpt2",
        display_name="DistilGPT-2 (82M)",
        family="gpt2",
        scale_tier=ScaleTier.TIER_0_NANO,
        parameter_count=81_912_576,
        num_layers=6,
        hidden_dim=768,
        num_attention_heads=12,
        vocab_size=50257,
        norm_type="layernorm",
        rotary_embeddings=False,
        gated_mlp=False,
        context_window=1024,
        recommended_min_vram_mb_fp16=300.0,
        recommended_min_ram_mb_ooc=600.0,
        hf_hub_url="https://huggingface.co/distilgpt2",
    ),

    # ── Tier 1: Small (100M - 300M) ──────────────────────────────────
    "gpt2": ModelZooEntry(
        model_id="gpt2",
        display_name="GPT-2 Base (124M)",
        family="gpt2",
        scale_tier=ScaleTier.TIER_1_SMALL,
        parameter_count=124_439_808,
        num_layers=12,
        hidden_dim=768,
        num_attention_heads=12,
        vocab_size=50257,
        norm_type="layernorm",
        rotary_embeddings=False,
        gated_mlp=False,
        context_window=1024,
        recommended_min_vram_mb_fp16=512.0,
        recommended_min_ram_mb_ooc=1024.0,
        hf_hub_url="https://huggingface.co/gpt2",
    ),
    "facebook/opt-125m": ModelZooEntry(
        model_id="facebook/opt-125m",
        display_name="OPT 125M",
        family="opt",
        scale_tier=ScaleTier.TIER_1_SMALL,
        parameter_count=125_237_760,
        num_layers=12,
        hidden_dim=768,
        num_attention_heads=12,
        vocab_size=50272,
        norm_type="layernorm",
        rotary_embeddings=False,
        gated_mlp=False,
        context_window=2048,
        recommended_min_vram_mb_fp16=512.0,
        recommended_min_ram_mb_ooc=1024.0,
        hf_hub_url="https://huggingface.co/facebook/opt-125m",
    ),
    "EleutherAI/gpt-neo-125m": ModelZooEntry(
        model_id="EleutherAI/gpt-neo-125m",
        display_name="GPT-Neo 125M",
        family="gpt_neo",
        scale_tier=ScaleTier.TIER_1_SMALL,
        parameter_count=125_198_592,
        num_layers=12,
        hidden_dim=768,
        num_attention_heads=12,
        vocab_size=50257,
        norm_type="layernorm",
        rotary_embeddings=False,
        gated_mlp=False,
        context_window=2048,
        recommended_min_vram_mb_fp16=512.0,
        recommended_min_ram_mb_ooc=1024.0,
        hf_hub_url="https://huggingface.co/EleutherAI/gpt-neo-125m",
    ),
    "EleutherAI/pythia-160m": ModelZooEntry(
        model_id="EleutherAI/pythia-160m",
        display_name="Pythia 160M",
        family="gpt_neox",
        scale_tier=ScaleTier.TIER_1_SMALL,
        parameter_count=162_322_432,
        num_layers=12,
        hidden_dim=768,
        num_attention_heads=12,
        vocab_size=50304,
        norm_type="layernorm",
        rotary_embeddings=True,
        gated_mlp=False,
        context_window=2048,
        recommended_min_vram_mb_fp16=650.0,
        recommended_min_ram_mb_ooc=1200.0,
        hf_hub_url="https://huggingface.co/EleutherAI/pythia-160m",
    ),

    # ── Tier 2: Medium (300M - 1.5B) ─────────────────────────────────
    "Qwen/Qwen2.5-0.5B": ModelZooEntry(
        model_id="Qwen/Qwen2.5-0.5B",
        display_name="Qwen 2.5 0.5B",
        family="qwen",
        scale_tier=ScaleTier.TIER_2_MEDIUM,
        parameter_count=494_032_896,
        num_layers=24,
        hidden_dim=896,
        num_attention_heads=14,
        vocab_size=151936,
        norm_type="rmsnorm",
        rotary_embeddings=True,
        gated_mlp=True,
        context_window=32768,
        recommended_min_vram_mb_fp16=1500.0,
        recommended_min_ram_mb_ooc=2048.0,
        hf_hub_url="https://huggingface.co/Qwen/Qwen2.5-0.5B",
    ),
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0": ModelZooEntry(
        model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        display_name="TinyLlama 1.1B",
        family="llama",
        scale_tier=ScaleTier.TIER_2_MEDIUM,
        parameter_count=1_100_048_384,
        num_layers=22,
        hidden_dim=2048,
        num_attention_heads=32,
        vocab_size=32000,
        norm_type="rmsnorm",
        rotary_embeddings=True,
        gated_mlp=True,
        context_window=2048,
        recommended_min_vram_mb_fp16=2500.0,
        recommended_min_ram_mb_ooc=3072.0,
        hf_hub_url="https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    ),
    "gpt2-medium": ModelZooEntry(
        model_id="gpt2-medium",
        display_name="GPT-2 Medium (355M)",
        family="gpt2",
        scale_tier=ScaleTier.TIER_2_MEDIUM,
        parameter_count=354_823_168,
        num_layers=24,
        hidden_dim=1024,
        num_attention_heads=16,
        vocab_size=50257,
        norm_type="layernorm",
        rotary_embeddings=False,
        gated_mlp=False,
        context_window=1024,
        recommended_min_vram_mb_fp16=1200.0,
        recommended_min_ram_mb_ooc=2048.0,
        hf_hub_url="https://huggingface.co/gpt2-medium",
    ),

    # ── Tier 3: Large (3B - 7B) ──────────────────────────────────────
    "Qwen/Qwen2.5-3B": ModelZooEntry(
        model_id="Qwen/Qwen2.5-3B",
        display_name="Qwen 2.5 3B",
        family="qwen",
        scale_tier=ScaleTier.TIER_3_LARGE,
        parameter_count=3_086_270_464,
        num_layers=36,
        hidden_dim=2048,
        num_attention_heads=16,
        vocab_size=151936,
        norm_type="rmsnorm",
        rotary_embeddings=True,
        gated_mlp=True,
        context_window=32768,
        recommended_min_vram_mb_fp16=6500.0,
        recommended_min_ram_mb_ooc=6144.0,
        hf_hub_url="https://huggingface.co/Qwen/Qwen2.5-3B",
    ),
    "mistralai/Mistral-7B-v0.1": ModelZooEntry(
        model_id="mistralai/Mistral-7B-v0.1",
        display_name="Mistral 7B v0.1",
        family="mistral",
        scale_tier=ScaleTier.TIER_3_LARGE,
        parameter_count=7_241_732_096,
        num_layers=32,
        hidden_dim=4096,
        num_attention_heads=32,
        vocab_size=32000,
        norm_type="rmsnorm",
        rotary_embeddings=True,
        gated_mlp=True,
        context_window=8192,
        recommended_min_vram_mb_fp16=15000.0,
        recommended_min_ram_mb_ooc=10240.0,
        hf_hub_url="https://huggingface.co/mistralai/Mistral-7B-v0.1",
    ),
}


def get_zoo_entry(model_id: str) -> Optional[ModelZooEntry]:
    """Retrieves metadata descriptor for a model from the registry."""
    return MODEL_ZOO_REGISTRY.get(model_id)


def list_zoo_models(tier: Optional[ScaleTier] = None, family: Optional[str] = None) -> List[ModelZooEntry]:
    """Lists model zoo entries filtered by scale tier or architecture family."""
    entries = list(MODEL_ZOO_REGISTRY.values())
    if tier is not None:
        entries = [e for e in entries if e.scale_tier == tier]
    if family is not None:
        entries = [e for e in entries if e.family.lower() == family.lower()]
    return entries
