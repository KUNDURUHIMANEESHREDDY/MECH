"""Self-describing model metadata — each model registers a ModelDescriptor."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ModelDescriptor:
    """Declarative metadata for a supported model architecture.

    The frontend uses this to dynamically adapt UI:
    - Which layers/heads exist
    - Which hook types are available
    - Model family-specific features
    """
    name: str
    hf_id: str
    family: str                       # "gpt2" | "gemma" | "llama" | "pythia" | ...
    architecture: str = "decoder"
    hidden_size: int = 0
    num_layers: int = 0
    num_heads: int = 0
    vocab_size: int = 0
    max_position: int = 0
    supports_attention: bool = True
    supports_residuals: bool = True
    supports_mlp: bool = True
    supports_embeddings: bool = True
    supports_logits: bool = True
    activation_function: str = ""
    description: str = ""


# ── Registry ─────────────────────────────────────────────────────

REGISTRY: dict[str, ModelDescriptor] = {}


def register(desc: ModelDescriptor) -> None:
    REGISTRY[desc.name] = desc


def get(name: str) -> ModelDescriptor | None:
    return REGISTRY.get(name)


def list_all() -> list[ModelDescriptor]:
    return list(REGISTRY.values())


# ── Built-in models ──────────────────────────────────────────────

register(ModelDescriptor(
    name="gpt2",
    hf_id="gpt2",
    family="gpt2",
    hidden_size=768,
    num_layers=12,
    num_heads=12,
    vocab_size=50257,
    max_position=1024,
    activation_function="gelu_new",
    description="GPT-2 small (124M parameters)",
))

register(ModelDescriptor(
    name="gpt2-medium",
    hf_id="gpt2-medium",
    family="gpt2",
    hidden_size=1024,
    num_layers=24,
    num_heads=16,
    vocab_size=50257,
    max_position=1024,
    activation_function="gelu_new",
    description="GPT-2 medium (355M parameters)",
))

register(ModelDescriptor(
    name="distilgpt2",
    hf_id="distilgpt2",
    family="gpt2",
    hidden_size=768,
    num_layers=6,
    num_heads=12,
    vocab_size=50257,
    max_position=1024,
    activation_function="gelu_new",
    description="Distilled GPT-2 (82M parameters)",
))

# Future models (declared but not yet loadable)
register(ModelDescriptor(
    name="gemma-2b",
    hf_id="google/gemma-2b",
    family="gemma",
    hidden_size=2048,
    num_layers=18,
    num_heads=8,
    vocab_size=256000,
    max_position=8192,
    activation_function="gelu_pytorch_tanh",
    description="Google Gemma 2B",
))

register(ModelDescriptor(
    name="llama-7b",
    hf_id="meta-llama/Llama-2-7b-hf",
    family="llama",
    hidden_size=4096,
    num_layers=32,
    num_heads=32,
    vocab_size=32000,
    max_position=4096,
    activation_function="silu",
    description="Meta Llama 2 7B",
))

register(ModelDescriptor(
    name="pythia-1b",
    hf_id="EleutherAI/pythia-1b",
    family="pythia",
    hidden_size=2048,
    num_layers=16,
    num_heads=8,
    vocab_size=50344,
    max_position=2048,
    activation_function="gelu",
    description="EleutherAI Pythia 1B",
))

register(ModelDescriptor(
    name="mistral-7b",
    hf_id="mistralai/Mistral-7B-v0.1",
    family="mistral",
    hidden_size=4096,
    num_layers=32,
    num_heads=32,
    vocab_size=32000,
    max_position=32768,
    activation_function="silu",
    description="Mistral 7B v0.1",
))

register(ModelDescriptor(
    name="qwen-1.5b",
    hf_id="Qwen/Qwen1.5-1.8B",
    family="qwen",
    hidden_size=2048,
    num_layers=24,
    num_heads=16,
    vocab_size=151936,
    max_position=32768,
    activation_function="silu",
    description="Qwen 1.5 1.8B",
))
