"""Model Configuration for MECH Platform."""

from pathlib import Path
from typing import Optional

from backend.core.config.settings import get_settings


class ModelConfig:
    """Model configuration and paths."""

    def __init__(self) -> None:
        settings = get_settings()
        self._default_model = settings.default_model
        self._model_zoo_path = settings.model_zoo_path
        self._model_cache_dir = settings.model_cache_dir

    @property
    def default_model(self) -> str:
        return self._default_model

    @property
    def model_zoo_path(self) -> Optional[Path]:
        if self._model_zoo_path:
            return Path(self._model_zoo_path)
        return None

    @property
    def model_cache_dir(self) -> Optional[Path]:
        if self._model_cache_dir:
            return Path(self._model_cache_dir)
        return None

    def get_model_path(self, model_name: str) -> Optional[Path]:
        """Get path for a specific model."""
        if self._model_zoo_path:
            return Path(self._model_zoo_path) / model_name
        return None

    def get_cache_path(self, model_name: str) -> Optional[Path]:
        """Get cache path for a specific model."""
        if self._model_cache_dir:
            return Path(self._model_cache_dir) / model_name
        return None


# Built-in model configurations
BUILTIN_MODELS = {
    "gpt2": {
        "name": "gpt2",
        "display_name": "GPT-2 Small (124M)",
        "layers": 12,
        "heads": 12,
        "hidden_size": 768,
        "vocab_size": 50257,
        "context_length": 1024,
        "source": "huggingface",
        "model_id": "gpt2",
    },
    "gpt2-medium": {
        "name": "gpt2-medium",
        "display_name": "GPT-2 Medium (355M)",
        "layers": 24,
        "heads": 16,
        "hidden_size": 1024,
        "vocab_size": 50257,
        "context_length": 1024,
        "source": "huggingface",
        "model_id": "gpt2-medium",
    },
    "gpt2-large": {
        "name": "gpt2-large",
        "display_name": "GPT-2 Large (774M)",
        "layers": 36,
        "heads": 20,
        "hidden_size": 1280,
        "vocab_size": 50257,
        "context_length": 1024,
        "source": "huggingface",
        "model_id": "gpt2-large",
    },
    "gpt2-xl": {
        "name": "gpt2-xl",
        "display_name": "GPT-2 XL (1.5B)",
        "layers": 48,
        "heads": 25,
        "hidden_size": 1600,
        "vocab_size": 50257,
        "context_length": 1024,
        "source": "huggingface",
        "model_id": "gpt2-xl",
    },
}


def get_model_config() -> ModelConfig:
    """Get model configuration."""
    return ModelConfig()


def get_builtin_model(model_name: str) -> Optional[dict]:
    """Get built-in model configuration."""
    return BUILTIN_MODELS.get(model_name)


def list_builtin_models() -> list[dict]:
    """List all built-in model configurations."""
    return list(BUILTIN_MODELS.values())