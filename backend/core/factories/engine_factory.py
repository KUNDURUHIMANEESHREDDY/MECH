"""Model Engine Factory for MECH Platform."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Type

from backend.core.factories.base import ConfigurableFactory

logger = logging.getLogger("MECH.factories.engine")


class ModelEngine(ABC):
    """Base class for model engines."""

    def __init__(self, model_name: str, **kwargs: Any) -> None:
        self.model_name = model_name
        self.config = kwargs

    @abstractmethod
    def load(self) -> None:
        """Load the model."""
        pass

    @abstractmethod
    def forward(self, input_ids: Any, **kwargs: Any) -> Any:
        """Run forward pass."""
        pass

    @abstractmethod
    def get_activations(self, layer: int, component: str) -> Any:
        """Get activations for a specific layer/component."""
        pass


class EngineFactory(ConfigurableFactory):
    """Factory for model engines."""

    def __init__(self) -> None:
        super().__init__()
        self._register_builtins()

    def get_type_name(self) -> str:
        return "engine"

    def _register_builtins(self) -> None:
        """Register built-in engine types."""
        # Register gpt2 engine
        try:
            from backend.services.gpt2_engine import load as gpt2_load, _model as gpt2_model, _tokenizer as gpt2_tokenizer
            from backend.runtime.models.gpt2 import GPT2ModelLoader, LoadedModel

            class GPT2Engine(ModelEngine):
                def __init__(self, model_name: str = "gpt2", device: Optional[str] = None, **kwargs):
                    super().__init__(model_name, **kwargs)
                    self.device = device or "cpu"
                    self.loader = GPT2ModelLoader(model_name=model_name, device=self.device)
                    self._loaded_model: Optional[LoadedModel] = None

                def load(self) -> None:
                    self._loaded_model = self.loader.load()

                def forward(self, input_ids: Any, **kwargs: Any) -> Any:
                    if self._loaded_model is None:
                        self.load()
                    with torch.no_grad():
                        return self._loaded_model.model(input_ids, **kwargs)

                def get_activations(self, layer: int, component: str) -> Any:
                    # Implementation would hook into model layers
                    pass

            self.register("gpt2", GPT2Engine)
            logger.debug("Registered gpt2 engine")
        except ImportError as e:
            logger.debug("GPT2 engine not available: %s", e)

        # Register transformer lens engine
        try:
            from backend.runtime.models import TransformerLensEngine
            self.register("transformer_lens", TransformerLensEngine)
        except ImportError:
            logger.debug("TransformerLensEngine not available")

        # Register HuggingFace engine
        try:
            from backend.runtime.models import HuggingFaceEngine
            self.register("huggingface", HuggingFaceEngine)
        except ImportError:
            logger.debug("HuggingFaceEngine not available")

    def create_engine(
        self,
        engine_type: str,
        model_name: str,
        device: Optional[str] = None,
        **kwargs: Any,
    ) -> ModelEngine:
        """Create a model engine with common parameters."""
        config = {"model_name": model_name}
        if device:
            config["device"] = device
        config.update(kwargs)
        return self.create(engine_type, **config)

    def list_engines(self) -> List[str]:
        """List available engine types."""
        return list(self._registered_types.keys())


# Global instance
_engine_factory: Optional[EngineFactory] = None


def get_engine_factory() -> EngineFactory:
    """Get the global engine factory."""
    global _engine_factory
    if _engine_factory is None:
        _engine_factory = EngineFactory()
    return _engine_factory