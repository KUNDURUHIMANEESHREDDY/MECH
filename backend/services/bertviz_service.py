"""BertViz Live Attention Visualization Service.

Directly utilizes the official bertviz library (Jesse Vig) to generate canonical
interactive Head View, Model View, and Neuron View visualizations powered
exclusively by live PyTorch model forward execution tensors.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import torch

try:
    import bertviz
    from bertviz import head_view, model_view, neuron_view
    _BERTVIZ_AVAILABLE = True
except ImportError:
    _BERTVIZ_AVAILABLE = False


class BertVizService:
    """Provides canonical BertViz interactive visualization bundles from live model execution."""

    def __init__(self, model_name: str = "gpt2") -> None:
        self.model_name = model_name

    def _ensure_live_execution(self, prompt: str):
        if not _BERTVIZ_AVAILABLE:
            raise RuntimeError("The official 'bertviz' library is not installed in the active Python environment.")

        if not prompt or not prompt.strip():
            raise ValueError("INVALID_INPUT: Prompt must not be empty or whitespace only.")

        import backend.services.gpt2_engine as gpt2_engine
        gpt2_engine.load()
        if not gpt2_engine.is_available() or gpt2_engine._model is None or gpt2_engine._tokenizer is None:
            raise RuntimeError("Live model is uninitialized. BertViz requires live PyTorch model execution.")

        model = gpt2_engine._model
        tokenizer = gpt2_engine._tokenizer

        enc = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model(**enc, output_attentions=True)

        if not hasattr(outputs, "attentions") or outputs.attentions is None or len(outputs.attentions) == 0:
            raise RuntimeError("Model forward pass did not return attention tensors. Cannot generate BertViz visualization.")

        # Convert token IDs to token strings
        input_ids = enc["input_ids"][0].tolist()
        tokens = [tokenizer.decode([tok_id]).replace("Ġ", " ") for tok_id in input_ids]

        return model, tokenizer, outputs.attentions, tokens

    def get_head_view(
        self,
        prompt: str,
        layer: Optional[int] = None,
        heads: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        """Generates canonical BertViz Head View interactive HTML using official bertviz package."""
        model, tokenizer, attentions, tokens = self._ensure_live_execution(prompt)

        # Generate official BertViz Head View HTML
        html_obj = bertviz.head_view(
            attention=attentions,
            tokens=tokens,
            layer=layer,
            heads=heads,
            html_action="return",
        )

        html_data = str(html_obj.data) if hasattr(html_obj, "data") else str(html_obj)

        return {
            "status": "success",
            "view_type": "head_view",
            "library": "bertviz",
            "library_version": getattr(bertviz, "__version__", "1.4.1"),
            "model_name": self.model_name,
            "prompt": prompt,
            "tokens": tokens,
            "layer": layer,
            "heads": heads,
            "html": html_data,
            "provenance": "OFFICIAL_BERTVIZ_LIVE_PYTORCH",
        }

    def get_model_view(self, prompt: str) -> Dict[str, Any]:
        """Generates canonical BertViz Model View interactive HTML using official bertviz package."""
        model, tokenizer, attentions, tokens = self._ensure_live_execution(prompt)

        html_obj = bertviz.model_view(
            attention=attentions,
            tokens=tokens,
            html_action="return",
        )

        html_data = str(html_obj.data) if hasattr(html_obj, "data") else str(html_obj)

        return {
            "status": "success",
            "view_type": "model_view",
            "library": "bertviz",
            "library_version": getattr(bertviz, "__version__", "1.4.1"),
            "model_name": self.model_name,
            "prompt": prompt,
            "tokens": tokens,
            "html": html_data,
            "provenance": "OFFICIAL_BERTVIZ_LIVE_PYTORCH",
        }
