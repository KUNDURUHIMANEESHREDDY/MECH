from __future__ import annotations

import unittest
from unittest.mock import patch

import torch
from transformers import GPT2Config, GPT2LMHeadModel

from backend.runtime import (
    GPT2ModelLoader,
    InvalidPromptError,
    MissingModelError,
    ModelExecutor,
    ModelOutOfMemoryError,
)
from backend.runtime.models.gpt2 import LoadedModel


class SimpleTokenizer:
    eos_token = "<eos>"
    pad_token = None

    def __call__(
        self,
        prompt: str,
        *,
        return_tensors: str,
        add_special_tokens: bool,
    ) -> dict[str, torch.Tensor]:
        del return_tensors, add_special_tokens
        ids = [ord(character) % 20 + 1 for character in prompt]
        input_ids = torch.tensor([ids], dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def convert_ids_to_tokens(self, ids: list[int]) -> list[str]:
        return [f"tok_{token_id}" for token_id in ids]


def make_tiny_gpt2() -> GPT2LMHeadModel:
    config = GPT2Config(
        vocab_size=32,
        n_positions=32,
        n_ctx=32,
        n_embd=16,
        n_layer=2,
        n_head=2,
        bos_token_id=0,
        eos_token_id=0,
    )
    model = GPT2LMHeadModel(config)
    model.eval()
    return model


class InMemoryLoader:
    def __init__(self, model: torch.nn.Module | None = None) -> None:
        self.model = model or make_tiny_gpt2()
        self.tokenizer = SimpleTokenizer()
        self.load_count = 0

    def load(self) -> LoadedModel:
        self.load_count += 1
        return LoadedModel(
            model_name="gpt2",
            model=self.model,
            tokenizer=self.tokenizer,
            device=torch.device("cpu"),
        )


class OOMModel(torch.nn.Module):
    def forward(self, *args: object, **kwargs: object) -> torch.Tensor:
        del args, kwargs
        raise RuntimeError("CUDA out of memory")


class ModelRuntimeTests(unittest.TestCase):
    def test_executor_runs_prompt_and_returns_activations(self) -> None:
        executor = ModelExecutor(model_loader=InMemoryLoader())

        result = executor.run_prompt("hello")

        self.assertEqual(result.model_name, "gpt2")
        self.assertEqual(len(result.tokens), 5)
        self.assertEqual(len(result.token_ids), 5)
        self.assertEqual(tuple(result.logits.shape[:2]), (1, 5))
        self.assertIn("embedding", result.activations)
        self.assertIn("layers.0.attention_output", result.activations)
        self.assertIn("layers.0.mlp_output", result.activations)
        self.assertIn("layers.0.residual", result.activations)
        self.assertIn("logits", result.activations)
        self.assertEqual(
            result.activations.get("layers.0.residual").metadata.layer,
            0,
        )
        self.assertEqual(
            result.activations.get("layers.0.mlp_output").metadata.kind,
            "mlp_output",
        )

    def test_result_can_be_serialized_without_tensors(self) -> None:
        result = ModelExecutor(model_loader=InMemoryLoader()).run_prompt("hi")

        payload = result.to_dict()

        self.assertEqual(payload["model_name"], "gpt2")
        self.assertIn("activations", payload)
        self.assertNotIn("activation_tensors", payload)
        self.assertEqual(payload["metadata"]["sequence_length"], 2)

    def test_invalid_prompt_raises_domain_error(self) -> None:
        loader = InMemoryLoader()
        executor = ModelExecutor(model_loader=loader)

        with self.assertRaises(InvalidPromptError):
            executor.run_prompt("")
        self.assertEqual(loader.load_count, 0)

    def test_oom_during_forward_raises_domain_error(self) -> None:
        executor = ModelExecutor(model_loader=InMemoryLoader(OOMModel()))

        with self.assertRaises(ModelOutOfMemoryError):
            executor.run_prompt("hello")

    @patch("backend.runtime.models.gpt2.GPT2LMHeadModel.from_pretrained")
    @patch("backend.runtime.models.gpt2.AutoTokenizer.from_pretrained")
    def test_gpt2_loader_loads_supported_model(
        self,
        tokenizer_factory,
        model_factory,
    ) -> None:
        tokenizer_factory.return_value = SimpleTokenizer()
        model_factory.return_value = make_tiny_gpt2()

        loaded = GPT2ModelLoader(model_name="gpt2", device="cpu").load()

        tokenizer_factory.assert_called_once()
        model_factory.assert_called_once()
        self.assertEqual(loaded.model_name, "gpt2")
        self.assertEqual(loaded.device, torch.device("cpu"))
        self.assertEqual(loaded.num_layers, 2)

    @patch("backend.runtime.models.gpt2.AutoTokenizer.from_pretrained")
    def test_missing_model_raises_domain_error(self, tokenizer_factory) -> None:
        tokenizer_factory.side_effect = OSError("not found")

        with self.assertRaises(MissingModelError):
            GPT2ModelLoader(model_name="gpt2", device="cpu").load()

    def test_unsupported_model_raises_missing_model_error(self) -> None:
        with self.assertRaises(MissingModelError):
            GPT2ModelLoader(model_name="bert-base-uncased", device="cpu").load()


if __name__ == "__main__":
    unittest.main()
