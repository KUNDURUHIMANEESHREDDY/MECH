"""Unit tests for LayerPager progressive forward execution and offloading."""

import pytest
import torch

from backend.runtime.model_manager import load_model, get_model_and_tokenizer, get_layer_pager
from backend.runtime.memory.layer_pager import LayerPager
from backend.runtime.artifacts.cas_store import ArtifactStore


@pytest.fixture
def temp_store(tmp_path):
    storage_dir = str(tmp_path / "artifacts")
    db_path = str(tmp_path / "artifacts.db")
    return ArtifactStore(storage_dir=storage_dir, db_path=db_path)


def test_layer_pager_sequential_gpt2_forward(temp_store):
    load_model("gpt2")
    model, tokenizer = get_model_and_tokenizer()

    pager = LayerPager(device="cpu", offload_to_cpu=False, store=temp_store)
    prompt = "The Eiffel Tower is in"

    # Run standard forward to compare logits
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        standard_out = model(**inputs)
    standard_logits = standard_out.logits

    # Run LayerPager sequential forward
    output = pager.run_sequential_forward(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        capture_layers=[0, 5, 11],
    )

    assert output.layers_executed == 12
    assert len(output.tokens) > 0
    assert 0 in output.layer_residuals
    assert 5 in output.layer_residuals
    assert 11 in output.layer_residuals

    # Logits from sequential layer execution should exactly match standard model forward pass!
    assert torch.allclose(output.logits, standard_logits, atol=1e-4)
    assert len(output.cached_artifact_keys) == 3


def test_layer_pager_with_hook_intervention(temp_store):
    model, tokenizer = get_model_and_tokenizer()
    pager = LayerPager(device="cpu", store=temp_store)

    # Define simple zeroing intervention at Layer 4
    def zero_intervention(tensor: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(tensor)

    output = pager.run_sequential_forward(
        model=model,
        tokenizer=tokenizer,
        prompt="Paris is a",
        interventions={4: zero_intervention},
        capture_layers=[4, 5],
    )

    assert output.layers_executed == 12
    # Layer 4 residual should have been zeroed before passing to layer 5
    assert torch.allclose(output.layer_residuals[4], torch.zeros_like(output.layer_residuals[4]))
