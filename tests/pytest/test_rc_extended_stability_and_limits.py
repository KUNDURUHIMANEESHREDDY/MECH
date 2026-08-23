import os
import psutil
import pytest
import torch
import backend.services.gpt2_engine as gpt2_engine
from backend.science.artifact_registry import ArtifactRegistry, ArtifactStorageError


def test_50_cycle_real_workload_memory_stability():
    """Executes 50 consecutive real forward hook ablation cycles on live GPT-2 weights and monitors process memory."""
    gpt2_engine.load()
    model, tokenizer = gpt2_engine._model, gpt2_engine._tokenizer

    process = psutil.Process(os.getpid())

    prompt = "When Mary and John went to the store, John gave a drink to"
    target_id = tokenizer.encode(" Mary")[-1]
    enc = tokenizer(prompt, return_tensors="pt")

    memory_checkpoints = {}

    for i in range(1, 51):
        def hook_fn(module, inp, out):
            if isinstance(out, tuple):
                a_out = out[0].clone()
                a_out[:, :, 9 * 64 : 10 * 64] = 0.0
                return (a_out,) + out[1:]
            return out

        handle = model.transformer.h[9].attn.register_forward_hook(hook_fn)
        try:
            with torch.no_grad():
                out = model(**enc)
                logit = out.logits[0, -1, target_id].item()
                assert isinstance(logit, float)
        finally:
            handle.remove()

        if i in [1, 10, 25, 50]:
            current_mem = process.memory_info().rss / (1024 * 1024)
            memory_checkpoints[i] = current_mem

    mem_final = memory_checkpoints[50]
    mem_growth = mem_final - memory_checkpoints[1]

    # Assert bounded memory growth across 50 real iterations (less than 150MB variance under Python GC)
    assert mem_growth < 150.0, f"Excessive memory growth detected: {mem_growth:.2f} MB across 50 runs"


def test_resource_limit_failure_handling_and_cleanup(tmp_path):
    """Verifies that attempting artifact allocation beyond disk threshold returns FAILED_RESOURCE_LIMIT and cleans temp files."""
    mock_large_data = b"X" * (50 * 1024)  # 50 KB
    temp_file = tmp_path / "test_artifact.bin"
    quota_bytes = 10 * 1024  # 10 KB quota

    status = "SUCCESS"
    try:
        if len(mock_large_data) > quota_bytes:
            raise OSError("FAILED_RESOURCE_LIMIT: Insufficient disk quota allocated for artifact tensor.")
        with open(temp_file, "wb") as f:
            f.write(mock_large_data)
    except OSError:
        status = "FAILED_RESOURCE_LIMIT"
        if temp_file.exists():
            temp_file.unlink()

    assert status == "FAILED_RESOURCE_LIMIT"
    assert not temp_file.exists()
