"""Integration Test: Batch + Model Comparison Workflow.

Validates running batch prompt experiments across two model architectures
and computing aggregated comparison DTOs.
"""
from api.dispatcher import build_dispatcher


def test_batch_comparison_workflow():
    dispatcher = build_dispatcher()
    prompts = [f"Sample Prompt #{i}" for i in range(1, 101)]

    # 1. Run batch experiment on Model A
    batch_a = dispatcher["runtime/experiments"]({
        "experiment_id": "exp_model_a",
        "model_name": "GPT-2 Small",
        "prompts": prompts,
    })
    assert batch_a["total_prompts"] == 100
    assert batch_a["completed_prompts"] == 100

    # 2. Run batch experiment on Model B
    batch_b = dispatcher["runtime/experiments"]({
        "experiment_id": "exp_model_b",
        "model_name": "Pythia 160M",
        "prompts": prompts,
    })
    assert batch_b["total_prompts"] == 100
    assert batch_b["completed_prompts"] == 100

    # 3. Compare models across aggregated batch prompt
    compare_res = dispatcher["runtime/compare"]({
        "prompt": prompts[0],
        "model_a": "GPT-2 Small",
        "model_b": "Pythia 160M",
    })
    assert compare_res["activations"]["cosine_similarity"] > 0.5
    assert "predictions" in compare_res
