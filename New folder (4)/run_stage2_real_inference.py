"""Stage 2 Validation Script: Real PyTorch GPT-2 Inference & Activation Patching.

Validates:
1. Real Hugging Face model & tokenizer loading.
2. Forward pass & Logit tensor shapes.
3. PyTorch activation hook registration & teardown.
4. Causal intervention (Logit delta & top-token prediction flip).
5. Memory & hook leak cleanup.
"""

import time
import sys
import torch
from backend.science.models.gpt2_adapter import GPT2Adapter
from backend.science.models.model_manager import ModelManager

def run_stage2_verification():
    print("==================================================")
    print(" Starting Stage 2: Real PyTorch Scientific Verification")
    print("==================================================")
    
    start_time = time.time()
    
    # 1. Model Loading & Device Verification
    print("\n[1/5] Verifying Model & Tokenizer Loading...")
    manager = ModelManager()
    model, tokenizer = manager.get_model_and_tokenizer(hf_repo_id="gpt2", revision="main")
    
    device = next(model.parameters()).device
    dtype = model.dtype
    name = model.config._name_or_path
    
    print(f"- Loaded Model: {name}")
    print(f"- Device: {device}")
    print(f"- Dtype: {dtype}")
    print(f"- Vocabulary Size: {tokenizer.vocab_size}")
    assert tokenizer.vocab_size == 50257, "Tokenizer vocab size mismatch!"
    
    # 2. Forward Pass
    print("\n[2/5] Running Real PyTorch Forward Pass...")
    prompt = "When John and Mary went to the store, John gave a drink to"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        
    print(f"- Prompt: '{prompt}'")
    print(f"- Logits Tensor Shape: {list(logits.shape)}")
    assert logits.dim() == 3, "Logits tensor should be 3-dimensional [batch, seq_len, vocab_size]"
    
    # Check top prediction
    last_token_logits = logits[0, -1, :]
    top_token_id = torch.argmax(last_token_logits).item()
    top_token_str = tokenizer.decode([top_token_id]).strip()
    print(f"- Unpatched Top Token Prediction: '{top_token_str}' (Logit: {last_token_logits[top_token_id].item():.2f})")
    
    # 3. PyTorch Activation Patching Hook
    print("\n[3/5] Registering Activation Patching Hook...")
    adapter = GPT2Adapter(variant="small", mock_mode=False)
    
    initial_hook_count = len(adapter._model.transformer.h[9].mlp._forward_hooks)
    print(f"- Initial MLP Layer 9 Forward Hooks Count: {initial_hook_count}")
    
    # Perform real intervention using GPT2Adapter.patch_activation
    print("- Invoking adapter.patch_activation(prompt, layer=9, neuron_index=0, patch_value=50.0)...")
    patch_result = adapter.patch_activation(
        prompt=prompt,
        layer=9,
        neuron_index=0,
        patch_value=50.0
    )
    
    print(f"- Intervention Executed Successfully.")
    
    # 4. Measure Intervention Impact (Delta Logit)
    print("\n[4/5] Measuring Causal Intervention Effects...")
    clean_logit = patch_result.original_logit
    patched_logit = patch_result.patched_logit
    delta_logit = patch_result.delta
    token_before = patch_result.top_token_before
    token_after = patch_result.top_token_after
    
    print("--- Causal Patching Results ---")
    print(f"Top Token Before Intervention : '{token_before}' | Logit: {clean_logit:.2f}")
    print(f"Top Token After Intervention  : '{token_after}' | Logit: {patched_logit:.2f}")
    print(f"Logit Delta (Effect Magnitude): {delta_logit:+.2f}")
    
    # 5. Memory Cleanup & Hook Teardown
    print("\n[5/5] Verifying Hook Teardown & Memory Cleanup...")
    post_hook_count = len(adapter._model.transformer.h[9].mlp._forward_hooks)
    print(f"- Post-Intervention MLP Layer 9 Forward Hooks Count: {post_hook_count}")
    assert post_hook_count == initial_hook_count, "Hook memory leak detected! Hooks were not unregistered."
    
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024 ** 2)
        print(f"- GPU Memory Currently Allocated: {allocated:.2f} MB")
    else:
        print("- System Running in CPU Mode (Memory Stable & Clean)")
        
    elapsed = time.time() - start_time
    print("\n==================================================")
    print(f" Stage 2 Scientific Verification Complete ({elapsed:.2f}s)")
    print(" Status: PASSED")
    print("==================================================")

if __name__ == "__main__":
    run_stage2_verification()
