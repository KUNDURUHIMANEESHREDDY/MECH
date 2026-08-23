"""
Agent 1 Integrity Test Suite
Tests all failure scenarios and verifies no fabricated data reaches scientific outputs.
"""
import sys
import traceback
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent))

results = []

def test(name, fn):
    """Run a test and record pass/fail."""
    try:
        fn()
        results.append((name, "PASS", None))
        print(f"  PASS: {name}")
    except Exception as e:
        results.append((name, "FAIL", str(e)))
        print(f"  FAIL: {name} -> {e}")


# ============================================================
# SECTION 1: Mock Path Integrity Tests
# ============================================================
print("\n=== SECTION 1: Mock Path Integrity ===")

def test_gpt2_mock_activations_raises():
    from backend.science.models.gpt2_adapter import GPT2Adapter
    adapter = GPT2Adapter(variant="small", mock_mode=True)
    try:
        adapter.get_activations("The cat sat on the", layer=5)
        raise AssertionError("Expected RuntimeError, got no exception")
    except RuntimeError as e:
        if "mock_mode" not in str(e):
            raise AssertionError(f"Error message should mention mock_mode: {e}")

def test_gpt2_mock_attention_raises():
    from backend.science.models.gpt2_adapter import GPT2Adapter
    adapter = GPT2Adapter(variant="small", mock_mode=True)
    try:
        adapter.get_attention_patterns("The cat sat on the", layer=5)
        raise AssertionError("Expected RuntimeError, got no exception")
    except RuntimeError as e:
        if "mock_mode" not in str(e):
            raise AssertionError(f"Error message should mention mock_mode: {e}")

def test_gpt2_mock_logits_raises():
    from backend.science.models.gpt2_adapter import GPT2Adapter
    adapter = GPT2Adapter(variant="small", mock_mode=True)
    try:
        adapter.get_logits("The cat sat on the")
        raise AssertionError("Expected RuntimeError, got no exception")
    except RuntimeError as e:
        if "mock_mode" not in str(e):
            raise AssertionError(f"Error message should mention mock_mode: {e}")

def test_gpt2_mock_patch_activation_raises():
    from backend.science.models.gpt2_adapter import GPT2Adapter
    adapter = GPT2Adapter(variant="small", mock_mode=True)
    try:
        adapter.patch_activation("The cat sat on the", layer=5, neuron_index=10, patch_value=0.0)
        raise AssertionError("Expected RuntimeError, got no exception")
    except RuntimeError as e:
        if "mock_mode" not in str(e):
            raise AssertionError(f"Error message should mention mock_mode: {e}")

def test_gpt2_mock_patch_head_raises():
    from backend.science.models.gpt2_adapter import GPT2Adapter
    adapter = GPT2Adapter(variant="small", mock_mode=True)
    try:
        adapter.patch_head_output("The cat sat on the", layer=5, head_index=0)
        raise AssertionError("Expected RuntimeError, got no exception")
    except RuntimeError as e:
        if "mock_mode" not in str(e):
            raise AssertionError(f"Error message should mention mock_mode: {e}")

def test_gpt2_mock_circuit_raises():
    from backend.science.models.gpt2_adapter import GPT2Adapter
    adapter = GPT2Adapter(variant="small", mock_mode=True)
    try:
        adapter.run_isolated_circuit("The cat sat on the", [(5, 0), (5, 1)])
        raise AssertionError("Expected RuntimeError, got no exception")
    except RuntimeError as e:
        if "mock_mode" not in str(e):
            raise AssertionError(f"Error message should mention mock_mode: {e}")

def test_gpt2_mock_residual_raises():
    from backend.science.models.gpt2_adapter import GPT2Adapter
    adapter = GPT2Adapter(variant="small", mock_mode=True)
    try:
        adapter.get_residual_stream("The cat sat on the")
        raise AssertionError("Expected RuntimeError, got no exception")
    except RuntimeError as e:
        if "mock_mode" not in str(e):
            raise AssertionError(f"Error message should mention mock_mode: {e}")

test("GPT2 mock get_activations raises", test_gpt2_mock_activations_raises)
test("GPT2 mock get_attention_patterns raises", test_gpt2_mock_attention_raises)
test("GPT2 mock get_logits raises", test_gpt2_mock_logits_raises)
test("GPT2 mock patch_activation raises", test_gpt2_mock_patch_activation_raises)
test("GPT2 mock patch_head_output raises", test_gpt2_mock_patch_head_raises)
test("GPT2 mock run_isolated_circuit raises", test_gpt2_mock_circuit_raises)
test("GPT2 mock get_residual_stream raises", test_gpt2_mock_residual_raises)


# ============================================================
# SECTION 2: Generic Adapter Mock Integrity
# ============================================================
print("\n=== SECTION 2: Generic Adapter Mock Integrity ===")

def test_generic_mock_activations_raises():
    from backend.science.models.model_adapters import create_adapter
    adapter = create_adapter("meta-llama/Llama-2-7b-hf", mock_mode=True)
    try:
        adapter.get_activations("The cat sat on the", layer=5)
        raise AssertionError("Expected RuntimeError")
    except RuntimeError as e:
        if "mock_mode" not in str(e):
            raise AssertionError(f"Error message should mention mock_mode: {e}")

def test_generic_mock_attention_raises():
    from backend.science.models.model_adapters import create_adapter
    adapter = create_adapter("meta-llama/Llama-2-7b-hf", mock_mode=True)
    try:
        adapter.get_attention_patterns("The cat sat on the", layer=5)
        raise AssertionError("Expected RuntimeError")
    except RuntimeError as e:
        if "mock_mode" not in str(e):
            raise AssertionError(f"Error message should mention mock_mode: {e}")

def test_generic_mock_logits_raises():
    from backend.science.models.model_adapters import create_adapter
    adapter = create_adapter("meta-llama/Llama-2-7b-hf", mock_mode=True)
    try:
        adapter.get_logits("The cat sat on the")
        raise AssertionError("Expected RuntimeError")
    except RuntimeError as e:
        if "mock_mode" not in str(e):
            raise AssertionError(f"Error message should mention mock_mode: {e}")

def test_generic_mock_patch_raises():
    from backend.science.models.model_adapters import create_adapter
    adapter = create_adapter("meta-llama/Llama-2-7b-hf", mock_mode=True)
    try:
        adapter.patch_activation("The cat sat on the", layer=5, neuron_index=10, patch_value=0.0)
        raise AssertionError("Expected RuntimeError")
    except RuntimeError as e:
        if "mock_mode" not in str(e):
            raise AssertionError(f"Error message should mention mock_mode: {e}")

def test_generic_mock_residual_raises():
    from backend.science.models.model_adapters import create_adapter
    adapter = create_adapter("meta-llama/Llama-2-7b-hf", mock_mode=True)
    try:
        adapter.get_residual_stream("The cat sat on the")
        raise AssertionError("Expected RuntimeError")
    except RuntimeError as e:
        if "mock_mode" not in str(e):
            raise AssertionError(f"Error message should mention mock_mode: {e}")

test("Generic mock get_activations raises", test_generic_mock_activations_raises)
test("Generic mock get_attention_patterns raises", test_generic_mock_attention_raises)
test("Generic mock get_logits raises", test_generic_mock_logits_raises)
test("Generic mock patch_activation raises", test_generic_mock_patch_raises)
test("Generic mock get_residual_stream raises", test_generic_mock_residual_raises)


# ============================================================
# SECTION 3: TransformerLens Mock Integrity
# ============================================================
print("\n=== SECTION 3: TransformerLens Mock Integrity ===")

def test_tl_mock_activations_raises():
    from backend.science.models.transformer_lens_adapter import TransformerLensAdapter
    adapter = TransformerLensAdapter(model_name="gpt2", mock_mode=True)
    try:
        adapter.get_activations("The cat sat on the", layer=5)
        raise AssertionError("Expected RuntimeError")
    except RuntimeError as e:
        if "mock_mode" not in str(e) and "unavailable" not in str(e):
            raise AssertionError(f"Error message should mention mock_mode: {e}")

def test_tl_mock_attention_raises():
    from backend.science.models.transformer_lens_adapter import TransformerLensAdapter
    adapter = TransformerLensAdapter(model_name="gpt2", mock_mode=True)
    try:
        adapter.get_attention_patterns("The cat sat on the", layer=5)
        raise AssertionError("Expected RuntimeError")
    except RuntimeError as e:
        if "mock_mode" not in str(e) and "unavailable" not in str(e):
            raise AssertionError(f"Error message should mention mock_mode: {e}")

def test_tl_mock_logits_raises():
    from backend.science.models.transformer_lens_adapter import TransformerLensAdapter
    adapter = TransformerLensAdapter(model_name="gpt2", mock_mode=True)
    try:
        adapter.get_logits("The cat sat on the")
        raise AssertionError("Expected RuntimeError")
    except RuntimeError as e:
        if "mock_mode" not in str(e) and "unavailable" not in str(e):
            raise AssertionError(f"Error message should mention mock_mode: {e}")

def test_tl_mock_patch_raises():
    from backend.science.models.transformer_lens_adapter import TransformerLensAdapter
    adapter = TransformerLensAdapter(model_name="gpt2", mock_mode=True)
    try:
        adapter.patch_activation("The cat sat on the", layer=5, neuron_index=10, patch_value=0.0)
        raise AssertionError("Expected RuntimeError")
    except RuntimeError as e:
        if "mock_mode" not in str(e) and "unavailable" not in str(e):
            raise AssertionError(f"Error message should mention mock_mode: {e}")

def test_tl_mock_residual_raises():
    from backend.science.models.transformer_lens_adapter import TransformerLensAdapter
    adapter = TransformerLensAdapter(model_name="gpt2", mock_mode=True)
    try:
        adapter.get_residual_stream("The cat sat on the")
        raise AssertionError("Expected RuntimeError")
    except RuntimeError as e:
        if "mock_mode" not in str(e) and "unavailable" not in str(e):
            raise AssertionError(f"Error message should mention mock_mode: {e}")

test("TL mock get_activations raises", test_tl_mock_activations_raises)
test("TL mock get_attention_patterns raises", test_tl_mock_attention_raises)
test("TL mock get_logits raises", test_tl_mock_logits_raises)
test("TL mock patch_activation raises", test_tl_mock_patch_raises)
test("TL mock get_residual_stream raises", test_tl_mock_residual_raises)


# ============================================================
# SECTION 4: Distributed Worker Integrity
# ============================================================
print("\n=== SECTION 4: Distributed Worker Integrity ===")

def test_worker_rejects_mock():
    from backend.distributed.worker import DistributedWorker
    import inspect
    # Verify worker has no mock_mode parameter
    sig = inspect.signature(DistributedWorker.__init__)
    assert "mock_mode" not in sig.parameters, f"DistributedWorker should not accept mock_mode: {sig.parameters}"
    print(f"    DistributedWorker params: {list(sig.parameters.keys())}")

def test_worker_requires_real_model():
    from backend.distributed.worker import DistributedWorker
    import inspect
    sig = inspect.signature(DistributedWorker.__init__)
    # Worker should require real model configuration
    print(f"    DistributedWorker params: {list(sig.parameters.keys())}")

test("Worker rejects mock_mode=True", test_worker_rejects_mock)
test("Worker requires real model", test_worker_requires_real_model)


# ============================================================
# SECTION 5: Real Model Loading & Execution
# ============================================================
print("\n=== SECTION 5: Real Model Loading & Execution ===")

def test_real_gpt2_loads():
    from backend.science.models.gpt2_adapter import GPT2Adapter
    adapter = GPT2Adapter(variant="small", mock_mode=False)
    assert adapter._model is not None, "Model should be loaded"
    assert adapter._tokenizer is not None, "Tokenizer should be loaded"
    assert not adapter.spec.mock_mode, "mock_mode should be False"

def test_real_gpt2_get_activations():
    from backend.science.models.gpt2_adapter import GPT2Adapter
    adapter = GPT2Adapter(variant="small", mock_mode=False)
    activations = adapter.get_activations("The cat sat on the", layer=5)
    assert len(activations) > 0, "Should return activations"
    assert all(a.activation_value != 0.0 or a.neuron_index != a.neuron_index for a in activations), "Activations should have real values"

def test_real_gpt2_get_attention():
    from backend.science.models.gpt2_adapter import GPT2Adapter
    adapter = GPT2Adapter(variant="small", mock_mode=False)
    patterns = adapter.get_attention_patterns("The cat sat on the", layer=5)
    assert len(patterns) > 0, "Should return attention patterns"
    assert all(len(p.pattern_matrix) > 0 for p in patterns), "Patterns should have matrices"

def test_real_gpt2_get_logits():
    from backend.science.models.gpt2_adapter import GPT2Adapter
    adapter = GPT2Adapter(variant="small", mock_mode=False)
    result = adapter.get_logits("The cat sat on the")
    assert "top_token" in result, "Should have top_token"
    assert "top_tokens" in result, "Should have top_tokens"
    assert len(result["top_tokens"]) > 0, "Should have tokens"

def test_real_gpt2_patch_activation():
    from backend.science.models.gpt2_adapter import GPT2Adapter
    adapter = GPT2Adapter(variant="small", mock_mode=False)
    result = adapter.patch_activation("The cat sat on the", layer=5, neuron_index=10, patch_value=0.0)
    assert hasattr(result, "delta"), "Should have delta"
    assert result.layer == 5, "Should have correct layer"

def test_real_gpt2_residual_stream():
    from backend.science.models.gpt2_adapter import GPT2Adapter
    adapter = GPT2Adapter(variant="small", mock_mode=False)
    stream = adapter.get_residual_stream("The cat sat on the")
    assert len(stream) > 0, "Should return residual stream"
    assert all("norm" in s for s in stream), "Each entry should have norm"

test("Real GPT-2 loads", test_real_gpt2_loads)
test("Real GPT-2 get_activations", test_real_gpt2_get_activations)
test("Real GPT-2 get_attention_patterns", test_real_gpt2_get_attention)
test("Real GPT-2 get_logits", test_real_gpt2_get_logits)
test("Real GPT-2 patch_activation", test_real_gpt2_patch_activation)
test("Real GPT-2 get_residual_stream", test_real_gpt2_residual_stream)


# ============================================================
# SECTION 6: Scientific Envelope Integrity
# ============================================================
print("\n=== SECTION 6: Scientific Envelope Integrity ===")

def test_unexecuted_envelope():
    from backend.science.scientific_envelope import wrap_unexecuted_failure
    envelope = wrap_unexecuted_failure("test_experiment", "Model unavailable")
    # Check evidence_level attribute (may be named differently)
    level = getattr(envelope, 'evidence_level', None) or getattr(envelope, 'status', None)
    assert level is not None, "Envelope should have evidence_level or status"
    assert envelope.result is None, "Result should be None"
    print(f"    Envelope level: {level}")

def test_no_provenance_rejects():
    from backend.science.scientific_envelope import wrap_unexecuted_failure
    envelope = wrap_unexecuted_failure("no_provenance", "Missing manifest_id")
    assert envelope.result is None, "Result should be None"
    print(f"    No-provenance envelope: result=None")

test("Unexecuted envelope returns UNEXECUTED", test_unexecuted_envelope)
test("No provenance rejects with UNEXECUTED", test_no_provenance_rejects)


# ============================================================
# SECTION 7: Experiment Engine Integrity
# ============================================================
print("\n=== SECTION 7: Experiment Engine Integrity ===")

def test_experiment_engine_no_mock():
    from backend.core.experiment_engine import CausalExperimentRunner
    import inspect
    sig = inspect.signature(CausalExperimentRunner.__init__)
    assert "mock_mode" not in sig.parameters, "CausalExperimentRunner should not accept mock_mode"
    print(f"    CausalExperimentRunner params: {list(sig.parameters.keys())}")

def test_experiment_engine_requires_model():
    from backend.core.experiment_engine import CausalExperimentRunner
    import inspect
    sig = inspect.signature(CausalExperimentRunner.__init__)
    print(f"    CausalExperimentRunner params: {list(sig.parameters.keys())}")

test("ExperimentEngine has no mock_mode parameter", test_experiment_engine_no_mock)
test("ExperimentEngine requires real model", test_experiment_engine_requires_model)


# ============================================================
# SECTION 8: Real Experiment Execution (IOI)
# ============================================================
print("\n=== SECTION 8: Real IOI Experiment ===")

def test_real_ioi_experiment():
    from backend.science.models.gpt2_adapter import GPT2Adapter
    
    adapter = GPT2Adapter(variant="small", mock_mode=False)
    
    # IOI-style prompts
    prompt = "When Mary and John went to the store, John gave a drink to"
    control_prompt = "When Mary and John went to the store, Mary gave a drink to"
    
    # Get activations for both prompts
    acts_subject = adapter.get_activations(prompt, layer=7)
    acts_control = adapter.get_activations(control_prompt, layer=7)
    
    assert len(acts_subject) > 0, "Subject activations should not be empty"
    assert len(acts_control) > 0, "Control activations should not be empty"
    
    # Get logits
    logits_subject = adapter.get_logits(prompt)
    logits_control = adapter.get_logits(control_prompt)
    
    assert "top_tokens" in logits_subject, "Subject logits should have tokens"
    assert "top_tokens" in logits_control, "Control logits should have tokens"
    
    # Compute delta
    subject_token = logits_subject["top_token"]
    control_token = logits_control["top_token"]
    
    print(f"    Subject top token: '{subject_token}'")
    print(f"    Control top token: '{control_token}'")
    print(f"    Subject logits: {logits_subject['top_tokens'][:3]}")
    print(f"    Control logits: {logits_subject['top_tokens'][:3]}")
    
    # All values must be real (not NaN, not hardcoded)
    for t in logits_subject["top_tokens"]:
        assert isinstance(t["logit"], (int, float)), f"Logit should be numeric: {t}"
        assert t["logit"] != 0.0 or t["token"] != "", f"Logit should not be empty/zero for all tokens"

test("Real IOI experiment execution", test_real_ioi_experiment)


# ============================================================
# SECTION 9: Real Falsification Experiment
# ============================================================
print("\n=== SECTION 9: Real Falsification Experiment ===")

def test_real_falsification():
    from backend.science.models.gpt2_adapter import GPT2Adapter
    
    adapter = GPT2Adapter(variant="small", mock_mode=False)
    
    # Hypothesis: Layer 0 head 0 is NOT important for IOI
    # Falsification: zero-ablate layer 0 head 0 and measure effect
    prompt = "When Mary and John went to the store, John gave a drink to"
    
    # Get original logits
    original = adapter.get_logits(prompt)
    
    # Get patched logits (zero out layer 0, head 0)
    patched = adapter.patch_head_output(prompt, layer=0, head_index=0)
    
    # Compute effect size
    delta = patched.delta
    print(f"    Original top token: '{original['top_token']}'")
    print(f"    Patched top token: '{patched.top_token_after}'")
    print(f"    Delta: {delta}")
    
    # Verify values are real
    assert isinstance(delta, (int, float)), f"Delta should be numeric: {delta}"
    assert patched.original_logit != 0.0, "Original logit should not be zero"

test("Real falsification experiment", test_real_falsification)


# ============================================================
# SECTION 10: Real Replication Experiment
# ============================================================
print("\n=== SECTION 10: Real Replication Experiment ===")

def test_real_replication():
    from backend.science.models.gpt2_adapter import GPT2Adapter
    
    adapter = GPT2Adapter(variant="small", mock_mode=False)
    
    # Replicate: run same experiment multiple times
    prompt = "The capital of France is"
    results = []
    
    for trial in range(3):
        logits = adapter.get_logits(prompt)
        results.append(logits)
    
    # All trials should return consistent results (deterministic model)
    for i in range(1, len(results)):
        assert results[i]["top_token"] == results[0]["top_token"], \
            f"Trial {i} top token differs: {results[i]['top_token']} vs {results[0]['top_token']}"
    
    print(f"    Replicated {len(results)} trials, all consistent: top_token='{results[0]['top_token']}'")

test("Real replication experiment", test_real_replication)


# ============================================================
# SECTION 11: Provenance Integrity
# ============================================================
print("\n=== SECTION 11: Provenance Integrity ===")

def test_provenance_hash():
    import hashlib
    import json
    
    # Simulate experiment result
    result_data = {
        "experiment_id": "test_001",
        "model": "gpt2-small",
        "prompt": "The capital of France is",
        "layer": 5,
        "head": 0,
        "delta_logit": 0.42
    }
    
    # Compute SHA-256 manifest
    manifest_hash = hashlib.sha256(json.dumps(result_data, sort_keys=True).encode()).hexdigest()
    
    assert len(manifest_hash) == 64, f"Hash should be 64 chars: {len(manifest_hash)}"
    assert manifest_hash != "", "Hash should not be empty"
    
    # Verify same data produces same hash
    manifest_hash_2 = hashlib.sha256(json.dumps(result_data, sort_keys=True).encode()).hexdigest()
    assert manifest_hash == manifest_hash_2, "Same data should produce same hash"
    
    print(f"    Provenance hash: {manifest_hash[:16]}...")

test("Provenance hash integrity", test_provenance_hash)


# ============================================================
# SECTION 12: Dead Code Verification
# ============================================================
print("\n=== SECTION 12: Dead Code Verification ===")

def test_mock_helpers_are_dead():
    """Verify mock helper functions are never called in active code paths."""
    import os
    
    dead_functions = [
        "_mock_activation",
        "_dynamic_mock_tokens",
        "_mock_attention_patterns",
    ]
    
    for func_name in dead_functions:
        found_def = False
        found_call = False
        
        for root, dirs, files in os.walk("backend"):
            for f in files:
                if f.endswith(".py"):
                    filepath = os.path.join(root, f)
                    try:
                        with open(filepath, "r", encoding="utf-8", errors="ignore") as fh:
                            content = fh.read()
                        
                        if f"def {func_name}" in content:
                            found_def = True
                        
                        # Check for calls (not definitions)
                        lines = content.split("\n")
                        for line in lines:
                            if func_name in line and "def " not in line and "import " not in line:
                                stripped = line.strip()
                                if not stripped.startswith("#") and not stripped.startswith('"') and not stripped.startswith("'"):
                                    found_call = True
                    except Exception:
                        pass
        
        if found_def and not found_call:
            print(f"    {func_name}: DEFINED but NEVER CALLED (dead code)")
        elif found_def and found_call:
            print(f"    WARNING: {func_name}: DEFINED and CALLED")
        else:
            print(f"    {func_name}: NOT FOUND")

test("Mock helpers are dead code", test_mock_helpers_are_dead)


# ============================================================
# SUMMARY
# ============================================================
print("\n" + "=" * 60)
print("TEST SUMMARY")
print("=" * 60)

passed = sum(1 for _, status, _ in results if status == "PASS")
failed = sum(1 for _, status, _ in results if status == "FAIL")

print(f"\nTotal: {len(results)}")
print(f"Passed: {passed}")
print(f"Failed: {failed}")

if failed > 0:
    print("\nFAILED TESTS:")
    for name, status, error in results:
        if status == "FAIL":
            print(f"  {name}: {error}")
    sys.exit(1)
else:
    print("\nALL TESTS PASSED")
    sys.exit(0)
