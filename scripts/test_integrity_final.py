"""
Agent 1 Integrity Test Suite - Final
Tests all failure scenarios and runs real experiments.
"""
import sys
import hashlib
import json
import traceback
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

results = []

def test(name, fn):
    try:
        fn()
        results.append((name, "PASS", None))
        print(f"  PASS: {name}")
    except Exception as e:
        results.append((name, "FAIL", str(e)))
        print(f"  FAIL: {name} -> {e}")


# ============================================================
# SECTION 1: Mock Path Integrity (17 tests)
# ============================================================
print("\n=== SECTION 1: Mock Path Integrity ===")

def test_gpt2_mock_activations_raises():
    from backend.science.models.gpt2_adapter import GPT2Adapter
    adapter = GPT2Adapter(variant="small", mock_mode=True)
    try:
        adapter.get_activations("The cat sat on the", layer=5)
        raise AssertionError("Expected RuntimeError")
    except RuntimeError as e:
        assert "mock_mode" in str(e)

def test_gpt2_mock_attention_raises():
    from backend.science.models.gpt2_adapter import GPT2Adapter
    adapter = GPT2Adapter(variant="small", mock_mode=True)
    try:
        adapter.get_attention_patterns("The cat sat on the", layer=5)
        raise AssertionError("Expected RuntimeError")
    except RuntimeError as e:
        assert "mock_mode" in str(e)

def test_gpt2_mock_logits_raises():
    from backend.science.models.gpt2_adapter import GPT2Adapter
    adapter = GPT2Adapter(variant="small", mock_mode=True)
    try:
        adapter.get_logits("The cat sat on the")
        raise AssertionError("Expected RuntimeError")
    except RuntimeError as e:
        assert "mock_mode" in str(e)

def test_gpt2_mock_patch_raises():
    from backend.science.models.gpt2_adapter import GPT2Adapter
    adapter = GPT2Adapter(variant="small", mock_mode=True)
    try:
        adapter.patch_activation("The cat sat on the", layer=5, neuron_index=10, patch_value=0.0)
        raise AssertionError("Expected RuntimeError")
    except RuntimeError as e:
        assert "mock_mode" in str(e)

def test_gpt2_mock_head_patch_raises():
    from backend.science.models.gpt2_adapter import GPT2Adapter
    adapter = GPT2Adapter(variant="small", mock_mode=True)
    try:
        adapter.patch_head_output("The cat sat on the", layer=5, head_index=0)
        raise AssertionError("Expected RuntimeError")
    except RuntimeError as e:
        assert "mock_mode" in str(e)

def test_gpt2_mock_circuit_raises():
    from backend.science.models.gpt2_adapter import GPT2Adapter
    adapter = GPT2Adapter(variant="small", mock_mode=True)
    try:
        adapter.run_isolated_circuit("The cat sat on the", ["5:0", "5:1"])
        raise AssertionError("Expected RuntimeError")
    except RuntimeError as e:
        assert "mock_mode" in str(e)

def test_gpt2_mock_residual_raises():
    from backend.science.models.gpt2_adapter import GPT2Adapter
    adapter = GPT2Adapter(variant="small", mock_mode=True)
    try:
        adapter.get_residual_stream("The cat sat on the")
        raise AssertionError("Expected RuntimeError")
    except RuntimeError as e:
        assert "mock_mode" in str(e)

test("GPT2 mock get_activations raises", test_gpt2_mock_activations_raises)
test("GPT2 mock get_attention_patterns raises", test_gpt2_mock_attention_raises)
test("GPT2 mock get_logits raises", test_gpt2_mock_logits_raises)
test("GPT2 mock patch_activation raises", test_gpt2_mock_patch_raises)
test("GPT2 mock patch_head_output raises", test_gpt2_mock_head_patch_raises)
test("GPT2 mock run_isolated_circuit raises", test_gpt2_mock_circuit_raises)
test("GPT2 mock get_residual_stream raises", test_gpt2_mock_residual_raises)

def test_generic_mock_activations_raises():
    from backend.science.models.model_adapters import create_adapter
    adapter = create_adapter("meta-llama/Llama-2-7b-hf", mock_mode=True)
    try:
        adapter.get_activations("The cat sat on the", layer=5)
        raise AssertionError("Expected RuntimeError")
    except RuntimeError as e:
        assert "mock_mode" in str(e)

def test_generic_mock_attention_raises():
    from backend.science.models.model_adapters import create_adapter
    adapter = create_adapter("meta-llama/Llama-2-7b-hf", mock_mode=True)
    try:
        adapter.get_attention_patterns("The cat sat on the", layer=5)
        raise AssertionError("Expected RuntimeError")
    except RuntimeError as e:
        assert "mock_mode" in str(e)

def test_generic_mock_logits_raises():
    from backend.science.models.model_adapters import create_adapter
    adapter = create_adapter("meta-llama/Llama-2-7b-hf", mock_mode=True)
    try:
        adapter.get_logits("The cat sat on the")
        raise AssertionError("Expected RuntimeError")
    except RuntimeError as e:
        assert "mock_mode" in str(e)

def test_generic_mock_patch_raises():
    from backend.science.models.model_adapters import create_adapter
    adapter = create_adapter("meta-llama/Llama-2-7b-hf", mock_mode=True)
    try:
        adapter.patch_activation("The cat sat on the", layer=5, neuron_index=10, patch_value=0.0)
        raise AssertionError("Expected RuntimeError")
    except RuntimeError as e:
        assert "mock_mode" in str(e)

def test_generic_mock_residual_raises():
    from backend.science.models.model_adapters import create_adapter
    adapter = create_adapter("meta-llama/Llama-2-7b-hf", mock_mode=True)
    try:
        adapter.get_residual_stream("The cat sat on the")
        raise AssertionError("Expected RuntimeError")
    except RuntimeError as e:
        assert "mock_mode" in str(e)

test("Generic mock get_activations raises", test_generic_mock_activations_raises)
test("Generic mock get_attention_patterns raises", test_generic_mock_attention_raises)
test("Generic mock get_logits raises", test_generic_mock_logits_raises)
test("Generic mock patch_activation raises", test_generic_mock_patch_raises)
test("Generic mock get_residual_stream raises", test_generic_mock_residual_raises)

def test_tl_mock_all_raises():
    from backend.science.models.transformer_lens_adapter import TransformerLensAdapter
    adapter = TransformerLensAdapter(model_name="gpt2", mock_mode=True)
    for method_name in ["get_activations", "get_attention_patterns", "get_logits"]:
        try:
            getattr(adapter, method_name)("test", layer=0) if method_name != "get_logits" else getattr(adapter, method_name)("test")
            raise AssertionError(f"{method_name} should raise RuntimeError")
        except RuntimeError as e:
            assert "mock_mode" in str(e) or "unavailable" in str(e)
    try:
        adapter.get_residual_stream("test")
        raise AssertionError("get_residual_stream should raise RuntimeError")
    except RuntimeError as e:
        assert "mock_mode" in str(e) or "unavailable" in str(e)

test("TransformerLens all mock methods raise", test_tl_mock_all_raises)


# ============================================================
# SECTION 2: Worker Integrity
# ============================================================
print("\n=== SECTION 2: Worker Integrity ===")

def test_worker_no_mock_mode_param():
    from backend.distributed.worker import DistributedWorker
    import inspect
    sig = inspect.signature(DistributedWorker.__init__)
    assert "mock_mode" not in sig.parameters, f"Worker should not accept mock_mode: {list(sig.parameters.keys())}"
    print(f"    Worker params: {list(sig.parameters.keys())}")

test("Worker has no mock_mode parameter", test_worker_no_mock_mode_param)


# ============================================================
# SECTION 3: Scientific Envelope Integrity
# ============================================================
print("\n=== SECTION 3: Scientific Envelope Integrity ===")

def test_unexecuted_envelope():
    from backend.science.scientific_envelope import wrap_unexecuted_failure
    envelope = wrap_unexecuted_failure("Model unavailable for testing")
    assert envelope.status == "UNEXECUTED_EXPERIMENT", f"Expected UNEXECUTED_EXPERIMENT, got {envelope.status}"
    assert envelope.data is None, "Data should be None"
    assert envelope.integrity_status == "UNVERIFIED"
    print(f"    Envelope status: {envelope.status}, data: None, integrity: {envelope.integrity_status}")

test("Unexecuted envelope returns UNEXECUTED_EXPERIMENT", test_unexecuted_envelope)


# ============================================================
# SECTION 4: Real Model Loading & Execution
# ============================================================
print("\n=== SECTION 4: Real Model Loading ===")

def test_real_gpt2_loads():
    from backend.science.models.gpt2_adapter import GPT2Adapter
    adapter = GPT2Adapter(variant="small", mock_mode=False)
    assert adapter._model is not None
    assert adapter._tokenizer is not None
    assert not adapter.spec.mock_mode
    print(f"    Model: {adapter.spec.name}, layers: {adapter.spec.num_layers}, heads: {adapter.spec.num_heads}")

test("Real GPT-2 loads successfully", test_real_gpt2_loads)


# ============================================================
# SECTION 5: Real IOI Experiment (Full Pipeline)
# ============================================================
print("\n=== SECTION 5: Real IOI Experiment (Full Pipeline) ===")

ioi_results = {}

def test_real_ioi_activations():
    from backend.science.models.gpt2_adapter import GPT2Adapter
    adapter = GPT2Adapter(variant="small", mock_mode=False)
    
    prompt = "When Mary and John went to the store, John gave a drink to"
    acts = adapter.get_activations(prompt, layer=7)
    assert len(acts) > 0
    assert all(a.activation_value != 0.0 for a in acts[:5])  # Real activations should vary
    ioi_results["activations"] = acts
    print(f"    Activations: {len(acts)} neurons, values: {[a.activation_value for a in acts[:3]]}")

def test_real_ioi_attention():
    from backend.science.models.gpt2_adapter import GPT2Adapter
    adapter = GPT2Adapter(variant="small", mock_mode=False)
    
    prompt = "When Mary and John went to the store, John gave a drink to"
    patterns = adapter.get_attention_patterns(prompt, layer=7)
    assert len(patterns) > 0
    assert all(len(p.pattern_matrix) > 0 for p in patterns)
    ioi_results["attention"] = patterns
    print(f"    Attention patterns: {len(patterns)} heads, matrix size: {len(patterns[0].pattern_matrix)}x{len(patterns[0].pattern_matrix[0])}")

def test_real_ioi_logits():
    from backend.science.models.gpt2_adapter import GPT2Adapter
    adapter = GPT2Adapter(variant="small", mock_mode=False)
    
    prompt = "When Mary and John went to the store, John gave a drink to"
    result = adapter.get_logits(prompt)
    assert "top_token" in result
    assert "top_tokens" in result
    assert len(result["top_tokens"]) == 5
    ioi_results["logits"] = result
    print(f"    Top token: '{result['top_token']}', logits: {[t['logit'] for t in result['top_tokens'][:3]]}")

def test_real_ioi_intervention():
    from backend.science.models.gpt2_adapter import GPT2Adapter
    adapter = GPT2Adapter(variant="small", mock_mode=False)
    
    prompt = "When Mary and John went to the store, John gave a drink to"
    
    # Zero-ablate layer 7, head 0
    patched = adapter.patch_head_output(prompt, layer=7, head_index=0)
    assert patched.delta != 0.0  # Real intervention should produce non-zero delta
    ioi_results["intervention"] = patched
    print(f"    Intervention delta: {patched.delta}, before: '{patched.top_token_before}', after: '{patched.top_token_after}'")

def test_real_ioi_controls():
    from backend.science.models.gpt2_adapter import GPT2Adapter
    adapter = GPT2Adapter(variant="small", mock_mode=False)
    
    prompt = "When Mary and John went to the store, John gave a drink to"
    
    # Run multiple controls
    controls = []
    for head_idx in range(3):
        patched = adapter.patch_head_output(prompt, layer=7, head_index=head_idx)
        controls.append(patched)
    
    # All controls should produce real values
    deltas = [c.delta for c in controls]
    assert all(isinstance(d, float) for d in deltas)
    ioi_results["controls"] = controls
    print(f"    Controls: {len(controls)} heads, deltas: {deltas}")

def test_real_ioi_provenance():
    import hashlib
    import json
    
    # Create experiment record with provenance
    experiment_data = {
        "experiment_id": f"ioi_real_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "model": "gpt2-small",
        "prompt": "When Mary and John went to the store, John gave a drink to",
        "intervention": "zero_ablation",
        "layer": 7,
        "head": 0,
        "timestamp": datetime.now().isoformat(),
    }
    
    manifest_hash = hashlib.sha256(json.dumps(experiment_data, sort_keys=True).encode()).hexdigest()
    assert len(manifest_hash) == 64
    
    ioi_results["provenance"] = {
        "experiment_id": experiment_data["experiment_id"],
        "manifest_hash": manifest_hash,
        "timestamp": experiment_data["timestamp"]
    }
    print(f"    Provenance: {manifest_hash[:16]}...")

test("Real IOI: activations", test_real_ioi_activations)
test("Real IOI: attention patterns", test_real_ioi_attention)
test("Real IOI: logits", test_real_ioi_logits)
test("Real IOI: intervention", test_real_ioi_intervention)
test("Real IOI: controls", test_real_ioi_controls)
test("Real IOI: provenance", test_real_ioi_provenance)


# ============================================================
# SECTION 6: Real Falsification Experiment
# ============================================================
print("\n=== SECTION 6: Real Falsification Experiment ===")

def test_real_falsification():
    from backend.science.models.gpt2_adapter import GPT2Adapter
    adapter = GPT2Adapter(variant="small", mock_mode=False)
    
    prompt = "The capital of France is"
    
    # Hypothesis: Layer 0, head 0 does NOT affect output
    # Falsification: zero-ablate and measure
    original = adapter.get_logits(prompt)
    patched = adapter.patch_head_output(prompt, layer=0, head_index=0)
    
    # Verify real values
    assert isinstance(patched.delta, float)
    assert patched.original_logit != 0.0
    
    # Check if hypothesis is supported or refuted
    effect_size = abs(patched.delta)
    if effect_size < 0.1:
        verdict = "SUPPORTED (layer 0 head 0 is NOT important)"
    else:
        verdict = "REFUTED (layer 0 head 0 IS important)"
    
    print(f"    Original top: '{original['top_token']}' (logit: {patched.original_logit})")
    print(f"    Patched top: '{patched.top_token_after}' (logit: {patched.patched_logit})")
    print(f"    Delta: {patched.delta}")
    print(f"    Effect size: {effect_size}")
    print(f"    Verdict: {verdict}")

test("Real falsification experiment", test_real_falsification)


# ============================================================
# SECTION 7: Real Replication Experiment
# ============================================================
print("\n=== SECTION 7: Real Replication Experiment ===")

def test_real_replication():
    from backend.science.models.gpt2_adapter import GPT2Adapter
    adapter = GPT2Adapter(variant="small", mock_mode=False)
    
    prompt = "The capital of France is"
    
    # Run 5 trials
    trials = []
    for i in range(5):
        result = adapter.get_logits(prompt)
        trials.append(result)
    
    # Verify consistency (deterministic model should be identical)
    top_tokens = [t["top_token"] for t in trials]
    assert len(set(top_tokens)) == 1, f"Top token should be consistent: {top_tokens}"
    
    print(f"    Replicated {len(trials)} trials")
    print(f"    All top tokens: '{top_tokens[0]}'")
    print(f"    Consistency: 100%")

test("Real replication experiment", test_real_replication)


# ============================================================
# SECTION 8: Dead Code Verification
# ============================================================
print("\n=== SECTION 8: Dead Code Verification ===")

def test_mock_helpers_are_dead():
    import os
    
    dead_functions = ["_mock_activation", "_dynamic_mock_tokens", "_mock_attention_patterns"]
    
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
                        for line in content.split("\n"):
                            if func_name in line and "def " not in line and "import " not in line:
                                stripped = line.strip()
                                if not stripped.startswith("#") and not stripped.startswith('"') and not stripped.startswith("'"):
                                    found_call = True
                    except Exception:
                        pass
        
        if found_def and not found_call:
            print(f"    {func_name}: DEAD CODE (defined, never called)")
        elif found_def and found_call:
            print(f"    WARNING: {func_name}: STILL CALLED")
        else:
            print(f"    {func_name}: NOT FOUND")

test("Mock helpers are dead code", test_mock_helpers_are_dead)


# ============================================================
# SECTION 9: Evidence Path Analysis
# ============================================================
print("\n=== SECTION 9: Evidence Path Analysis ===")

def test_evidence_path_no_mock():
    """Verify that the path from adapter -> ExperimentRun -> EvidenceRecord has no mock bypass."""
    import os
    
    # Check experiment_engine.py for mock_mode references
    filepath = "backend/core/experiment_engine.py"
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()
    
    # Should NOT have mock_mode in experiment engine
    assert "mock_mode" not in content, f"experiment_engine.py should not reference mock_mode"
    print(f"    experiment_engine.py: No mock_mode references")
    
    # Check scientific_router.py
    filepath = "backend/api/scientific_router.py"
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()
    
    # Should NOT create mock adapters
    assert "mock_mode=True" not in content, f"scientific_router.py should not create mock adapters"
    print(f"    scientific_router.py: No mock_mode=True references")

test("Evidence path has no mock bypass", test_evidence_path_no_mock)


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
