"""Automate every API endpoint against the running MECH backend."""
import json, urllib.request, urllib.error, sys

BASE = "http://localhost:8000"
errors = []

def call(method, path, body=None):
    url = f"{BASE}{path}"
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    if data:
        req.add_header("Content-Type", "application/json")
    try:
        resp = urllib.request.urlopen(req, timeout=30)
        status = resp.status
        text = resp.read().decode()
        try:
            parsed = json.loads(text)
        except:
            parsed = text[:200]
        print(f"  [{method} {path}] {status} -> {json.dumps(parsed)[:120] if not isinstance(parsed, str) else parsed[:120]}")
        return status, parsed
    except urllib.error.HTTPError as e:
        text = e.read().decode()[:200]
        print(f"  [{method} {path}] ERROR {e.code}: {text}")
        errors.append(f"HTTPError {method} {path}: {e.code} {text}")
        return e.code, None
    except Exception as e:
        print(f"  [{method} {path}] EXCEPTION: {e}")
        errors.append(f"Exception {method} {path}: {e}")
        return None, None

print("=" * 60)
print("API Endpoint Automation")
print("=" * 60)

# GET endpoints
print("\n--- GET endpoints ---")
call("GET", "/")
call("GET", "/health")
call("GET", "/api/status")
call("GET", "/api/models")
call("GET", "/api/benchmarks")
call("GET", "/api/research_catalog")
call("GET", "/api/experiments")
call("GET", "/api/sessions")
call("GET", "/api/discoveries")
call("GET", "/api/portal/summary")
call("GET", "/api/interpretability/inspectors")
call("GET", "/api/runtime/engines")
call("GET", "/api/agents")
call("GET", "/api/knowledge-graph")
call("GET", "/api/models/gpt2-small")

# POST endpoints
print("\n--- POST endpoints ---")
call("POST", "/api/ping", {})
call("POST", "/api/models/load", {"model_name": "gpt2-small"})
call("POST", "/api/infer", {"prompt": "Hello world", "model_name": "gpt2-small"})
call("POST", "/api/benchmarks/run", {"benchmark_name": "IOI"})
call("POST", "/api/experiments", {"name": "test_exp", "description": "auto test"})
call("POST", "/api/sessions", {"name": "test_sess"})
call("POST", "/api/runtime/status", {})
call("POST", "/api/runtime/analyze_tokens", {"prompt": "Hello world test"})
call("POST", "/api/gpt2/load", {"model_name": "gpt2"})
call("POST", "/api/gpt2/run_prompt", {"prompt": "The capital of France is"})
call("POST", "/api/gpt2/activations", {"layer": 5})
call("POST", "/api/gpt2/attention_head", {"layer": 5, "head": 3})
call("POST", "/api/gpt2/patchhead", {"layer": 9, "head": 9, "pos_token": "Paris", "neg_token": "London"})
call("POST", "/api/gpt2/patch_head", {"layer": 9, "head": 9, "pos_token": "Paris", "neg_token": "London"})
call("POST", "/api/gpt2/ioi", {"io_name": "John", "subj_name": "Mary"})
call("POST", "/api/gpt2/architecture", {})
call("POST", "/api/gpt2/layer", {"layer": 5})
call("POST", "/api/gpt2/neurons", {"layer": 5, "component": "mlp"})
call("POST", "/api/gpt2/neuron", {"layer": 5, "neuron_index": 42})
call("POST", "/api/gpt2/head", {"layer": 5, "head": 3})
call("POST", "/api/gpt2/patch_neuron", {"layer": 5, "neuron_index": 42, "patch_value": 3.5})

# DELETE endpoints
print("\n--- DELETE endpoints ---")
# First create, then delete
_, exp = call("POST", "/api/experiments", {"name": "del_test", "description": "to be deleted"})
call("DELETE", "/api/experiments/exp_9999")
_, sess = call("POST", "/api/sessions", {"name": "del_sess"})
call("DELETE", "/api/sessions/sess_9999")

# Test research_catalog with different item_types
print("\n--- research_catalog variants ---")
call("GET", "/api/research_catalog?item_type=models")
call("GET", "/api/research_catalog?item_type=papers")
call("GET", "/api/research_catalog?item_type=all")

print("\n" + "=" * 60)
if errors:
    print(f"FAILED: {len(errors)} errors found:")
    for e in errors:
        print(f"  - {e}")
    sys.exit(1)
else:
    print("ALL API ENDPOINTS PASSED - no errors")
    sys.exit(0)
