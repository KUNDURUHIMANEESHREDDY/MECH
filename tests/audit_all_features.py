import urllib.request
import urllib.error
import json
import time

BASE_URL = "http://127.0.0.1:8000"

def request(method, path, body=None, timeout=30):
    url = f"{BASE_URL}{path}"
    headers = {"Content-Type": "application/json"}
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    start = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            elapsed = round((time.time() - start) * 1000, 2)
            content_type = resp.headers.get("Content-Type", ""); raw_bytes = resp.read(); raw = f"<{len(raw_bytes)} bytes binary image>" if "image" in content_type else raw_bytes.decode("utf-8", errors="replace")
            try:
                res_json = json.loads(raw)
            except Exception:
                res_json = raw[:200]
            return {"status": resp.status, "data": res_json, "latency_ms": elapsed, "error": None}
    except urllib.error.HTTPError as e:
        elapsed = round((time.time() - start) * 1000, 2)
        raw_err = e.read().decode("utf-8")
        try:
            err_json = json.loads(raw_err)
        except Exception:
            err_json = raw_err[:200]
        return {"status": e.code, "data": err_json, "latency_ms": elapsed, "error": f"HTTP {e.code}"}
    except Exception as e:
        elapsed = round((time.time() - start) * 1000, 2)
        return {"status": 0, "data": None, "latency_ms": elapsed, "error": str(e)}

features = [
    # 1. Health & System
    ("Health Check", "GET", "/health", None),
    ("API Status", "GET", "/api/status", None),
    ("API Ping", "POST", "/api/ping", {}),
    ("Portal Summary", "GET", "/api/portal/summary", None),
    
    # 2. Model Catalog & Management
    ("List Models", "GET", "/api/models", None),
    ("Get Model Info (gpt2-small)", "GET", "/api/models/gpt2-small", None),
    ("Load Model", "POST", "/api/models/load", {"model_name": "gpt2-small"}),
    ("GPT2 Load", "POST", "/api/gpt2/load", {"model_name": "gpt2"}),
    
    # 3. GPT-2 Inference & Probing
    ("Infer (Prompt)", "POST", "/api/infer", {"prompt": "The capital of France is"}),
    ("GPT2 Run Prompt", "POST", "/api/gpt2/run_prompt", {"prompt": "Neural networks learn"}),
    ("GPT2 Architecture", "POST", "/api/gpt2/architecture", {}),
    ("GPT2 Layer Detail (L0)", "POST", "/api/gpt2/layer", {"layer": 0}),
    ("GPT2 Head Detail (L0H0)", "POST", "/api/gpt2/head", {"layer": 0, "head": 0}),
    ("GPT2 Attention Head (L5H5)", "POST", "/api/gpt2/attention_head", {"prompt": "The cat sat on the mat", "layer": 5, "head": 5}),
    ("GPT2 Activations (L2)", "POST", "/api/gpt2/activations", {"prompt": "Hello world", "layer": 2}),
    ("GPT2 Layer Activations", "POST", "/api/gpt2/layer_activations", {"prompt": "Quick brown fox", "layer": 1}),
    ("GPT2 Single Neuron (L0N10)", "POST", "/api/gpt2/neuron", {"layer": 0, "neuron_index": 10}),
    ("GPT2 Neurons Page", "POST", "/api/gpt2/neurons", {"layer": 0, "page": 1, "page_size": 10}),
    ("GPT2 Patch Head", "POST", "/api/gpt2/patch_head", {"layer": 9, "head": 9, "pos_token": "Paris", "neg_token": "London"}),
    ("GPT2 Patch Neuron", "POST", "/api/gpt2/patch_neuron", {"layer": 0, "neuron": 10, "pos_token": "Paris", "neg_token": "London"}),
    ("GPT2 Logit Lens All", "POST", "/api/gpt2/logit_lens_all", {"prompt": "France capital is"}),
    ("Figure Attention", "GET", "/api/figures/attention?layer=0&head=0&prompt=Hello", None),
    
    # 4. Mechanistic Circuits & Interpretability
    ("GPT2 IOI Reproduction", "POST", "/api/gpt2/ioi", {"io_name": "Mary", "subj_name": "John"}),
    ("List Circuits", "GET", "/api/circuits", None),
    ("Get Circuit Detail", "GET", "/api/circuits/ioi_circuit_gpt2_small", None),
    ("List Discoveries", "GET", "/api/discoveries", None),
    ("List Inspectors", "GET", "/api/interpretability/inspectors", None),
    ("Runtime Analyze Tokens", "POST", "/api/runtime/analyze_tokens", {"text": "Transformers process tokens"}),
    ("Runtime Engines", "GET", "/api/runtime/engines", None),
    ("Runtime Status", "POST", "/api/runtime/status", {}),
    
    # 5. Autonomous AI Scientist & Society
    ("List Agents", "GET", "/api/agents", None),
    ("Research Catalog", "GET", "/api/research_catalog", None),
    ("Society Run Proposal", "POST", "/api/society/run", {"hypothesis": "Layer 9 mediates indirect object identification", "model": "gpt2-small"}),
    ("Society Runs List", "GET", "/api/society/runs", None),
    
    # 6. Knowledge Graph & Storage Persistence
    ("Knowledge Graph Info", "GET", "/api/knowledge-graph", None),
    ("List Experiments", "GET", "/api/experiments", None),
    ("Create Experiment", "POST", "/api/experiments", {"id": "audit_exp_001", "title": "Comprehensive Feature Audit", "status": "running"}),
    ("Delete Experiment", "DELETE", "/api/experiments/audit_exp_001", None),
    ("List Sessions", "GET", "/api/sessions", None),
    ("Create Session", "POST", "/api/sessions", {"id": "audit_sess_001", "name": "Audit Session", "model": "gpt2-small"}),
    ("Delete Session", "DELETE", "/api/sessions/audit_sess_001", None),
    ("List Benchmarks", "GET", "/api/benchmarks", None),
    ("Run Benchmark", "POST", "/api/benchmarks/run", {"benchmark_name": "induction_test"}),
]

print(f"Starting Systematic Feature Audit across {len(features)} platform capabilities...\n")
results = []
failures = []

for name, method, path, body in features:
    res = request(method, path, body)
    status = res["status"]
    lat = res["latency_ms"]
    err = res["error"]
    ok = (200 <= status < 300)
    
    status_str = f"HTTP {status}" if status else "FAILED"
    mark = "PASS" if ok else "FAIL"
    print(f"[{mark}] {name:<35} | {method:<6} {path:<40} | {status_str:<8} | {lat:7.1f}ms")
    
    results.append({"name": name, "ok": ok, "status": status, "path": path, "latency": lat, "error": err, "data": res["data"]})
    if not ok:
        failures.append({"name": name, "method": method, "path": path, "status": status, "data": res["data"]})

passed = sum(1 for r in results if r["ok"])
total = len(results)
print("\n" + "="*80)
print(f"AUDIT SUMMARY: {passed}/{total} features passed ({round(passed/total*100, 1)}%)")
print("="*80)

if failures:
    print(f"\nDiscovered {len(failures)} Issues/Loopholes:")
    for f in failures:
        print(f" - {f['name']} ({f['method']} {f['path']}) -> Status: {f['status']} | Response: {f['data']}")
else:
    print("\nALL TESTED PLATFORM FEATURES ARE 100% OPERATIONAL!")

