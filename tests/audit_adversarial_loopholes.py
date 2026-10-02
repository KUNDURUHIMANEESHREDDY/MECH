import urllib.request
import urllib.error
import json

BASE_URL = "http://127.0.0.1:8000"

def probe(method, path, body=None):
    url = f"{BASE_URL}{path}"
    headers = {"Content-Type": "application/json"}
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return {"status": resp.status, "body": resp.read().decode("utf-8", errors="replace")}
    except urllib.error.HTTPError as e:
        return {"status": e.code, "body": e.read().decode("utf-8", errors="replace")}
    except Exception as e:
        return {"status": 0, "body": str(e)}

cases = [
    # 1. Out-of-bounds layer / head values (should handle gracefully without 500)
    ("OOB Layer in attention_head", "POST", "/api/gpt2/attention_head", {"layer": 9999, "head": 9999, "prompt": "Hi"}),
    ("Negative Layer in attention_head", "POST", "/api/gpt2/attention_head", {"layer": -5, "head": -2, "prompt": "Hi"}),
    ("OOB Layer in activations", "POST", "/api/gpt2/activations", {"layer": 100, "prompt": "Test"}),
    ("OOB Neuron Index", "POST", "/api/gpt2/neuron", {"layer": 0, "neuron_index": 999999}),
    
    # 2. Type Mismatch / Invalid Types
    ("Float layer in attention_head", "POST", "/api/gpt2/attention_head", {"layer": 1.5, "head": "abc"}),
    ("Invalid pos_token type", "POST", "/api/gpt2/patch_head", {"layer": 0, "head": 0, "pos_token": 123, "neg_token": None}),
    ("Null prompt in infer", "POST", "/api/infer", {"prompt": None}),
    ("Integer prompt in run_prompt", "POST", "/api/gpt2/run_prompt", {"prompt": 9999}),
    
    # 3. Path Traversal / SQL Injection in CRUD & Routes
    ("SQL Injection in Experiment ID", "POST", "/api/experiments", {"id": "'; DROP TABLE experiments; --", "title": "Injected"}),
    ("Nonexistent Circuit Query", "GET", "/api/circuits/non_existent_fake_circuit", None),
    ("Nonexistent Society Run Query", "GET", "/api/society/runs/fake_run_12345", None),
    
    # 4. Empty / Boundary strings
    ("Empty Prompt in infer", "POST", "/api/infer", {"prompt": ""}),
    ("Very Long Prompt (2000 chars)", "POST", "/api/infer", {"prompt": "AI interpretability " * 100}),
    ("Special Unicode & Emojis", "POST", "/api/gpt2/run_prompt", {"prompt": "?? Transformer ? ?? \u2603 \x00"}),
]

print(f"Running Adversarial Edge-Case Audit across {len(cases)} vectors...\n")
crashes = []

for label, method, path, payload in cases:
    res = probe(method, path, payload)
    status = res["status"]
    is_500 = (status >= 500 or status == 0)
    
    tag = "CRASH (500)" if is_500 else ("HANDLED (4xx)" if status >= 400 else "SUCCESS (200)")
    print(f"[{tag:<13}] {label:<35} | {method:<6} {path:<30} -> HTTP {status}")
    if is_500:
        crashes.append((label, path, status, res["body"]))

print("\n" + "="*80)
if crashes:
    print(f"FAILED: Found {len(crashes)} unhandled server crashes (HTTP 500):")
    for label, path, status, body in crashes:
        print(f" - {label} ({path}): HTTP {status}\n   Body: {body[:300]}")
else:
    print("SUCCESS: Zero unhandled 500 crashes found! All edge cases either handled or validated with 4xx.")
print("="*80)
