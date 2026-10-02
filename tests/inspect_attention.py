import urllib.request
import json

url = "http://127.0.0.1:8000/api/gpt2/attention_head"

prompts = [
    ("Prompt A (Fact Retrieval)", "The capital of France is Paris"),
    ("Prompt B (Indirect Object)", "When Mary and John went to the store, John gave a drink to Mary")
]

heads = [
    (0, 0, "L0H0  (Early Positional)"),
    (5, 5, "L5H5  (Middle Syntactic)"),
    (9, 9, "L9H9  (Name Mover)"),
    (10, 7, "L10H7 (Induction Head)")
]

for p_title, prompt in prompts:
    print("\n" + "="*70)
    print(f" {p_title}: \"{prompt}\"")
    print("="*70)
    
    for layer, head, head_label in heads:
        payload = json.dumps({"prompt": prompt, "layer": layer, "head": head}).encode("utf-8")
        req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            
        tokens = data["str_tokens"]
        matrix = data["matrix"]
        n = len(tokens)
        
        print(f"\n--- {head_label} | Matrix size: {n}x{n} ---")
        
        # Display the attention heatmap for the last 3 destination tokens
        dest_indices = range(max(0, n - 3), n)
        header = f"{'Query Token':<16} | " + " ".join([f"{t.strip()[:5]:>6}" for t in tokens])
        print(header)
        print("-" * len(header))
        
        for i in dest_indices:
            q_tok = tokens[i].strip()[:14]
            row_vals = " ".join([f"{matrix[i][j]:6.2f}" for j in range(n)])
            print(f"[{i:2d}] {q_tok:<12} | {row_vals}")
