import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def test_load():
    print("Testing GPT-2 Small load...")
    try:
        model = AutoModelForCausalLM.from_pretrained("gpt2")
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        print("Model loaded successfully.")

        prompt = "When John and Mary went to the store, John gave a drink to"
        inputs = tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits[0, -1, :]
            top_id = torch.argmax(logits).item()
            print(f"Top prediction: '{tokenizer.decode([top_id])}'")
    except Exception as e:
        print(f"Load failed: {e}")

if __name__ == "__main__":
    test_load()
