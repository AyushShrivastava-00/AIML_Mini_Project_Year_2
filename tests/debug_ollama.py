import json
import time
import requests

url = "http://localhost:11434/api/generate"
payload = {
    "model": "qwen2.5:1.5b",
    "prompt": "You are a credit auditor. Return JSON with key 'underwriting_rationale' summarizing approval for Suresh Patel.",
    "stream": False,
    "format": "json",
    "options": {"temperature": 0.1, "num_predict": 150}
}

print("⏳ Sending request to Ollama (qwen2.5:1.5b)...")
start = time.time()
try:
    res = requests.post(url, json=payload, timeout=40)
    elapsed = time.time() - start
    print(f"⏱️ Time taken: {elapsed:.2f} seconds")
    print(f"📡 Status Code: {res.status_code}")
    print(f"📄 Raw Response: {res.text[:300]}...")

    res_json = res.json()
    model_response = res_json.get("response", "")
    print(f"\n🧠 Model Output String:\n{model_response}")

    parsed = json.loads(model_response)
    print(f"\n✅ Successfully parsed JSON keys: {list(parsed.keys())}")
    print(f"Content: {parsed}")
except requests.exceptions.Timeout:
    print(f"❌ TIMEOUT: Ollama took longer than 40 seconds to generate on CPU.")
except Exception as e:
    print(f"❌ ERROR: {type(e).__name__}: {e}")
