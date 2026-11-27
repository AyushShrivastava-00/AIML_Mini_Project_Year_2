import os
import sys

# 1. Check if groq package is installed
try:
    from groq import Groq
    print("✅ 'groq' Python package is installed.")
except ImportError:
    print("❌ ERROR: 'groq' package is NOT installed in this virtual environment!")
    print("👉 Run: pip install groq")
    sys.exit(1)

# 2. Check for API key in environment or Streamlit secrets
api_key = os.environ.get("GROQ_API_KEY")

if not api_key:
    # Check .streamlit/secrets.toml
    secrets_path = os.path.join(".streamlit", "secrets.toml")
    if os.path.exists(secrets_path):
        with open(secrets_path, "r") as f:
            for line in f:
                if "GROQ_API_KEY" in line and "=" in line:
                    api_key = line.split("=", 1)[1].strip().strip('"').strip("'")
                    print(f"🔑 Found key in .streamlit/secrets.toml: {api_key[:8]}...{api_key[-4:]}")

if not api_key:
    print("❌ ERROR: GROQ_API_KEY is not set anywhere!")
    print("👉 You must set it in .streamlit/secrets.toml or via terminal environment variable.")
    sys.exit(1)

# 3. Test a live API call to Groq
print(f"⏳ Connecting to Groq Cloud using model 'llama-3.1-8b-instant'...")
try:
    client = Groq(api_key=api_key)
    completion = client.chat.completions.create(
        messages=[
            {"role": "system", "content": "You are a credit auditor. Output JSON only."},
            {"role": "user", "content": "Write a 1-sentence approval memo in JSON format with key 'underwriting_rationale'."}
        ],
        model="llama-3.1-8b-instant",
        temperature=0.1,
        max_tokens=100,
        response_format={"type": "json_object"}
    )
    response_text = completion.choices[0].message.content
    print(f"\n✅ [GROQ SUCCESS] Live response received:")
    print(response_text)
except Exception as e:
    print(f"\n❌ [GROQ API ERROR] {type(e).__name__}: {e}")
