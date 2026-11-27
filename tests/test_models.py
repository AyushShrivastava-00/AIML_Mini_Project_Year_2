import os
from groq import Groq

# Extract key from .streamlit/secrets.toml
key = None
with open(".streamlit/secrets.toml", "r") as f:
    for line in f:
        if "GROQ_API_KEY" in line and "=" in line:
            key = line.split("=", 1)[1].strip().strip('"').strip("'")
            break

if not key:
    print("❌ Key not found in .streamlit/secrets.toml")
    exit(1)

client = Groq(api_key=key)
models = [m.id for m in client.models.list().data]

print("📋 ACTIVE GROQ MODELS ON YOUR ACCOUNT:")
for m in sorted(models):
    print(f" - {m}")
