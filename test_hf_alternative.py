"""Test with alternative models available on HF Inference API."""

import os
from dotenv import load_dotenv
import requests
import json

load_dotenv()

api_key = os.environ.get("HUGGINGFACE_API_KEY")
headers = {"Authorization": f"Bearer {api_key}"}

print("="*70)
print("Testing Alternative Models on HuggingFace Inference API")
print("="*70)

# Models that are typically available on free tier
test_models = [
    ("microsoft/phi-2", "Phi-2 (2.7B)"),
    ("mistralai/Mistral-7B-v0.1", "Mistral 7B"),
    ("facebook/opt-125m", "OPT-125M"),
    ("EleutherAI/gpt-neo-125M", "GPT-Neo 125M"),
    ("bigscience/bloom-560m", "BLOOM 560M"),
]

print("\nTesting which models are available...")
print("-" * 50)

available_models = []

for model_id, model_name in test_models:
    url = f"https://api-inference.huggingface.co/models/{model_id}"
    payload = {
        "inputs": "What is 2+2? Answer:",
        "parameters": {
            "max_new_tokens": 10,
            "temperature": 0.7
        }
    }
    
    response = requests.post(url, headers=headers, json=payload)
    
    if response.status_code == 200:
        try:
            result = response.json()
            if isinstance(result, list) and len(result) > 0:
                output = result[0].get("generated_text", "")
            else:
                output = str(result)
            print(f"✓ {model_name}: {output[:50]}...")
            available_models.append((model_id, model_name))
        except:
            print(f"✗ {model_name}: Invalid response")
    elif response.status_code == 503:
        print(f"⏳ {model_name}: Loading (may take 20s on first call)")
        available_models.append((model_id, model_name))
    else:
        print(f"✗ {model_name}: Not available ({response.status_code})")

print("\n" + "="*70)
print("RESULTS:")
print("="*70)

if available_models:
    print(f"\nFound {len(available_models)} available model(s):")
    for model_id, model_name in available_models:
        print(f"  • {model_name} ({model_id})")
    
    print("\nNOTE: Since Qwen2-0.5B is not available on free tier, you can:")
    print("1. Use one of the above models as a substitute")
    print("2. Deploy Qwen2-0.5B to HF Spaces (free with some limits)")
    print("3. Use HF Inference Endpoints (paid, ~$0.06/hour)")
    print("4. Run locally with the original dense_lm.py")
else:
    print("\nNo models immediately available. This might be due to:")
    print("- API key issues")
    print("- Rate limiting")
    print("- Models need to warm up (try again in 20s)")

print("\nFor dense vector experiments, we recommend:")
print("- Continue using MockLM for testing")
print("- Run locally with real Qwen2-0.5B when needed")
print("- The architecture works with any transformer model")