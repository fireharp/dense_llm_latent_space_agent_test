"""Simple test to check HuggingFace API access."""

import os
from dotenv import load_dotenv
from huggingface_hub import InferenceClient, list_models
import requests

load_dotenv()

api_key = os.environ.get("HUGGINGFACE_API_KEY")
print(f"API Key present: {bool(api_key)}")
print(f"API Key prefix: {api_key[:8]}..." if api_key else "No key")

# Test 1: Direct API call
print("\n1. Testing direct API call to Qwen2-0.5B:")
print("-" * 50)

headers = {"Authorization": f"Bearer {api_key}"}
API_URL = "https://api-inference.huggingface.co/models/Qwen/Qwen2-0.5B"

response = requests.post(
    API_URL,
    headers=headers,
    json={"inputs": "What is 2+2? The answer is"},
)

print(f"Status code: {response.status_code}")
print(f"Response: {response.text[:200]}")

# Test 2: Try with a known working model
print("\n2. Testing with GPT-2 (known to work):")
print("-" * 50)

try:
    client = InferenceClient(token=api_key)
    result = client.text_generation(
        "What is 2+2? The answer is",
        model="gpt2",
        max_new_tokens=10
    )
    print(f"GPT-2 response: {result}")
except Exception as e:
    print(f"Error: {e}")

# Test 3: Check if Qwen2-0.5B needs different endpoint
print("\n3. Checking Qwen2-0.5B status:")
print("-" * 50)

# Try text-generation-inference endpoint
API_URL_TGI = "https://api-inference.huggingface.co/models/Qwen/Qwen2-0.5B"
response = requests.get(
    API_URL_TGI + "/status",
    headers=headers,
)
print(f"Model status: {response.text if response.status_code == 200 else 'Not available'}")

# Test 4: Try with explicit parameters
print("\n4. Testing Qwen2-0.5B with explicit parameters:")
print("-" * 50)

payload = {
    "inputs": "The capital of France is",
    "parameters": {
        "max_new_tokens": 10,
        "temperature": 0.7,
        "return_full_text": False
    }
}

response = requests.post(API_URL, headers=headers, json=payload)
print(f"Status: {response.status_code}")
if response.status_code == 200:
    print(f"Response: {response.json()}")
else:
    print(f"Error: {response.text}")

# Test 5: List available text generation models
print("\n5. Checking available models:")
print("-" * 50)
print("Visit: https://huggingface.co/models?pipeline_tag=text-generation&sort=likes")
print("Large models like Qwen2-0.5B may require:")
print("- Inference Endpoints (paid)")
print("- Spaces deployment")
print("- Local inference")