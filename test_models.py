import os
import requests
import time
from dotenv import load_dotenv

load_dotenv("/home/joa/groq-bot/.env")
nv_key = os.getenv("NVIDIA_API_KEY")
groq_key = os.getenv("GROQ_API_KEY")

def test_api(name, url, headers, json_payload):
    print(f"--- Testing {name} ---")
    start = time.time()
    try:
        res = requests.post(url, headers=headers, json=json_payload, timeout=5.0)
        elapsed = time.time() - start
        print(f"Status: {res.status_code} (took {elapsed:.2f}s)")
        if res.status_code != 200:
            print(f"Error: {res.text[:200]}")
    except Exception as e:
        print(f"Failed! Exception: {e}")
    print()

test_api(
    "NVIDIA Scout (Llama 3.1 8B)",
    "https://integrate.api.nvidia.com/v1/chat/completions",
    {"Authorization": f"Bearer {nv_key}"},
    {"model": "meta/llama-3.1-8b-instruct", "messages": [{"role": "user", "content": "Hi"}], "max_tokens": 10}
)

test_api(
    "NVIDIA Brain (Llama 3.3 70B)",
    "https://integrate.api.nvidia.com/v1/chat/completions",
    {"Authorization": f"Bearer {nv_key}"},
    {"model": "meta/llama-3.3-70b-instruct", "messages": [{"role": "user", "content": "Hi"}], "max_tokens": 10}
)

test_api(
    "NVIDIA Flux Image Gen",
    "https://ai.api.nvidia.com/v1/genai/black-forest-labs/flux.1-schnell",
    {"Authorization": f"Bearer {nv_key}", "Accept": "application/json"},
    {"prompt": "a red car", "width": 1024, "height": 1024, "seed": 0, "steps": 4}
)

test_api(
    "NVIDIA Vision",
    "https://integrate.api.nvidia.com/v1/chat/completions",
    {"Authorization": f"Bearer {nv_key}"},
    {"model": "meta/llama-3.2-11b-vision-instruct", "messages": [{"role": "user", "content": "Hi"}], "max_tokens": 10}
)

test_api(
    "Groq Fallback (llama-3.3-70b-versatile)",
    "https://api.groq.com/openai/v1/chat/completions",
    {"Authorization": f"Bearer {groq_key}"},
    {"model": "llama-3.3-70b-versatile", "messages": [{"role": "user", "content": "Hi"}], "max_tokens": 10}
)
