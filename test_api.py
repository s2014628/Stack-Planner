import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from scripts.qa_bench_pipeline.summary_agent import (
    SUMMARY_API_KEY as api_key,
    SUMMARY_BASE_URL as base_url,
    SUMMARY_MODEL as model,
)

print("=" * 50)
print("OpenRouter API Diagnostics")
print("=" * 50)

print("\n[1] Proxy env vars:")
for k in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "no_proxy", "NO_PROXY", "ALL_PROXY"):
    v = os.environ.get(k)
    if v:
        print(f"  {k} = {v}")
if not any(os.environ.get(k) for k in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY")):
    print("  (none set)")

print("\n[2] SSL info:")
import ssl, certifi
print(f"  OpenSSL: {ssl.OPENSSL_VERSION}")
print(f"  certifi: {certifi.where()}")

print("\n[3] DNS resolution:")
import socket
try:
    ips = socket.getaddrinfo("openrouter.ai", 443)
    print(f"  openrouter.ai -> {ips[0][4][0]}")
except Exception as e:
    print(f"  DNS FAILED: {e}")

print("\n[4] Raw socket connect (port 443):")
try:
    s = socket.create_connection(("openrouter.ai", 443), timeout=10)
    print("  TCP connect OK")
    s.close()
except Exception as e:
    print(f"  TCP connect FAILED: {e}")

print("\n[5] httpx GET test:")
import httpx
try:
    r = httpx.get(f"{base_url}/models", timeout=15)
    print(f"  Status: {r.status_code}")
except Exception:
    traceback.print_exc()

print(f"\n[6] OpenAI chat completion (model={model}):")
from openai import OpenAI
try:
    client = OpenAI(api_key=api_key, base_url=base_url)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "Say hello in one sentence."}],
        max_tokens=50,
        timeout=30,
    )
    print(f"  SUCCESS: {response.choices[0].message.content}")
except Exception:
    traceback.print_exc()
