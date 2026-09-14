"""Capture the actual HTTP request LiteLLM sends to Anthropic.

Monkey-patch httpx to intercept the outgoing POST body so we can see
whether cache_control markers reach Anthropic at all.
"""

import json
import os
from dotenv import load_dotenv
import httpx
import litellm

load_dotenv()

_orig_send = httpx.Client.send


def _spy_send(self, request, *a, **kw):
    if "anthropic.com" in str(request.url):
        try:
            body = json.loads(request.content.decode("utf-8"))
        except Exception:
            body = "<decode failed>"
        print("=== OUTGOING REQUEST ===")
        print("URL:", request.url)
        print("HEADERS:")
        for k, v in request.headers.items():
            if k.lower() in ("authorization", "x-api-key"):
                v = v[:8] + "...redacted"
            print(f"  {k}: {v}")
        with open("out/anthropic_request.json", "w", encoding="utf-8") as f:
            json.dump(body, f, indent=2, default=str)
        print("BODY WRITTEN to out/anthropic_request.json")
        if isinstance(body, dict):
            def _find_cache(obj, path=""):
                if isinstance(obj, dict):
                    for k, v in obj.items():
                        if k == "cache_control":
                            print(f"  found cache_control at {path}: {v}")
                        _find_cache(v, f"{path}.{k}")
                elif isinstance(obj, list):
                    for i, v in enumerate(obj):
                        _find_cache(v, f"{path}[{i}]")
            print("cache_control markers in outgoing body:")
            _find_cache(body)
    return _orig_send(self, request, *a, **kw)


httpx.Client.send = _spy_send


r = litellm.completion(
    model="anthropic/claude-haiku-4-5",
    messages=[
        {"role": "system", "content": [
            {"type": "text", "text": "You are terse.", "cache_control": {"type": "ephemeral"}}
        ]},
        {"role": "user", "content": [
            {"type": "text", "text": "STABLE_PREFIX. " * 400, "cache_control": {"type": "ephemeral"}},
            {"type": "text", "text": "What is 2+2?"},
        ]},
    ],
    max_tokens=20,
)
print("\n=== USAGE ===")
print(r.usage)
