"""Replicate _ensemble_allocate: same messages hit gpt-4o-mini then haiku.

Does hitting OpenAI first with the same message list corrupt the Anthropic
cache path somehow?
"""

import os
os.environ.setdefault("AGENTIC_CITE_TO_TRADE", "1")

from dotenv import load_dotenv
load_dotenv()

from agentic_investor.orchestrator.state import Allocation  # noqa
from agentic_investor.llm.client import structured_complete  # noqa

SLOW_PREFIX = (
    "# 1. Portfolio rules\n"
    + ("- Cash floor is 10% of NAV. " * 200)
    + "\n# 2. Universe\n"
    + ("- Consider stocks in SP500 with market cap > $10B. " * 200)
    + "\n# 3. Recent regime\n"
    + ("VIX 17.5, SPY -1.8% 20d, sideways. " * 100)
)
FAST_TAIL = "Given AAPL just reported earnings, recommend allocation."

messages = [
    {"role": "system", "content": [
        {"type": "text", "text": "You are a portfolio allocator. Return a valid Allocation.",
         "cache_control": {"type": "ephemeral"}},
    ]},
    {"role": "user", "content": [
        {"type": "text", "text": SLOW_PREFIX, "cache_control": {"type": "ephemeral"}},
        {"type": "text", "text": FAST_TAIL},
    ]},
]

import json
import httpx

_orig = httpx.Client.send


def _spy(self, request, *a, **kw):
    resp = _orig(self, request, *a, **kw)
    url = str(request.url)
    if "anthropic.com" in url:
        try:
            body = json.loads(resp.content.decode("utf-8"))
            u = body.get("usage", {})
            print(f"  [ANTHROPIC] prompt={u.get('input_tokens')} "
                  f"cache_create={u.get('cache_creation_input_tokens')} "
                  f"cache_read={u.get('cache_read_input_tokens')}")
            # inspect outgoing body for cache_control markers
            req_body = json.loads(request.content.decode("utf-8"))
            def _find(o, path=""):
                if isinstance(o, dict):
                    for k, v in o.items():
                        if k == "cache_control":
                            print(f"    outgoing marker at {path}: {v}")
                        _find(v, f"{path}.{k}")
                elif isinstance(o, list):
                    for i, v in enumerate(o):
                        _find(v, f"{path}[{i}]")
            _find(req_body)
        except Exception as e:
            print(f"  parse fail: {e}")
    elif "openai.com" in url:
        try:
            body = json.loads(resp.content.decode("utf-8"))
            u = body.get("usage", {})
            ptd = u.get("prompt_tokens_details") or {}
            print(f"  [OPENAI] prompt={u.get('prompt_tokens')} "
                  f"cached={ptd.get('cached_tokens')}")
        except Exception as e:
            print(f"  parse fail: {e}")
    return resp


httpx.Client.send = _spy

# Tick 1: same-tick ensemble (gpt then haiku)
print("=== TICK 1 (gpt-4o-mini then haiku) ===")
r1 = structured_complete(Allocation, messages, model="gpt-4o-mini")
print(f"  gpt returned {len(r1.positions)} positions")
r2 = structured_complete(Allocation, messages, model="anthropic/claude-haiku-4-5")
print(f"  haiku returned {len(r2.positions)} positions")

# Tick 2: same messages, should now hit cache on haiku
print("\n=== TICK 2 (gpt-4o-mini then haiku, same messages) ===")
r3 = structured_complete(Allocation, messages, model="gpt-4o-mini")
print(f"  gpt returned {len(r3.positions)} positions")
r4 = structured_complete(Allocation, messages, model="anthropic/claude-haiku-4-5")
print(f"  haiku returned {len(r4.positions)} positions")
