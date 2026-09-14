"""Replicate the exact structured_complete call path with the actual
Allocation Pydantic model and a big padded slow_prefix.

If this shows cache_read > 0 on call 2, then our production code is fine
and something in the running arm C is subtly different. If it shows
cache_creation=0, the bug is reproducible.
"""

import os
os.environ.setdefault("AGENTIC_CITE_TO_TRADE", "1")

from dotenv import load_dotenv
load_dotenv()

from agentic_investor.orchestrator.state import Allocation  # noqa
from agentic_investor.llm.client import structured_complete  # noqa

# Realistic slow_prefix ~5000 tokens (matches what graph.py produces)
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
    if "anthropic.com" in str(request.url):
        try:
            body = json.loads(resp.content.decode("utf-8"))
            u = body.get("usage", {})
            print(f"  ANTHROPIC USAGE prompt={u.get('input_tokens')} "
                  f"cache_create={u.get('cache_creation_input_tokens')} "
                  f"cache_read={u.get('cache_read_input_tokens')}")
        except Exception as e:
            print(f"  parse fail: {e}")
    return resp


httpx.Client.send = _spy

print(f"slow_prefix chars={len(SLOW_PREFIX)} est_tokens={len(SLOW_PREFIX)//4}")
print("\n=== call 1 (should create cache) ===")
r1 = structured_complete(Allocation, messages, model="anthropic/claude-haiku-4-5")
print(f"  Allocation returned with {len(r1.positions)} positions")

print("\n=== call 2 (should read cache) ===")
r2 = structured_complete(Allocation, messages, model="anthropic/claude-haiku-4-5")
print(f"  Allocation returned with {len(r2.positions)} positions")
