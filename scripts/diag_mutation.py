"""Confirm the mutation hypothesis: does calling structured_complete with
gpt-4o-mini strip cache_control from the messages list in place?
"""

import copy
import json
import os

os.environ.setdefault("AGENTIC_CITE_TO_TRADE", "1")

from dotenv import load_dotenv
load_dotenv()

from agentic_investor.orchestrator.state import Allocation  # noqa
from agentic_investor.llm.client import structured_complete  # noqa

SLOW_PREFIX = "STABLE. " * 1500

messages = [
    {"role": "system", "content": [
        {"type": "text", "text": "You are terse.", "cache_control": {"type": "ephemeral"}},
    ]},
    {"role": "user", "content": [
        {"type": "text", "text": SLOW_PREFIX, "cache_control": {"type": "ephemeral"}},
        {"type": "text", "text": "Recommend allocation."},
    ]},
]

before = json.dumps(messages, sort_keys=True)
print("BEFORE gpt-4o-mini call:")
print("  cache_control markers found:", before.count('"cache_control"'))

structured_complete(Allocation, messages, model="gpt-4o-mini")

after = json.dumps(messages, sort_keys=True)
print("AFTER gpt-4o-mini call:")
print("  cache_control markers found:", after.count('"cache_control"'))
print("  messages identical?", before == after)

if before != after:
    print("\n  DIFF found. Messages mutated by LiteLLM OpenAI adapter.")
