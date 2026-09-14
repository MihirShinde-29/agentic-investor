"""Test whether adding tools/tool_choice (like Instructor does) breaks
Anthropic caching for otherwise identical requests.
"""

import os
from dotenv import load_dotenv
import litellm

load_dotenv()


BIG_PREFIX = "STABLE. " * 1500  # ~3000 tokens, above haiku minimum


def _u(r):
    u = r.usage.model_dump() if hasattr(r.usage, "model_dump") else dict(r.usage)
    return {
        "prompt": u.get("prompt_tokens"),
        "cache_create": u.get("cache_creation_input_tokens"),
        "cache_read": u.get("cache_read_input_tokens"),
    }


print("=== no tools, plain completion, 2 back-to-back calls ===")
for i in (1, 2):
    r = litellm.completion(
        model="anthropic/claude-haiku-4-5",
        messages=[
            {"role": "user", "content": [
                {"type": "text", "text": BIG_PREFIX, "cache_control": {"type": "ephemeral"}},
                {"type": "text", "text": "What is 2+2?"},
            ]},
        ],
        max_tokens=20,
    )
    print(f"  call {i}: {_u(r)}")


print("\n=== WITH tools param + tool_choice, 2 back-to-back calls ===")
tools = [{
    "type": "function",
    "function": {
        "name": "answer",
        "description": "Answer the question",
        "parameters": {"type": "object", "properties": {"verdict": {"type": "string"}}, "required": ["verdict"]},
    },
}]
for i in (1, 2):
    r = litellm.completion(
        model="anthropic/claude-haiku-4-5",
        messages=[
            {"role": "user", "content": [
                {"type": "text", "text": BIG_PREFIX, "cache_control": {"type": "ephemeral"}},
                {"type": "text", "text": "What is 2+2?"},
            ]},
        ],
        tools=tools,
        tool_choice={"type": "function", "function": {"name": "answer"}},
        max_tokens=20,
    )
    print(f"  call {i}: {_u(r)}")


print("\n=== WITH tools + cache_control ALSO on the last tool ===")
tools_with_cache = [{
    "type": "function",
    "function": {
        "name": "answer",
        "description": "Answer the question",
        "parameters": {"type": "object", "properties": {"verdict": {"type": "string"}}, "required": ["verdict"]},
    },
    "cache_control": {"type": "ephemeral"},
}]
for i in (1, 2):
    r = litellm.completion(
        model="anthropic/claude-haiku-4-5",
        messages=[
            {"role": "user", "content": [
                {"type": "text", "text": BIG_PREFIX, "cache_control": {"type": "ephemeral"}},
                {"type": "text", "text": "What is 2+2?"},
            ]},
        ],
        tools=tools_with_cache,
        tool_choice={"type": "function", "function": {"name": "answer"}},
        max_tokens=20,
    )
    print(f"  call {i}: {_u(r)}")
