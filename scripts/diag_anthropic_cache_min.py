"""Test Anthropic's minimum cacheable block size for claude-haiku-4-5.

If Haiku requires >= 2048 tokens per cacheable block, our slow_prefix
sometimes falls below that. Test escalating prefix sizes to find where
the cache actually kicks in.
"""

import os
from dotenv import load_dotenv
import litellm

load_dotenv()


def _try(prefix_tokens_target: int) -> None:
    prefix = "STABLE_PREFIX_TOKEN. " * prefix_tokens_target
    r = litellm.completion(
        model="anthropic/claude-haiku-4-5",
        messages=[
            {"role": "user", "content": [
                {"type": "text", "text": prefix, "cache_control": {"type": "ephemeral"}},
                {"type": "text", "text": "What is 2+2?"},
            ]},
        ],
        max_tokens=10,
    )
    u = r.usage.model_dump() if hasattr(r.usage, "model_dump") else dict(r.usage)
    print(
        f"target={prefix_tokens_target} prompt_tokens={u.get('prompt_tokens')} "
        f"cache_creation={u.get('cache_creation_input_tokens')} "
        f"cache_read={u.get('cache_read_input_tokens')}"
    )


for target in (500, 1000, 1500, 2000, 2500, 3000, 5000):
    _try(target)
