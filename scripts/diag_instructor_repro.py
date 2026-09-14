"""Mimic production Instructor path exactly, with a >2000 token prefix.

If cache_create=0 here, the bug is in Instructor's handling.
"""

import os
from dotenv import load_dotenv
import litellm
import instructor
from pydantic import BaseModel

load_dotenv()

BIG_PREFIX = "STABLE_PREFIX. " * 1500  # ~3000 tokens

client = instructor.from_litellm(litellm.completion)


class Answer(BaseModel):
    verdict: str


for i in (1, 2):
    obj, raw = client.chat.completions.create_with_completion(
        model="anthropic/claude-haiku-4-5",
        messages=[
            {"role": "system", "content": [
                {"type": "text", "text": "You are terse.", "cache_control": {"type": "ephemeral"}},
            ]},
            {"role": "user", "content": [
                {"type": "text", "text": BIG_PREFIX, "cache_control": {"type": "ephemeral"}},
                {"type": "text", "text": "What is 2+2?"},
            ]},
        ],
        response_model=Answer,
        max_retries=1,
        max_tokens=100,
    )
    u = raw.usage.model_dump() if hasattr(raw.usage, "model_dump") else dict(raw.usage)
    print(f"call {i}: prompt={u.get('prompt_tokens')} "
          f"cache_create={u.get('cache_creation_input_tokens')} "
          f"cache_read={u.get('cache_read_input_tokens')} verdict={obj.verdict[:40]!r}")
