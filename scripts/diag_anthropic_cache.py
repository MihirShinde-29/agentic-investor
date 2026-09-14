"""Diagnose why Anthropic prompt caching is not firing on arm C.

Runs three variants of the same prompt against claude-haiku-4-5:
  1. Raw litellm.completion with cache_control on system + user slow prefix
  2. Same but via instructor.from_litellm + a Pydantic response_model
  3. Same but with an explicit anthropic-beta header

Prints cache_read_input_tokens and cache_creation_input_tokens after each.

Run twice back-to-back so we can see call-2 read from cache.
"""

import os
from dotenv import load_dotenv
import litellm
import instructor
from pydantic import BaseModel

load_dotenv()


SLOW_PREFIX = (
    "You are an investment analyst. Below are stable rules and background "
    "that will not change between calls in this session.\n\n"
    + ("Rule: consider risk. " * 300)  # bulk it up past Anthropic's 1024-token min
)

FAST_TAIL = "Question: is AAPL a buy today?"

SYSTEM = "You are a concise financial analyst. Answer in one sentence."


def _extract(u) -> dict:
    if hasattr(u, "model_dump"):
        u = u.model_dump()
    return {
        "prompt_tokens": u.get("prompt_tokens"),
        "cache_read_input_tokens": u.get("cache_read_input_tokens"),
        "cache_creation_input_tokens": u.get("cache_creation_input_tokens"),
    }


def variant_1_raw_litellm() -> None:
    print("\n=== variant 1: raw litellm.completion (no instructor) ===")
    messages = [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": SYSTEM, "cache_control": {"type": "ephemeral"}}
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": SLOW_PREFIX, "cache_control": {"type": "ephemeral"}},
                {"type": "text", "text": FAST_TAIL},
            ],
        },
    ]
    for i in (1, 2):
        r = litellm.completion(
            model="anthropic/claude-haiku-4-5",
            messages=messages,
            max_tokens=50,
        )
        print(f"  call {i}: {_extract(r.usage)}")


class Answer(BaseModel):
    verdict: str


def variant_2_instructor_structured() -> None:
    print("\n=== variant 2: instructor.from_litellm + response_model ===")
    client = instructor.from_litellm(litellm.completion)
    messages = [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": SYSTEM, "cache_control": {"type": "ephemeral"}}
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": SLOW_PREFIX, "cache_control": {"type": "ephemeral"}},
                {"type": "text", "text": FAST_TAIL},
            ],
        },
    ]
    for i in (1, 2):
        obj, raw = client.chat.completions.create_with_completion(
            model="anthropic/claude-haiku-4-5",
            messages=messages,
            response_model=Answer,
            max_retries=1,
        )
        print(f"  call {i}: {_extract(raw.usage)} verdict={obj.verdict!r}")


def main() -> None:
    if not os.getenv("ANTHROPIC_API_KEY"):
        raise SystemExit("ANTHROPIC_API_KEY missing")
    print(f"slow_prefix chars={len(SLOW_PREFIX)} est_tokens={len(SLOW_PREFIX)//4}")
    variant_1_raw_litellm()
    variant_2_instructor_structured()


if __name__ == "__main__":
    main()
