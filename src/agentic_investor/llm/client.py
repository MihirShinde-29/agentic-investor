"""Provider-agnostic LLM access.

Wraps LiteLLM (many providers behind one call) with instructor (schema-guided,
validated, auto-retried structured output). Callers ask for a Pydantic type and
get a validated instance back, whichever provider is configured.
"""

import instructor
import litellm
from dotenv import load_dotenv
from pydantic import BaseModel

from agentic_investor.config import get_settings

# Let LiteLLM pick up provider keys (OPENAI_API_KEY, GEMINI_API_KEY, ...) from .env.
load_dotenv()

_client = instructor.from_litellm(litellm.completion)


def structured_complete[T: BaseModel](
    response_model: type[T],
    messages: list[dict],
    *,
    model: str | None = None,
    temperature: float | None = None,
    max_retries: int = 2,
) -> T:
    """Call the LLM and return a validated instance of response_model.

    instructor re-prompts with the validation error if the model returns
    something that doesn't fit the schema, up to max_retries times.
    """
    s = get_settings()
    return _client.chat.completions.create(
        model=model or s.llm_model,
        messages=messages,
        response_model=response_model,
        temperature=s.llm_temperature if temperature is None else temperature,
        max_retries=max_retries,
    )
