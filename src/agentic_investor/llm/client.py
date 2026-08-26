"""Provider-agnostic LLM access.

Wraps LiteLLM (many providers behind one call) with instructor (schema-guided,
validated, auto-retried structured output) and a tenacity retry-on-transient
loop for provider-side 503 / 429 / timeouts. Instructor handles validation
retries; tenacity handles infrastructure retries.
"""

import instructor
import litellm
from dotenv import load_dotenv
from pydantic import BaseModel
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from agentic_investor.config import get_settings

# Let LiteLLM pick up provider keys (OPENAI_API_KEY, GEMINI_API_KEY, ...) from .env.
load_dotenv()

_client = instructor.from_litellm(litellm.completion)

# Provider-side transient errors worth backing off on. Bad-schema or auth
# errors fall through immediately.
_TRANSIENT = (
    litellm.exceptions.ServiceUnavailableError,
    litellm.exceptions.RateLimitError,
    litellm.exceptions.APIConnectionError,
    litellm.exceptions.Timeout,
    litellm.exceptions.InternalServerError,
)


def _is_transient(exc: BaseException) -> bool:
    # instructor wraps the underlying provider error in InstructorRetryException,
    # so isinstance on the top-level exception misses it. Walk the cause chain.
    current: BaseException | None = exc
    while current is not None:
        if isinstance(current, _TRANSIENT):
            return True
        current = current.__cause__ or current.__context__
    return False


@retry(
    reraise=True,
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=2, min=3, max=30),
    retry=retry_if_exception(_is_transient),
)
def structured_complete[T: BaseModel](
    response_model: type[T],
    messages: list[dict],
    *,
    model: str | None = None,
    temperature: float | None = None,
    max_retries: int = 1,
) -> T:
    """Call the LLM and return a validated instance of response_model.

    Two retry loops layered here:
    - instructor's max_retries re-prompts the LLM when the reply fails schema
      validation (kept small since the outer loop handles infrastructure).
    - the outer tenacity decorator retries on provider transient errors
      (503, 429, connection blips) with exponential backoff, walking the
      exception chain because instructor wraps provider errors.
    """
    s = get_settings()
    return _client.chat.completions.create(
        model=model or s.llm_model,
        messages=messages,
        response_model=response_model,
        temperature=s.llm_temperature if temperature is None else temperature,
        max_retries=max_retries,
    )
