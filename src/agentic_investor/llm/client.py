"""Provider-agnostic LLM access with per-run call + cost tracking.

Wraps LiteLLM (many providers behind one call) with instructor (schema-guided,
validated, auto-retried structured output), a tenacity retry-on-transient loop
for provider-side 503 / 429 / timeouts, and a lightweight per-command usage
tracker (calls, tokens, estimated cost) so every CLI command can print an LLM
usage summary.
"""

import threading
from dataclasses import dataclass, field

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


# Cost tracking: per-run counter of LLM calls, tokens, and estimated $ cost.
# Prices are (input, output) per 1M tokens in USD, approximate as of 2026-08.
# Verify against the provider's live pricing when it matters.
_PRICES: dict[str, tuple[float, float]] = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4.1-nano": (0.10, 0.40),
    "gemini/gemini-flash-latest": (0.30, 2.50),
    "gemini/gemini-flash-lite-latest": (0.075, 0.30),
    "gemini/gemini-2.5-flash": (0.30, 2.50),
    "anthropic/claude-haiku-4-5": (1.00, 5.00),
    "anthropic/claude-sonnet-4-5": (3.00, 15.00),
}


def _estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    prices = _PRICES.get(model)
    if prices is None:
        # Substring fallback (e.g. "openai/gpt-4o-mini" or provider prefixes).
        for name, p in _PRICES.items():
            if name in model or model in name:
                prices = p
                break
    if prices is None:
        return 0.0
    in_rate, out_rate = prices
    return (prompt_tokens * in_rate + completion_tokens * out_rate) / 1_000_000


@dataclass
class _CallStats:
    n_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    estimated_cost_usd: float = 0.0
    by_model: dict[str, dict] = field(default_factory=dict)


_stats = _CallStats()
_stats_lock = threading.Lock()


def reset_call_stats() -> None:
    """Zero out the LLM usage counters (call at the start of each CLI command)."""
    global _stats
    with _stats_lock:
        _stats = _CallStats()


def get_call_stats() -> _CallStats:
    """Return a snapshot of the current LLM usage counters."""
    with _stats_lock:
        return _CallStats(
            n_calls=_stats.n_calls,
            prompt_tokens=_stats.prompt_tokens,
            completion_tokens=_stats.completion_tokens,
            estimated_cost_usd=_stats.estimated_cost_usd,
            by_model={k: dict(v) for k, v in _stats.by_model.items()},
        )


def format_call_stats(stats: _CallStats | None = None) -> str:
    """Human-readable one-block summary for CLI display."""
    s = stats if stats is not None else get_call_stats()
    if s.n_calls == 0:
        return "  LLM usage: 0 calls (no LLM hit this run)"
    lines = [
        f"  LLM usage: {s.n_calls} calls, "
        f"{s.prompt_tokens:,} input + {s.completion_tokens:,} output tokens, "
        f"~${s.estimated_cost_usd:.4f} estimated"
    ]
    for model, m in s.by_model.items():
        lines.append(
            f"    - {model}: {m['calls']} calls, "
            f"{m['prompt']:,}/{m['completion']:,} tokens, ~${m['cost']:.4f}"
        )
    return "\n".join(lines)


def _track_usage(kwargs, completion_response, start_time, end_time) -> None:
    """LiteLLM success callback: pull token counts + estimate cost, accumulate."""
    try:
        usage = getattr(completion_response, "usage", None) or {}
        # usage may be a dict or a pydantic-like object.
        if hasattr(usage, "model_dump"):
            usage = usage.model_dump()
        elif hasattr(usage, "__dict__"):
            usage = {**getattr(usage, "__dict__", {})}
        prompt = int(usage.get("prompt_tokens", 0) or 0)
        completion = int(usage.get("completion_tokens", 0) or 0)
        model = kwargs.get("model", "unknown")
        cost = _estimate_cost(model, prompt, completion)
        with _stats_lock:
            _stats.n_calls += 1
            _stats.prompt_tokens += prompt
            _stats.completion_tokens += completion
            _stats.estimated_cost_usd += cost
            m = _stats.by_model.setdefault(
                model, {"calls": 0, "prompt": 0, "completion": 0, "cost": 0.0}
            )
            m["calls"] += 1
            m["prompt"] += prompt
            m["completion"] += completion
            m["cost"] += cost
    except Exception:  # noqa: BLE001 - never let telemetry break a real call
        pass


# Register the tracker with LiteLLM. Extend rather than replace so we don't
# clobber any other callbacks a caller might have set.
if _track_usage not in (litellm.success_callback or []):
    litellm.success_callback = [*(litellm.success_callback or []), _track_usage]


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
    Every successful call is logged into the module-level usage tracker via
    litellm's success_callback so CLI commands can print a per-run summary.
    """
    s = get_settings()
    return _client.chat.completions.create(
        model=model or s.llm_model,
        messages=messages,
        response_model=response_model,
        temperature=s.llm_temperature if temperature is None else temperature,
        max_retries=max_retries,
    )
