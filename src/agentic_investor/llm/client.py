"""Provider-agnostic LLM access with per-run call + cost tracking.

Wraps LiteLLM (many providers behind one call) with instructor (schema-guided,
validated, auto-retried structured output), a tenacity retry-on-transient loop
for provider-side 503 / 429 / timeouts, and a lightweight per-command usage
tracker (calls, tokens, estimated cost) so every CLI command can print an LLM
usage summary.
"""

import logging
import os
import threading
from dataclasses import dataclass, field

import instructor
import litellm
from dotenv import load_dotenv
from instructor.core import InstructorRetryException
from pydantic import BaseModel, ValidationError

# LiteLLM logs every completion call at INFO twice ("LiteLLM completion()..."
# once from litellm.utils, once from Wrapper: Completed Call). Fills the Fly
# log buffer and drowns real events. Quiet it unless the user opted in via
# LITELLM_LOG.
if os.getenv("LITELLM_LOG") is None:
    for _name in ("LiteLLM", "LiteLLM Proxy", "litellm"):
        logging.getLogger(_name).setLevel(logging.WARNING)
from tenacity import (  # noqa: E402
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


def _estimate_cost(
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    *,
    cached_tokens: int = 0,
) -> float:
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
    # OpenAI charges cached input at 50% of the base input rate.
    uncached = max(0, prompt_tokens - cached_tokens)
    input_cost = (uncached * in_rate + cached_tokens * in_rate * 0.5) / 1_000_000
    output_cost = completion_tokens * out_rate / 1_000_000
    return input_cost + output_cost


@dataclass
class _CallStats:
    n_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0
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
            cached_tokens=_stats.cached_tokens,
            estimated_cost_usd=_stats.estimated_cost_usd,
            by_model={k: dict(v) for k, v in _stats.by_model.items()},
        )


def format_call_stats(stats: _CallStats | None = None) -> str:
    """Human-readable one-block summary for CLI display."""
    s = stats if stats is not None else get_call_stats()
    if s.n_calls == 0:
        return "  LLM usage: 0 calls (no LLM hit this run)"
    cache_hint = ""
    if s.prompt_tokens > 0 and s.cached_tokens > 0:
        pct = s.cached_tokens / s.prompt_tokens * 100
        cache_hint = f" ({s.cached_tokens:,} cached, {pct:.0f}% hit)"
    lines = [
        f"  LLM usage: {s.n_calls} calls, "
        f"{s.prompt_tokens:,} input{cache_hint} + {s.completion_tokens:,} output "
        f"tokens, ~${s.estimated_cost_usd:.4f} estimated"
    ]
    for model, m in s.by_model.items():
        cached = m.get("cached", 0)
        c_hint = f" ({cached:,} cached)" if cached else ""
        lines.append(
            f"    - {model}: {m['calls']} calls, "
            f"{m['prompt']:,}{c_hint}/{m['completion']:,} tokens, ~${m['cost']:.4f}"
        )
    return "\n".join(lines)


def _extract_cached_tokens(usage: dict) -> int:
    """Pull cached-input-tokens from OpenAI or Anthropic usage payload."""
    # OpenAI: usage.prompt_tokens_details.cached_tokens
    details = usage.get("prompt_tokens_details") or {}
    if hasattr(details, "model_dump"):
        details = details.model_dump()
    elif hasattr(details, "__dict__"):
        details = {**getattr(details, "__dict__", {})}
    cached = int(details.get("cached_tokens", 0) or 0)
    # Anthropic: usage.cache_read_input_tokens (LiteLLM passes through)
    cached += int(usage.get("cache_read_input_tokens", 0) or 0)
    return cached


def _track_usage(kwargs, completion_response, start_time, end_time) -> None:
    """LiteLLM success callback: pull token counts + estimate cost, accumulate."""
    try:
        usage = getattr(completion_response, "usage", None) or {}
        if hasattr(usage, "model_dump"):
            usage = usage.model_dump()
        elif hasattr(usage, "__dict__"):
            usage = {**getattr(usage, "__dict__", {})}
        prompt = int(usage.get("prompt_tokens", 0) or 0)
        completion = int(usage.get("completion_tokens", 0) or 0)
        cached = _extract_cached_tokens(usage)
        model = kwargs.get("model", "unknown")
        cost = _estimate_cost(model, prompt, completion, cached_tokens=cached)
        with _stats_lock:
            _stats.n_calls += 1
            _stats.prompt_tokens += prompt
            _stats.completion_tokens += completion
            _stats.cached_tokens += cached
            _stats.estimated_cost_usd += cost
            m = _stats.by_model.setdefault(
                model,
                {"calls": 0, "prompt": 0, "completion": 0, "cached": 0, "cost": 0.0},
            )
            m["calls"] += 1
            m["prompt"] += prompt
            m["completion"] += completion
            m["cached"] += cached
            m["cost"] += cost
    except Exception:  # noqa: BLE001 - never let telemetry break a real call
        pass


# Register the tracker with LiteLLM. Extend rather than replace so we don't
# clobber any other callbacks a caller might have set.
if _track_usage not in (litellm.success_callback or []):
    litellm.success_callback = [*(litellm.success_callback or []), _track_usage]


def _maybe_enable_langfuse() -> None:
    """Wire LiteLLM's built-in Langfuse callback if credentials are set.

    Langfuse traces every LLM call (prompts, completions, tokens, cost) into
    a UI you can browse to debug allocator decisions. Opt-in via env vars so
    a fresh clone with no keys still works.
    """
    import os

    if not os.getenv("LANGFUSE_PUBLIC_KEY") or not os.getenv("LANGFUSE_SECRET_KEY"):
        return
    for hook in ("success_callback", "failure_callback"):
        current = list(getattr(litellm, hook, None) or [])
        if "langfuse" not in current:
            setattr(litellm, hook, [*current, "langfuse"])


_maybe_enable_langfuse()


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
    timeout: float = 30.0,
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

    `timeout` bounds each individual completion call so a stuck provider
    response can't freeze the whole loop while tenacity waits for it.
    """
    s = get_settings()
    try:
        return _client.chat.completions.create(
            model=model or s.llm_model,
            messages=messages,
            response_model=response_model,
            temperature=s.llm_temperature if temperature is None else temperature,
            max_retries=max_retries,
            timeout=timeout,
        )
    except InstructorRetryException as exc:
        _log_instructor_failure(exc, response_model, messages)
        raise


def _log_instructor_failure(
    exc: InstructorRetryException,
    response_model: type[BaseModel],
    messages: list[dict],
) -> None:
    """Dump every failed attempt with field-level validation errors and the
    LLM's raw output so we can see WHICH field the model keeps mangling.
    Instructor's default message truncates at "1 validation error for X",
    which is useless for diagnosis.
    """
    log = logging.getLogger(__name__)
    total_msg_chars = sum(len(str(m.get("content", ""))) for m in messages)
    log.error(
        "instructor_failed: model=%s attempts=%d messages=%d prompt_chars=%d",
        response_model.__name__,
        len(exc.failed_attempts or []),
        len(messages),
        total_msg_chars,
    )
    for att in exc.failed_attempts or []:
        cause = att.exception
        if isinstance(cause, ValidationError):
            for err in cause.errors():
                loc = ".".join(str(p) for p in err.get("loc", ()))
                log.error(
                    "  attempt=%d field=%s type=%s msg=%s input=%r",
                    att.attempt_number,
                    loc,
                    err.get("type", "?"),
                    err.get("msg", "?"),
                    _truncate(err.get("input"), 200),
                )
        else:
            log.error(
                "  attempt=%d non_validation_error=%s: %s",
                att.attempt_number,
                type(cause).__name__,
                _truncate(str(cause), 500),
            )
        try:
            content = att.completion.choices[0].message.content
        except (AttributeError, IndexError, TypeError):
            content = None
        if content:
            log.error(
                "  attempt=%d raw_completion=%s",
                att.attempt_number,
                _truncate(content, 800),
            )


def _truncate(v, n: int) -> str:
    s = repr(v) if not isinstance(v, str) else v
    return s if len(s) <= n else s[:n] + f"...<+{len(s) - n} chars>"
