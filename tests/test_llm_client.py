"""Tests for the LLM client's retry-on-transient chain walker.

We only test _is_transient here (the tenacity-decorated call is exercised
indirectly by the orchestrator tests, which mock structured_complete).
"""

import pytest
from litellm.exceptions import (
    APIConnectionError,
    RateLimitError,
    ServiceUnavailableError,
)

from agentic_investor.llm.client import _is_transient


def _bare(cls):
    # Bypass __init__ so we don't have to satisfy each litellm exception's
    # varying constructor signature.
    return cls.__new__(cls)


def test_direct_transient_is_detected():
    assert _is_transient(_bare(ServiceUnavailableError)) is True
    assert _is_transient(_bare(RateLimitError)) is True
    assert _is_transient(_bare(APIConnectionError)) is True


def test_wrapped_transient_is_detected_via_cause_chain():
    inner = _bare(ServiceUnavailableError)
    outer = RuntimeError("wrapped by instructor")
    outer.__cause__ = inner
    assert _is_transient(outer) is True


def test_wrapped_transient_is_detected_via_context_chain():
    inner = _bare(RateLimitError)
    outer = RuntimeError("wrapped without `from`")
    outer.__context__ = inner
    assert _is_transient(outer) is True


def test_non_transient_returns_false():
    assert _is_transient(ValueError("not a network problem")) is False
    assert _is_transient(TypeError("bad arg")) is False


# Cost tracking


def test_estimate_cost_for_known_model():
    from agentic_investor.llm.client import _estimate_cost

    # gpt-4o-mini is $0.15 / $0.60 per 1M tokens
    cost = _estimate_cost("gpt-4o-mini", 1_000_000, 1_000_000)
    assert cost == pytest.approx(0.75)


def test_estimate_cost_unknown_model_is_zero():
    from agentic_investor.llm.client import _estimate_cost

    assert _estimate_cost("some-unknown-model", 1000, 500) == 0.0


def test_estimate_cost_substring_match():
    from agentic_investor.llm.client import _estimate_cost

    # "openai/gpt-4o-mini" should match the "gpt-4o-mini" price entry.
    cost = _estimate_cost("openai/gpt-4o-mini", 100_000, 50_000)
    expected = (100_000 * 0.15 + 50_000 * 0.60) / 1_000_000
    assert cost == pytest.approx(expected)


def test_reset_and_track_via_callback():
    from agentic_investor.llm.client import (
        _track_usage,
        format_call_stats,
        get_call_stats,
        reset_call_stats,
    )

    reset_call_stats()
    assert get_call_stats().n_calls == 0

    # Simulate two litellm success callback invocations.
    class _FakeResp:
        def __init__(self, prompt, completion):
            self.usage = {"prompt_tokens": prompt, "completion_tokens": completion}

    _track_usage({"model": "gpt-4o-mini"}, _FakeResp(1000, 500), 0, 0)
    _track_usage({"model": "gpt-4o-mini"}, _FakeResp(2000, 800), 0, 0)

    stats = get_call_stats()
    assert stats.n_calls == 2
    assert stats.prompt_tokens == 3000
    assert stats.completion_tokens == 1300
    # (3000 * 0.15 + 1300 * 0.60) / 1M = (450 + 780) / 1M = 0.001230
    assert stats.estimated_cost_usd == pytest.approx(0.00123)
    assert "gpt-4o-mini" in stats.by_model
    # format_call_stats renders without error
    assert "2 calls" in format_call_stats(stats)


def test_format_zero_calls():
    from agentic_investor.llm.client import (
        _CallStats,
        format_call_stats,
    )

    assert "0 calls" in format_call_stats(_CallStats())
