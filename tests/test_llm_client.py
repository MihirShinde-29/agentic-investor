"""Tests for the LLM client's retry-on-transient chain walker.

We only test _is_transient here (the tenacity-decorated call is exercised
indirectly by the orchestrator tests, which mock structured_complete).
"""

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
