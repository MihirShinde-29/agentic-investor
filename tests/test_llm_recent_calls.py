"""Tests for the per-call ring buffer + `pop_recent_calls` accessor
(task #162 / E).

The paper-loop drains this buffer after each regen so the dashboard
can show live prompt-cache hit-rate per call rather than aggregated
over the whole session.
"""

from __future__ import annotations

from agentic_investor.llm import client as llm_client


def _fake_usage_row(prompt: int, completion: int, cached: int = 0,
                    creation: int = 0):
    """Build a shape LiteLLM's success callback would hand `_track_usage`.

    completion_response.usage is a dict-ish attribute; the tracker calls
    `.model_dump()` or `.__dict__` on it. A plain dict works too.
    """
    class _Resp:
        usage = {
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "cache_read_input_tokens": cached,
            "cache_creation_input_tokens": creation,
        }
    return _Resp


def test_pop_recent_calls_starts_empty():
    llm_client.reset_call_stats()
    assert llm_client.pop_recent_calls() == []


def test_track_usage_appends_one_row_per_call():
    llm_client.reset_call_stats()
    llm_client._track_usage(
        {"model": "gpt-4o-mini"}, _fake_usage_row(1000, 200), 0, 0,
    )
    llm_client._track_usage(
        {"model": "gpt-4o-mini"}, _fake_usage_row(1200, 250, cached=800), 0, 0,
    )
    got = llm_client.pop_recent_calls()
    assert len(got) == 2
    assert got[0]["model"] == "gpt-4o-mini"
    assert got[0]["prompt_tokens"] == 1000
    assert got[0]["completion_tokens"] == 200
    assert got[0]["cached_tokens"] == 0
    assert got[0]["cache_hit_ratio"] == 0.0
    assert got[1]["cached_tokens"] == 800
    # 800 / 1200 = 0.666...
    assert 0.66 < got[1]["cache_hit_ratio"] < 0.67


def test_pop_recent_calls_drains_buffer():
    llm_client.reset_call_stats()
    llm_client._track_usage(
        {"model": "gpt-4o-mini"}, _fake_usage_row(100, 20), 0, 0,
    )
    first = llm_client.pop_recent_calls()
    second = llm_client.pop_recent_calls()
    assert len(first) == 1
    assert second == []


def test_cache_hit_ratio_zero_when_no_prompt_tokens():
    """A callback with prompt_tokens=0 (edge case from a provider that
    strips usage on error paths) must not ZeroDivisionError."""
    llm_client.reset_call_stats()
    llm_client._track_usage(
        {"model": "gpt-4o-mini"}, _fake_usage_row(0, 0), 0, 0,
    )
    got = llm_client.pop_recent_calls()
    assert got[0]["cache_hit_ratio"] == 0.0


def test_ring_buffer_caps_at_max():
    """Beyond _RECENT_CALLS_CAP the buffer trims to half its cap so an
    arm that never drains doesn't leak."""
    llm_client.reset_call_stats()
    for _ in range(llm_client._RECENT_CALLS_CAP + 20):
        llm_client._track_usage(
            {"model": "gpt-4o-mini"}, _fake_usage_row(10, 5), 0, 0,
        )
    got = llm_client.pop_recent_calls()
    assert len(got) <= llm_client._RECENT_CALLS_CAP


def test_reset_call_stats_clears_recent_buffer():
    """`reset_call_stats` at command entry must also clear the drain
    buffer so a prior CLI invocation's tail doesn't leak into a new
    run's first regen event stream.
    """
    llm_client.reset_call_stats()
    llm_client._track_usage(
        {"model": "gpt-4o-mini"}, _fake_usage_row(10, 5), 0, 0,
    )
    llm_client.reset_call_stats()
    assert llm_client.pop_recent_calls() == []


def test_anthropic_cache_read_tokens_populate_ratio():
    """Anthropic's usage carries cache_read_input_tokens (not
    prompt_tokens_details.cached_tokens like OpenAI). Ratio math should
    use the same field so both providers show up in the dashboard.
    """
    llm_client.reset_call_stats()
    llm_client._track_usage(
        {"model": "anthropic/claude-haiku-4-5"},
        _fake_usage_row(2000, 300, cached=1500),
        0, 0,
    )
    got = llm_client.pop_recent_calls()
    assert got[0]["cached_tokens"] == 1500
    assert 0.74 < got[0]["cache_hit_ratio"] < 0.76
