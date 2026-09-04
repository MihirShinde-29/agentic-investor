"""Tests for the news-effect journal: staging, attribution, dropping, TTL."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from agentic_investor.orchestrator.loop import (
    LoopState,
    _attribute_pending_news,
    _drop_pending_news,
    _prune_news_effect_log,
    _render_news_effect_log,
    _stage_pending_news,
)
from agentic_investor.orchestrator.state import (
    Allocation,
    OrchestratorRequest,
    Position,
    Recommendation,
)


@dataclass
class _E:
    ticker: str
    headline: str
    summary: str = ""
    source: str = ""


def _rec(positions_pct: dict[str, float]) -> Recommendation:
    positions = [
        Position(ticker=t, weight_pct=w, dollars=w * 100, rationale="x")
        for t, w in positions_pct.items()
    ]
    cash = max(0.0, 100.0 - sum(positions_pct.values()))
    return Recommendation(
        request=OrchestratorRequest(tickers=list(positions_pct), amount=10_000),
        allocation=Allocation(
            positions=positions,
            cash_pct=cash, cash_dollars=cash * 100, portfolio_rationale="x",
        ),
    )


def test_stage_adds_events_to_pending():
    state = LoopState()
    now = datetime.now(UTC)
    events = [
        _E("AAPL", "Morgan Stanley Overweight, PT $250"),
        _E("MSFT", "UBS Buy, PT $500"),
    ]
    _stage_pending_news(state, events, now)
    assert len(state.pending_news_events) == 2
    assert {e["ticker"] for e in state.pending_news_events} == {"AAPL", "MSFT"}


def test_stage_ignores_events_with_empty_ticker():
    state = LoopState()
    now = datetime.now(UTC)
    _stage_pending_news(state, [_E("", "orphan")], now)
    assert state.pending_news_events == []


def test_attribute_promotes_only_movers():
    # LLM went from AAPL 20 -> 30 (+10pp) and MSFT 10 -> 11 (+1pp on threshold).
    # NVDA had news but didn't move -> dropped.
    state = LoopState()
    now = datetime.now(UTC)
    _stage_pending_news(state, [
        _E("AAPL", "AAPL upgraded"),
        _E("MSFT", "MSFT upgraded"),
        _E("NVDA", "NVDA upgraded"),
    ], now)
    prev = _rec({"AAPL": 20, "MSFT": 10})
    new = _rec({"AAPL": 30, "MSFT": 11, "NVDA": 0.5})
    _attribute_pending_news(state, new, prev, min_delta_pp=1.0)
    assert state.pending_news_events == []  # staging cleared
    assert "AAPL" in state.news_effect_log
    assert state.news_effect_log["AAPL"][0]["delta_pp"] == 10.0
    # MSFT at exactly 1pp passes >= threshold
    assert "MSFT" in state.news_effect_log
    # NVDA's 0.5pp move is below threshold -> dropped
    assert "NVDA" not in state.news_effect_log


def test_attribute_first_rec_treats_prev_as_empty():
    state = LoopState()
    now = datetime.now(UTC)
    _stage_pending_news(state, [_E("AAPL", "AAPL opens")], now)
    new = _rec({"AAPL": 20})
    _attribute_pending_news(state, new, prev_rec=None, min_delta_pp=1.0)
    assert state.news_effect_log["AAPL"][0]["delta_pp"] == 20.0


def test_drop_clears_pending_without_touching_log():
    state = LoopState()
    now = datetime.now(UTC)
    state.news_effect_log["AAPL"] = [{
        "ts": now.isoformat(), "ticker": "AAPL",
        "headline": "prior effective news", "delta_pp": 5.0,
        "source": "x", "summary": "",
    }]
    _stage_pending_news(state, [_E("MSFT", "noise")], now)
    _drop_pending_news(state)
    assert state.pending_news_events == []
    # Prior journal entries untouched.
    assert len(state.news_effect_log["AAPL"]) == 1


def test_prune_removes_entries_older_than_ttl():
    state = LoopState()
    now = datetime.now(UTC)
    old = (now - timedelta(hours=25)).isoformat()
    fresh = (now - timedelta(hours=1)).isoformat()
    state.news_effect_log["AAPL"] = [
        {"ts": old, "ticker": "AAPL", "headline": "stale", "delta_pp": 3.0,
         "source": "", "summary": ""},
        {"ts": fresh, "ticker": "AAPL", "headline": "recent", "delta_pp": 2.0,
         "source": "", "summary": ""},
    ]
    state.news_effect_log["MSFT"] = [
        {"ts": old, "ticker": "MSFT", "headline": "all stale", "delta_pp": 4.0,
         "source": "", "summary": ""},
    ]
    _prune_news_effect_log(state, now, ttl_seconds=86400)
    # AAPL keeps only the fresh entry
    assert len(state.news_effect_log["AAPL"]) == 1
    assert state.news_effect_log["AAPL"][0]["headline"] == "recent"
    # MSFT emptied -> dropped from log entirely
    assert "MSFT" not in state.news_effect_log


def test_render_produces_compact_per_ticker_journal():
    state = LoopState()
    ts = "2026-09-04T10:14:23.000000+00:00"
    state.news_effect_log["SNOW"] = [
        {"ts": ts, "ticker": "SNOW", "headline": "RBC Outperform PT $440",
         "delta_pp": 2.5, "source": "Benzinga", "summary": ""},
    ]
    rendered = _render_news_effect_log(state)
    assert "SNOW" in rendered
    assert "10:14" in rendered
    assert "+2.5pp" in rendered


def test_render_returns_empty_when_log_empty():
    assert _render_news_effect_log(LoopState()) == ""


def test_state_serialization_roundtrip_preserves_journal():
    state = LoopState()
    state.pending_news_events = [{
        "ts": "2026-09-04T10:00:00+00:00", "ticker": "AAPL",
        "headline": "x", "summary": "y", "source": "z",
    }]
    state.news_effect_log["AAPL"] = [{
        "ts": "2026-09-04T09:00:00+00:00", "ticker": "AAPL",
        "headline": "prior", "summary": "", "source": "MS", "delta_pp": 3.0,
    }]
    restored = LoopState.from_dict(state.to_dict())
    assert restored.pending_news_events == state.pending_news_events
    assert restored.news_effect_log == state.news_effect_log
