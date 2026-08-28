"""Tests for the micro-batched decision engine."""

from datetime import UTC, datetime, timedelta

from agentic_investor.orchestrator.decision_engine import (
    DecisionState,
    _tag,
    build_batch,
    ingest,
    should_fire,
)
from agentic_investor.tools.news_stream import NewsEvent


def _news(ticker="AAPL", minutes_ago=0.0, headline="h"):
    ts = datetime.now(UTC) - timedelta(minutes=minutes_ago)
    return NewsEvent(
        ticker=ticker, headline=headline, summary="",
        published_at=ts.isoformat(),
        received_at=datetime.now(UTC).isoformat(),
    )


def test_tag_classifies_fresh_as_hot():
    tagged = _tag(_news(minutes_ago=1), now=datetime.now(UTC))
    assert tagged.tag == "HOT"


def test_tag_classifies_15min_old_as_cooked():
    tagged = _tag(_news(minutes_ago=16), now=datetime.now(UTC))
    assert tagged.tag == "COOKED"


def test_tag_classifies_over_hour_as_stale():
    tagged = _tag(_news(minutes_ago=61), now=datetime.now(UTC))
    assert tagged.tag == "STALE"


def test_should_not_fire_when_no_news():
    state = DecisionState()
    fire, reason = should_fire(state, datetime.now(UTC))
    assert fire is False
    assert reason == "no-unprocessed-news"


def test_should_not_fire_within_global_cap():
    now = datetime.now(UTC)
    state = DecisionState(
        unprocessed=[_news()],
        last_fire_at=now - timedelta(seconds=10),
        last_batch_window_started=now - timedelta(seconds=70),
    )
    fire, reason = should_fire(state, now)
    assert fire is False
    assert reason.startswith("global-cap")


def test_should_fire_when_batch_window_closed():
    now = datetime.now(UTC)
    state = DecisionState(unprocessed=[_news()])
    ingest(state, [_news()], now - timedelta(seconds=61))
    fire, reason = should_fire(state, now)
    assert fire is True
    assert reason == "batch-window-closed"


def test_should_fire_when_cooked_news_ready():
    now = datetime.now(UTC)
    # Old event forces a fire even mid-batch-window.
    state = DecisionState(unprocessed=[_news(minutes_ago=16)])
    state.last_batch_window_started = now  # window just opened
    fire, reason = should_fire(state, now)
    assert fire is True
    assert reason == "cooked-news-ready"


def test_build_batch_tags_and_resets_state():
    now = datetime.now(UTC)
    state = DecisionState(unprocessed=[
        _news("AAPL", 1),   # HOT
        _news("NVDA", 16),  # COOKED
        _news("MSFT", 70),  # STALE
    ])
    state.last_batch_window_started = now - timedelta(seconds=60)
    batch = build_batch(state, now)
    assert len(batch.hot) == 1 and batch.hot[0].event.ticker == "AAPL"
    assert len(batch.cooked) == 1 and batch.cooked[0].event.ticker == "NVDA"
    assert len(batch.stale) == 1 and batch.stale[0].event.ticker == "MSFT"
    assert batch.all_tickers() == {"AAPL", "NVDA", "MSFT"}
    # State should be reset for the next batch.
    assert state.unprocessed == []
    assert state.last_batch_window_started is None
    assert state.last_fire_at == now


def test_ingest_opens_batch_window_on_first_event():
    state = DecisionState()
    now = datetime.now(UTC)
    ingest(state, [_news()], now)
    assert state.last_batch_window_started == now
    # Another arrival within window does NOT reset the window.
    later = now + timedelta(seconds=15)
    ingest(state, [_news()], later)
    assert state.last_batch_window_started == now
    assert len(state.unprocessed) == 2
