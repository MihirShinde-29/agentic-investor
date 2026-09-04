"""Tests for headline normalization + cross-ticker fanout detection."""

from __future__ import annotations

import queue

from agentic_investor.tools.news_stream import (
    NewsStreamer,
    _normalize_headline,
)


def test_normalize_strips_wording_modifiers_that_dont_carry_meaning():
    # These are the same story in different wrappings - should hash the same.
    variants = [
        "VBNK Q2 2026 Earnings Call: Full Transcript",
        "VBNK Q2 2026 Earnings Call: Transcript",
        "UPDATE: VBNK Q2 2026 Earnings Call: Transcript",
        "Reported Earlier: VBNK Q2 2026 Earnings Call: Full Transcript",
        "  VBNK Q2 2026 Earnings Call: Complete Transcript  ",
    ]
    baseline = _normalize_headline(variants[0])
    for v in variants[1:]:
        assert _normalize_headline(v) == baseline, f"variant differed: {v!r}"


def test_normalize_preserves_meaning_bearing_tokens():
    # Direction and PT MUST change the hash - "beats" vs "misses" are
    # opposite meanings that happen to share sentence structure.
    beats = _normalize_headline("Apple beats Q3 estimates")
    misses = _normalize_headline("Apple misses Q3 estimates")
    assert beats != misses

    up = _normalize_headline("Morgan Stanley Raises PT to $500")
    down = _normalize_headline("Morgan Stanley Lowers PT to $500")
    assert up != down

    pt500 = _normalize_headline("Morgan Stanley Raises PT to $500")
    pt550 = _normalize_headline("Morgan Stanley Raises PT to $550")
    assert pt500 != pt550


def test_normalize_strips_trailing_source_attribution():
    a = _normalize_headline("Nvidia hits new high - Benzinga")
    b = _normalize_headline("Nvidia hits new high - Reuters")
    c = _normalize_headline("Nvidia hits new high")
    assert a == b == c


def test_normalize_folds_html_entities():
    a = _normalize_headline("Apple &amp; Microsoft partner on AI")
    b = _normalize_headline("Apple   Microsoft partner on AI")
    assert a == b


class _FakeItem:
    def __init__(self, symbols, headline, url=""):
        self.symbols = symbols
        self.headline = headline
        self.summary = ""
        self.created_at = None
        self.url = url
        self.source = "test"


async def _run_stream_on(streamer, items):
    for i in items:
        await streamer._on_news(i)


def test_dedup_drops_repeats_within_ttl_per_ticker():
    import asyncio

    q: queue.Queue = queue.Queue()
    s = NewsStreamer(["AAPL"], event_queue=q)
    items = [
        _FakeItem(["AAPL"], "Apple posts Q3 results"),
        _FakeItem(["AAPL"], "Apple posts Q3 results"),  # exact dup
        _FakeItem(["AAPL"], "UPDATE: Apple posts Q3 results"),  # normalized dup
    ]
    asyncio.run(_run_stream_on(s, items))
    assert q.qsize() == 1


def test_fanout_diagnostic_fires_when_story_reaches_many_tickers(caplog):
    import asyncio
    import logging

    q: queue.Queue = queue.Queue()
    s = NewsStreamer(["*"], event_queue=q)
    # Same headline tagged to 6 tickers = above the 5-ticker fanout threshold.
    tickers = ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META"]
    items = [_FakeItem([t], "Big Tech AI Investment Wave Accelerates") for t in tickers]
    with caplog.at_level(logging.INFO):
        asyncio.run(_run_stream_on(s, items))
    # Per-ticker events all pass (attribution preserved).
    assert q.qsize() == 6
    # One fanout diagnostic emitted once the threshold trips at the 5th
    # ticker. Should not re-fire on the 6th - we log once per story.
    fanout_logs = [r for r in caplog.records if "news fanout" in r.getMessage()]
    assert len(fanout_logs) == 1
    assert "5 tickers" in fanout_logs[0].getMessage()
