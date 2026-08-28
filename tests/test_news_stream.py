"""Tests for NewsStreamer wildcard subscription + ticker-mention extraction."""

from __future__ import annotations

import asyncio
import queue
from types import SimpleNamespace

from agentic_investor.tools.news_stream import (
    NewsEvent,
    NewsStreamer,
    extract_tickers_from_text,
)


def _make_item(symbols, headline="", summary=""):
    return SimpleNamespace(
        symbols=symbols,
        headline=headline,
        summary=summary,
        created_at="2026-08-28T15:00:00Z",
        url="https://example.test/x",
        source="alpaca",
    )


def _drain(q):
    out = []
    while True:
        try:
            out.append(q.get_nowait())
        except queue.Empty:
            return out


def test_extract_cashtags_and_parenthesized():
    body = "Nvidia ($NVDA) and Alphabet (GOOGL) rally; also AAPL up."
    tickers = extract_tickers_from_text(body)
    assert "NVDA" in tickers
    assert "GOOGL" in tickers
    assert "AAPL" not in tickers


def test_extract_handles_empty():
    assert extract_tickers_from_text("") == set()


def test_extract_exchange_prefixed():
    body = "Apple (NASDAQ:AAPL) and Berkshire (NYSE:BRK.B) reported today."
    tickers = extract_tickers_from_text(body)
    assert "AAPL" in tickers
    assert "BRK" in tickers


def test_non_wildcard_filters_to_subscribed_tickers():
    q: queue.Queue[NewsEvent] = queue.Queue()
    s = NewsStreamer(["AAPL"], event_queue=q)
    asyncio.run(s._on_news(_make_item(["MSFT"])))
    assert _drain(q) == []


def test_non_wildcard_accepts_matching_ticker():
    q: queue.Queue[NewsEvent] = queue.Queue()
    s = NewsStreamer(["AAPL"], event_queue=q)
    asyncio.run(s._on_news(_make_item(["AAPL"], headline="Apple earnings")))
    events = _drain(q)
    assert len(events) == 1
    assert events[0].ticker == "AAPL"


def test_wildcard_accepts_any_ticker():
    q: queue.Queue[NewsEvent] = queue.Queue()
    s = NewsStreamer(["*"], event_queue=q)
    assert s.wildcard is True
    asyncio.run(s._on_news(_make_item(["MSFT", "GOOGL"])))
    tickers = {e.ticker for e in _drain(q)}
    assert tickers == {"MSFT", "GOOGL"}


def test_wildcard_extracts_extra_tickers_from_body():
    q: queue.Queue[NewsEvent] = queue.Queue()
    s = NewsStreamer(["*"], event_queue=q)
    item = _make_item(
        ["NVDA"],
        headline="Nvidia beats; suppliers $TSM and (AVGO) also mentioned",
    )
    asyncio.run(s._on_news(item))
    tickers = {e.ticker for e in _drain(q)}
    assert tickers == {"NVDA", "TSM", "AVGO"}


def test_extraction_dedupes_across_alpaca_tags_and_body():
    q: queue.Queue[NewsEvent] = queue.Queue()
    s = NewsStreamer(["*"], event_queue=q)
    item = _make_item(["NVDA"], headline="$NVDA up on strong earnings")
    asyncio.run(s._on_news(item))
    events = _drain(q)
    assert len(events) == 1
    assert events[0].ticker == "NVDA"
