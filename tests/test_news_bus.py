"""Shared-news-bus fanout: writer -> SQLite -> N reader arms.

The point is proving that the bus can substitute for the Alpaca websocket
in NewsStreamer._build_stream, without changing any of NewsStreamer's own
processing logic. So the tests drive the reader directly and check that
the callback receives BusItems that look like Alpaca payloads.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
import threading
import time
from datetime import UTC, datetime
from pathlib import Path


def _write_bus_row(
    db_path: Path,
    *,
    symbols: list[str],
    headline: str,
    summary: str = "",
) -> None:
    with sqlite3.connect(str(db_path)) as conn:
        conn.execute(
            "INSERT INTO bus_events "
            "(ts_received, ts_published, symbols_json, headline, "
            " summary, url, source) VALUES (?,?,?,?,?,?,?)",
            (
                datetime.now(UTC).isoformat(),
                datetime.now(UTC).isoformat(),
                json.dumps(symbols),
                headline,
                summary,
                "",
                "",
            ),
        )


def test_init_bus_table_creates_schema(tmp_path):
    from agentic_investor.experiments.news_bus import init_bus_table

    db = tmp_path / "bus.db"
    init_bus_table(db)
    with sqlite3.connect(str(db)) as conn:
        cols = [r[1] for r in conn.execute("PRAGMA table_info(bus_events)")]
    assert "headline" in cols
    assert "symbols_json" in cols
    assert "id" in cols


def test_shared_bus_reader_delivers_rows_written_after_start(tmp_path):
    """Prove the poll loop picks up rows appended after subscribe_news."""
    from agentic_investor.experiments.news_bus import (
        SharedBusStream,
        init_bus_table,
    )

    db = tmp_path / "bus.db"
    init_bus_table(db)
    stream = SharedBusStream(f"sqlite:///{db}", poll_interval=0.05)

    received: list = []

    async def cb(item):
        received.append(item)

    stream.subscribe_news(cb, "*")

    def _run():
        stream.run()

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()

    # Give the reader a tick to enter its poll loop, then append.
    time.sleep(0.2)
    _write_bus_row(db, symbols=["AAPL"], headline="Apple beats Q3")
    _write_bus_row(db, symbols=["MSFT", "GOOGL"], headline="Cloud outage")

    # Poll for delivery (bounded wait).
    deadline = time.monotonic() + 3.0
    while len(received) < 2 and time.monotonic() < deadline:
        time.sleep(0.05)
    stream.stop()
    thread.join(timeout=2.0)

    assert len(received) == 2
    assert received[0].headline == "Apple beats Q3"
    assert received[0].symbols == ["AAPL"]
    assert received[1].symbols == ["MSFT", "GOOGL"]


def test_two_readers_each_get_every_row(tmp_path):
    """Two arms polling the same bus must each see every headline once."""
    from agentic_investor.experiments.news_bus import (
        SharedBusStream,
        init_bus_table,
    )

    db = tmp_path / "bus.db"
    init_bus_table(db)

    received_a: list = []
    received_b: list = []

    async def cb_a(item):
        received_a.append(item.headline)

    async def cb_b(item):
        received_b.append(item.headline)

    stream_a = SharedBusStream(f"sqlite:///{db}", poll_interval=0.05)
    stream_b = SharedBusStream(f"sqlite:///{db}", poll_interval=0.05)
    stream_a.subscribe_news(cb_a, "*")
    stream_b.subscribe_news(cb_b, "*")

    ta = threading.Thread(target=stream_a.run, daemon=True)
    tb = threading.Thread(target=stream_b.run, daemon=True)
    ta.start()
    tb.start()

    time.sleep(0.2)
    for i in range(3):
        _write_bus_row(db, symbols=["AAPL"], headline=f"headline {i}")

    deadline = time.monotonic() + 3.0
    while (len(received_a) < 3 or len(received_b) < 3) and \
            time.monotonic() < deadline:
        time.sleep(0.05)

    stream_a.stop()
    stream_b.stop()
    ta.join(timeout=2.0)
    tb.join(timeout=2.0)

    assert received_a == ["headline 0", "headline 1", "headline 2"]
    assert received_b == ["headline 0", "headline 1", "headline 2"]


def test_reader_raises_if_bus_never_appears(tmp_path):
    """If the writer never came up, reader should fail loud after startup wait."""
    from agentic_investor.experiments import news_bus
    from agentic_investor.experiments.news_bus import SharedBusStream

    # Shrink the startup wait so the test doesn't sit for 30 seconds.
    monkey_wait = 0.5
    original = news_bus._READER_STARTUP_WAIT_SEC
    news_bus._READER_STARTUP_WAIT_SEC = monkey_wait
    try:
        stream = SharedBusStream(
            f"sqlite:///{tmp_path / 'never-created.db'}",
        )

        async def cb(_item):
            pass

        stream.subscribe_news(cb, "*")

        import pytest
        with pytest.raises(FileNotFoundError):
            stream.run()
    finally:
        news_bus._READER_STARTUP_WAIT_SEC = original


def test_bus_path_from_url_roundtrip():
    from agentic_investor.experiments.news_bus import bus_path_from_url

    assert bus_path_from_url("sqlite:///abs/path.db") == Path("abs/path.db")


def test_bus_path_rejects_non_sqlite():
    import pytest

    from agentic_investor.experiments.news_bus import bus_path_from_url

    with pytest.raises(ValueError, match="only sqlite"):
        bus_path_from_url("postgres://...")


def test_news_stream_picks_shared_bus_when_env_set(tmp_path, monkeypatch):
    """NewsStreamer._build_stream must swap to SharedBusStream when env is set."""
    from agentic_investor.experiments.news_bus import (
        SharedBusStream,
        init_bus_table,
    )
    from agentic_investor.tools.news_stream import NewsStreamer

    db = tmp_path / "bus.db"
    init_bus_table(db)
    monkeypatch.setenv("AGENTIC_NEWS_BUS", f"sqlite:///{db}")

    import queue as _q
    ns = NewsStreamer(tickers=["*"], event_queue=_q.Queue())
    stream = ns._build_stream()
    assert isinstance(stream, SharedBusStream)


def test_news_stream_uses_alpaca_when_env_absent(monkeypatch):
    """Without AGENTIC_NEWS_BUS, _build_stream must return a real Alpaca client."""
    from agentic_investor.tools.news_stream import NewsStreamer

    monkeypatch.delenv("AGENTIC_NEWS_BUS", raising=False)
    # Point the real Alpaca stream at fake creds so we don't dial out.
    monkeypatch.setenv("ALPACA_API_KEY", "fake")
    monkeypatch.setenv("ALPACA_API_SECRET", "fake")

    import queue as _q
    ns = NewsStreamer(tickers=["*"], event_queue=_q.Queue())
    stream = ns._build_stream()
    # We don't call .run() on it - just verify it's NOT a SharedBusStream.
    from agentic_investor.experiments.news_bus import SharedBusStream
    assert not isinstance(stream, SharedBusStream)


def _drain_asyncio():
    """Ensure event loop teardown between tests."""
    try:
        asyncio.get_event_loop()
    except RuntimeError:
        pass
