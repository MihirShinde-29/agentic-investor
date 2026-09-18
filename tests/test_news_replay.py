"""Tests for the news-replay driver (task #159 / A).

Feeds recorded news rows into an arm's event_queue in place of a live
Alpaca websocket when AGENTIC_REPLAY_FROM is set. Complements the
recorder capture already exercised in test_recorder.py.
"""

from __future__ import annotations

import json
import queue as _q
import threading
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from agentic_investor.orchestrator import recorder
from agentic_investor.orchestrator.news_replay import (
    NewsReplayDriver,
    _current_speed,
    _parse_ts,
    _row_to_event,
    replay_active,
)


def _write_recording(dir_: Path, rows: list[dict]) -> None:
    dir_.mkdir(parents=True, exist_ok=True)
    with (dir_ / "recording.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def _news_row(seq: int, *, ticker: str, received_at: str,
              headline: str = "h") -> dict:
    return {
        "kind": "news",
        "seq": seq,
        "ts": received_at,
        "ticker": ticker,
        "headline": headline,
        "summary": "s",
        "published_at": received_at,
        "received_at": received_at,
        "url": "https://ex/",
        "source": "test",
    }


@pytest.fixture(autouse=True)
def _reset_recorder():
    recorder.reset_for_tests()
    yield
    recorder.reset_for_tests()


def test_replay_active_reflects_env(monkeypatch):
    monkeypatch.delenv("AGENTIC_REPLAY_FROM", raising=False)
    assert replay_active() is False
    monkeypatch.setenv("AGENTIC_REPLAY_FROM", "/tmp/x")
    assert replay_active() is True


def test_parse_ts_round_trips_isoformat():
    now = datetime.now(UTC)
    got = _parse_ts(now.isoformat())
    assert got is not None
    assert abs(got - now.timestamp()) < 1e-6


def test_parse_ts_tolerates_missing_and_bad_input():
    assert _parse_ts(None) is None
    assert _parse_ts("") is None
    assert _parse_ts("not-a-date") is None


def test_row_to_event_populates_fields():
    row = _news_row(1, ticker="AAPL", received_at="2026-01-01T00:00:00+00:00")
    evt = _row_to_event(row)
    assert evt.ticker == "AAPL"
    assert evt.headline == "h"
    assert evt.source == "test"


def test_current_speed_defaults_to_one(monkeypatch):
    monkeypatch.delenv("AGENTIC_REPLAY_NEWS_SPEED", raising=False)
    assert _current_speed() == 1.0


def test_current_speed_parses_env(monkeypatch):
    monkeypatch.setenv("AGENTIC_REPLAY_NEWS_SPEED", "2.5")
    assert _current_speed() == 2.5


def test_current_speed_bad_value_falls_back(monkeypatch):
    monkeypatch.setenv("AGENTIC_REPLAY_NEWS_SPEED", "banana")
    assert _current_speed() == 1.0


def test_driver_noop_when_no_recording(tmp_path, monkeypatch):
    """No AGENTIC_REPLAY_FROM => start() finds zero rows, thread never
    spawns. Silent no-op keeps live-mode behavior identical when a
    caller wires this in unconditionally.
    """
    monkeypatch.delenv("AGENTIC_REPLAY_FROM", raising=False)
    q: _q.Queue = _q.Queue()
    drv = NewsReplayDriver(q, speed=0)
    drv.start()
    assert drv.total == 0
    assert q.empty()


def test_driver_injects_all_rows_in_order(tmp_path, monkeypatch):
    """speed=0 fires every recorded event as fast as possible; the
    queue should end up with them in seq order, matching what a live
    NewsStreamer would have delivered.
    """
    base = datetime.now(UTC)
    rows = [
        _news_row(1, ticker="AAPL",
                  received_at=(base).isoformat(),
                  headline="a1"),
        _news_row(2, ticker="NVDA",
                  received_at=(base + timedelta(seconds=30)).isoformat(),
                  headline="n1"),
        _news_row(3, ticker="TSLA",
                  received_at=(base + timedelta(seconds=90)).isoformat(),
                  headline="t1"),
    ]
    _write_recording(tmp_path, rows)
    monkeypatch.setenv("AGENTIC_REPLAY_FROM", str(tmp_path))

    q: _q.Queue = _q.Queue()
    drv = NewsReplayDriver(q, speed=0)
    drv.start()
    drv.join(timeout=5.0)

    got = []
    while not q.empty():
        got.append(q.get_nowait())
    assert [e.ticker for e in got] == ["AAPL", "NVDA", "TSLA"]
    assert [e.headline for e in got] == ["a1", "n1", "t1"]
    assert drv.injected == 3


def test_driver_respects_recorded_cadence_at_real_time(
    tmp_path, monkeypatch,
):
    """speed=1.0 should honour recorded inter-event gaps roughly. Uses
    a tiny 200 ms gap so the test doesn't add real latency to the
    suite. Wall-clock delta must clear at least the gap.
    """
    base = datetime.now(UTC)
    rows = [
        _news_row(1, ticker="AAPL", received_at=base.isoformat()),
        _news_row(2, ticker="NVDA",
                  received_at=(base + timedelta(milliseconds=200)).isoformat()),
    ]
    _write_recording(tmp_path, rows)
    monkeypatch.setenv("AGENTIC_REPLAY_FROM", str(tmp_path))

    q: _q.Queue = _q.Queue()
    drv = NewsReplayDriver(q, speed=1.0)
    import time
    t0 = time.monotonic()
    drv.start()
    drv.join(timeout=5.0)
    elapsed = time.monotonic() - t0
    assert drv.injected == 2
    # At least ~180 ms should have passed - allow small scheduling
    # slack so we don't flake on a busy CI host.
    assert elapsed >= 0.18, f"expected ~0.2s cadence, got {elapsed:.3f}"


def test_driver_stop_is_prompt(tmp_path, monkeypatch):
    """stop() should unblock the sleep between events so shutdown is
    fast even with large recorded gaps.
    """
    base = datetime.now(UTC)
    rows = [
        _news_row(1, ticker="AAPL", received_at=base.isoformat()),
        _news_row(2, ticker="NVDA",
                  received_at=(base + timedelta(seconds=30)).isoformat()),
    ]
    _write_recording(tmp_path, rows)
    monkeypatch.setenv("AGENTIC_REPLAY_FROM", str(tmp_path))

    q: _q.Queue = _q.Queue()
    drv = NewsReplayDriver(q, speed=1.0)
    drv.start()
    # First event fires immediately; give the thread a moment to reach
    # the wait on the second.
    import time
    time.sleep(0.1)
    t0 = time.monotonic()
    drv.stop()
    drv.join(timeout=2.0)
    stop_latency = time.monotonic() - t0
    assert stop_latency < 1.0, (
        f"stop() took {stop_latency:.2f}s; wait() should have "
        f"broken immediately on the event"
    )
    # First event landed; second either did or didn't depending on
    # timing, but we shouldn't have waited 30s for it.
    assert drv.injected >= 1


def test_driver_idempotent_start_while_running(tmp_path, monkeypatch):
    """start() called twice while the first thread is still alive is a
    no-op. Uses a slow recorded gap so thread1 is definitely still
    running when we call start() again. (Once the first thread has
    finished, a fresh start() is allowed to spawn a new one - the
    idempotency guard only covers overlap.)
    """
    base = datetime.now(UTC)
    rows = [
        _news_row(1, ticker="AAPL", received_at=base.isoformat()),
        _news_row(2, ticker="NVDA",
                  received_at=(base + timedelta(seconds=10)).isoformat()),
    ]
    _write_recording(tmp_path, rows)
    monkeypatch.setenv("AGENTIC_REPLAY_FROM", str(tmp_path))

    q: _q.Queue = _q.Queue()
    drv = NewsReplayDriver(q, speed=1.0)
    drv.start()
    thread1 = drv._thread
    drv.start()
    thread2 = drv._thread
    assert thread1 is thread2
    drv.stop()
    drv.join(timeout=2.0)


def test_driver_ignores_non_news_rows(tmp_path, monkeypatch):
    """recording.jsonl mixes kinds; the driver only cares about news."""
    base = datetime.now(UTC)
    rows = [
        {"kind": "clock", "seq": 1, "ts": base.isoformat(),
         "is_open": True},
        _news_row(2, ticker="AAPL", received_at=base.isoformat()),
        {"kind": "price", "seq": 3, "ts": base.isoformat(),
         "ticker": "AAPL", "price": 100.0},
        _news_row(4, ticker="NVDA", received_at=base.isoformat()),
    ]
    _write_recording(tmp_path, rows)
    monkeypatch.setenv("AGENTIC_REPLAY_FROM", str(tmp_path))

    q: _q.Queue = _q.Queue()
    drv = NewsReplayDriver(q, speed=0)
    drv.start()
    drv.join(timeout=5.0)
    tickers = []
    while not q.empty():
        tickers.append(q.get_nowait().ticker)
    assert tickers == ["AAPL", "NVDA"]


def test_speed_multiplier_shrinks_wall_time(tmp_path, monkeypatch):
    """A 400ms recorded gap at speed=4.0 should take ~100 ms."""
    base = datetime.now(UTC)
    rows = [
        _news_row(1, ticker="AAPL", received_at=base.isoformat()),
        _news_row(2, ticker="NVDA",
                  received_at=(base + timedelta(milliseconds=400)).isoformat()),
    ]
    _write_recording(tmp_path, rows)
    monkeypatch.setenv("AGENTIC_REPLAY_FROM", str(tmp_path))

    q: _q.Queue = _q.Queue()
    drv = NewsReplayDriver(q, speed=4.0)
    import time
    t0 = time.monotonic()
    drv.start()
    drv.join(timeout=5.0)
    elapsed = time.monotonic() - t0
    # Under 250 ms (400 / 4 = 100, +scheduling slack). Real-time would
    # have taken >= 400 ms so any comfortable ceiling under that
    # confirms the multiplier's doing something.
    assert elapsed < 0.30, (
        f"speed=4.0 with 400ms gap should stay well under 300ms, "
        f"got {elapsed:.3f}"
    )


def test_driver_survives_bad_timestamp_gracefully(tmp_path, monkeypatch):
    """A recording with a garbled received_at should still deliver the
    event - just without an inter-event wait.
    """
    base = datetime.now(UTC)
    good = _news_row(1, ticker="AAPL", received_at=base.isoformat())
    bad = _news_row(2, ticker="NVDA", received_at="garbage-not-a-date")
    _write_recording(tmp_path, [good, bad])
    monkeypatch.setenv("AGENTIC_REPLAY_FROM", str(tmp_path))

    q: _q.Queue = _q.Queue()
    drv = NewsReplayDriver(q, speed=1.0)
    drv.start()
    drv.join(timeout=5.0)
    got = []
    while not q.empty():
        got.append(q.get_nowait().ticker)
    assert got == ["AAPL", "NVDA"]


def test_driver_thread_is_daemon(tmp_path, monkeypatch):
    """Daemon so a pytest crash / KeyboardInterrupt in the caller
    doesn't hang the process on the replay thread.
    """
    base = datetime.now(UTC)
    rows = [_news_row(1, ticker="AAPL", received_at=base.isoformat())]
    _write_recording(tmp_path, rows)
    monkeypatch.setenv("AGENTIC_REPLAY_FROM", str(tmp_path))

    q: _q.Queue = _q.Queue()
    drv = NewsReplayDriver(q, speed=0)
    drv.start()
    assert isinstance(drv._thread, threading.Thread)
    assert drv._thread.daemon is True
    drv.join(timeout=2.0)
