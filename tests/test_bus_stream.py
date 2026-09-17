"""Tests for the shared bus reconnect + status helpers (task #149).

Covers:
- BusStatus singleton row lifecycle (init, mark_*, record_event, snapshot)
- run_with_reconnect happy path (one clean connect + one stop_event)
- run_with_reconnect retry-with-backoff on stream_factory failure
- run_with_reconnect reconnect-on-error triggers the on_reconnect hook
  with the correct before_ts window
- max_reconnects ceiling exits with nonzero rc
"""

from __future__ import annotations

import sqlite3
import threading
import time
from pathlib import Path

from agentic_investor.experiments._bus_stream import (
    BusStatus,
    init_status_table,
    run_with_reconnect,
)


def _tmp_db(tmp_path: Path) -> Path:
    return tmp_path / "bus.db"


def test_init_creates_singleton_row(tmp_path):
    db = _tmp_db(tmp_path)
    init_status_table(db)
    # Second init is a no-op (INSERT OR IGNORE guards).
    init_status_table(db)
    with sqlite3.connect(str(db)) as conn:
        rows = conn.execute(
            "SELECT id, total_events, reconnect_count FROM bus_status"
        ).fetchall()
    assert rows == [(1, 0, 0)]


def test_bus_status_transitions(tmp_path):
    status = BusStatus.open(_tmp_db(tmp_path))
    snap = status.snapshot()
    assert snap["is_connected"] is False
    assert snap["total_events"] == 0
    assert snap["reconnect_count"] == 0

    status.mark_connected()
    assert status.snapshot()["is_connected"] is True

    status.record_event("2026-09-17T10:00:00+00:00")
    status.record_event("2026-09-17T10:00:05+00:00")
    snap = status.snapshot()
    assert snap["total_events"] == 2
    assert snap["last_event_at"] == "2026-09-17T10:00:05+00:00"

    n = status.bump_reconnect()
    assert n == 1
    assert status.snapshot()["reconnect_count"] == 1

    status.mark_disconnected()
    assert status.snapshot()["is_connected"] is False


def test_record_event_with_count_advances_total(tmp_path):
    """Backfill hook increments total_events by the batch size."""
    status = BusStatus.open(_tmp_db(tmp_path))
    status.record_event("2026-09-17T10:00:00+00:00", count=17)
    assert status.snapshot()["total_events"] == 17


def test_run_with_reconnect_stops_cleanly_on_event(tmp_path):
    """One connect, stop_event set, factory called once."""
    status = BusStatus.open(_tmp_db(tmp_path))
    calls = {"factory": 0, "run": 0}
    stop = threading.Event()

    def _factory():
        calls["factory"] += 1
        return object()

    def _run(stream):
        calls["run"] += 1
        # Wait for stop to be set so we don't tightloop.
        stop.wait()

    def _driver():
        run_with_reconnect(
            stream_factory=_factory,
            run_stream=_run,
            on_reconnect=None,
            status=status,
            label="test",
            stop_event=stop,
            initial_backoff_sec=0.01,
        )

    t = threading.Thread(target=_driver, daemon=True)
    t.start()
    time.sleep(0.2)  # let the loop connect
    stop.set()
    t.join(timeout=2.0)
    assert not t.is_alive()
    assert calls["factory"] == 1
    assert calls["run"] == 1
    assert status.snapshot()["reconnect_count"] == 0  # first connect isn't a reconnect


def test_run_with_reconnect_backs_off_when_factory_fails(tmp_path):
    """When stream_factory raises, we back off + retry; eventually a
    successful factory unblocks the loop.
    """
    status = BusStatus.open(_tmp_db(tmp_path))
    calls = {"n": 0}
    stop = threading.Event()

    def _factory():
        calls["n"] += 1
        if calls["n"] < 3:
            raise RuntimeError("simulated auth blip")
        return object()

    def _run(stream):
        stop.wait()

    def _driver():
        run_with_reconnect(
            stream_factory=_factory,
            run_stream=_run,
            on_reconnect=None,
            status=status,
            label="test",
            stop_event=stop,
            initial_backoff_sec=0.02,
            max_backoff_sec=0.05,
        )

    t = threading.Thread(target=_driver, daemon=True)
    t.start()
    time.sleep(0.5)  # allow ~3 factory attempts with backoff
    stop.set()
    t.join(timeout=2.0)
    assert not t.is_alive()
    assert calls["n"] >= 3


def test_run_with_reconnect_fires_backfill_hook_on_second_connect(tmp_path):
    """First connect: no backfill. Stream errors and we reconnect ->
    backfill hook fires with the (pre-drop last_event_at, now) window.
    """
    status = BusStatus.open(_tmp_db(tmp_path))
    # Pre-populate last_event_at so the backfill window is meaningful.
    status.record_event("2026-09-17T10:00:00+00:00")

    stop = threading.Event()
    factory_calls = {"n": 0}
    backfill_calls: list[tuple[str | None, str]] = []
    run_signal = threading.Event()

    def _factory():
        factory_calls["n"] += 1
        return object()

    def _run(stream):
        # First call errors immediately to trigger reconnect. Second call
        # blocks until stop is set.
        if factory_calls["n"] == 1:
            raise RuntimeError("simulated ws drop")
        run_signal.set()
        stop.wait()

    def _backfill(before, now):
        backfill_calls.append((before, now))
        return 3

    def _driver():
        run_with_reconnect(
            stream_factory=_factory,
            run_stream=_run,
            on_reconnect=_backfill,
            status=status,
            label="test",
            stop_event=stop,
            initial_backoff_sec=0.02,
            max_backoff_sec=0.05,
        )

    t = threading.Thread(target=_driver, daemon=True)
    t.start()
    assert run_signal.wait(timeout=3.0), "second connect never happened"
    stop.set()
    t.join(timeout=2.0)
    assert not t.is_alive()

    assert factory_calls["n"] >= 2
    # Backfill fired exactly once (on the second connect, not the first).
    assert len(backfill_calls) == 1
    before, now = backfill_calls[0]
    assert before == "2026-09-17T10:00:00+00:00"
    # `now` is a wall-clock ISO from datetime.now(UTC); just sanity-check.
    assert "T" in now


def test_max_reconnects_ceiling_exits(tmp_path):
    """After max_reconnects, the loop returns nonzero + marks disconnected."""
    status = BusStatus.open(_tmp_db(tmp_path))
    stop = threading.Event()

    def _factory():
        return object()

    def _run(stream):
        # Always error - forces immediate reconnect.
        raise RuntimeError("simulated permanent drop")

    rc_holder: list[int] = []

    def _driver():
        rc = run_with_reconnect(
            stream_factory=_factory,
            run_stream=_run,
            on_reconnect=None,
            status=status,
            label="test",
            stop_event=stop,
            initial_backoff_sec=0.001,
            max_backoff_sec=0.005,
            max_reconnects=3,
        )
        rc_holder.append(rc)

    t = threading.Thread(target=_driver, daemon=True)
    t.start()
    t.join(timeout=3.0)
    assert not t.is_alive(), "loop should have given up by max_reconnects"
    assert rc_holder == [1]
    assert status.snapshot()["is_connected"] is False
    assert status.snapshot()["reconnect_count"] >= 3
