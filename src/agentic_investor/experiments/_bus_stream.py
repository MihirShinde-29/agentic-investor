"""Shared reconnect + status plumbing for the news-bus and price-bus writers.

Both writers were single-shot on 2026-09-17 pre-#149: `stream.run()` blocks
until Alpaca closes the socket; when the WebSocket drops (transient network
blip, laptop sleep/wake, Alpaca-side rotation), the writer subprocess exits
and the supervisor respawns it - but any events during the disconnect are
silently gone.

This module gives both writers a common lifecycle:
- `run_with_reconnect(stream_factory, run_stream, label, ...)` loops around
  the blocking stream call with exponential backoff (1s -> 2 -> 4 -> 8 -> cap 30s)
- `BusStatus` dataclass owns the singleton bus_status row (last_event_at,
  total_events, reconnect_count, is_connected)
- Callers pass an optional `on_reconnect(before_ts_iso, now_ts_iso)` hook
  that runs after each successful reconnect to backfill missed events via
  Alpaca's REST endpoints

Deliberately shared so both writers use identical reconnect semantics + a
single place to fix bugs.
"""

from __future__ import annotations

import logging
import sqlite3
import threading
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

logger = logging.getLogger(__name__)


_INITIAL_BACKOFF_SEC = 1.0
_MAX_BACKOFF_SEC = 30.0
_MAX_RECONNECTS = 100  # give up after this many so a truly-broken auth
                      # eventually exits the process cleanly (supervisor
                      # then decides whether to respawn)


_STATUS_SCHEMA = """
CREATE TABLE IF NOT EXISTS bus_status (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    updated_at TEXT NOT NULL,
    last_event_at TEXT,
    total_events INTEGER NOT NULL DEFAULT 0,
    reconnect_count INTEGER NOT NULL DEFAULT 0,
    is_connected INTEGER NOT NULL DEFAULT 0
)
"""


def init_status_table(db_path: Path) -> None:
    with sqlite3.connect(str(db_path)) as conn:
        conn.execute(_STATUS_SCHEMA)
        conn.execute(
            "INSERT OR IGNORE INTO bus_status "
            "(id, updated_at, total_events, reconnect_count, is_connected) "
            "VALUES (1, ?, 0, 0, 0)",
            (datetime.now(UTC).isoformat(),),
        )


@dataclass
class BusStatus:
    """Threadsafe wrapper over the singleton bus_status row.

    Every mutator updates the row's `updated_at` in the same UPDATE so
    a stale-status observer (dashboard, tests) can tell how fresh the
    numbers are.
    """

    db_path: Path
    _lock: threading.Lock

    @classmethod
    def open(cls, db_path: Path) -> BusStatus:
        init_status_table(db_path)
        return cls(db_path=db_path, _lock=threading.Lock())

    def _write(self, sql: str, params: tuple) -> None:
        with self._lock, sqlite3.connect(str(self.db_path)) as conn:
            conn.execute(sql, params)

    def mark_connected(self) -> None:
        self._write(
            "UPDATE bus_status SET is_connected=1, updated_at=? WHERE id=1",
            (datetime.now(UTC).isoformat(),),
        )

    def mark_disconnected(self) -> None:
        self._write(
            "UPDATE bus_status SET is_connected=0, updated_at=? WHERE id=1",
            (datetime.now(UTC).isoformat(),),
        )

    def bump_reconnect(self) -> int:
        """Increment the reconnect counter; return the new value."""
        now = datetime.now(UTC).isoformat()
        with self._lock, sqlite3.connect(str(self.db_path)) as conn:
            conn.execute(
                "UPDATE bus_status SET "
                "reconnect_count = reconnect_count + 1, updated_at=? "
                "WHERE id=1",
                (now,),
            )
            row = conn.execute(
                "SELECT reconnect_count FROM bus_status WHERE id=1"
            ).fetchone()
        return int(row[0]) if row else 0

    def record_event(self, event_ts_iso: str, count: int = 1) -> None:
        """Every event handler calls this so `last_event_at` stays fresh.

        Backfill callers pass count>1 so the running total reflects
        recovered rows too.
        """
        now = datetime.now(UTC).isoformat()
        with self._lock, sqlite3.connect(str(self.db_path)) as conn:
            conn.execute(
                "UPDATE bus_status SET "
                "last_event_at = COALESCE(?, last_event_at), "
                "total_events = total_events + ?, "
                "updated_at = ? "
                "WHERE id=1",
                (event_ts_iso, int(count), now),
            )

    def snapshot(self) -> dict:
        """Return a plain dict for tests and dashboard consumers."""
        with self._lock, sqlite3.connect(str(self.db_path)) as conn:
            row = conn.execute(
                "SELECT updated_at, last_event_at, total_events, "
                "reconnect_count, is_connected FROM bus_status WHERE id=1"
            ).fetchone()
        if row is None:
            return {}
        return {
            "updated_at": row[0],
            "last_event_at": row[1],
            "total_events": int(row[2] or 0),
            "reconnect_count": int(row[3] or 0),
            "is_connected": bool(row[4]),
        }


def run_with_reconnect(
    *,
    stream_factory: Callable[[], object],
    run_stream: Callable[[object], None],
    on_reconnect: Callable[[str | None, str], int] | None,
    status: BusStatus,
    label: str,
    stop_event: threading.Event | None = None,
    initial_backoff_sec: float = _INITIAL_BACKOFF_SEC,
    max_backoff_sec: float = _MAX_BACKOFF_SEC,
    max_reconnects: int = _MAX_RECONNECTS,
) -> int:
    """Blocking supervisor loop: call `stream_factory()` -> `run_stream()`
    in a try/except that reconnects on any exception with exponential
    backoff, bounded by `max_reconnects`.

    - `run_stream(stream)` is the caller's blocking connect+run function
      (typically `stream.subscribe_*(); stream.run()`).
    - `on_reconnect(last_event_at_before, now_iso)` fires after each
      successful reconnect (not the first connect) so callers can query
      REST for the gap window. Returns the count of backfilled rows for
      logging.

    Returns 0 on clean stop_event trigger; nonzero if max_reconnects
    exhausted.
    """
    stop = stop_event or threading.Event()
    backoff = initial_backoff_sec
    first = True
    logger.info("%s writer: reconnect supervisor armed", label)
    while not stop.is_set():
        # Snapshot `last_event_at` BEFORE (re)connect so the backfill
        # hook knows the gap window.
        pre_snap = status.snapshot()
        before_ts = pre_snap.get("last_event_at")

        try:
            stream = stream_factory()
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "%s stream_factory failed: %s; retrying in %.1fs",
                label, e, backoff,
            )
            if stop.wait(backoff):
                return 0
            backoff = min(backoff * 2, max_backoff_sec)
            continue

        status.mark_connected()
        if not first:
            n_recon = status.bump_reconnect()
            logger.warning(
                "%s reconnected (attempt %d); backfill window: after %s",
                label, n_recon, before_ts,
            )
            if on_reconnect is not None:
                try:
                    now_iso = datetime.now(UTC).isoformat()
                    backfilled = on_reconnect(before_ts, now_iso)
                    if backfilled:
                        logger.info(
                            "%s backfilled %d events after reconnect",
                            label, backfilled,
                        )
                except Exception as e:  # noqa: BLE001
                    logger.warning("%s backfill failed: %s", label, e)
            if n_recon >= max_reconnects:
                logger.error(
                    "%s exhausted %d reconnect attempts; exiting",
                    label, max_reconnects,
                )
                status.mark_disconnected()
                return 1

        try:
            run_stream(stream)
            # Clean return from run_stream = stream closed on its own
            # (Alpaca-side rotation or our stop hook). Treat as a
            # reconnect trigger not a hard failure.
            logger.warning("%s stream returned cleanly; will reconnect", label)
            status.mark_disconnected()
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "%s stream error: %s; reconnect in %.1fs",
                label, e, backoff,
            )
            status.mark_disconnected()
        first = False
        # After the first successful connect, reset backoff so the next
        # transient drop retries fast. Only prolonged failure stretches
        # the backoff.
        if not first:
            backoff = initial_backoff_sec
        if stop.wait(backoff):
            return 0
        backoff = min(backoff * 2, max_backoff_sec)
    return 0
