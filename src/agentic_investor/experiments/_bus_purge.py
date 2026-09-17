"""Periodic TTL purge for the shared bus SQLite files.

News-bus and price-bus writers each spawn one purge daemon at startup;
it wakes every `interval_sec`, deletes rows older than the retention
cutoff, and runs `PRAGMA wal_checkpoint(TRUNCATE)` so the file actually
shrinks (otherwise DELETE just moves rows to the free list and the file
size stays put).

Deliberately shared between news_bus.py and price_bus.py so both use
identical purge semantics + a single place to fix bugs in.

Motivated by 2026-09-17: news_bus.db grew unbounded through a trading
day, at market close it held ~3000 stale news items which the STALE-drop
in `render_batch_context` (commit 96505e4) papered over. This is the
upstream fix.
"""

from __future__ import annotations

import logging
import sqlite3
import threading
from collections.abc import Callable
from pathlib import Path

logger = logging.getLogger(__name__)


PurgeQuery = Callable[[sqlite3.Connection], int]
"""Callable that runs the DELETE and returns the rows-affected count."""


def _purge_once(
    db_path: Path,
    purge: PurgeQuery,
    *,
    conn_setup: Callable[[sqlite3.Connection], None] | None = None,
) -> int:
    """Run one purge + WAL checkpoint. Returns rows deleted, or -1 on error.

    Commits the DELETE first (so wal_checkpoint(TRUNCATE) doesn't run
    inside the write transaction, which raises SQLITE_BUSY). Then opens
    a separate connection for the checkpoint pragma.

    `conn_setup` runs against each freshly opened connection before
    handing it to `purge`. Callers that need sqlite-vec loaded (news
    store purge touching vec_news) pass a setup that calls
    `sqlite_vec.load(conn)`.
    """
    try:
        conn = sqlite3.connect(str(db_path), isolation_level=None)
        try:
            if conn_setup is not None:
                conn_setup(conn)
            n = purge(conn)
        finally:
            conn.close()
        # Separate connection: WAL checkpoint only works outside a write
        # transaction on the same conn.
        ck = sqlite3.connect(str(db_path), isolation_level=None)
        try:
            ck.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        finally:
            ck.close()
        return n
    except sqlite3.OperationalError as e:
        logger.debug("purge transient error: %s", e)
        return -1


def start_purge_thread(
    db_path: Path,
    purge: PurgeQuery,
    *,
    interval_sec: float,
    label: str,
    conn_setup: Callable[[sqlite3.Connection], None] | None = None,
    stop_event: threading.Event | None = None,
) -> threading.Thread:
    """Kick off a daemon thread that runs `purge` every `interval_sec`.

    `purge(conn)` gets an autocommit sqlite3.Connection to the bus DB.
    It should perform its DELETE(s) and return the total rows deleted;
    we handle the WAL checkpoint after.

    `conn_setup(conn)` (optional) runs once per opened connection before
    `purge` sees it - use it to load sqlite-vec / attach extensions.

    `stop_event` is optional - pass it to allow clean shutdown; without
    it the thread exits when the process exits (it's a daemon).

    Returns the started thread so callers that want to join it can.
    """
    stop = stop_event or threading.Event()

    def _loop() -> None:
        # First sweep after a short delay so the writer has time to
        # populate the schema before we start swinging DELETE at it.
        if stop.wait(min(30.0, interval_sec)):
            return
        while not stop.is_set():
            n = _purge_once(db_path, purge, conn_setup=conn_setup)
            if n > 0:
                logger.info("purge[%s]: dropped %d rows", label, n)
            if stop.wait(interval_sec):
                return

    t = threading.Thread(
        target=_loop, name=f"purge-{label}", daemon=True,
    )
    t.start()
    logger.info(
        "purge[%s] armed: interval=%.0fs on %s",
        label, interval_sec, db_path,
    )
    return t


# Note: `env_ttl_hours` was removed in the #150 flags-registry migration.
# TTLs are now read via `agentic_investor.flags.flags.NEWS_BUS_TTL_HOURS`
# (etc.) at the call site.
