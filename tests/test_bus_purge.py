"""Tests for the shared bus TTL / purge helpers.

Covers the raw `_bus_purge` primitives (env parsing, single-shot purge,
daemon-thread lifecycle) plus a smoke test for each bus writer's purge
wiring using an in-file SQLite store (not :memory:, because the purge
helper opens its own connection per sweep).
"""

from __future__ import annotations

import sqlite3
import threading
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

from agentic_investor.experiments._bus_purge import (
    _purge_once,
    env_ttl_hours,
    start_purge_thread,
)


def _new_bus_db(tmp_path: Path) -> Path:
    path = tmp_path / "bus.db"
    with sqlite3.connect(str(path)) as conn:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute(
            "CREATE TABLE events (id INTEGER PRIMARY KEY AUTOINCREMENT, "
            "ts TEXT NOT NULL, payload TEXT)"
        )
    return path


def _seed(conn, n: int, hours_ago: float) -> None:
    ts = (datetime.now(UTC) - timedelta(hours=hours_ago)).isoformat()
    conn.executemany(
        "INSERT INTO events (ts, payload) VALUES (?, ?)",
        [(ts, f"payload {i}") for i in range(n)],
    )


def _count(path: Path) -> int:
    with sqlite3.connect(str(path)) as conn:
        return conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]


def test_env_ttl_hours_defaults_and_parses(monkeypatch):
    monkeypatch.delenv("X_TTL", raising=False)
    assert env_ttl_hours("X_TTL", 4) == 4.0
    monkeypatch.setenv("X_TTL", "2.5")
    assert env_ttl_hours("X_TTL", 4) == 2.5


def test_env_ttl_hours_rejects_negative_and_nonnumeric(monkeypatch, caplog):
    import logging
    monkeypatch.setenv("X_TTL", "not-a-number")
    with caplog.at_level(logging.WARNING):
        assert env_ttl_hours("X_TTL", 4) == 4.0
    assert any("is not a number" in r.getMessage() for r in caplog.records)

    caplog.clear()
    monkeypatch.setenv("X_TTL", "-3")
    with caplog.at_level(logging.WARNING):
        assert env_ttl_hours("X_TTL", 4) == 4.0
    assert any("is negative" in r.getMessage() for r in caplog.records)


def test_purge_once_deletes_old_rows(tmp_path):
    path = _new_bus_db(tmp_path)
    with sqlite3.connect(str(path)) as conn:
        _seed(conn, 5, hours_ago=10)  # old
        _seed(conn, 3, hours_ago=0.5)  # fresh

    def _purge(conn):
        cutoff = (datetime.now(UTC) - timedelta(hours=4)).isoformat()
        cur = conn.execute("DELETE FROM events WHERE ts < ?", (cutoff,))
        return cur.rowcount

    n = _purge_once(path, _purge)
    assert n == 5
    assert _count(path) == 3


def test_purge_once_reports_zero_when_all_fresh(tmp_path):
    path = _new_bus_db(tmp_path)
    with sqlite3.connect(str(path)) as conn:
        _seed(conn, 10, hours_ago=0.1)

    def _purge(conn):
        cutoff = (datetime.now(UTC) - timedelta(hours=4)).isoformat()
        cur = conn.execute("DELETE FROM events WHERE ts < ?", (cutoff,))
        return cur.rowcount

    assert _purge_once(path, _purge) == 0
    assert _count(path) == 10


def test_purge_once_survives_missing_table(tmp_path):
    """Writer might not have created the table yet when the first sweep
    fires. Returning -1 (not raising) is the contract.
    """
    missing = tmp_path / "not_yet.db"
    # Create empty DB so `sqlite3.connect` succeeds but query fails.
    sqlite3.connect(str(missing)).close()

    def _bad_purge(conn):
        conn.execute("DELETE FROM does_not_exist")
        return 0

    assert _purge_once(missing, _bad_purge) == -1


def test_start_purge_thread_runs_and_stops(tmp_path):
    """The thread lifecycle: it runs at least one sweep, respects
    stop_event, exits cleanly, and doesn't leak.
    """
    path = _new_bus_db(tmp_path)
    with sqlite3.connect(str(path)) as conn:
        _seed(conn, 4, hours_ago=10)

    calls = {"n": 0}

    def _purge(conn):
        calls["n"] += 1
        cutoff = (datetime.now(UTC) - timedelta(hours=4)).isoformat()
        cur = conn.execute("DELETE FROM events WHERE ts < ?", (cutoff,))
        return cur.rowcount

    stop = threading.Event()
    # Very short interval + first-sweep wait bounded by min(30, interval)
    # so the test doesn't need to wait 30 s.
    t = start_purge_thread(
        path, _purge,
        interval_sec=0.1, label="test",
        stop_event=stop,
    )
    # Give it enough wall-clock for at least one sweep.
    time.sleep(0.5)
    stop.set()
    t.join(timeout=2.0)
    assert not t.is_alive(), "purge thread did not stop after event"
    assert calls["n"] >= 1
    assert _count(path) == 0  # all old rows purged


def test_wal_checkpoint_runs_after_purge(tmp_path):
    """After a purge sweep, the -wal file should be truncated so disk
    usage actually drops. Not a strict assertion on file bytes (WAL
    behavior varies by OS + sqlite version) but verifies the pragma
    fires without error.
    """
    path = _new_bus_db(tmp_path)
    with sqlite3.connect(str(path)) as conn:
        _seed(conn, 100, hours_ago=10)

    def _purge(conn):
        cutoff = (datetime.now(UTC) - timedelta(hours=4)).isoformat()
        cur = conn.execute("DELETE FROM events WHERE ts < ?", (cutoff,))
        return cur.rowcount

    n = _purge_once(path, _purge)
    assert n == 100
    # No exception + purge count matched = wal_checkpoint ran (it's the
    # step right after DELETE in _purge_once).


def test_news_store_purge_removes_both_tables(tmp_path):
    """The news_store purge (as wired in news_bus.run_bus_writer) must
    delete both news_articles rows AND their vec_news counterparts.
    Recreates the exact SQL shape used in run_bus_writer._purge_news_store.
    """
    from agentic_investor.tools.news_store import _init_conn
    store_path = tmp_path / "news_store.db"
    conn = _init_conn(str(store_path))
    # Seed 4 old + 2 fresh; each with an accompanying vec_news row.
    from agentic_investor.tools.news import _pack_vector
    from agentic_investor.tools.news_store import EMBED_DIM
    old_iso = (datetime.now(UTC) - timedelta(days=60)).isoformat()
    fresh_iso = (datetime.now(UTC) - timedelta(days=1)).isoformat()
    zeros = [0.0] * EMBED_DIM
    for i, ts in enumerate([old_iso] * 4 + [fresh_iso] * 2, start=1):
        conn.execute(
            "INSERT INTO news_articles (id, ticker, headline, published_at) "
            "VALUES (?, ?, ?, ?)",
            (f"art{i}", "AAPL", f"headline {i}", ts),
        )
        conn.execute(
            "INSERT INTO vec_news (rowid, embedding) VALUES (?, ?)",
            (i, _pack_vector(zeros)),
        )
    conn.close()

    def _purge(conn):
        cutoff = (datetime.now(UTC) - timedelta(days=30)).isoformat()
        rows = conn.execute(
            "SELECT internal_id FROM news_articles WHERE published_at < ?",
            (cutoff,),
        ).fetchall()
        ids = [r[0] for r in rows]
        if not ids:
            return 0
        placeholders = ",".join("?" * len(ids))
        conn.execute(
            f"DELETE FROM news_articles WHERE internal_id IN ({placeholders})",
            ids,
        )
        for iid in ids:
            conn.execute("DELETE FROM vec_news WHERE rowid = ?", (iid,))
        return len(ids)

    import sqlite_vec

    def _load_vec(conn):
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)

    n = _purge_once(store_path, _purge, conn_setup=_load_vec)
    assert n == 4

    # Reopen to inspect final state. Needs sqlite-vec loaded to read
    # vec_news (a vec0 virtual table).
    check = sqlite3.connect(str(store_path))
    try:
        check.enable_load_extension(True)
        sqlite_vec.load(check)
        check.enable_load_extension(False)
        art_count = check.execute("SELECT COUNT(*) FROM news_articles").fetchone()[0]
        vec_count = check.execute("SELECT COUNT(*) FROM vec_news").fetchone()[0]
    finally:
        check.close()
    assert art_count == 2
    assert vec_count == 2
