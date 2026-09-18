"""Regression: news_store connection must be usable from any thread.

Bug that motivated the tests: LangGraph parallelises the news-agent
per-ticker calls on a threadpool. The `news_store` module holds a
process-wide singleton connection, and sqlite3.connect() defaults to
`check_same_thread=True`, so every worker except the init thread
threw "SQLite objects created in a thread can only be used in that
same thread". Every news-agent call for every ticker failed, silently
degrading the LLM regen's news signal to zero.

Fix: `check_same_thread=False` on _init_conn, plus a module-level
RLock (`get_stmt_lock`) that callers wrap around writes and vec MATCH
queries.
"""

from __future__ import annotations

import concurrent.futures
import threading

from agentic_investor.tools.news_store import (
    get_stmt_lock,
    open_memory_conn,
)


def test_stmt_lock_is_a_reentrant_lock():
    """Nested acquisition needs to work (e.g. a caller inside a
    `with get_stmt_lock():` block calling another helper that also
    acquires). Rules out a plain Lock regression.
    """
    lock = get_stmt_lock()
    assert isinstance(lock, type(threading.RLock()))
    with lock:
        with lock:
            pass


def test_shared_connection_usable_from_worker_thread():
    """The pre-fix bug: a conn opened on thread A raised on any
    execute() from thread B. Post-fix, a shared conn works from any
    thread. Uses an in-memory conn so the test needs no real store.
    """
    conn = open_memory_conn()

    result: list = []
    err: list = []

    def _write_and_read(i: int) -> None:
        try:
            # Real callers wrap the whole write+lookup under the lock -
            # a shared conn's implicit cursor state isn't safe across
            # thread-interleaved statements even with autocommit.
            with get_stmt_lock():
                conn.execute(
                    "INSERT INTO news_articles "
                    "(id, ticker, headline, summary, source, url, published_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (f"id-{i}", "NVDA", f"h{i}", "s", "src", "u", "2026-09-18"),
                )
                row = conn.execute(
                    "SELECT ticker FROM news_articles WHERE id = ?", (f"id-{i}",),
                ).fetchone()
            result.append(row[0])
        except Exception as e:  # noqa: BLE001
            err.append(e)

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(_write_and_read, i) for i in range(10)]
        for f in futures:
            f.result()

    assert err == [], f"unexpected errors: {err}"
    assert sorted(result) == ["NVDA"] * 10


def test_parallel_writes_do_not_interleave_under_lock():
    """`with get_stmt_lock(), conn:` serialises the multi-statement
    upsert (insert -> lookup internal_id -> delete vec_news -> insert
    vec_news). Without the lock, a concurrent worker could delete the
    vec row that another worker just wrote, corrupting the join.

    Here we just verify all 20 rows land with matching internal_id ->
    vec_news rowids - if interleaving happened, we'd see mismatched
    or missing rows.
    """
    import struct

    from agentic_investor.tools.news_store import EMBED_DIM

    conn = open_memory_conn()

    def _upsert(i: int) -> None:
        with get_stmt_lock(), conn:
            conn.execute(
                "INSERT INTO news_articles "
                "(id, ticker, headline, summary, source, url, published_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (f"n-{i}", "AAPL", f"h{i}", "", "", "", "2026-09-18"),
            )
            row = conn.execute(
                "SELECT internal_id FROM news_articles WHERE id = ?", (f"n-{i}",),
            ).fetchone()
            internal_id = int(row[0])
            conn.execute(
                "DELETE FROM vec_news WHERE rowid = ?", (internal_id,),
            )
            emb = struct.pack(f"{EMBED_DIM}f", *([float(i)] * EMBED_DIM))
            conn.execute(
                "INSERT INTO vec_news (rowid, embedding) VALUES (?, ?)",
                (internal_id, emb),
            )

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(_upsert, i) for i in range(20)]
        for f in futures:
            f.result()

    n_meta = conn.execute("SELECT COUNT(*) FROM news_articles").fetchone()[0]
    n_vec = conn.execute("SELECT COUNT(*) FROM vec_news").fetchone()[0]
    assert n_meta == 20
    assert n_vec == 20
    # Every meta row must have a matching vec row on the join.
    matched = conn.execute(
        "SELECT COUNT(*) FROM news_articles a "
        "JOIN vec_news v ON v.rowid = a.internal_id"
    ).fetchone()[0]
    assert matched == 20
