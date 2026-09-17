"""sqlite-vec backing store for the M17 recommendations index.

Replaces the earlier chromadb-backed index. The chroma variant corrupted
its HNSW segments during a SIGKILL cascade on 2026-09-17, reserved 44 GB
of address space attempting to load the bad header, and threw - the
exception was caught upstream but the reservation stayed pinned. See
docs/INTERVIEW_NOTES.md B14 for the incident writeup.

Design:
  - One sqlite3 file (`settings.rec_store_path`, default `./.rec_store.db`)
  - `recs` table for metadata + doc text (regular sqlite, ACID)
  - `vec_recs` virtual table via sqlite-vec (`vec0`) for embeddings; its
    rowid ties to `recs.rec_id` 1:1
  - No compaction / segment files -> no partial-write corruption class

Shared across arm subprocesses. Each arm sqlite writer opens its own
connection; sqlite journaling handles concurrent access. Retrieval is
read-only from arms' perspective, plus the outcome-sweeper's periodic
UPDATE pass.
"""

from __future__ import annotations

import logging
import sqlite3
import threading
from pathlib import Path

import sqlite_vec

from agentic_investor.config import get_settings

logger = logging.getLogger(__name__)


# Kept in module scope so callers can pass a specific dim in tests; matches
# sentence-transformers/all-MiniLM-L6-v2 which is what tools/news.py loads.
EMBED_DIM = 384


_SCHEMA_SQL_META = """
CREATE TABLE IF NOT EXISTS recs (
    rec_id INTEGER PRIMARY KEY,
    source TEXT NOT NULL,
    created_at TEXT NOT NULL,
    tickers TEXT,
    text TEXT NOT NULL,
    n_positions INTEGER,
    avg_confidence REAL,
    cash_pct REAL,
    risk TEXT,
    db_url TEXT,
    outcome_pl_pct_15m REAL,
    outcome_pl_pct_60m REAL,
    outcome_pl_pct_1d REAL,
    outcome_pl_pct_1w REAL
);
CREATE INDEX IF NOT EXISTS recs_source_idx ON recs(source);
CREATE INDEX IF NOT EXISTS recs_created_at_idx ON recs(created_at);
"""

_SCHEMA_SQL_VEC = (
    f"CREATE VIRTUAL TABLE IF NOT EXISTS vec_recs USING vec0("
    f"embedding float[{EMBED_DIM}])"
)


# One process-wide connection would be simpler but not thread-safe under
# sqlite3's default check_same_thread=True. Arm subprocesses are single-
# threaded on the write path so a per-process singleton is fine; the M17
# outcome-sweeper is a distinct subprocess and gets its own singleton
# there. Guard the init with a lock so first-touch races don't double-
# initialize the schema.
_conn_lock = threading.Lock()
_conn: sqlite3.Connection | None = None


def _init_conn(path: str) -> sqlite3.Connection:
    """Open a sqlite connection, load the vec0 extension, ensure schema."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, isolation_level=None)  # autocommit-per-statement
    # sqlite-vec requires load_extension. On some Python builds this is
    # disabled by default; enable then load, then disable to avoid third-
    # party code loading extensions later.
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    # Reasonable pragmas for a small local file with concurrent readers.
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA busy_timeout=5000")
    # Idempotent schema (two statements because CREATE INDEX can't be in
    # the same executescript block on some sqlite builds when the table
    # was created earlier).
    conn.executescript(_SCHEMA_SQL_META)
    conn.execute(_SCHEMA_SQL_VEC)
    return conn


def get_connection(path: str | None = None) -> sqlite3.Connection:
    """Return the process-wide connection to the rec store.

    Passing an explicit `path` bypasses the singleton and returns a fresh
    connection - useful for tests and for the migration script that
    writes to a different file.
    """
    if path is not None:
        return _init_conn(path)
    global _conn
    with _conn_lock:
        if _conn is None:
            _conn = _init_conn(get_settings().rec_store_path)
        return _conn


def reset_process_connection() -> None:
    """Drop the process-wide connection. Tests use this between runs."""
    global _conn
    with _conn_lock:
        if _conn is not None:
            try:
                _conn.close()
            except Exception:  # noqa: BLE001
                pass
            _conn = None


def open_memory_conn() -> sqlite3.Connection:
    """Return a fresh in-memory connection with schema. Test convenience."""
    return _init_conn(":memory:")
