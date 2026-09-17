"""sqlite-vec backing store for the news-article collection.

Second half of the 2026-09-17 chromadb retirement (first half was
`memory/store.py` for the M17 recommendations index). Same schema
pattern: metadata table + vec0 virtual table joined on rowid.

Kept in `tools/` (not `memory/`) because news articles are an ingestion-
side asset, not a decision-side memory. Different lifecycle, different
TTL policy, no cross-arm A/B invariant to enforce.
"""

from __future__ import annotations

import logging
import sqlite3
import threading
from pathlib import Path

import sqlite_vec

from agentic_investor.config import get_settings

logger = logging.getLogger(__name__)


# Matches sentence-transformers/all-MiniLM-L6-v2 (the embed model that
# tools/news.py loads and that ml_service serves).
EMBED_DIM = 384


_SCHEMA_SQL_META = """
CREATE TABLE IF NOT EXISTS news_articles (
    -- Internal integer PK so vec0's rowid can join cleanly. External `id`
    -- from the provider (Alpaca/Benzinga UUIDs, feed-native ids) is a
    -- separate column with a UNIQUE constraint for idempotent upserts.
    internal_id INTEGER PRIMARY KEY AUTOINCREMENT,
    id TEXT NOT NULL UNIQUE,
    ticker TEXT NOT NULL,
    headline TEXT NOT NULL,
    summary TEXT,
    source TEXT,
    url TEXT,
    published_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS news_articles_ticker_idx ON news_articles(ticker);
CREATE INDEX IF NOT EXISTS news_articles_published_at_idx
    ON news_articles(published_at);
"""

_SCHEMA_SQL_VEC = (
    f"CREATE VIRTUAL TABLE IF NOT EXISTS vec_news USING vec0("
    f"embedding float[{EMBED_DIM}])"
)


_conn_lock = threading.Lock()
_conn: sqlite3.Connection | None = None


def _init_conn(path: str) -> sqlite3.Connection:
    """Open a sqlite connection, load vec0, ensure schema."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, isolation_level=None)
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA busy_timeout=5000")
    conn.executescript(_SCHEMA_SQL_META)
    conn.execute(_SCHEMA_SQL_VEC)
    return conn


def get_connection(path: str | None = None) -> sqlite3.Connection:
    """Return the process-wide news-store connection.

    Passing an explicit `path` returns a fresh connection instead of the
    singleton - used by tests and the retrieval-eval ephemeral harness.
    """
    if path is not None:
        return _init_conn(path)
    global _conn
    with _conn_lock:
        if _conn is None:
            _conn = _init_conn(get_settings().news_store_path)
        return _conn


def reset_process_connection() -> None:
    global _conn
    with _conn_lock:
        if _conn is not None:
            try:
                _conn.close()
            except Exception:  # noqa: BLE001
                pass
            _conn = None


def open_memory_conn() -> sqlite3.Connection:
    """Fresh in-memory conn with schema. Test convenience."""
    return _init_conn(":memory:")
