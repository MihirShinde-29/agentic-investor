"""Shared Alpaca news feed for N-arm experiments.

Alpaca allows one concurrent news websocket per API key. A single writer
subscribes and appends every headline to SQLite; each arm's NewsStreamer
uses SharedBusStream (duck-types NewsDataStream) to poll that table.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
import threading
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

logger = logging.getLogger(__name__)

_READER_POLL_INTERVAL_SEC = 0.5
_READER_STARTUP_WAIT_SEC = 30.0


def bus_path_from_url(bus_url: str) -> Path:
    if not bus_url.startswith("sqlite:///"):
        raise ValueError(
            f"only sqlite:/// bus URLs supported (got {bus_url!r})"
        )
    return Path(bus_url.removeprefix("sqlite:///"))


def init_bus_table(db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(str(db_path)) as conn:
        # WAL so N readers don't block the writer or each other.
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS bus_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts_received TEXT NOT NULL,
                ts_published TEXT NOT NULL,
                symbols_json TEXT NOT NULL,
                headline TEXT NOT NULL,
                summary TEXT NOT NULL DEFAULT '',
                url TEXT NOT NULL DEFAULT '',
                source TEXT NOT NULL DEFAULT ''
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS ix_bus_id ON bus_events(id)"
        )


@dataclass
class _BusItem:
    symbols: list[str]
    headline: str
    summary: str
    created_at: str
    url: str
    source: str


def run_bus_writer(bus_url: str) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        force=True,
    )
    from alpaca.data.live.news import NewsDataStream

    from agentic_investor.config import get_settings
    from agentic_investor.experiments._bus_purge import start_purge_thread
    from agentic_investor.flags import flags

    s = get_settings()
    db_path = bus_path_from_url(bus_url)
    init_bus_table(db_path)
    logger.info("news bus writer starting -> %s", db_path)

    # TTL purge: drop bus_events older than NEWS_BUS_TTL_HOURS (default
    # 4h - 4x the STALE cutoff used by the decision engine's
    # render_batch_context, comfortable margin). Sweeps every 5 min.
    news_bus_ttl_h = flags.NEWS_BUS_TTL_HOURS

    def _purge_bus(conn):
        # Use ts_received (writer-side clock, always sane) rather than
        # ts_published (which is provider-set, occasionally weird).
        from datetime import UTC, datetime, timedelta
        cutoff = (
            datetime.now(UTC) - timedelta(hours=news_bus_ttl_h)
        ).isoformat()
        cur = conn.execute(
            "DELETE FROM bus_events WHERE ts_received < ?", (cutoff,),
        )
        return cur.rowcount

    start_purge_thread(
        db_path, _purge_bus,
        interval_sec=300.0,  # 5 min
        label="news_bus",
    )

    # Also purge the news_articles + vec_news store owned by tools/
    # news.py (post-#146). Uses NEWS_STORE_TTL_DAYS, default 30 (news
    # recency for RAG matters most in the near term; older articles are
    # diluted by fresher signal anyway).
    news_store_ttl_days = flags.NEWS_STORE_TTL_DAYS
    news_store_path = Path(s.news_store_path)

    def _purge_news_store(conn):
        # Two-table cleanup: DELETE metadata rows first, then remove
        # their vec_news counterparts by joined-on internal_id. Doing it
        # in one transaction is fine since the writer thread owns the
        # only other lock and it's tiny per-insert.
        from datetime import UTC, datetime, timedelta
        cutoff = (
            datetime.now(UTC) - timedelta(days=news_store_ttl_days)
        ).isoformat()
        # Collect internal_ids we're about to delete so vec_news can drop
        # them by rowid (sqlite-vec vec0 tables don't support subquery
        # DELETE the same way regular tables do).
        rows = conn.execute(
            "SELECT internal_id FROM news_articles WHERE published_at < ?",
            (cutoff,),
        ).fetchall()
        if not rows:
            return 0
        ids = [r[0] for r in rows]
        # DELETE metadata first
        placeholders = ",".join("?" * len(ids))
        conn.execute(
            f"DELETE FROM news_articles WHERE internal_id IN ({placeholders})",
            ids,
        )
        # DELETE vec rows one-by-one; vec0 accepts single-rowid DELETEs.
        for iid in ids:
            conn.execute("DELETE FROM vec_news WHERE rowid = ?", (iid,))
        return len(ids)

    # The news store is a separate sqlite file with a vec0 virtual table,
    # so each purge connection needs sqlite_vec.load() before it can
    # touch vec_news. start_purge_thread opens its own connection per
    # sweep so no cross-file locking.
    import sqlite_vec

    def _load_vec(conn):
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)

    if news_store_path.exists():
        start_purge_thread(
            news_store_path, _purge_news_store,
            interval_sec=3600.0,  # once an hour is plenty for a days-TTL
            label="news_store",
            conn_setup=_load_vec,
        )
    else:
        logger.info(
            "news store path not present yet (%s); skipping TTL thread",
            news_store_path,
        )

    stream = NewsDataStream(
        api_key=s.alpaca_api_key,
        secret_key=s.alpaca_api_secret,
    )

    async def _on_news(item) -> None:
        symbols = getattr(item, "symbols", None) or []
        published = getattr(item, "created_at", None) or datetime.now(UTC)
        row = (
            datetime.now(UTC).isoformat(),
            str(published),
            json.dumps([str(x) for x in symbols]),
            str(getattr(item, "headline", "")),
            str(getattr(item, "summary", "")),
            str(getattr(item, "url", "")),
            str(getattr(item, "source", "")),
        )
        with sqlite3.connect(str(db_path)) as conn:
            conn.execute(
                "INSERT INTO bus_events "
                "(ts_received, ts_published, symbols_json, "
                " headline, summary, url, source) "
                "VALUES (?,?,?,?,?,?,?)",
                row,
            )

    stream.subscribe_news(_on_news, "*")
    stream.run()
    return 0


class SharedBusStream:
    """Duck-types NewsDataStream: subscribe_news + run.

    Each reader tracks its own last-seen id so N arms polling the same
    bus each get every headline exactly once.
    """

    def __init__(
        self,
        bus_url: str,
        *,
        poll_interval: float = _READER_POLL_INTERVAL_SEC,
    ):
        self._url = bus_url
        self._poll = poll_interval
        self._callback = None
        self._stop = threading.Event()
        self._last_id = 0

    def subscribe_news(self, cb, *_symbols) -> None:
        # Writer already subscribes to "*"; NewsStreamer's own dedup +
        # fanout decides which tickers reach the arm's decision pipeline.
        self._callback = cb

    def stop(self) -> None:
        self._stop.set()

    def run(self) -> None:
        if self._callback is None:
            raise RuntimeError(
                "SharedBusStream.run called before subscribe_news"
            )
        db_path = bus_path_from_url(self._url)
        deadline = time.monotonic() + _READER_STARTUP_WAIT_SEC
        while not db_path.exists() and time.monotonic() < deadline:
            if self._stop.wait(0.5):
                return
        if not db_path.exists():
            raise FileNotFoundError(
                f"news bus DB never appeared: {db_path}"
            )
        # NewsStreamer._on_news is async, so we need an event loop.
        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            self._poll_loop(db_path, loop)
        finally:
            loop.close()

    def _poll_loop(self, db_path: Path, loop) -> None:
        while not self._stop.is_set():
            rows = self._fetch_new_rows(db_path)
            for row in rows:
                id_, _ts_recv, ts_pub, symj, headline, summary, url, source = row
                try:
                    symbols = json.loads(symj) if symj else []
                except json.JSONDecodeError:
                    symbols = []
                item = _BusItem(
                    symbols=[str(x) for x in symbols],
                    headline=headline,
                    summary=summary,
                    created_at=ts_pub,
                    url=url,
                    source=source,
                )
                loop.run_until_complete(self._callback(item))
                self._last_id = id_
            if self._stop.wait(self._poll):
                return

    def _fetch_new_rows(self, db_path: Path) -> list[tuple]:
        try:
            with sqlite3.connect(str(db_path)) as conn:
                return conn.execute(
                    "SELECT id, ts_received, ts_published, symbols_json, "
                    "headline, summary, url, source "
                    "FROM bus_events WHERE id > ? ORDER BY id",
                    (self._last_id,),
                ).fetchall()
        except sqlite3.OperationalError as e:
            # Writer may be mid-init; next poll retries.
            logger.debug("news bus poll transient error: %s", e)
            return []
