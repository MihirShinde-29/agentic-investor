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


def _insert_bus_event(
    db_path: Path,
    *,
    ts_received: str,
    ts_published: str,
    symbols: list[str],
    headline: str,
    summary: str,
    url: str,
    source: str,
) -> None:
    """One-shot insert helper - single-column write path shared between
    the websocket handler and the REST backfill.
    """
    with sqlite3.connect(str(db_path)) as conn:
        conn.execute(
            "INSERT INTO bus_events "
            "(ts_received, ts_published, symbols_json, "
            " headline, summary, url, source) "
            "VALUES (?,?,?,?,?,?,?)",
            (
                ts_received, ts_published,
                json.dumps([str(x) for x in symbols]),
                headline, summary, url, source,
            ),
        )


def _backfill_news_since(
    db_path: Path,
    since_iso: str | None,
    until_iso: str,
    *,
    max_window_hours: float = 4.0,
) -> int:
    """REST-query Alpaca News for `[since_iso, until_iso]` and insert
    any missed events. Returns the count inserted.

    `since_iso=None` means "no prior events" - we cap the window at
    `max_window_hours` so a first-start / cold-boot doesn't pull the
    entire historical corpus.
    """
    from datetime import timedelta

    from alpaca.data.historical.news import NewsClient
    from alpaca.data.requests import NewsRequest

    from agentic_investor.config import get_settings

    s = get_settings()
    if not (s.alpaca_api_key and s.alpaca_api_secret):
        return 0
    client = NewsClient(api_key=s.alpaca_api_key, secret_key=s.alpaca_api_secret)

    until_dt = datetime.fromisoformat(until_iso.replace("Z", "+00:00"))
    if since_iso:
        since_dt = datetime.fromisoformat(since_iso.replace("Z", "+00:00"))
    else:
        since_dt = until_dt - timedelta(hours=max_window_hours)
    # Cap: don't chew Alpaca quota on huge windows.
    if until_dt - since_dt > timedelta(hours=max_window_hours):
        since_dt = until_dt - timedelta(hours=max_window_hours)

    req = NewsRequest(
        symbols=None,  # match the wildcard subscription
        start=since_dt,
        end=until_dt,
        limit=50,
    )
    try:
        resp = client.get_news(req)
    except Exception as e:  # noqa: BLE001
        logger.warning("news backfill REST call failed: %s", e)
        return 0

    now_iso = datetime.now(UTC).isoformat()
    n = 0
    # Dedup against what's already in the DB by (headline, ts_published)
    # so backfill after a partial catch doesn't double-insert.
    with sqlite3.connect(str(db_path)) as conn:
        for arts in resp.data.values():
            for a in arts:
                headline = str(getattr(a, "headline", ""))
                if not headline:
                    continue
                ts_pub = str(getattr(a, "created_at", None) or now_iso)
                row = conn.execute(
                    "SELECT 1 FROM bus_events "
                    "WHERE headline = ? AND ts_published = ? LIMIT 1",
                    (headline, ts_pub),
                ).fetchone()
                if row is not None:
                    continue
                symbols = list(getattr(a, "symbols", None) or [])
                _insert_bus_event(
                    db_path,
                    ts_received=now_iso,
                    ts_published=ts_pub,
                    symbols=symbols,
                    headline=headline,
                    summary=str(getattr(a, "summary", "")),
                    url=str(getattr(a, "url", "")),
                    source=str(getattr(a, "source", "")),
                )
                n += 1
    return n


def run_bus_writer(bus_url: str) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        force=True,
    )
    from alpaca.data.live.news import NewsDataStream

    from agentic_investor.config import get_settings
    from agentic_investor.experiments._bus_purge import start_purge_thread
    from agentic_investor.experiments._bus_stream import (
        BusStatus,
        run_with_reconnect,
    )
    from agentic_investor.flags import flags

    s = get_settings()
    db_path = bus_path_from_url(bus_url)
    init_bus_table(db_path)
    status = BusStatus.open(db_path)
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

    async def _on_news(item) -> None:
        symbols = list(getattr(item, "symbols", None) or [])
        published = getattr(item, "created_at", None) or datetime.now(UTC)
        ts_pub = str(published)
        ts_recv = datetime.now(UTC).isoformat()
        _insert_bus_event(
            db_path,
            ts_received=ts_recv,
            ts_published=ts_pub,
            symbols=symbols,
            headline=str(getattr(item, "headline", "")),
            summary=str(getattr(item, "summary", "")),
            url=str(getattr(item, "url", "")),
            source=str(getattr(item, "source", "")),
        )
        status.record_event(ts_pub)

    def _factory():
        return NewsDataStream(
            api_key=s.alpaca_api_key,
            secret_key=s.alpaca_api_secret,
        )

    def _run(stream):
        stream.subscribe_news(_on_news, "*")
        stream.run()

    def _backfill(before_ts_iso, now_iso):
        n = _backfill_news_since(db_path, before_ts_iso, now_iso)
        if n > 0:
            status.record_event(now_iso, count=n)
        return n

    return run_with_reconnect(
        stream_factory=_factory,
        run_stream=_run,
        on_reconnect=_backfill,
        status=status,
        label="news-bus",
    )


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
