"""Shared news bus for parallel experiment arms.

Alpaca's news websocket permits only one concurrent connection per API
key. When N arms each try to open their own subscription, N-1 of them
get "connection limit exceeded". The bus fixes this: one writer owns
the single Alpaca websocket and appends every headline to a shared
SQLite table; every arm's news streamer polls that table instead of
talking to Alpaca directly.

Two moving pieces:
  - `run_bus_writer(bus_url)`  - the one-and-only Alpaca subscriber;
    typically spawned by the experiment runner as its own subprocess.
  - `SharedBusStream` - duck-types Alpaca's NewsDataStream (implements
    `subscribe_news(cb, *symbols)` + `run()`), so the existing
    NewsStreamer wrapper can use it transparently.

Enabled per-arm by setting env `AGENTIC_NEWS_BUS=sqlite:///...`.
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

# Poll cadence for arm-side readers. News is not high-frequency (dozens
# per minute during a busy session), so half-second poll is way faster
# than we need and keeps CPU near-zero.
_READER_POLL_INTERVAL_SEC = 0.5

# How long a reader will wait for the bus DB file to appear before it
# gives up and lets the outer NewsStreamer retry loop take over.
_READER_STARTUP_WAIT_SEC = 30.0


def bus_path_from_url(bus_url: str) -> Path:
    if not bus_url.startswith("sqlite:///"):
        raise ValueError(
            f"only sqlite:/// bus URLs supported (got {bus_url!r})"
        )
    return Path(bus_url.removeprefix("sqlite:///"))


def init_bus_table(db_path: Path) -> None:
    """Create the append-only bus table + enable WAL for concurrent readers."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(str(db_path)) as conn:
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
    """Duck-typed replacement for alpaca News payloads.

    Only the attributes NewsStreamer._on_news actually reads are populated
    (symbols, headline, summary, created_at, url, source). If future code
    reaches for another field, it'll show up as an AttributeError in
    testing rather than silently returning None from the real payload.
    """

    symbols: list[str]
    headline: str
    summary: str
    created_at: str
    url: str
    source: str


def run_bus_writer(bus_url: str) -> int:
    """Blocking. Owns the single Alpaca news websocket for the experiment.

    Uses the primary account's news creds (secondary/tertiary accounts
    aren't news subscribers - they're purely broker routing for the
    per-arm books).
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        force=True,
    )
    from alpaca.data.live.news import NewsDataStream

    from agentic_investor.config import get_settings

    s = get_settings()
    db_path = bus_path_from_url(bus_url)
    init_bus_table(db_path)
    logger.info("news bus writer starting -> %s", db_path)

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
    """Duck-types NewsDataStream so NewsStreamer can use it unchanged.

    On `run()`, tails the shared bus_events table and calls the registered
    subscribe_news callback for each new row. Each reader tracks its own
    last-seen id in memory, so N arms polling the same bus all get every
    headline exactly once.
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
        # We ignore *symbols here because the writer already subscribes to
        # "*" upstream; NewsStreamer's own dedup + fanout logic decides
        # which tickers to forward to the arm's decision pipeline.
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
            # Bus never came up. Raise so NewsStreamer's outer backoff loop
            # will retry - the writer might come online later.
            raise FileNotFoundError(
                f"news bus DB never appeared: {db_path}"
            )
        # NewsStreamer's _on_news is `async`, so we need an event loop to
        # drive it. One loop for the lifetime of run().
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
            # Transient - writer might be initializing the table right
            # now. Skip this tick, next poll will retry.
            logger.debug("news bus poll transient error: %s", e)
            return []
