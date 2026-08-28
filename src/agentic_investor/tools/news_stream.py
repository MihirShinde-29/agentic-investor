"""Alpaca news websocket streamer running in a background thread.

Subscribes to real-time news for a fixed set of tickers, drops each event into
a threadsafe queue, and reconnects on socket drops. The main loop drains the
queue at each decision-moment check.

We intentionally keep the streamer dumb: it does not decide, batch, or fire
the LLM. The decision engine in loop.py owns micro-batching + phase logic.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime

from agentic_investor.config import get_settings

logger = logging.getLogger(__name__)


@dataclass
class NewsEvent:
    ticker: str
    headline: str
    summary: str
    published_at: str  # ISO 8601 UTC
    received_at: str  # ISO 8601 UTC (when the streamer got it)
    url: str = ""
    source: str = ""


class NewsStreamer:
    """Background-thread wrapper over alpaca-py's NewsDataStream."""

    def __init__(
        self,
        tickers: list[str],
        *,
        event_queue: queue.Queue[NewsEvent],
        stream_factory: Callable | None = None,
    ):
        self.tickers = [t.upper() for t in tickers]
        self.event_queue = event_queue
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._stream_factory = stream_factory  # for tests

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="news-stream", daemon=True)
        self._thread.start()
        logger.info("news streamer started for %s", ",".join(self.tickers))

    def stop(self) -> None:
        self._stop.set()

    def update_tickers(self, tickers: list[str]) -> None:
        """Live-update the subscribed set. Restart is simplest + robust."""
        self.tickers = [t.upper() for t in tickers]
        self.stop()
        time.sleep(0.5)
        self.start()

    def _run(self) -> None:
        # Reconnect loop: any exception in the stream tries again after backoff.
        backoff = 1.0
        while not self._stop.is_set():
            try:
                stream = self._build_stream()
                stream.subscribe_news(self._on_news, *self.tickers)
                stream.run()
                backoff = 1.0
            except Exception as e:  # noqa: BLE001
                logger.warning("news stream error: %s; retrying in %.1fs", e, backoff)
                self._stop.wait(backoff)
                backoff = min(backoff * 2, 60.0)

    def _build_stream(self):
        if self._stream_factory is not None:
            return self._stream_factory()
        from alpaca.data.live.news import NewsDataStream

        s = get_settings()
        return NewsDataStream(api_key=s.alpaca_api_key, secret_key=s.alpaca_api_secret)

    async def _on_news(self, item) -> None:
        # alpaca-py sends symbols as a list per item.
        symbols = getattr(item, "symbols", None) or []
        headline = str(getattr(item, "headline", ""))
        summary = str(getattr(item, "summary", ""))
        published = getattr(item, "created_at", None) or datetime.now(UTC)
        url = str(getattr(item, "url", ""))
        source = str(getattr(item, "source", ""))
        for sym in symbols:
            sym = str(sym).upper()
            if self.tickers and sym not in self.tickers:
                continue
            evt = NewsEvent(
                ticker=sym,
                headline=headline,
                summary=summary,
                published_at=str(published),
                received_at=datetime.now(UTC).isoformat(),
                url=url,
                source=source,
            )
            self.event_queue.put(evt)
