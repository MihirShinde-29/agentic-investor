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
import re
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime

from agentic_investor.config import get_settings

logger = logging.getLogger(__name__)


# Cashtag + parenthesized symbols only - bare all-caps standalones would need
# a known-ticker gate to avoid false positives like "AI" / "CEO".
_CASHTAG_RE = re.compile(r"\$([A-Z]{1,5})\b")
# Matches (TSM), (NVDA), (NASDAQ:AAPL), (NYSE:BRK.B) etc.
_PAREN_RE = re.compile(r"\((?:[A-Z]{2,10}:)?([A-Z]{1,5})(?:\.[A-Z])?\)")


def extract_tickers_from_text(text: str) -> set[str]:
    """Pull ticker mentions from a news body (cashtag + parenthesized only)."""
    if not text:
        return set()
    tickers = set(_CASHTAG_RE.findall(text))
    tickers.update(_PAREN_RE.findall(text))
    return tickers


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
    """Background-thread wrapper over alpaca-py's NewsDataStream.

    Pass tickers=["*"] to subscribe to ALL news (wildcard); tickers filter
    is disabled and every event reaches the queue with whatever symbols
    Alpaca tagged.
    """

    def __init__(
        self,
        tickers: list[str],
        *,
        event_queue: queue.Queue[NewsEvent],
        stream_factory: Callable | None = None,
    ):
        self.wildcard = "*" in tickers
        self.tickers = [] if self.wildcard else [t.upper() for t in tickers]
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
        backoff = 1.0
        while not self._stop.is_set():
            try:
                stream = self._build_stream()
                subs = ("*",) if self.wildcard else tuple(self.tickers)
                stream.subscribe_news(self._on_news, *subs)
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
        symbols = getattr(item, "symbols", None) or []
        headline = str(getattr(item, "headline", ""))
        summary = str(getattr(item, "summary", ""))
        published = getattr(item, "created_at", None) or datetime.now(UTC)
        url = str(getattr(item, "url", ""))
        source = str(getattr(item, "source", ""))

        # 0p: extract additional tickers from the body that Alpaca may not
        # have tagged (secondary beneficiaries named in the article).
        extra = extract_tickers_from_text(f"{headline}\n{summary}")
        all_symbols = list(dict.fromkeys(
            [str(s).upper() for s in symbols] + list(extra)
        ))

        for sym in all_symbols:
            # Non-wildcard mode: still filter to subscribed tickers.
            if not self.wildcard and self.tickers and sym not in self.tickers:
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
