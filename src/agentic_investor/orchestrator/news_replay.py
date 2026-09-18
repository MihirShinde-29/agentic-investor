"""News-replay driver: pump recorded news events back into an arm's
event queue at their recorded cadence.

Complements the existing capture in `tools.news_stream` (line ~260)
which persists every incoming `NewsEvent` to `recording.jsonl` when
`AGENTIC_RECORD_TO` is set. This module closes the loop: on a replay
run (`AGENTIC_REPLAY_FROM=<dir>`), instead of a live Alpaca websocket
we spawn a thread that walks the recorded rows in order, sleeps for
the inter-event interval, and drops reconstructed `NewsEvent`s into
the same queue the live streamer would have used.

Playback speed is controlled by `AGENTIC_REPLAY_NEWS_SPEED`:
    1.0 (default) - real-time playback (respects recorded gaps)
    2.0           - twice real-time
    0             - fire every event as fast as possible
                    (useful for the integration test in task #160)

Why a separate module rather than folding into recorder.py:
- Keeps recorder.py stdlib-only + import-cheap (loop.py imports it
  from the hot path). This module pulls in `NewsEvent` and `queue`.
- Gives us a place to hang playback controls (speed, gate on
  published_at vs received_at) without noising up the capture side.
"""

from __future__ import annotations

import logging
import os
import queue as _q
import threading
import time
from datetime import datetime

logger = logging.getLogger(__name__)

_ENV_SPEED = "AGENTIC_REPLAY_NEWS_SPEED"


def _parse_ts(iso: str | None) -> float | None:
    """Return POSIX seconds for an ISO-8601 stamp or None on parse error.
    Recorded events use `received_at` which the recorder writes with
    `.isoformat()` on a UTC-aware datetime, so parsing is round-trippable.
    """
    if not iso:
        return None
    try:
        s = iso.replace("Z", "+00:00")
        return datetime.fromisoformat(s).timestamp()
    except ValueError:
        return None


def _row_to_event(row: dict):
    """Rebuild a NewsEvent from a recorded row. Lazy import to avoid a
    tools -> orchestrator import cycle at module load."""
    from agentic_investor.tools.news_stream import NewsEvent
    return NewsEvent(
        ticker=row.get("ticker", ""),
        headline=row.get("headline", ""),
        summary=row.get("summary", ""),
        published_at=row.get("published_at", ""),
        received_at=row.get("received_at", ""),
        url=row.get("url", ""),
        source=row.get("source", ""),
    )


def _current_speed() -> float:
    raw = os.environ.get(_ENV_SPEED)
    if raw is None or raw == "":
        return 1.0
    try:
        v = float(raw)
    except ValueError:
        logger.warning("%s=%r not float; using 1.0", _ENV_SPEED, raw)
        return 1.0
    # Negative or non-finite values collapse to real-time; 0 stays 0
    # (as-fast-as-possible sentinel).
    if v < 0 or v != v:  # noqa: PLR0124 - NaN check
        return 1.0
    return v


class NewsReplayDriver:
    """Background thread that pumps recorded news into `event_queue`.

    Threading model matches NewsStreamer: one daemon thread; caller
    holds a stop event via `stop()`. Idempotent start() so a
    reconnect-style retry is a no-op.

    The driver reads its rows via `recorder.iter_recorded_news()` at
    start() time and pins them; subsequent recording appends (unusual,
    but possible under RECORD+REPLAY chaining) are not reflected in
    the current run. Kept simple: replay is a bounded playback, not a
    live tail.
    """

    def __init__(
        self,
        event_queue: _q.Queue,
        *,
        speed: float | None = None,
    ) -> None:
        self.event_queue = event_queue
        self._explicit_speed = speed
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._rows: list[dict] = []
        self._injected = 0

    def _load(self) -> None:
        from agentic_investor.orchestrator.recorder import iter_recorded_news
        self._rows = list(iter_recorded_news())

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._load()
        if not self._rows:
            logger.info(
                "news-replay: no recorded news rows; driver stays idle",
            )
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, name="news-replay", daemon=True,
        )
        self._thread.start()
        logger.info(
            "news-replay: injecting %d recorded rows at speed=%.2f",
            len(self._rows), self._effective_speed(),
        )

    def stop(self) -> None:
        self._stop.set()

    def join(self, timeout: float | None = None) -> None:
        if self._thread is not None:
            self._thread.join(timeout=timeout)

    def _effective_speed(self) -> float:
        return (
            self._explicit_speed if self._explicit_speed is not None
            else _current_speed()
        )

    def _run(self) -> None:
        speed = self._effective_speed()
        prev_wall: float | None = None
        prev_recorded: float | None = None
        for row in self._rows:
            if self._stop.is_set():
                break
            recorded_ts = _parse_ts(row.get("received_at"))
            if speed and prev_recorded is not None and recorded_ts is not None:
                # Wall-clock gap since we injected the previous event
                # subtracted from the recorded gap. Keeps drift bounded
                # when downstream drain is slower than the sleep math.
                gap_recorded = max(recorded_ts - prev_recorded, 0.0)
                gap_wall = time.monotonic() - (prev_wall or time.monotonic())
                sleep_for = (gap_recorded / speed) - gap_wall
                if sleep_for > 0:
                    # Wait on the stop event so shutdown is prompt.
                    if self._stop.wait(timeout=sleep_for):
                        break
            try:
                self.event_queue.put(_row_to_event(row))
                self._injected += 1
            except Exception as e:  # noqa: BLE001
                logger.warning("news-replay: queue put failed: %s", e)
            prev_wall = time.monotonic()
            if recorded_ts is not None:
                prev_recorded = recorded_ts
        logger.info(
            "news-replay: finished (injected %d/%d)",
            self._injected, len(self._rows),
        )

    @property
    def injected(self) -> int:
        return self._injected

    @property
    def total(self) -> int:
        return len(self._rows)


def replay_active() -> bool:
    """True when AGENTIC_REPLAY_FROM is set. Cheap - no I/O."""
    return bool(os.environ.get("AGENTIC_REPLAY_FROM"))
