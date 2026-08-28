"""In-memory pub/sub bus for streaming session events to WebSocket clients.

The paper loop already writes each event to a JSONL file. This module adds a
parallel in-process fanout: the loop calls `publish()` (thread-safe from any
worker thread), and every subscribed asyncio.Queue receives a copy. The
FastAPI WebSocket handler subscribes one queue per client.

A ring buffer of recent events is kept so late-connecting clients see the last
N events immediately (typical for a browser refresh mid-session).
"""

from __future__ import annotations

import asyncio
import logging
import threading
from collections import deque
from typing import Any

logger = logging.getLogger(__name__)


class EventBus:
    """Cross-thread pub/sub. Publishers run in any thread; subscribers are asyncio.Queues."""

    def __init__(self, buffer_size: int = 500):
        self._subscribers: set[asyncio.Queue[dict[str, Any]]] = set()
        self._recent: deque[dict[str, Any]] = deque(maxlen=buffer_size)
        self._lock = threading.Lock()
        self._loop: asyncio.AbstractEventLoop | None = None

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Attach the FastAPI event loop so cross-thread puts can be scheduled."""
        self._loop = loop

    def subscribe(self) -> asyncio.Queue[dict[str, Any]]:
        """Register a new WS client's queue and preload it with recent history."""
        q: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        with self._lock:
            self._subscribers.add(q)
            for evt in list(self._recent):
                q.put_nowait(evt)
        return q

    def unsubscribe(self, q: asyncio.Queue[dict[str, Any]]) -> None:
        with self._lock:
            self._subscribers.discard(q)

    def publish(self, event: dict[str, Any]) -> None:
        """Thread-safe publish. Called from any worker thread."""
        with self._lock:
            self._recent.append(event)
            subs = list(self._subscribers)
            loop = self._loop
        if loop is None or not subs:
            return
        for q in subs:
            try:
                loop.call_soon_threadsafe(q.put_nowait, event)
            except RuntimeError:
                # Loop is closed / shutting down; drop silently.
                pass

    def recent(self, limit: int = 100) -> list[dict[str, Any]]:
        """Snapshot of the last N events for REST hydration."""
        with self._lock:
            data = list(self._recent)
        return data[-limit:]


_bus = EventBus()


def get_bus() -> EventBus:
    return _bus
