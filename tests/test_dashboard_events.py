"""Tests for the dashboard event bus + FastAPI endpoints."""

from __future__ import annotations

import time
import urllib.request

from agentic_investor.dashboard.events import EventBus, get_bus
from agentic_investor.dashboard.server import serve_in_thread


def test_bus_publish_delivers_to_late_subscriber_from_ring_buffer():
    bus = EventBus(buffer_size=10)
    bus.publish({"ts": "t1", "event": "a"})
    bus.publish({"ts": "t2", "event": "b"})
    # No loop bound -> no fanout, but the ring buffer still holds them.
    recent = bus.recent(limit=10)
    assert [e["event"] for e in recent] == ["a", "b"]


def test_bus_recent_respects_limit():
    bus = EventBus(buffer_size=100)
    for i in range(20):
        bus.publish({"ts": str(i), "event": f"e{i}"})
    recent = bus.recent(limit=5)
    assert len(recent) == 5
    assert recent[-1]["event"] == "e19"


def test_ring_buffer_drops_oldest_when_full():
    bus = EventBus(buffer_size=3)
    for i in range(5):
        bus.publish({"ts": str(i), "event": f"e{i}"})
    recent = bus.recent(limit=10)
    assert [e["event"] for e in recent] == ["e2", "e3", "e4"]


def test_server_health_endpoint_returns_ok():
    # Use a unique port to avoid collisions across test runs.
    port = 8791
    serve_in_thread(port=port)
    # Give uvicorn a moment to bind.
    time.sleep(1.5)
    r = urllib.request.urlopen(f"http://localhost:{port}/api/health")
    body = r.read().decode()
    assert '"status":"ok"' in body


def test_events_endpoint_returns_recent_publications():
    port = 8792
    serve_in_thread(port=port)
    time.sleep(1.5)
    get_bus().publish({"ts": "x", "event": "unit-test-marker"})
    time.sleep(0.2)
    r = urllib.request.urlopen(f"http://localhost:{port}/api/events?limit=200")
    body = r.read().decode()
    assert "unit-test-marker" in body
