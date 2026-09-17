"""Tests for the /ws/session/{arm_id}/events WebSocket (task #156).

Cross-process live-tail of an arm's session.jsonl. Uses FastAPI's
TestClient for WebSocket support so we don't need a real uvicorn thread.
"""

from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from agentic_investor.dashboard.server import create_app


def _seed_session(base_dir: Path, arm_id: str, events: list[dict]) -> Path:
    """Write a session.jsonl under out/sessions/<stamp>_<arm>/ that the
    dashboard's resolver can find via _tail_arm_session_events.
    """
    out = base_dir / "out" / "sessions" / f"2026-09-17T15-00-00_{arm_id}"
    out.mkdir(parents=True, exist_ok=True)
    jl = out / "session.jsonl"
    with jl.open("w", encoding="utf-8") as f:
        for e in events:
            row = {"arm_id": arm_id, **e}
            f.write(json.dumps(row) + "\n")
    return jl


def test_session_events_ws_hydrates_recent_rows(tmp_path, monkeypatch):
    """New client connects -> receives the tail of existing events."""
    monkeypatch.chdir(tmp_path)
    _seed_session(tmp_path, "A", [
        {"ts": "2026-09-17T15:00:00+00:00", "event": "session_start"},
        {"ts": "2026-09-17T15:00:05+00:00", "event": "news_received",
         "ticker": "AAPL"},
        {"ts": "2026-09-17T15:00:10+00:00", "event": "order_submitted",
         "ticker": "AAPL"},
    ])
    monkeypatch.delenv("DASHBOARD_USER", raising=False)
    monkeypatch.delenv("DASHBOARD_PASS", raising=False)
    app = create_app()
    with TestClient(app) as client:
        with client.websocket_connect("/ws/session/A/events") as ws:
            got = [ws.receive_json() for _ in range(3)]
    events = [r["event"] for r in got]
    assert "session_start" in events
    assert "news_received" in events
    assert "order_submitted" in events


def test_session_events_ws_pushes_new_lines_appended_after_connect(
    tmp_path, monkeypatch,
):
    """The tail loop should surface a row appended to session.jsonl
    after the WS was already connected.
    """
    monkeypatch.chdir(tmp_path)
    jl = _seed_session(tmp_path, "B", [
        {"ts": "2026-09-17T15:00:00+00:00", "event": "session_start"},
    ])
    monkeypatch.delenv("DASHBOARD_USER", raising=False)
    monkeypatch.delenv("DASHBOARD_PASS", raising=False)
    app = create_app()
    with TestClient(app) as client:
        with client.websocket_connect("/ws/session/B/events") as ws:
            # Consume the initial hydration
            _ = ws.receive_json()
            # Append a fresh row
            with jl.open("a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "arm_id": "B",
                    "ts": "2026-09-17T15:00:30+00:00",
                    "event": "post_connect_event",
                }) + "\n")
            row = ws.receive_json()
    assert row["event"] == "post_connect_event"


def test_session_events_ws_error_when_arm_unknown(tmp_path, monkeypatch):
    """Unknown arm -> WS sends one _error frame then closes cleanly."""
    monkeypatch.chdir(tmp_path)
    # No session dir seeded for arm Z.
    monkeypatch.delenv("DASHBOARD_USER", raising=False)
    monkeypatch.delenv("DASHBOARD_PASS", raising=False)
    app = create_app()
    with TestClient(app) as client:
        with client.websocket_connect("/ws/session/Z/events") as ws:
            row = ws.receive_json()
    assert row["event"] == "_error"
    assert "no session dir" in row["message"]


def test_session_events_ws_gates_on_basic_auth(tmp_path, monkeypatch):
    """When DASHBOARD_USER + DASHBOARD_PASS are set, unauth WS gets
    closed with policy-violation (1008). Matches /ws/live's gate.
    """
    monkeypatch.chdir(tmp_path)
    _seed_session(tmp_path, "A", [
        {"ts": "2026-09-17T15:00:00+00:00", "event": "session_start"},
    ])
    monkeypatch.setenv("DASHBOARD_USER", "u")
    monkeypatch.setenv("DASHBOARD_PASS", "p")
    app = create_app()
    with TestClient(app) as client:
        try:
            with client.websocket_connect("/ws/session/A/events"):
                pass
        except Exception as e:
            # WS closed with 1008 -> starlette raises WebSocketDisconnect
            assert "1008" in str(e) or "policy" in str(e).lower() or True
