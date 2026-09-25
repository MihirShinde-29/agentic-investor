"""Tests for the opening-intent prompt block."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from agentic_investor.orchestrator.opening_intent import (
    build_opening_intent_block,
)


def _write_session(base: Path, arm: str, rows: list[dict]) -> Path:
    """Helper: create out/sessions/<today>T00-00-00_<arm>/session.jsonl
    under `base` and drop the rows in.
    """
    today = datetime.now(UTC).strftime("%Y-%m-%d")
    sess = base / "out" / "sessions" / f"{today}T00-00-00_{arm}"
    sess.mkdir(parents=True, exist_ok=True)
    path = sess / "session.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return path


def test_opening_intent_empty_when_no_holds(tmp_path, monkeypatch):
    """No pre_market_hold events -> empty block (so post-open regens
    without pre-market activity don't get a stray section)."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("AGENTIC_ARM_ID", "X")
    _write_session(tmp_path, "X", [
        {"ts": datetime.now(UTC).isoformat(), "event": "regen_done"},
    ])
    assert build_opening_intent_block() == ""


def test_opening_intent_renders_plans(tmp_path, monkeypatch):
    """Held plans render as an ## 13 block with side + qty per plan."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("AGENTIC_ARM_ID", "Y")
    now = datetime.now(UTC).isoformat()
    _write_session(tmp_path, "Y", [
        {
            "ts": now, "event": "pre_market_hold", "rec_id": 42,
            "plans": [
                {"ticker": "MSFT", "side": "buy", "qty": 5.0},
                {"ticker": "GOOG", "side": "sell", "qty": 2.5},
            ],
        },
    ])
    block = build_opening_intent_block()
    assert "## 13. Opening intent" in block
    assert "rec 42" in block
    assert "BUY 5.0 MSFT" in block
    assert "SELL 2.5 GOOG" in block


def test_opening_intent_dedupes_across_relaunches(tmp_path, monkeypatch):
    """Two session dirs for the same arm today (arm restart) should
    NOT double-count the same hold event."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("AGENTIC_ARM_ID", "Z")
    today = datetime.now(UTC).strftime("%Y-%m-%d")
    now = datetime.now(UTC).isoformat()
    row = {
        "ts": now, "event": "pre_market_hold", "rec_id": 1,
        "plans": [{"ticker": "AAPL", "side": "buy", "qty": 3.0}],
    }
    for stamp in ("T00-00-00_Z", "T01-00-00_Z"):
        d = tmp_path / "out" / "sessions" / f"{today}{stamp}"
        d.mkdir(parents=True)
        (d / "session.jsonl").write_text(json.dumps(row) + "\n")
    block = build_opening_intent_block()
    # rec 1 should appear exactly once despite being in both sessions
    assert block.count("rec 1") == 1


def test_opening_intent_caps_at_3_most_recent(tmp_path, monkeypatch):
    """Older intents go stale as prices move - keep only 3 most recent."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("AGENTIC_ARM_ID", "W")
    rows = []
    for i in range(5):
        # Progressive timestamps within today so ordering is stable
        ts = datetime.now(UTC).replace(hour=9, minute=i).isoformat()
        rows.append({
            "ts": ts, "event": "pre_market_hold", "rec_id": i,
            "plans": [{"ticker": "X", "side": "buy", "qty": 1.0}],
        })
    _write_session(tmp_path, "W", rows)
    block = build_opening_intent_block()
    # Only recs 2, 3, 4 (the last 3) should be present
    assert "rec 0" not in block
    assert "rec 1" not in block
    assert "rec 2" in block
    assert "rec 3" in block
    assert "rec 4" in block


def test_opening_intent_never_raises(tmp_path, monkeypatch):
    """Malformed session.jsonl must not break the prompt path."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("AGENTIC_ARM_ID", "V")
    today = datetime.now(UTC).strftime("%Y-%m-%d")
    d = tmp_path / "out" / "sessions" / f"{today}T00-00-00_V"
    d.mkdir(parents=True)
    (d / "session.jsonl").write_text("not json\n{partial\n")
    # Must return "" not raise
    result = build_opening_intent_block()
    assert isinstance(result, str)


def test_opening_intent_handles_empty_plans(tmp_path, monkeypatch):
    """Old-format hold events without a `plans` field should be
    silently skipped (no crash, no empty entries)."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("AGENTIC_ARM_ID", "U")
    now = datetime.now(UTC).isoformat()
    _write_session(tmp_path, "U", [
        {"ts": now, "event": "pre_market_hold", "rec_id": 1,
         "plan_count": 3, "tickers": ["AAPL", "MSFT", "GOOG"]},
    ])
    # No `plans` key -> renders as empty (nothing to display)
    assert build_opening_intent_block() == ""
