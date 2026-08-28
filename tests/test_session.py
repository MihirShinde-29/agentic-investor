"""Tests for the session recorder."""

import json

from agentic_investor.ops.session import SessionRecorder


def test_recorder_writes_jsonl_and_counts_events(tmp_path):
    rec = SessionRecorder.start(base_dir=str(tmp_path))
    rec.log("news_received", {"ticker": "AAPL", "headline": "hi"})
    rec.log("news_received", {"ticker": "NVDA", "headline": "hi"})
    rec.log("order_submitted", {"ticker": "AAPL", "side": "buy", "qty": 10})

    lines = rec.jsonl_path.read_text(encoding="utf-8").strip().split("\n")
    # session_start + 3 explicit
    assert len(lines) == 4
    parsed = [json.loads(line) for line in lines]
    assert parsed[0]["event"] == "session_start"
    assert parsed[-1]["event"] == "order_submitted"
    assert rec._counts["news_received"] == 2
    assert rec._counts["order_submitted"] == 1


def test_finalize_writes_summary_markdown(tmp_path):
    rec = SessionRecorder.start(base_dir=str(tmp_path))
    rec.log("news_received", {"ticker": "AAPL"})
    path = rec.finalize()

    md = path.read_text(encoding="utf-8")
    assert "# Session" in md
    assert "news_received" in md
    assert "session_start" in md
