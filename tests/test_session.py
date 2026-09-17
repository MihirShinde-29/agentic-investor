"""Tests for the session recorder."""

import json
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

from pydantic import BaseModel

from agentic_investor.ops.session import (
    SessionRecorder,
    _safe_json_payload,
    iter_events,
)


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


# --- _safe_json_payload -----------------------------------------------------


class _Sample(BaseModel):
    ticker: str
    weight: float


def test_sanitizer_pydantic_model_serializes_via_model_dump():
    m = _Sample(ticker="AAPL", weight=0.25)
    out = _safe_json_payload({"pos": m})
    assert out["pos"] == {"ticker": "AAPL", "weight": 0.25}
    # Round-trip through json.dumps must succeed.
    json.dumps(out)


def test_sanitizer_datetime_to_iso():
    ts = datetime(2026, 9, 17, 15, 30, tzinfo=UTC)
    out = _safe_json_payload({"ts": ts})
    assert out["ts"].startswith("2026-09-17T15:30")
    json.dumps(out)


def test_sanitizer_path_to_str():
    p = Path("out") / "x.log"
    out = _safe_json_payload({"log": p})
    assert isinstance(out["log"], str)
    # Path.__str__ uses os.sep, so accept either separator.
    assert "x.log" in out["log"]


def test_sanitizer_decimal_to_float():
    out = _safe_json_payload({"cash": Decimal("100.50")})
    assert isinstance(out["cash"], float)
    assert out["cash"] == 100.5


def test_sanitizer_set_and_tuple_to_list():
    out = _safe_json_payload({"tickers": {"AAPL", "MSFT"}, "shape": (1, 2)})
    assert isinstance(out["tickers"], list)
    assert set(out["tickers"]) == {"AAPL", "MSFT"}
    assert out["shape"] == [1, 2]


def test_sanitizer_nested_mix():
    m = _Sample(ticker="NVDA", weight=0.5)
    payload = {
        "positions": [m, m],
        "meta": {
            "ts": datetime(2026, 9, 17, tzinfo=UTC),
            "path": Path("a") / "b",
            "cash": Decimal("50"),
            "excluded": frozenset({"XYZ"}),
        },
    }
    out = _safe_json_payload(payload)
    assert out["positions"] == [
        {"ticker": "NVDA", "weight": 0.5},
        {"ticker": "NVDA", "weight": 0.5},
    ]
    assert out["meta"]["ts"].startswith("2026-09-17T")
    assert isinstance(out["meta"]["path"], str)
    assert out["meta"]["cash"] == 50.0
    assert out["meta"]["excluded"] == ["XYZ"]
    json.dumps(out)


def test_sanitizer_never_raises_on_weird_type():
    class _Weird:
        # Pydantic-shaped decoy: has model_dump but it explodes.
        def model_dump(self, mode=None):
            raise RuntimeError("nope")

    out = _safe_json_payload({"w": _Weird()})
    # Falls through to str() via the outer try/except.
    assert isinstance(out["w"], str)


def test_recorder_serializes_pydantic_payload(tmp_path):
    rec = SessionRecorder.start(base_dir=str(tmp_path))
    rec.log("test_event", {"model": _Sample(ticker="AAPL", weight=0.3)})
    rows = list(iter_events(rec.out_dir))
    hit = [r for r in rows if r.get("event") == "test_event"]
    assert len(hit) == 1
    assert hit[0]["model"] == {"ticker": "AAPL", "weight": 0.3}


# --- iter_events -----------------------------------------------------------


def test_iter_events_reads_all_rows(tmp_path):
    rec = SessionRecorder.start(base_dir=str(tmp_path))
    rec.log("news_received", {"ticker": "AAPL"})
    rec.log("order_submitted", {"ticker": "AAPL"})
    rec.log("news_received", {"ticker": "MSFT"})

    rows = list(iter_events(rec.out_dir))
    # session_start + 3 explicit
    assert len(rows) == 4
    events = [r["event"] for r in rows]
    assert events.count("news_received") == 2
    assert events.count("order_submitted") == 1


def test_iter_events_filters_by_event_type(tmp_path):
    rec = SessionRecorder.start(base_dir=str(tmp_path))
    rec.log("news_received", {"ticker": "AAPL"})
    rec.log("order_submitted", {"ticker": "AAPL"})
    rows = list(iter_events(rec.out_dir, event_types=["order_submitted"]))
    assert len(rows) == 1
    assert rows[0]["ticker"] == "AAPL"


def test_iter_events_filters_by_since_ts(tmp_path):
    rec = SessionRecorder.start(base_dir=str(tmp_path))
    rec.log("first", {})
    # Bump timestamp cutoff to now-ish so both `session_start` and `first`
    # get filtered out.
    cutoff = datetime.now(UTC).isoformat()
    # Sleep so the next log's ts is definitely after cutoff.
    import time
    time.sleep(0.01)
    rec.log("second", {})
    rec.log("third", {})
    rows = list(iter_events(rec.out_dir, since_ts=cutoff))
    events = [r["event"] for r in rows]
    assert "first" not in events
    assert events == ["second", "third"]


def test_iter_events_missing_file_yields_nothing(tmp_path):
    rows = list(iter_events(tmp_path / "does-not-exist"))
    assert rows == []


def test_iter_events_skips_malformed_lines(tmp_path):
    """A partially-written jsonl (mid-crash) must not raise; malformed
    lines are logged at DEBUG and skipped.
    """
    (tmp_path / "session.jsonl").write_text(
        '{"ts": "2026-01-01T00:00:00", "event": "a"}\n'
        '{"malformed\n'  # truncated
        '{"ts": "2026-01-01T00:00:01", "event": "b"}\n',
        encoding="utf-8",
    )
    rows = list(iter_events(tmp_path))
    events = [r["event"] for r in rows]
    assert events == ["a", "b"]


def test_iter_events_accepts_direct_jsonl_path(tmp_path):
    """Calling with the file path directly (not the containing dir)
    should also work - some callers have the exact path from a picker.
    """
    (tmp_path / "session.jsonl").write_text(
        '{"ts": "2026-01-01T00:00:00", "event": "x"}\n',
        encoding="utf-8",
    )
    rows = list(iter_events(tmp_path / "session.jsonl"))
    assert [r["event"] for r in rows] == ["x"]
