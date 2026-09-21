"""Tests for the recent-decision verdict-feedback prompt block.

The block reads the arm's own paper_snapshots (latest positions) +
paper_orders (recent buys on held tickers), pairs each with its
current unrealized P&L%, and renders one line per fill with a
Jev-or-fallback verdict label.

Flag-gated behind AGENTIC_JEV_VERDICT_ENABLED so arms with the flag
off (A + C in the current 3-arm A/B) get no extra prompt content.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from agentic_investor.orchestrator.verdict_feedback import (
    _fetch_open_position_fills,
    _latest_positions_from_snapshots,
    build_verdict_feedback_block,
)


def _seed_db(db: Path,
             positions: list[dict] | None,
             orders: list[dict] | None) -> None:
    """Create the two tables the block reads + insert one snapshot +
    N buys. Mirrors the shape paper_store uses at runtime.
    """
    with sqlite3.connect(str(db)) as c:
        c.execute("""
            CREATE TABLE IF NOT EXISTS paper_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                captured_at TEXT NOT NULL,
                account_json TEXT NOT NULL,
                positions_json TEXT NOT NULL
            )
        """)
        c.execute("""
            CREATE TABLE IF NOT EXISTS paper_orders (
                client_order_id TEXT PRIMARY KEY,
                broker_order_id TEXT,
                ticker TEXT NOT NULL,
                side TEXT NOT NULL,
                qty REAL NOT NULL,
                order_type TEXT NOT NULL,
                status TEXT NOT NULL,
                submitted_at TEXT NOT NULL,
                filled_at TEXT,
                filled_avg_price REAL,
                stop_loss REAL,
                take_profit REAL,
                source TEXT,
                rec_id INTEGER,
                triggering_news_ids TEXT
            )
        """)
        if positions is not None:
            c.execute(
                "INSERT INTO paper_snapshots (captured_at, account_json, "
                "positions_json) VALUES (?, ?, ?)",
                ("2026-09-21T20:00:00+00:00",
                 json.dumps({"equity": 100000}),
                 json.dumps(positions)),
            )
        for o in orders or []:
            c.execute(
                "INSERT INTO paper_orders "
                "(client_order_id, ticker, side, qty, order_type, status, "
                "submitted_at, filled_at, filled_avg_price, "
                "triggering_news_ids) "
                "VALUES (?, ?, ?, ?, 'market', 'filled', ?, ?, ?, ?)",
                (
                    o["client_order_id"], o["ticker"], o["side"],
                    o["qty"], o["submitted_at"], o["filled_at"],
                    o["price"], json.dumps(o.get("news_ids") or []),
                ),
            )


# --- _latest_positions_from_snapshots -------------------------------------

def test_latest_positions_reads_most_recent_snapshot(tmp_path):
    db = tmp_path / "arm.db"
    _seed_db(db,
             positions=[
                 {"ticker": "NVDA", "qty": 3.5, "market_value": 700.0,
                  "avg_entry_price": 195.0, "unrealized_pl_pct": 2.4},
             ],
             orders=[])
    got = _latest_positions_from_snapshots(db)
    assert len(got) == 1
    assert got[0]["ticker"] == "NVDA"


def test_latest_positions_empty_when_db_missing(tmp_path):
    assert _latest_positions_from_snapshots(tmp_path / "missing.db") == []


def test_latest_positions_empty_when_no_snapshots(tmp_path):
    db = tmp_path / "arm.db"
    _seed_db(db, positions=None, orders=[])  # tables but no snapshot row
    assert _latest_positions_from_snapshots(db) == []


# --- _fetch_open_position_fills -------------------------------------------

def test_fetch_open_fills_returns_only_buys_on_held(tmp_path):
    db = tmp_path / "arm.db"
    _seed_db(db, positions=[], orders=[
        {"client_order_id": "1", "ticker": "AAPL", "side": "buy",
         "qty": 1.5, "price": 200.0,
         "submitted_at": "2026-09-21T19:30:00+00:00",
         "filled_at": "2026-09-21T19:30:01+00:00", "news_ids": []},
        {"client_order_id": "2", "ticker": "NVDA", "side": "buy",
         "qty": 2.0, "price": 100.0,
         "submitted_at": "2026-09-21T19:35:00+00:00",
         "filled_at": "2026-09-21T19:35:01+00:00", "news_ids": ["N123"]},
        {"client_order_id": "3", "ticker": "AAPL", "side": "sell",
         "qty": 0.5, "price": 205.0,
         "submitted_at": "2026-09-21T19:40:00+00:00",
         "filled_at": "2026-09-21T19:40:01+00:00", "news_ids": []},
    ])
    got = _fetch_open_position_fills(db, {"AAPL", "NVDA"}, limit=10)
    tickers = {r["ticker"].upper() for r in got}
    sides = {r["side"] for r in got}
    assert tickers == {"AAPL", "NVDA"}
    assert sides == {"buy"}
    # NVDA came in later - should be first (ORDER BY filled_at DESC)
    assert got[0]["ticker"] == "NVDA"
    assert got[0]["news_ids"] == ["N123"]


def test_fetch_open_fills_filters_out_unheld(tmp_path):
    db = tmp_path / "arm.db"
    _seed_db(db, positions=[], orders=[
        {"client_order_id": "1", "ticker": "TSLA", "side": "buy",
         "qty": 1.0, "price": 250.0,
         "submitted_at": "2026-09-21T19:30:00+00:00",
         "filled_at": "2026-09-21T19:30:01+00:00", "news_ids": []},
    ])
    # We DON'T hold TSLA anymore.
    got = _fetch_open_position_fills(db, {"AAPL"}, limit=10)
    assert got == []


# --- build_verdict_feedback_block (end-to-end) ----------------------------

def test_block_returns_empty_when_flag_off(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENTIC_JEV_VERDICT_ENABLED", "0")
    db = tmp_path / "arm.db"
    _seed_db(db, positions=[
        {"ticker": "NVDA", "qty": 3.0, "market_value": 900.0,
         "avg_entry_price": 295.0, "unrealized_pl_pct": 1.7},
    ], orders=[
        {"client_order_id": "1", "ticker": "NVDA", "side": "buy",
         "qty": 3.0, "price": 295.0,
         "submitted_at": "2026-09-21T18:00:00+00:00",
         "filled_at": "2026-09-21T18:00:01+00:00", "news_ids": []},
    ])
    assert build_verdict_feedback_block(db) == ""


def test_block_returns_empty_when_no_positions(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENTIC_JEV_VERDICT_ENABLED", "1")
    db = tmp_path / "arm.db"
    _seed_db(db, positions=[], orders=[])
    assert build_verdict_feedback_block(db) == ""


def test_block_returns_empty_when_no_fills_on_held(tmp_path, monkeypatch):
    """Held NVDA but no BUY on NVDA in paper_orders (maybe imported
    position, or fills predate the arm's DB history). Block emits
    nothing rather than a misleading empty section."""
    monkeypatch.setenv("AGENTIC_JEV_VERDICT_ENABLED", "1")
    db = tmp_path / "arm.db"
    _seed_db(db, positions=[
        {"ticker": "NVDA", "qty": 3.0, "market_value": 900.0,
         "avg_entry_price": 295.0, "unrealized_pl_pct": 1.7},
    ], orders=[])
    assert build_verdict_feedback_block(db) == ""


def test_block_renders_lines_for_held_positions(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENTIC_JEV_VERDICT_ENABLED", "1")
    # Jev SDK is optional and may not be importable in CI; falling
    # back to deterministic verdicts is the intended path here so we
    # don't need a mocked client.
    db = tmp_path / "arm.db"
    _seed_db(db, positions=[
        {"ticker": "NVDA", "qty": 3.0, "market_value": 900.0,
         "avg_entry_price": 295.0, "unrealized_pl_pct": 1.7},
        {"ticker": "AAPL", "qty": 4.0, "market_value": 800.0,
         "avg_entry_price": 205.0, "unrealized_pl_pct": -0.8},
    ], orders=[
        {"client_order_id": "1", "ticker": "NVDA", "side": "buy",
         "qty": 3.0, "price": 295.0,
         "submitted_at": "2026-09-21T18:00:00+00:00",
         "filled_at": "2026-09-21T18:00:01+00:00", "news_ids": ["N1abc"]},
        {"client_order_id": "2", "ticker": "AAPL", "side": "buy",
         "qty": 4.0, "price": 205.0,
         "submitted_at": "2026-09-21T19:00:00+00:00",
         "filled_at": "2026-09-21T19:00:01+00:00", "news_ids": []},
    ])
    block = build_verdict_feedback_block(db)
    assert "## 12" in block
    assert "NVDA" in block
    assert "AAPL" in block
    assert "%" in block  # pnl_pct formatted
    # Verdict label from the deterministic labeller must appear
    assert any(lbl in block for lbl in ("WORKING", "WRONG", "UNCLEAR",
                                          "TOO_EARLY"))


def test_block_omits_positions_with_zero_qty(tmp_path, monkeypatch):
    """Broker sometimes reports positions with qty=0 (fully closed
    but not yet purged). Filter them so we don't render verdicts on
    ghost positions.
    """
    monkeypatch.setenv("AGENTIC_JEV_VERDICT_ENABLED", "1")
    db = tmp_path / "arm.db"
    _seed_db(db, positions=[
        {"ticker": "NVDA", "qty": 0, "market_value": 0,
         "avg_entry_price": 0, "unrealized_pl_pct": 0},
    ], orders=[
        {"client_order_id": "1", "ticker": "NVDA", "side": "buy",
         "qty": 3.0, "price": 295.0,
         "submitted_at": "2026-09-21T18:00:00+00:00",
         "filled_at": "2026-09-21T18:00:01+00:00", "news_ids": []},
    ])
    assert build_verdict_feedback_block(db) == ""


def test_block_respects_limit_fills(tmp_path, monkeypatch):
    """limit_fills=2 with 5 fills on the held ticker → only 2 lines."""
    monkeypatch.setenv("AGENTIC_JEV_VERDICT_ENABLED", "1")
    db = tmp_path / "arm.db"
    orders = [
        {"client_order_id": str(i), "ticker": "NVDA", "side": "buy",
         "qty": 1.0, "price": 200.0 + i,
         "submitted_at": f"2026-09-21T19:{i:02d}:00+00:00",
         "filled_at": f"2026-09-21T19:{i:02d}:01+00:00", "news_ids": []}
        for i in range(5)
    ]
    _seed_db(db, positions=[
        {"ticker": "NVDA", "qty": 5.0, "market_value": 1000.0,
         "avg_entry_price": 200.0, "unrealized_pl_pct": 0.0},
    ], orders=orders)
    block = build_verdict_feedback_block(db, limit_fills=2)
    # 2 dashes = 2 lines (one bullet per fill)
    assert block.count("\n-") == 2
