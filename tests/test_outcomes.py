"""Multi-horizon outcome attribution for indexed recommendations."""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime, timedelta

import chromadb

from agentic_investor.memory.outcomes import (
    attach_outcomes_to_index,
    compute_outcomes_for_rec,
)


def _seed_db(path):
    """Create a DB with the schema we need for outcome computation."""
    conn = sqlite3.connect(str(path))
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS recommendations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            payload_json TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS paper_snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            captured_at TEXT NOT NULL,
            account_json TEXT NOT NULL,
            positions_json TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS daily_bars (
            ticker TEXT NOT NULL,
            date TEXT NOT NULL,
            close REAL NOT NULL,
            PRIMARY KEY (ticker, date)
        );
        """
    )
    conn.commit()
    return conn


def _insert_snapshot(conn, captured_at: datetime, equity: float):
    conn.execute(
        "INSERT INTO paper_snapshots (captured_at, account_json, positions_json) "
        "VALUES (?, ?, ?)",
        (
            captured_at.isoformat(),
            json.dumps({"equity": equity, "cash": 0, "portfolio_value": equity}),
            "[]",
        ),
    )
    conn.commit()


def _insert_bar(conn, ticker: str, date: str, close: float):
    conn.execute(
        "INSERT OR REPLACE INTO daily_bars (ticker, date, close) VALUES (?, ?, ?)",
        (ticker, date, close),
    )
    conn.commit()


def _payload(positions: list[tuple[str, float]]) -> dict:
    """Minimal rec payload for outcome tests."""
    return {
        "allocation": {
            "positions": [
                {"ticker": t, "weight_pct": w, "dollars": 100.0,
                 "rationale": "test", "confidence": 0.7}
                for t, w in positions
            ],
            "cash_pct": 0.0,
            "cash_dollars": 0.0,
            "portfolio_rationale": "test rationale",
        },
        "request": {
            "tickers": [t for t, _ in positions], "amount": 10000.0,
            "risk": "moderate", "target": "12-month growth",
        },
    }


def test_intraday_equity_delta_from_snapshots(tmp_path):
    db = tmp_path / "seed.db"
    conn = _seed_db(db)
    rec_time = datetime(2026, 9, 1, 10, 0, 0, tzinfo=UTC)
    _insert_snapshot(conn, rec_time, equity=100_000.0)
    _insert_snapshot(conn, rec_time + timedelta(minutes=15), equity=101_000.0)
    _insert_snapshot(conn, rec_time + timedelta(minutes=60), equity=100_500.0)

    payload = _payload([("AAPL", 100.0)])
    result = compute_outcomes_for_rec(
        payload, rec_time.isoformat(), f"sqlite:///{db}",
    )
    assert result["outcome_pl_pct_15m"] == 1.0
    assert result["outcome_pl_pct_60m"] == 0.5
    assert result["outcome_available"] is True


def test_missing_snapshot_returns_none(tmp_path):
    db = tmp_path / "seed.db"
    conn = _seed_db(db)
    rec_time = datetime(2026, 9, 1, 10, 0, 0, tzinfo=UTC)
    _insert_snapshot(conn, rec_time, equity=100_000.0)

    payload = _payload([("AAPL", 100.0)])
    result = compute_outcomes_for_rec(
        payload, rec_time.isoformat(), f"sqlite:///{db}",
    )
    assert result["outcome_pl_pct_15m"] is None
    assert result["outcome_pl_pct_60m"] is None


def test_daily_bars_weighted_move(tmp_path):
    db = tmp_path / "seed.db"
    conn = _seed_db(db)
    rec_time = datetime(2026, 9, 1, 10, 0, 0, tzinfo=UTC)
    _insert_bar(conn, "AAPL", "2026-09-01", 100.0)
    _insert_bar(conn, "AAPL", "2026-09-02", 102.0)   # +2%
    _insert_bar(conn, "MSFT", "2026-09-01", 400.0)
    _insert_bar(conn, "MSFT", "2026-09-02", 396.0)   # -1%

    # 50/50 split; weighted = (2 + -1) / 2 = 0.5
    payload = _payload([("AAPL", 50.0), ("MSFT", 50.0)])
    result = compute_outcomes_for_rec(
        payload, rec_time.isoformat(), f"sqlite:///{db}",
    )
    assert result["outcome_pl_pct_1d"] == 0.5


def test_weekend_rolls_to_next_trading_day(tmp_path):
    db = tmp_path / "seed.db"
    conn = _seed_db(db)
    # Friday rec, +1 day = Saturday (no bar). Should roll to Monday.
    rec_time = datetime(2026, 9, 4, 10, 0, 0, tzinfo=UTC)  # Fri
    _insert_bar(conn, "AAPL", "2026-09-04", 100.0)
    _insert_bar(conn, "AAPL", "2026-09-07", 103.0)  # Mon +3%

    payload = _payload([("AAPL", 100.0)])
    result = compute_outcomes_for_rec(
        payload, rec_time.isoformat(), f"sqlite:///{db}",
    )
    assert result["outcome_pl_pct_1d"] == 3.0


def test_insufficient_weight_data_returns_none(tmp_path):
    """If <50% of allocation weight has forward bar data, outcome is None."""
    db = tmp_path / "seed.db"
    conn = _seed_db(db)
    rec_time = datetime(2026, 9, 1, 10, 0, 0, tzinfo=UTC)
    # AAPL 70% weight, but no forward bar. MSFT 30%, has bars.
    _insert_bar(conn, "AAPL", "2026-09-01", 100.0)
    _insert_bar(conn, "MSFT", "2026-09-01", 400.0)
    _insert_bar(conn, "MSFT", "2026-09-02", 408.0)

    payload = _payload([("AAPL", 70.0), ("MSFT", 30.0)])
    result = compute_outcomes_for_rec(
        payload, rec_time.isoformat(), f"sqlite:///{db}",
    )
    # Only 30% weight has data → below threshold, return None
    assert result["outcome_pl_pct_1d"] is None


def test_attach_outcomes_updates_chroma_metadata(tmp_path):
    db = tmp_path / "seed.db"
    conn = _seed_db(db)
    rec_time = datetime(2026, 9, 1, 10, 0, 0, tzinfo=UTC)

    # Persist a rec
    payload = _payload([("AAPL", 100.0)])
    conn.execute(
        "INSERT INTO recommendations (created_at, payload_json) VALUES (?, ?)",
        (rec_time.isoformat(), json.dumps(payload)),
    )
    conn.commit()
    _insert_snapshot(conn, rec_time, equity=100_000.0)
    _insert_snapshot(conn, rec_time + timedelta(minutes=60), equity=100_500.0)
    _insert_bar(conn, "AAPL", "2026-09-01", 100.0)
    _insert_bar(conn, "AAPL", "2026-09-02", 101.0)

    # Seed Chroma with the rec (mimics M17.A having run)
    client = chromadb.PersistentClient(path=str(tmp_path / "chroma"))
    coll = client.get_or_create_collection(
        name="recommendations", metadata={"hnsw:space": "cosine"},
    )
    coll.upsert(
        ids=["rec:historical:1"],
        embeddings=[[0.1, 0.2, 0.3, 0.4]],
        documents=["test doc"],
        metadatas=[{
            "rec_id": 1,
            "source": "historical",
            "created_at": rec_time.isoformat(),
            "tickers": "AAPL",
            "n_positions": 1,
            "avg_confidence": 0.7,
            "cash_pct": 0.0,
            "risk": "moderate",
        }],
    )

    n_updated, n_with = attach_outcomes_to_index(
        db_url=f"sqlite:///{db}", collection=coll,
    )
    assert n_updated == 1
    assert n_with == 1

    # Confirm outcomes made it into the metadata
    res = coll.get(ids=["rec:historical:1"])
    meta = res["metadatas"][0]
    assert meta["outcome_pl_pct_60m"] == 0.5
    assert meta["outcome_pl_pct_1d"] == 1.0
    assert meta["outcome_available"] is True
    # 15m had no snapshot → sentinel
    assert meta["outcome_pl_pct_15m"] == -9999.0


def test_attach_outcomes_empty_collection_is_noop(tmp_path):
    db = tmp_path / "seed.db"
    _seed_db(db)
    client = chromadb.PersistentClient(path=str(tmp_path / "chroma"))
    coll = client.get_or_create_collection(
        name="recommendations", metadata={"hnsw:space": "cosine"},
    )
    n_updated, n_with = attach_outcomes_to_index(
        db_url=f"sqlite:///{db}", collection=coll,
    )
    assert (n_updated, n_with) == (0, 0)


def test_malformed_created_at_returns_unavailable(tmp_path):
    db = tmp_path / "seed.db"
    _seed_db(db)
    payload = _payload([("AAPL", 100.0)])
    result = compute_outcomes_for_rec(payload, "not-a-timestamp", f"sqlite:///{db}")
    assert result["outcome_available"] is False
    assert result["outcome_pl_pct_15m"] is None
