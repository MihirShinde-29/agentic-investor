"""Multi-horizon outcome attribution for indexed recommendations (sqlite-vec)."""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime, timedelta

from agentic_investor.memory.outcomes import (
    attach_outcomes_to_index,
    compute_outcomes_for_rec,
)
from agentic_investor.memory.store import open_memory_conn


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


def _any_available(outcomes: dict) -> bool:
    """Historical helper: after we dropped the outcome_available flag,
    availability is just 'any horizon is not None'.
    """
    return any(v is not None for v in outcomes.values())


def _seed_rec_row(conn, rec_id: int, created_at: str, source: str = "historical",
                  db_url: str | None = None):
    """Insert a bare row into the rec store's recs table (skip vec_recs;
    the outcomes sweep doesn't need embeddings)."""
    conn.execute(
        """
        INSERT INTO recs (
            rec_id, source, created_at, tickers, text, n_positions,
            avg_confidence, cash_pct, risk, db_url
        ) VALUES (?, ?, ?, 'AAPL', 'test doc', 1, 0.7, 0.0, 'moderate', ?)
        """,
        (rec_id, source, created_at, db_url),
    )


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
    assert _any_available(result) is True


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


def test_attach_outcomes_updates_recs_row(tmp_path):
    db = tmp_path / "seed.db"
    src_conn = _seed_db(db)
    rec_time = datetime(2026, 9, 1, 10, 0, 0, tzinfo=UTC)

    # Persist a rec in the source DB
    payload = _payload([("AAPL", 100.0)])
    src_conn.execute(
        "INSERT INTO recommendations (created_at, payload_json) VALUES (?, ?)",
        (rec_time.isoformat(), json.dumps(payload)),
    )
    src_conn.commit()
    _insert_snapshot(src_conn, rec_time, equity=100_000.0)
    _insert_snapshot(src_conn, rec_time + timedelta(minutes=60), equity=100_500.0)
    _insert_bar(src_conn, "AAPL", "2026-09-01", 100.0)
    _insert_bar(src_conn, "AAPL", "2026-09-02", 101.0)

    # Seed the rec store with the row (mimics M17.A having run)
    store = open_memory_conn()
    _seed_rec_row(store, rec_id=1, created_at=rec_time.isoformat())

    n_updated, n_with = attach_outcomes_to_index(
        db_url=f"sqlite:///{db}", conn=store,
    )
    assert n_updated == 1
    assert n_with == 1

    row = store.execute(
        "SELECT outcome_pl_pct_15m, outcome_pl_pct_60m, outcome_pl_pct_1d, "
        "outcome_pl_pct_1w FROM recs WHERE rec_id = 1"
    ).fetchone()
    pl_15m, pl_60m, pl_1d, pl_1w = row
    assert pl_60m == 0.5
    assert pl_1d == 1.0
    # 15m had no snapshot → NULL (no more -9999.0 sentinel)
    assert pl_15m is None
    # 1w has no future bars either → NULL
    assert pl_1w is None


def test_attach_outcomes_sweeps_arm_sources_and_uses_per_rec_db(tmp_path):
    """Arm recs stash their own db_url in the row; the sweep must read
    snapshots from THAT db, not the caller default."""
    arm_db = tmp_path / "arm_A.db"
    conn_a = _seed_db(arm_db)
    rec_time = datetime(2026, 9, 1, 10, 0, 0, tzinfo=UTC)
    payload = _payload([("AAPL", 100.0)])
    conn_a.execute(
        "INSERT INTO recommendations (created_at, payload_json) VALUES (?, ?)",
        (rec_time.isoformat(), json.dumps(payload)),
    )
    conn_a.commit()
    _insert_snapshot(conn_a, rec_time, equity=100_000.0)
    _insert_snapshot(conn_a, rec_time + timedelta(minutes=60), equity=101_000.0)

    # Caller default DB is empty; the arm row's db_url is what should
    # route the outcome lookup.
    default_db = tmp_path / "default.db"
    _seed_db(default_db)

    store = open_memory_conn()
    _seed_rec_row(store, rec_id=1, created_at=rec_time.isoformat(),
                  source="arm_A", db_url=f"sqlite:///{arm_db}")

    n_updated, n_with = attach_outcomes_to_index(
        db_url=f"sqlite:///{default_db}", conn=store,
    )
    assert n_updated == 1
    assert n_with == 1
    row = store.execute(
        "SELECT outcome_pl_pct_60m FROM recs WHERE rec_id = 1"
    ).fetchone()
    assert row[0] == 1.0


def test_attach_outcomes_empty_store_is_noop(tmp_path):
    db = tmp_path / "seed.db"
    _seed_db(db)
    store = open_memory_conn()
    n_updated, n_with = attach_outcomes_to_index(
        db_url=f"sqlite:///{db}", conn=store,
    )
    assert (n_updated, n_with) == (0, 0)


def test_malformed_created_at_returns_unavailable(tmp_path):
    db = tmp_path / "seed.db"
    _seed_db(db)
    payload = _payload([("AAPL", 100.0)])
    result = compute_outcomes_for_rec(payload, "not-a-timestamp", f"sqlite:///{db}")
    assert _any_available(result) is False
    assert result["outcome_pl_pct_15m"] is None
