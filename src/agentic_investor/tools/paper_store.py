"""SQLite audit trail for paper-trading orders and P&L snapshots.

The broker (Alpaca) is the source of truth for account/position state. This
mirror exists for offline reports, drift analysis, and reconciliation checks -
useful when you want to answer "what did the AI decide at 4pm on Tuesday"
without a round-trip to Alpaca.

Two tables:
- paper_orders: one row per submitted order, keyed by our client_order_id.
- paper_snapshots: periodic account+positions snapshots for equity-curve plots.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

from agentic_investor.config import get_settings
from agentic_investor.tools.paper_broker import (
    PaperAccount,
    PaperOrder,
    PaperPosition,
)


def _db_path(url: str) -> Path:
    if not url.startswith("sqlite:///"):
        raise ValueError(f"only sqlite:/// URLs supported for now (got {url!r})")
    return Path(url.removeprefix("sqlite:///"))


def _connect(url: str) -> sqlite3.Connection:
    path = _db_path(url)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute(
        """
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
            rec_id INTEGER
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS paper_snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            captured_at TEXT NOT NULL,
            account_json TEXT NOT NULL,
            positions_json TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS loop_state (
            account_key TEXT PRIMARY KEY,
            saved_at TEXT NOT NULL,
            state_json TEXT NOT NULL
        )
        """
    )
    # Attribution log: every filter skip records the would-be allocation so
    # we can later simulate the counterfactual and measure false-positive rate.
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS filter_skips (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            skipped_at TEXT NOT NULL,
            skip_reason TEXT NOT NULL,
            trigger_reason TEXT,
            avg_drift_pp REAL,
            max_delta_pp REAL,
            max_delta_ticker TEXT,
            n_tickers INTEGER,
            prev_rec_id INTEGER,
            would_be_allocation_json TEXT NOT NULL,
            actual_positions_json TEXT NOT NULL,
            equity_at_skip REAL,
            deltas_json TEXT
        )
        """
    )
    return conn


def _resolve_url(url: str | None) -> str:
    return url if url is not None else get_settings().database_url


def record_order(
    order: PaperOrder,
    *,
    source: str = "manual",
    rec_id: int | None = None,
    url: str | None = None,
) -> None:
    """Upsert an order row. Idempotent by client_order_id."""
    with _connect(_resolve_url(url)) as conn:
        conn.execute(
            """
            INSERT INTO paper_orders (
                client_order_id, broker_order_id, ticker, side, qty,
                order_type, status, submitted_at, filled_at, filled_avg_price,
                stop_loss, take_profit, source, rec_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(client_order_id) DO UPDATE SET
                broker_order_id=excluded.broker_order_id,
                status=excluded.status,
                filled_at=excluded.filled_at,
                filled_avg_price=excluded.filled_avg_price
            """,
            (
                order.client_order_id,
                order.id,
                order.ticker,
                order.side,
                order.qty,
                order.order_type,
                order.status,
                order.submitted_at,
                order.filled_at,
                order.filled_avg_price,
                order.stop_loss,
                order.take_profit,
                source,
                rec_id,
            ),
        )
        conn.commit()


def list_orders(
    *, limit: int = 50, url: str | None = None
) -> list[dict]:
    with _connect(_resolve_url(url)) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM paper_orders ORDER BY submitted_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [dict(r) for r in rows]


def record_filter_skip(
    *,
    skip_reason: str,
    trigger_reason: str | None,
    would_be_allocation: dict,
    actual_positions: list[dict],
    equity_at_skip: float,
    stats: dict | None = None,
    deltas: dict | None = None,
    prev_rec_id: int | None = None,
    url: str | None = None,
) -> int:
    """Record a filter skip + its would-be allocation for counterfactual eval."""
    stats = stats or {}
    with _connect(_resolve_url(url)) as conn:
        cur = conn.execute(
            """
            INSERT INTO filter_skips (
                skipped_at, skip_reason, trigger_reason,
                avg_drift_pp, max_delta_pp, max_delta_ticker, n_tickers,
                prev_rec_id, would_be_allocation_json, actual_positions_json,
                equity_at_skip, deltas_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                datetime.now(UTC).isoformat(),
                skip_reason,
                trigger_reason,
                stats.get("avg_drift"),
                stats.get("max_delta"),
                stats.get("max_delta_ticker"),
                stats.get("n_tickers"),
                prev_rec_id,
                json.dumps(would_be_allocation, default=str),
                json.dumps(actual_positions, default=str),
                float(equity_at_skip),
                json.dumps(deltas or {}, default=str),
            ),
        )
        conn.commit()
        return int(cur.lastrowid)


def list_filter_skips(*, limit: int = 100, url: str | None = None) -> list[dict]:
    with _connect(_resolve_url(url)) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM filter_skips ORDER BY skipped_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [dict(r) for r in rows]


def save_loop_state(
    state_dict: dict, *, account_key: str = "default", url: str | None = None
) -> None:
    """Upsert LoopState so it survives crashes / restarts.

    account_key will distinguish concurrent loops for M13 A/B testing;
    for now everyone uses 'default'.
    """
    with _connect(_resolve_url(url)) as conn:
        conn.execute(
            """
            INSERT INTO loop_state (account_key, saved_at, state_json)
            VALUES (?, ?, ?)
            ON CONFLICT(account_key) DO UPDATE SET
                saved_at=excluded.saved_at,
                state_json=excluded.state_json
            """,
            (account_key, datetime.now(UTC).isoformat(), json.dumps(state_dict)),
        )
        conn.commit()


def load_loop_state(
    *, account_key: str = "default", url: str | None = None
) -> dict | None:
    """Return the last-saved loop state dict, or None if none exists."""
    with _connect(_resolve_url(url)) as conn:
        row = conn.execute(
            "SELECT state_json FROM loop_state WHERE account_key = ?",
            (account_key,),
        ).fetchone()
    if row is None:
        return None
    return json.loads(row[0])


def record_snapshot(
    account: PaperAccount,
    positions: list[PaperPosition],
    *,
    url: str | None = None,
) -> int:
    """Persist a point-in-time account + positions snapshot for later P&L plots."""
    with _connect(_resolve_url(url)) as conn:
        cur = conn.execute(
            """
            INSERT INTO paper_snapshots (captured_at, account_json, positions_json)
            VALUES (?, ?, ?)
            """,
            (
                datetime.now(UTC).isoformat(),
                json.dumps(account.__dict__),
                json.dumps([p.__dict__ for p in positions]),
            ),
        )
        conn.commit()
        return int(cur.lastrowid)


def list_snapshots(*, limit: int = 100, url: str | None = None) -> list[dict]:
    with _connect(_resolve_url(url)) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT id, captured_at, account_json, positions_json "
            "FROM paper_snapshots ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        d["account"] = json.loads(d.pop("account_json"))
        d["positions"] = json.loads(d.pop("positions_json"))
        out.append(d)
    return out
