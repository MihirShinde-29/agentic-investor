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
    # Daily bar cache: past-day rows are immutable, so we keep them forever
    # and only refetch today's row (still moving) + any gaps. Correlation
    # compute pulls the same 60d history per ticker on every regen without
    # this table.
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS daily_bars (
            ticker TEXT NOT NULL,
            date TEXT NOT NULL,
            close REAL NOT NULL,
            PRIMARY KEY (ticker, date)
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


def reconcile_orders(broker, *, limit: int = 200, url: str | None = None) -> int:
    """Poll the broker for recent orders and update our mirror's fill status.

    Alpaca is source of truth for state. We only see status="pending_new" at
    submission time; fills, partial fills, cancels, rejects arrive
    asynchronously. This closes the gap so paper-report shows real fill
    prices instead of "-".

    Returns count of rows whose state ACTUALLY changed. SQLite's UPDATE
    rowcount counts matched-rows not changed-rows, so we add IS-NOT-DISTINCT
    guards to only touch a row when at least one field would differ.
    Otherwise every reconcile "updated" every mirrored order, spamming the
    dashboard once per poll with a fake count of 96.
    """
    updated = 0
    try:
        remote = broker.list_orders(limit=limit, status="all")
    except Exception:  # noqa: BLE001 - broker outage mustn't crash reconcile
        return 0
    if not remote:
        return 0
    with _connect(_resolve_url(url)) as conn:
        for o in remote:
            coid = getattr(o, "client_order_id", None) or ""
            if not coid:
                continue
            new_status = getattr(o, "status", None)
            new_filled_at = getattr(o, "filled_at", None)
            new_price = getattr(o, "filled_avg_price", None)
            new_id = getattr(o, "id", None)
            cur = conn.execute(
                """
                UPDATE paper_orders SET
                    status = ?, filled_at = ?, filled_avg_price = ?,
                    broker_order_id = COALESCE(NULLIF(broker_order_id, ''), ?)
                WHERE client_order_id = ?
                  AND (
                    status IS NOT ?
                    OR filled_at IS NOT ?
                    OR filled_avg_price IS NOT ?
                    OR (broker_order_id IS NULL OR broker_order_id = '')
                  )
                """,
                (
                    new_status, new_filled_at, new_price, new_id,
                    coid,
                    new_status, new_filled_at, new_price,
                ),
            )
            if cur.rowcount > 0:
                updated += cur.rowcount
        conn.commit()
    return updated


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


def recent_sold_tickers(
    *, since_iso: str, url: str | None = None,
) -> list[str]:
    """Return distinct tickers that had a SELL trade after `since_iso`.

    Used by the correlation-hint block so the allocator sees that a
    ticker it's about to buy correlates with something we recently exited
    (hidden factor re-exposure).
    """
    with _connect(_resolve_url(url)) as conn:
        rows = conn.execute(
            "SELECT DISTINCT UPPER(ticker) FROM paper_orders "
            "WHERE side = 'sell' AND submitted_at >= ?",
            (since_iso,),
        ).fetchall()
    return [r[0] for r in rows]


def recent_trades_for_tickers(
    tickers: list[str],
    *,
    since_iso: str | None = None,
    per_ticker_limit: int = 3,
    url: str | None = None,
) -> dict[str, list[dict]]:
    """Return {ticker: [most-recent-trade, ...]} for the given tickers.

    Used by the allocator prompt to give the LLM its own recent trade
    history per ticker so it can factor in "I just sold this 3 min ago"
    when deciding whether to flip.
    """
    if not tickers:
        return {}
    out: dict[str, list[dict]] = {t.upper(): [] for t in tickers}
    with _connect(_resolve_url(url)) as conn:
        conn.row_factory = sqlite3.Row
        for t in {t.upper() for t in tickers}:
            query = (
                "SELECT ticker, side, qty, submitted_at, filled_at, "
                "filled_avg_price, status FROM paper_orders "
                "WHERE UPPER(ticker) = ?"
            )
            params: list = [t]
            if since_iso:
                query += " AND submitted_at >= ?"
                params.append(since_iso)
            query += " ORDER BY submitted_at DESC LIMIT ?"
            params.append(per_ticker_limit)
            rows = conn.execute(query, params).fetchall()
            out[t] = [dict(r) for r in rows]
    return out


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


def load_cached_daily_closes(
    ticker: str,
    start_date: str,
    end_date: str,
    *,
    url: str | None = None,
) -> dict[str, float]:
    """Return {YYYY-MM-DD: close} for a ticker over an inclusive date range.

    Empty dict if nothing cached. Caller is responsible for filling gaps
    from the provider and calling save_daily_closes to persist new bars.
    """
    with _connect(_resolve_url(url)) as conn:
        rows = conn.execute(
            "SELECT date, close FROM daily_bars "
            "WHERE ticker = ? AND date >= ? AND date <= ? "
            "ORDER BY date",
            (ticker.upper(), start_date, end_date),
        ).fetchall()
    return {r[0]: float(r[1]) for r in rows}


def save_daily_closes(
    ticker: str,
    closes: dict[str, float],
    *,
    url: str | None = None,
) -> int:
    """Upsert (ticker, date, close) rows. Returns row count written."""
    if not closes:
        return 0
    rows = [(ticker.upper(), d, float(c)) for d, c in closes.items()]
    with _connect(_resolve_url(url)) as conn:
        conn.executemany(
            "INSERT INTO daily_bars (ticker, date, close) VALUES (?, ?, ?) "
            "ON CONFLICT(ticker, date) DO UPDATE SET close = excluded.close",
            rows,
        )
        conn.commit()
    return len(rows)
