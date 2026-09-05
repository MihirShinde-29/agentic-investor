"""Multi-horizon outcome attribution for indexed recommendations.

For each rec: portfolio-weighted P/L at 15m, 60m, 1D, 1W after created_at.
Short horizons use paper_snapshots equity delta (actual realized).
Long horizons use daily_bars per-ticker close weighted by allocation
(synthetic - reflects the rec's thesis regardless of filter execution).
"""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)

# Snapshots are captured on loop ticks (~3-5min apart during active
# sessions). If no snapshot lands within this many minutes of the target
# horizon, treat the outcome as unavailable rather than pick a stale one.
_SNAPSHOT_TOLERANCE_MIN = 8


def _db_path(url: str) -> Path:
    if not url.startswith("sqlite:///"):
        raise ValueError(f"only sqlite:/// URLs supported (got {url!r})")
    return Path(url.removeprefix("sqlite:///"))


def _parse_ts(iso: str) -> datetime | None:
    try:
        return datetime.fromisoformat(iso.replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None


def _equity_at(conn: sqlite3.Connection, target: datetime) -> tuple[datetime, float] | None:
    """Snapshot with captured_at closest to target within _SNAPSHOT_TOLERANCE_MIN.

    Returns (captured_at, equity) or None. Considers snapshots on either
    side of `target` so we can find the nearest match.
    """
    lo = (target - timedelta(minutes=_SNAPSHOT_TOLERANCE_MIN)).isoformat()
    hi = (target + timedelta(minutes=_SNAPSHOT_TOLERANCE_MIN)).isoformat()
    rows = conn.execute(
        "SELECT captured_at, account_json FROM paper_snapshots "
        "WHERE captured_at BETWEEN ? AND ? ORDER BY captured_at",
        (lo, hi),
    ).fetchall()
    if not rows:
        return None
    best = None
    best_dt_delta: float | None = None
    for captured_at, account_json in rows:
        ts = _parse_ts(captured_at)
        if ts is None:
            continue
        try:
            equity = float(json.loads(account_json).get("equity") or 0.0)
        except (json.JSONDecodeError, TypeError, ValueError):
            continue
        if equity <= 0:
            continue
        delta = abs((ts - target).total_seconds())
        if best_dt_delta is None or delta < best_dt_delta:
            best = (ts, equity)
            best_dt_delta = delta
    return best


def _compute_intraday_outcomes(
    conn: sqlite3.Connection, rec_time: datetime, horizons_min: list[int],
) -> dict[int, float | None]:
    """Portfolio equity pct-change at each horizon from paper_snapshots."""
    baseline = _equity_at(conn, rec_time)
    if baseline is None:
        return {h: None for h in horizons_min}
    _, base_equity = baseline
    out: dict[int, float | None] = {}
    for h in horizons_min:
        later = _equity_at(conn, rec_time + timedelta(minutes=h))
        if later is None:
            out[h] = None
            continue
        _, later_equity = later
        out[h] = round((later_equity / base_equity - 1) * 100, 3)
    return out


def _close_on_or_after(
    conn: sqlite3.Connection, ticker: str, target_date: str,
) -> tuple[str, float] | None:
    """First daily close on or after target_date for ticker (handles weekends/holidays)."""
    row = conn.execute(
        "SELECT date, close FROM daily_bars "
        "WHERE ticker=? AND date>=? ORDER BY date LIMIT 1",
        (ticker.upper(), target_date),
    ).fetchone()
    if row is None:
        return None
    return (row[0], float(row[1]))


def _compute_daily_outcomes(
    conn: sqlite3.Connection,
    rec_payload: dict,
    rec_time: datetime,
    horizons_days: list[int],
) -> dict[int, float | None]:
    """Weight per-ticker daily close move by allocation weight_pct."""
    positions = list((rec_payload.get("allocation") or {}).get("positions") or [])
    if not positions:
        return {h: None for h in horizons_days}
    base_date = rec_time.date().isoformat()
    base_closes: dict[str, tuple[str, float]] = {}
    for p in positions:
        ticker = str(p.get("ticker") or "").upper()
        if not ticker:
            continue
        base = _close_on_or_after(conn, ticker, base_date)
        if base is not None:
            base_closes[ticker] = base

    out: dict[int, float | None] = {}
    for days in horizons_days:
        target = (rec_time + timedelta(days=days)).date().isoformat()
        weighted_move = 0.0
        weight_total = 0.0
        for p in positions:
            ticker = str(p.get("ticker") or "").upper()
            weight = float(p.get("weight_pct") or 0.0)
            if not ticker or weight <= 0 or ticker not in base_closes:
                continue
            base = base_closes[ticker]
            later = _close_on_or_after(conn, ticker, target)
            if later is None or later[0] == base[0]:
                # No forward bar yet (same-day dupe or missing data).
                continue
            move_pct = (later[1] / base[1] - 1) * 100
            weighted_move += move_pct * weight
            weight_total += weight
        # If less than half the target weight has forward data, don't
        # report a partial outcome - it would misrepresent the rec.
        if weight_total < 50.0:
            out[days] = None
        else:
            out[days] = round(weighted_move / weight_total, 3)
    return out


def compute_outcomes_for_rec(
    rec_payload: dict, created_at: str, db_url: str,
) -> dict:
    """Return the outcome-metadata dict to merge into the Chroma doc."""
    rec_time = _parse_ts(created_at)
    if rec_time is None:
        return {
            "outcome_pl_pct_15m": None,
            "outcome_pl_pct_60m": None,
            "outcome_pl_pct_1d": None,
            "outcome_pl_pct_1w": None,
            "outcome_available": False,
        }
    path = _db_path(db_url)
    with sqlite3.connect(str(path)) as conn:
        intraday = _compute_intraday_outcomes(conn, rec_time, [15, 60])
        daily = _compute_daily_outcomes(conn, rec_payload, rec_time, [1, 7])
    result = {
        "outcome_pl_pct_15m": intraday.get(15),
        "outcome_pl_pct_60m": intraday.get(60),
        "outcome_pl_pct_1d": daily.get(1),
        "outcome_pl_pct_1w": daily.get(7),
    }
    result["outcome_available"] = any(v is not None for v in result.values())
    return result


def attach_outcomes_to_index(
    db_url: str | None = None, *, collection=None,
) -> tuple[int, int]:
    """Update every historical Chroma doc with computed outcome metadata.

    Returns (n_updated, n_with_any_outcome).
    """
    from agentic_investor.config import get_settings
    from agentic_investor.memory.rec_index import _default_collection

    resolved_url = db_url or get_settings().database_url
    path = _db_path(resolved_url)
    coll = collection if collection is not None else _default_collection()

    # Chroma metadata field values can be null (None) but the KEY set is
    # unioned across all docs on read, so we need to pull docs then update
    # one-at-a-time (chromadb.update requires the id + full metadata).
    result = coll.get(where={"source": "historical"})
    ids = result.get("ids") or []
    existing_metas = result.get("metadatas") or []
    if not ids:
        logger.info("no historical docs in Chroma; run memory-index --historical first")
        return (0, 0)

    with sqlite3.connect(str(path)) as conn:
        rec_rows = {
            rec_id: (created_at, payload_json)
            for rec_id, created_at, payload_json in conn.execute(
                "SELECT id, created_at, payload_json FROM recommendations"
            )
        }

    n_updated = 0
    n_with_outcome = 0
    for doc_id, meta in zip(ids, existing_metas, strict=False):
        rec_id = int(meta.get("rec_id") or 0)
        if rec_id not in rec_rows:
            continue
        created_at, payload_json = rec_rows[rec_id]
        try:
            payload = json.loads(payload_json)
        except json.JSONDecodeError:
            continue
        outcomes = compute_outcomes_for_rec(payload, created_at, resolved_url)
        merged = {**meta, **outcomes}
        # Chroma requires scalar (str/int/float/bool) - convert None to a
        # sentinel we can filter on later. Using -9999.0 signals "no data"
        # without conflicting with real P/L magnitudes.
        for k in list(merged.keys()):
            if merged[k] is None:
                merged[k] = -9999.0 if k.startswith("outcome_pl_pct_") else False
        coll.update(ids=[doc_id], metadatas=[merged])
        n_updated += 1
        if outcomes.get("outcome_available"):
            n_with_outcome += 1
    logger.info(
        "attached outcomes to %d docs (%d have at least one horizon)",
        n_updated, n_with_outcome,
    )
    return (n_updated, n_with_outcome)
