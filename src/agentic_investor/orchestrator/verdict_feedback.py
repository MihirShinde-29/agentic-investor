"""Recent-decision verdict-feedback block for the allocator prompt.

Renders one line per recent fill: `SIDE qty TICKER Nm ago @ $entry ->
+X.YY% now: VERDICT`. Fed into `fast_sections` (post-cache-marker)
so it doesn't invalidate the M14 prefix cache.

Scope for MVP:
- Only OPEN positions get verdicts (closed round-trips need post-exit
  price history that we don't have easy access to at render time; add
  in a follow-up).
- Aggregate cap of `AGENTIC_VERDICT_FEEDBACK_MAX_FILLS` (default 8)
  matching the memory design (2026-09-15 arm-D plan).
- Empty output (no held positions, or flag off) returns "" so the
  caller can `if verdict_block:` around the append.

Verdict labels come from `jev_client.verdict_for_trade` which is
itself flag-gated + falls back deterministically. So the render path
runs whether or not Jev is enabled; the difference is who assigns
the label.
"""

from __future__ import annotations

import logging
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

logger = logging.getLogger(__name__)


def _fetch_open_position_fills(
    db_path: Path, held_tickers: set[str], limit: int,
) -> list[dict]:
    """Return the most recent BUY fills for tickers we still hold.

    Uses the arm's own paper_orders DB. Filters to buys (opening
    trades) on tickers currently in the portfolio so we don't emit
    verdicts on positions the arm has since fully exited.
    """
    if not held_tickers or not db_path.exists():
        return []
    up = ",".join("?" for _ in held_tickers)
    with sqlite3.connect(str(db_path)) as c:
        rows = c.execute(
            f"""SELECT client_order_id, ticker, side, qty, filled_avg_price,
                       filled_at, rec_id, triggering_news_ids
                FROM paper_orders
                WHERE filled_at IS NOT NULL
                  AND side = 'buy'
                  AND upper(ticker) IN ({up})
                ORDER BY filled_at DESC
                LIMIT ?""",
            (*sorted(t.upper() for t in held_tickers), limit),
        ).fetchall()
    out = []
    for r in rows:
        cid, tk, side, qty, price, filled_at, rec_id, news_ids_json = r
        news_ids: list = []
        if news_ids_json:
            try:
                import json as _j
                parsed = _j.loads(news_ids_json)
                if isinstance(parsed, list):
                    news_ids = parsed
            except Exception:  # noqa: BLE001
                pass
        out.append({
            "client_order_id": cid,
            "ticker": tk,
            "side": side,
            "qty": float(qty or 0),
            "price": float(price or 0),
            "filled_at": filled_at,
            "rec_id": rec_id,
            "news_ids": news_ids,
        })
    return out


def _age_min(iso_ts: str, now: datetime) -> float:
    try:
        t = datetime.fromisoformat(iso_ts.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        return 0.0
    return max(0.0, (now - t).total_seconds() / 60.0)


def _latest_positions_from_snapshots(db_path: Path) -> list[dict]:
    """Read the most recent paper_snapshots row and return its positions
    list. Each element is a dict with keys `ticker`, `qty`,
    `avg_entry_price`, `market_value`, `unrealized_pl_pct`.

    Self-serving from the arm's own DB means the prompt builder doesn't
    need a broker in scope. Snapshots update every tick, so the data is
    at most one tick stale which is fine for verdict-attribution.
    """
    if not db_path.exists():
        return []
    import json as _j
    with sqlite3.connect(str(db_path)) as c:
        row = c.execute(
            "SELECT positions_json FROM paper_snapshots "
            "ORDER BY captured_at DESC LIMIT 1",
        ).fetchone()
    if not row or not row[0]:
        return []
    try:
        parsed = _j.loads(row[0])
    except Exception:  # noqa: BLE001
        return []
    return parsed if isinstance(parsed, list) else []


def build_verdict_feedback_block(
    db_path: Path,
    limit_fills: int = 8,
) -> str:
    """Return a markdown block for the allocator prompt.

    `db_path`: path to the arm's paper SQLite (paper_orders +
               paper_snapshots). Both positions and recent fills come
               from here so the caller doesn't need a broker in scope.

    Emits "" when there's nothing to show (empty portfolio, no fills
    on held names within the DB history, verdict flag off entirely).
    """
    from agentic_investor.flags import flags
    if not flags.JEV_VERDICT_ENABLED:
        # No block at all when the flag is off. Arm A stays clean;
        # only arm B (or any arm with the flag on) sees the section.
        return ""
    positions = _latest_positions_from_snapshots(db_path)
    if not positions:
        return ""
    from agentic_investor.llm.jev_client import verdict_for_trade

    held = {str(p.get("ticker", "")).upper() for p in positions
            if float(p.get("qty") or 0) != 0}
    if not held:
        return ""

    fills = _fetch_open_position_fills(db_path, held, limit_fills)
    if not fills:
        return ""

    unrealized_by_ticker = {
        str(p.get("ticker", "")).upper():
            float(p.get("unrealized_pl_pct") or 0)
        for p in positions
    }
    now = datetime.now(UTC)
    lines: list[str] = []
    for f in fills:
        tk = str(f["ticker"]).upper()
        pnl_pct = unrealized_by_ticker.get(tk, 0.0)
        age_min = _age_min(f["filled_at"] or "", now)
        trade = {
            "ticker": tk,
            "side": f["side"],
            "qty": f["qty"],
            "price": f["price"],
            "filled_at": f["filled_at"],
            "age_min": age_min,
            "news_ids": f["news_ids"],
        }
        trajectory = [{"minutes": int(age_min), "pnl_pct": pnl_pct}]
        try:
            v = verdict_for_trade(trade, trajectory)
        except Exception as e:  # noqa: BLE001 — never break the prompt
            logger.debug("verdict_for_trade failed for %s: %s", tk, e)
            continue
        news_hint = ""
        if f["news_ids"]:
            first = str(f["news_ids"][0])[:12]
            news_hint = f" (cited {first})"
        lines.append(
            f"- {f['side'].upper()} {f['qty']:.2f} {tk} "
            f"{age_min:.0f}m ago @ ${f['price']:.2f} "
            f"-> {pnl_pct:+.2f}% now: {v.label}{news_hint}"
        )

    if not lines:
        return ""

    return (
        "## 12. Recent decision verdicts (self-attribution feedback)\n"
        "Your recent fills, evaluated against their current P&L. "
        "WORKING = thesis validated, WRONG = thesis contradicted, "
        "UNCLEAR = chop, TOO_EARLY = less than 30 minutes elapsed. "
        "Use this to update your conviction on similar current calls.\n"
        + "\n".join(lines)
    )
