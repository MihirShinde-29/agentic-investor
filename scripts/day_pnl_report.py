"""Day-level P&L attribution across a paper-experiment.

Walks the arm sqlite DBs at `out/experiments/<name>/{A,B,C}.db` and
answers:

- Total P&L per arm (equity delta first vs last snapshot for the day).
- Per-ticker realized+unrealized attribution:
    - realized = SUM(sell_qty * sell_price) - SUM(buy_qty * buy_price)
                 on tickers whose net qty is 0 at day close
    - unrealized = (last_snapshot.market_value - first_snapshot.market_value)
                   for tickers still held
  We report both under one number per ticker: total-contribution-to-day-P&L.
- Trigger-attribution of each fill:
    - "fresh regen" if the order landed within 5s of a `regen_done` event
    - "drift-band" if it landed under a `regen_attribution` row that
      wasn't preceded by a fresh `regen_done`
    - "unattributed" otherwise
  Reads session.jsonl per arm (multiple sessions per arm on relaunch days,
  concatenated).
- Simple hit-rate: how many fills closed at a better price than they opened,
  measured against the day's last snapshot for held positions and against
  fill price for round-trips.

Usage:
    python scripts/day_pnl_report.py                      # today
    python scripts/day_pnl_report.py --date 2026-09-18
    python scripts/day_pnl_report.py --experiment reasoning-quality

Output is plain text so the human running end-of-day can eyeball;
--json prints one dict per arm for downstream tooling (e.g. an arm D
verdict-feedback reward signal that needs a machine-readable summary).
"""

from __future__ import annotations

import argparse
import glob
import json
import sqlite3
import sys
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path


def _load_snapshots(db: Path, date: str) -> list[tuple[str, dict, list[dict]]]:
    """(captured_at, account, positions) for one arm, filtered to date."""
    out = []
    with sqlite3.connect(str(db)) as c:
        rows = c.execute(
            "SELECT captured_at, account_json, positions_json "
            "FROM paper_snapshots WHERE captured_at LIKE ? "
            "ORDER BY captured_at",
            (f"{date}%",),
        ).fetchall()
    for ts, acct_j, pos_j in rows:
        out.append((ts, json.loads(acct_j), json.loads(pos_j)))
    return out


def _load_orders(db: Path, date: str) -> list[dict]:
    with sqlite3.connect(str(db)) as c:
        rows = c.execute(
            "SELECT client_order_id, ticker, side, qty, filled_avg_price, "
            "filled_at, submitted_at, rec_id "
            "FROM paper_orders WHERE submitted_at LIKE ? "
            "AND filled_at IS NOT NULL "
            "ORDER BY filled_at",
            (f"{date}%",),
        ).fetchall()
    keys = ("client_order_id", "ticker", "side", "qty", "price",
            "filled_at", "submitted_at", "rec_id")
    return [dict(zip(keys, r, strict=True)) for r in rows]


def _load_session_events(arm: str, date: str) -> list[dict]:
    """Concatenate all session.jsonl for this arm on `date`. Multiple
    launches per day => multiple session dirs; we merge and sort by ts.
    """
    events: list[dict] = []
    for d in sorted(glob.glob(f"out/sessions/{date}T*_{arm}/")):
        p = Path(d) / "session.jsonl"
        if not p.exists():
            continue
        with p.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    events.sort(key=lambda e: e.get("ts", ""))
    return events


def _attribute_fills_to_trigger(
    orders: list[dict], events: list[dict],
) -> dict[str, str]:
    """client_order_id -> {"fresh_regen"|"drift_band"|"unattributed"}.

    Rule: for each order, find the most recent `regen_attribution` in
    the last 30s. If that rec_id also had a `regen_done` in the last
    30s, mark fresh. If only `regen_attribution` (no fresh done), it
    came from the drift-band path (opinion_drift_skip fires without
    regen_done). Otherwise unattributed.
    """
    def _ts(s: str) -> float:
        try:
            return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()
        except (ValueError, AttributeError):
            return 0.0

    attribs = [(e, _ts(e["ts"])) for e in events
               if e.get("event") == "regen_attribution"]
    regen_done_by_recid: dict[int, list[float]] = defaultdict(list)
    for e in events:
        if e.get("event") == "regen_done":
            rid = e.get("rec_id")
            if rid is not None:
                regen_done_by_recid[int(rid)].append(_ts(e["ts"]))

    out: dict[str, str] = {}
    for o in orders:
        # Prefer the filled_at ts; fall back to submitted_at.
        o_ts = _ts(o.get("filled_at") or o.get("submitted_at") or "")
        # Nearest preceding regen_attribution within 30s.
        candidate = None
        for ev, ev_ts in attribs:
            if ev_ts <= o_ts and (o_ts - ev_ts) <= 30.0:
                candidate = ev
        if candidate is None:
            out[o["client_order_id"]] = "unattributed"
            continue
        rec_id = candidate.get("rec_id")
        # Was there a regen_done for this rec_id within ~5s before the
        # attribution? If yes, fresh regen; if no, drift-band.
        attr_ts = _ts(candidate.get("ts", ""))
        fresh = any(
            0 <= (attr_ts - dts) <= 5.0
            for dts in regen_done_by_recid.get(int(rec_id or 0), [])
        )
        out[o["client_order_id"]] = "fresh_regen" if fresh else "drift_band"
    return out


def _per_ticker_pnl(
    orders: list[dict],
    first_snap: list[dict],
    last_snap: list[dict],
) -> dict[str, dict]:
    """Contribution to day P&L per ticker.

    realized = sum(sell qty * sell price) - sum(buy qty * buy price)
    unrealized_delta = last_market_value - first_market_value for names
                       held across the day.
    total = realized + unrealized_delta. Sums to the equity delta modulo
    fees + cash-drag; small residual is normal.
    """
    first_mv = {p["ticker"].upper(): float(p.get("market_value") or 0)
                for p in first_snap}
    last_mv = {p["ticker"].upper(): float(p.get("market_value") or 0)
               for p in last_snap}
    tickers = set(first_mv) | set(last_mv) | {o["ticker"].upper() for o in orders}

    # Cash flow per ticker from fills: buy = cash out (negative), sell = in.
    cash_flow: dict[str, float] = defaultdict(float)
    for o in orders:
        tk = o["ticker"].upper()
        px = float(o.get("price") or 0)
        qty = float(o.get("qty") or 0)
        if o["side"] == "buy":
            cash_flow[tk] -= px * qty
        else:
            cash_flow[tk] += px * qty

    out = {}
    for tk in sorted(tickers):
        m0 = first_mv.get(tk, 0.0)
        m1 = last_mv.get(tk, 0.0)
        cf = cash_flow.get(tk, 0.0)
        # P&L contribution: end-position - start-position + net-cash-received
        # (i.e. sold more than bought is a positive contribution).
        pnl = (m1 - m0) + cf
        if abs(pnl) < 0.005 and m0 == 0 and m1 == 0 and cf == 0:
            continue
        out[tk] = {
            "start_mv": round(m0, 2),
            "end_mv": round(m1, 2),
            "net_cash_flow": round(cf, 2),
            "pnl": round(pnl, 2),
        }
    return out


def _report_arm(arm: str, exp: str, date: str) -> dict:
    db = Path(f"out/experiments/{exp}/{arm}.db")
    if not db.exists():
        print(f"  arm {arm}: no DB at {db}", file=sys.stderr)
        return {}
    snaps = _load_snapshots(db, date)
    if len(snaps) < 2:
        print(f"  arm {arm}: fewer than 2 snapshots on {date}", file=sys.stderr)
        return {}
    orders = _load_orders(db, date)
    events = _load_session_events(arm, date)
    attributions = _attribute_fills_to_trigger(orders, events)

    first_ts, first_acct, first_pos = snaps[0]
    last_ts, last_acct, last_pos = snaps[-1]
    day_pnl = float(last_acct["equity"]) - float(first_acct["equity"])
    per_ticker = _per_ticker_pnl(orders, first_pos, last_pos)

    # Trigger split of order count + total notional.
    by_trigger = defaultdict(lambda: {"n": 0, "notional": 0.0, "pnl_bucket": 0.0})
    for o in orders:
        tag = attributions.get(o["client_order_id"], "unattributed")
        px = float(o.get("price") or 0)
        qty = float(o.get("qty") or 0)
        by_trigger[tag]["n"] += 1
        by_trigger[tag]["notional"] += px * qty

    return {
        "arm": arm,
        "date": date,
        "day_pnl": round(day_pnl, 2),
        "start_equity": round(float(first_acct["equity"]), 2),
        "end_equity": round(float(last_acct["equity"]), 2),
        "start_ts": first_ts,
        "end_ts": last_ts,
        "n_orders": len(orders),
        "per_ticker_pnl": per_ticker,
        "by_trigger": {k: {"n": v["n"],
                           "notional": round(v["notional"], 2)}
                       for k, v in by_trigger.items()},
    }


def _print_report(rep: dict) -> None:
    arm = rep["arm"]
    print(f"\n== arm {arm} ==")
    print(f"  window: {rep['start_ts'][:19]} -> {rep['end_ts'][:19]}")
    print(f"  equity: ${rep['start_equity']:,.2f} -> ${rep['end_equity']:,.2f}")
    print(f"  day P&L: ${rep['day_pnl']:+,.2f}  ({rep['n_orders']} fills)")

    print("  fills by trigger:")
    for tag, m in sorted(rep["by_trigger"].items()):
        print(f"    {tag}: {m['n']} fills, ${m['notional']:,.2f} notional")

    ranked = sorted(
        rep["per_ticker_pnl"].items(),
        key=lambda kv: kv[1]["pnl"],
    )
    winners = [(t, m) for t, m in ranked if m["pnl"] > 0][-6:][::-1]
    losers = [(t, m) for t, m in ranked if m["pnl"] < 0][:6]
    print("  top P&L winners:")
    for t, m in winners:
        print(f"    {t:<6} ${m['pnl']:+,.2f}  "
              f"(start_mv=${m['start_mv']:,.0f} end_mv=${m['end_mv']:,.0f} "
              f"cash=${m['net_cash_flow']:+,.0f})")
    print("  top P&L losers:")
    for t, m in losers:
        print(f"    {t:<6} ${m['pnl']:+,.2f}  "
              f"(start_mv=${m['start_mv']:,.0f} end_mv=${m['end_mv']:,.0f} "
              f"cash=${m['net_cash_flow']:+,.0f})")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=datetime.now(UTC).strftime("%Y-%m-%d"),
                    help="YYYY-MM-DD (UTC); default today")
    ap.add_argument("--experiment", default="reasoning-quality",
                    help="experiment folder under out/experiments/")
    ap.add_argument("--json", action="store_true",
                    help="emit JSON instead of the text report")
    args = ap.parse_args()

    reports = []
    for arm in "ABC":
        rep = _report_arm(arm, args.experiment, args.date)
        if rep:
            reports.append(rep)

    if args.json:
        json.dump(reports, sys.stdout, indent=2, default=str)
        print()
        return 0

    if not reports:
        print(f"no snapshots found for {args.date} under "
              f"out/experiments/{args.experiment}/", file=sys.stderr)
        return 1

    print(f"day P&L report - {args.date} - experiment '{args.experiment}'")
    for rep in reports:
        _print_report(rep)

    # Cross-arm summary at the bottom.
    print("\n== cross-arm summary ==")
    total = sum(r["day_pnl"] for r in reports)
    print(f"  combined day P&L (paper): ${total:+,.2f}")
    for r in reports:
        print(f"    {r['arm']}: ${r['day_pnl']:+,.2f} "
              f"({r['n_orders']} fills)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
