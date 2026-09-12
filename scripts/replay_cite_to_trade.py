"""Replay cite-to-trade against a historical session with full accuracy.

Unlike a quick session.jsonl grep, this script:
  - Loads real Recommendation objects from the arm's SQLite so
    positions[] rationales are visible (structural citation source)
  - Loads the prev-rec target weights and applies the drift-rebalance
    exemption exactly as production does
  - Uses the same _apply_cite_to_trade function the loop uses, no
    duplicated logic

Point it at an experiment directory + arm; prints per-rec counts and a
per-trade breakdown of blocks / direction violations / clean.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from agentic_investor.orchestrator.loop import _apply_cite_to_trade
from agentic_investor.orchestrator.rebalancer import TradePlan
from agentic_investor.orchestrator.store import load_recommendation


class _CaptureSession:
    def __init__(self):
        self.events: list[tuple[str, dict]] = []

    def log(self, name, payload):
        self.events.append((name, dict(payload)))


@dataclass
class _Order:
    rec_id: int
    ticker: str
    side: str
    qty: float


def _load_orders_with_rec_ids(session_jsonl: Path) -> list[_Order]:
    """Walk the arm's session.jsonl; associate each order with the most
    recent regen_done rec_id preceding it."""
    orders: list[_Order] = []
    current_rec_id: int | None = None
    for line in session_jsonl.read_text(encoding="utf-8").splitlines():
        try:
            o = json.loads(line)
        except json.JSONDecodeError:
            continue
        e = o.get("event")
        if e == "regen_done":
            current_rec_id = o.get("rec_id")
        elif e == "order_submitted" and current_rec_id is not None:
            orders.append(_Order(
                rec_id=current_rec_id,
                ticker=str(o.get("ticker") or ""),
                side=str(o.get("side") or ""),
                qty=float(o.get("qty") or 0.0),
            ))
    return orders


def _order_to_plan(order: _Order) -> TradePlan:
    """Make a TradePlan-shaped object so _apply_cite_to_trade can run.
    Only the fields the gate reads matter (ticker/side/qty/reason)."""
    return TradePlan(
        ticker=order.ticker, side=order.side, dollars=0.0,
        qty=order.qty, target_pct=0.0, current_pct=0.0,
        reason="replay",
    )


def _find_prev_rec_id(arm_db: Path, target_rec_id: int) -> int | None:
    """Return the id of the most recent rec BEFORE target_rec_id."""
    with sqlite3.connect(str(arm_db)) as conn:
        row = conn.execute(
            "SELECT id FROM recommendations WHERE id < ? "
            "ORDER BY id DESC LIMIT 1",
            (target_rec_id,),
        ).fetchone()
    return row[0] if row else None


def replay_arm(exp_dir: Path, arm: str, session_glob_prefix: str) -> dict:
    arm_db = exp_dir / f"{arm}.db"
    if not arm_db.exists():
        return {"arm": arm, "error": f"missing {arm_db}"}
    # Find latest session.jsonl matching the glob.
    session_root = Path("out") / "sessions"
    candidates = sorted(
        session_root.glob(f"{session_glob_prefix}*_{arm}/session.jsonl"),
        key=lambda p: p.stat().st_mtime,
    )
    if not candidates:
        return {"arm": arm, "error": "no session.jsonl match"}
    session_jsonl = candidates[-1]

    orders = _load_orders_with_rec_ids(session_jsonl)
    # Load rec DB URL directly so store.load_recommendation reads the
    # right sqlite file (default reads settings.database_url).
    db_url = f"sqlite:///{arm_db.resolve()}"

    # Group orders by rec so we build one plan-list per rec and hand it
    # to _apply_cite_to_trade with the correct prev_rec.
    from collections import defaultdict
    by_rec: dict[int, list[_Order]] = defaultdict(list)
    for o in orders:
        by_rec[o.rec_id].append(o)

    summary = {
        "arm": arm,
        "total_orders": len(orders),
        "blocked": 0,
        "direction_violations": 0,
        "allowed": 0,
        "drift_exempt": 0,
        "block_examples": [],
    }

    for rec_id, arm_orders in sorted(by_rec.items()):
        rec = load_recommendation(rec_id, url=db_url)
        if rec is None:
            continue
        prev_rec_id = _find_prev_rec_id(arm_db, rec_id)
        prev_rec = (
            load_recommendation(prev_rec_id, url=db_url)
            if prev_rec_id is not None else None
        )
        plans = [_order_to_plan(o) for o in arm_orders]
        session = _CaptureSession()
        kept = _apply_cite_to_trade(
            plans, rec, prev_rec, session, rec_id,
        )
        # Cross-reference which kept plans came through as drift-exempt
        # (session got no cite_violation for them).
        blocked_tickers = {
            ev["ticker"] for name, ev in session.events
            if name == "cite_violation"
        }
        summary["blocked"] += len(blocked_tickers)
        summary["direction_violations"] += sum(
            1 for name, _ in session.events if name == "direction_violation"
        )
        summary["allowed"] += len(kept)
        for name, ev in session.events:
            if name == "cite_violation" and len(summary["block_examples"]) < 8:
                summary["block_examples"].append({
                    "rec_id": ev["rec_id"], "ticker": ev["ticker"],
                    "side": ev["side"], "qty": ev["qty"],
                    "verdict": ev.get("cot", {}).get("verdict", "")[:80],
                })
    # Best-effort drift-exempt estimate: total minus (blocked + allowed).
    # Some allowed trades ARE drift-exempt, so this bounds nothing tight,
    # but a mismatch flags a bookkeeping issue.
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", default="reasoning-quality",
                    help="experiment name under out/experiments/")
    ap.add_argument("--session-prefix", default="2026-09-11T15-22-",
                    help="glob prefix under out/sessions/ (matches "
                         "specific run timestamp)")
    ap.add_argument("--arms", default="A,B,C")
    args = ap.parse_args()

    exp_dir = Path("out") / "experiments" / args.experiment
    print(f"replay: experiment={args.experiment} session_prefix={args.session_prefix}")
    print(f"{'arm':<4} {'orders':<7} {'blocked':<8} {'dir_viol':<9} {'allowed':<8}")
    for arm in args.arms.split(","):
        s = replay_arm(exp_dir, arm.strip(), args.session_prefix)
        if "error" in s:
            print(f"{arm:<4} ERR: {s['error']}")
            continue
        print(f"{s['arm']:<4} {s['total_orders']:<7} "
              f"{s['blocked']:<8} {s['direction_violations']:<9} "
              f"{s['allowed']:<8}")
        if s["block_examples"]:
            print("  block examples:")
            for ex in s["block_examples"]:
                print(f"    rec #{ex['rec_id']}: {ex['side']} {ex['ticker']} "
                      f"qty={ex['qty']} | verdict: {ex['verdict']!r}")


if __name__ == "__main__":
    main()
