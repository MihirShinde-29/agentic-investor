"""Deep 2-day A/B analysis. Cross-references three data sources:

1. paper_orders.db per arm -> FIFO round-trip P&L per ticker
2. skip_events.jsonl -> filter drops with price trajectory
3. arm_contradictions.jsonl -> opposite-direction cross-arm trades

Answers three questions that inform whether phase-2 aborted-intent
feedback is worth building:

A) Which tickers did each arm actually make/lose money on?
B) On opinion_drift skips, would the LLM's aborted trades have been
   right if the filter had let them through?
C) On cross-arm contradictions, which arm was right?
"""

import json
import pathlib
import sqlite3
from collections import defaultdict

BASE = pathlib.Path("out/experiments/reasoning-quality")


def fifo_pnl_per_ticker(arm: str) -> dict:
    """Return {ticker: {realized, open_pnl, buys_notional, sells_notional,
    n_round_trips, most_recent_price}}."""
    c = sqlite3.connect(str(BASE / f"{arm}.db"))
    rows = c.execute(
        "SELECT ticker, side, qty, filled_avg_price, submitted_at "
        "FROM paper_orders WHERE status='filled' ORDER BY submitted_at ASC"
    ).fetchall()

    lots = defaultdict(list)
    realized = defaultdict(float)
    trips = defaultdict(int)
    buys_n = defaultdict(float)
    sells_n = defaultdict(float)
    last_price = {}

    for t, side, qty, px, _ts in rows:
        if px is None or qty is None:
            continue
        qty, px = float(qty), float(px)
        last_price[t] = px
        if side == "buy":
            lots[t].append([qty, px])
            buys_n[t] += qty * px
        else:
            sells_n[t] += qty * px
            remaining = qty
            while remaining > 0 and lots[t]:
                lot_qty, lot_px = lots[t][0]
                close = min(remaining, lot_qty)
                realized[t] += close * (px - lot_px)
                remaining -= close
                lot_qty -= close
                if lot_qty > 1e-9:
                    lots[t][0][0] = lot_qty
                else:
                    lots[t].pop(0)
                trips[t] += 1

    out = {}
    for t in set(list(realized) + list(lots)):
        open_qty = sum(q for q, _ in lots.get(t, []))
        avg_cost = (
            sum(q * p for q, p in lots.get(t, [])) / open_qty if open_qty else 0
        )
        open_pnl = (
            (last_price.get(t, avg_cost) - avg_cost) * open_qty if open_qty else 0
        )
        out[t] = {
            "realized": round(realized.get(t, 0), 2),
            "open_pnl_approx": round(open_pnl, 2),
            "total": round(realized.get(t, 0) + open_pnl, 2),
            "buys_notional": round(buys_n.get(t, 0), 0),
            "sells_notional": round(sells_n.get(t, 0), 0),
            "n_round_trips": trips.get(t, 0),
            "last_seen_price": round(last_price.get(t, 0), 2),
        }
    return out


def analyze_skip_events() -> list[dict]:
    """For each opinion_drift skip with a named ticker, compute whether the
    aborted intent would have been right at various horizons.

    Direction of intent isn't stored — we infer from what the arm eventually
    did on that ticker (executed_side). If the arm never executed post-skip,
    we mark direction unknown.
    """
    p = BASE / "skip_events.jsonl"
    if not p.exists():
        return []
    events = [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines()]
    drifts = [e for e in events if e.get("skip_type") == "opinion_drift"
              and e.get("ticker") not in (None, "?")
              and e.get("price_at_skip") is not None]
    return drifts


def analyze_contradictions() -> list[dict]:
    p = BASE / "arm_contradictions.jsonl"
    if not p.exists():
        return []
    return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines()]


def main() -> None:
    print("=" * 76)
    print("2-DAY DEEP ANALYSIS: M14.5 A/B (2026-09-14 + 2026-09-15)")
    print("=" * 76)

    # A) P&L per ticker per arm
    print("\n### A) Per-ticker realized+open P&L (FIFO, both days) ###\n")
    arm_pnl = {}
    for arm in ("A", "B", "C"):
        arm_pnl[arm] = fifo_pnl_per_ticker(arm)
        tickers = sorted(arm_pnl[arm].keys(),
                         key=lambda t: -abs(arm_pnl[arm][t]["total"]))
        total = sum(v["total"] for v in arm_pnl[arm].values())
        print(f"--- {arm} (net P&L across all tickers: ${total:+.2f}) ---")
        print(f"  {'ticker':<7} {'trips':>5} {'realized':>10} {'open':>10} "
              f"{'total':>10}  buys/sells")
        for t in tickers:
            v = arm_pnl[arm][t]
            print(f"  {t:<7} {v['n_round_trips']:>5} ${v['realized']:>+9.2f} "
                  f"${v['open_pnl_approx']:>+9.2f} ${v['total']:>+9.2f}  "
                  f"${v['buys_notional']:>5,.0f}/${v['sells_notional']:>5,.0f}")

    # B) Skip-event outcomes
    print("\n### B) Opinion-drift skips — would the LLM have been right? ###\n")
    drifts = analyze_skip_events()
    if not drifts:
        print("  no drift events with price data")
    else:
        by_ticker = defaultdict(list)
        for e in drifts:
            by_ticker[e["ticker"]].append(e)

        # For each ticker, compute the abs move t0->t+60m as a proxy for
        # "would trading it have mattered." Big move = LLM might've caught
        # something; flat = LLM was just noisy.
        print(f"  {'ticker':<7} {'n_skips':>8} {'skip_price':>10} "
              f"{'p+15m':>8} {'p+30m':>8} {'p+60m':>8} {'p_now':>8}  "
              f"|move_60m|")
        for t, evs in sorted(by_ticker.items(),
                             key=lambda x: -len(x[1])):
            first = evs[0]
            p_at = first["price_at_skip"]
            p_15 = first.get("price_15m")
            p_30 = first.get("price_30m")
            p_60 = first.get("price_60m")
            p_now = first.get("price_now") or p_at
            m60 = ((p_60 - p_at) / p_at * 100) if (p_at and p_60) else None
            m60s = f"{m60:+.2f}%" if m60 is not None else "n/a"
            row = (f"  {t:<7} {len(evs):>8} ${p_at:>8.2f} "
                   f"${p_15 or 0:>6.2f} ${p_30 or 0:>6.2f} "
                   f"${p_60 or 0:>6.2f} ${p_now:>6.2f}  {m60s:>8}")
            print(row)

        big_movers = [e for e in drifts
                      if e.get("price_60m") and e.get("price_at_skip")
                      and abs((e["price_60m"] - e["price_at_skip"])
                              / e["price_at_skip"]) > 0.005]
        print(f"\n  {len(big_movers)}/{len(drifts)} skips saw >0.5% price move "
              f"within 60min — these are potential missed-alpha events "
              f"the filter blocked.")

    # C) Cross-arm contradictions
    print("\n### C) Cross-arm contradictions — who was right? ###\n")
    contras = analyze_contradictions()
    if not contras:
        print("  no contradiction events with price data")
    else:
        opp = [c for c in contras
               if c.get("kind") == "opposite_direction"
               and c.get("price_at_first") and c.get("price_now")]
        print(f"  {'time':<20} {'ticker':<6} {'first':<15} vs "
              f"{'second':<15}  {'move_since':>10}  who_right")
        buyer_right = 0
        seller_right = 0
        chop = 0
        for c in opp:
            first = c["first"]
            second = c["second"]
            move = c.get("move_since_first_pct", 0)
            if abs(move) < 0.2:
                verdict = "chop"
                chop += 1
            elif move > 0:
                verdict = "BUY was right" if first["side"] == "buy" \
                    else "SELL was wrong"
                buyer_right += 1 if first["side"] == "buy" else 0
                seller_right += 0 if first["side"] == "buy" else 1
            else:
                verdict = "SELL was right" if first["side"] == "sell" \
                    else "BUY was wrong"
                seller_right += 1 if first["side"] == "sell" else 0
                buyer_right += 0 if first["side"] == "sell" else 1
            a1 = f"{first['arm']} {first['side'][:4]}/{first['qty']:.1f}"
            a2 = f"{second['arm']} {second['side'][:4]}/{second['qty']:.1f}"
            print(f"  {first['ts_local']:<20} {c['ticker']:<6} "
                  f"{a1:<15} vs {a2:<15}  {move:>+8.2f}%  {verdict}")
        total_directional = buyer_right + seller_right
        if total_directional > 0:
            print(f"\n  Directional resolutions: buyer_right={buyer_right}  "
                  f"seller_right={seller_right}  chop={chop}")

    # D) Cross-cutting: for names where BOTH arms and skip events happened,
    # show whether the "skipping arm" would have been better off flipping.
    print("\n### D) Does the drift filter earn its keep? ###\n")
    # Sum realized P&L on drift-thrashed names vs successfully-traded names
    thrash_names = {e["ticker"] for e in drifts} if drifts else set()
    print(f"  {'arm':<3} {'thrash_pnl':>12} {'clean_pnl':>12}  "
          f"(across tickers with >=3 drift skips vs no drift skips)")
    heavy = {t for t, evs in
             defaultdict(list, {t: [] for t in thrash_names}).items()}
    heavy_ct = defaultdict(int)
    for e in drifts:
        heavy_ct[e["ticker"]] += 1
    heavy = {t for t, n in heavy_ct.items() if n >= 3}
    for arm in ("A", "B", "C"):
        thrash = sum(v["total"] for t, v in arm_pnl[arm].items() if t in heavy)
        clean = sum(v["total"] for t, v in arm_pnl[arm].items()
                    if t not in heavy)
        print(f"  {arm:<3} ${thrash:>+11.2f} ${clean:>+11.2f}")

    # Executive summary
    print("\n### Executive read ###\n")
    totals = {arm: sum(v["total"] for v in arm_pnl[arm].values())
              for arm in ("A", "B", "C")}
    winner = max(totals.items(), key=lambda x: x[1])
    print(f"  2-day net P&L: A={totals['A']:+.2f} B={totals['B']:+.2f} "
          f"C={totals['C']:+.2f}  -> leader: {winner[0]} ${winner[1]:+.2f}")


if __name__ == "__main__":
    main()
