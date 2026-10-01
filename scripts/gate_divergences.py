"""Emit a JSONL log of B/C materiality-gate divergences.

Parses `out/logs/experiment.out` for arm B laya_materiality_gate and
arm C jev_materiality_gate events, pairs them by (n_headlines,
timestamp proximity <=15s), and writes one JSONL row per divergent
pair to `out/analytics/gate_divergences_<date>.jsonl`.

Roadmap item #11 from docs/INTERVIEW_NOTES.md, promoted after Mon
2026-09-28 open showed ~30% B/C disagreement rate on shared batches
- data that only surfaced by tailing the log by line. This is the
offline / add-only version of the recorder (no arm-code changes,
respects the mid-week freeze rule).

Usage:
    python scripts/gate_divergences.py                       # today
    python scripts/gate_divergences.py --date 2026-09-28
    python scripts/gate_divergences.py --stdout              # skip file
    python scripts/gate_divergences.py --with-forward-pnl    # ex-post score

With `--with-forward-pnl`, each divergence is additionally scored by
pulling Alpaca 1-min bars for the tickers the acting arm actually
traded within 5 min of the gate, then marking P&L to T+15 min from
the gate. Alpaca creds (.env) required; silently skipped per-row if
a fetch fails.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path

LOG = Path("out/logs/experiment.out")
OUT_DIR = Path("out/analytics")

GATE_RE = re.compile(
    r"^\[([BC])\] (\d{4}-\d{2}-\d{2}) (\d{2}:\d{2}:\d{2}),\d{3} .*"
    r"\[(laya_materiality_gate|jev_materiality_gate)\] "
    r"material=(True|False) confidence=([0-9.]+) "
    r"from_(?:laya|jev)=(?:True|False) n_headlines=(\d+)"
)

# Order submissions on the acting arm, used by the optional
# forward-P&L scorer.
ORDER_RE = re.compile(
    r"^\[([BC])\] (\d{4}-\d{2}-\d{2}) (\d{2}:\d{2}:\d{2}),\d{3} .*"
    r"\[order_submitted\] ticker=([A-Z][A-Z0-9.\-]+) side=(buy|sell) "
    r"qty=([0-9.]+)"
)

# Batches close within a few seconds across arms; 15s is generous
# enough to survive news-bus jitter but tight enough that we don't
# accidentally pair adjacent unrelated batches.
PAIR_WINDOW_SEC = 15.0

# Window after the gate in which we attribute order_submitted events
# to the gate's regen. 5 min is the longest regen tail we see.
ORDER_WINDOW_SEC = 300

# How far forward from the gate we mark the price. 15 min is enough
# to let a reaction develop without drifting into unrelated news.
FORWARD_WINDOW_MIN = 15


def _parse_events(path: Path, day: str) -> list[dict]:
    events: list[dict] = []
    if not path.exists():
        print(f"log not found: {path}", file=sys.stderr)
        return events
    with path.open(encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = GATE_RE.match(line)
            if not m:
                continue
            arm, d, hms, gate, mat, conf, n = m.groups()
            if d != day:
                continue
            events.append({
                "arm": arm,
                "gate": gate,
                "ts": datetime.strptime(f"{d} {hms}", "%Y-%m-%d %H:%M:%S"),
                "material": mat == "True",
                "confidence": float(conf),
                "n_headlines": int(n),
            })
    return events


def _pair(events: list[dict]) -> tuple[list[dict], int, int]:
    """Return (divergent_pairs, total_pairs, agree_count)."""
    by_n: dict[int, list[dict]] = defaultdict(list)
    for e in events:
        by_n[e["n_headlines"]].append(e)

    divergent: list[dict] = []
    total, agree = 0, 0
    for group in by_n.values():
        group.sort(key=lambda x: x["ts"])
        used = [False] * len(group)
        for i, e in enumerate(group):
            if e["arm"] != "B" or used[i]:
                continue
            best_j, best_dt = None, PAIR_WINDOW_SEC + 1
            for j, o in enumerate(group):
                if o["arm"] != "C" or used[j]:
                    continue
                dt = abs((o["ts"] - e["ts"]).total_seconds())
                if dt <= PAIR_WINDOW_SEC and dt < best_dt:
                    best_j, best_dt = j, dt
            if best_j is None:
                continue
            c = group[best_j]
            used[i] = True
            used[best_j] = True
            total += 1
            if e["material"] == c["material"]:
                agree += 1
                continue
            divergent.append({
                "ts": min(e["ts"], c["ts"]).isoformat(),
                "n_headlines": e["n_headlines"],
                "laya_material": e["material"],
                "laya_conf": e["confidence"],
                "jev_material": c["material"],
                "jev_conf": c["confidence"],
                "acting_arm": "B" if e["material"] else "C",
                "pair_gap_sec": round(best_dt, 2),
            })
    divergent.sort(key=lambda x: x["ts"])
    return divergent, total, agree


def _parse_orders(path: Path, day: str) -> list[dict]:
    """order_submitted events on arm B or C for the given day."""
    rows: list[dict] = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = ORDER_RE.match(line)
            if not m:
                continue
            arm, d, hms, ticker, side, qty = m.groups()
            if d != day:
                continue
            rows.append({
                "arm": arm,
                "ts": datetime.strptime(f"{d} {hms}", "%Y-%m-%d %H:%M:%S"),
                "ticker": ticker,
                "side": side,
                "qty": float(qty),
            })
    return rows


class _BarFetcher:
    """Thin wrapper over Alpaca's StockHistoricalDataClient.

    Caches per-(ticker, minute) so a batch of divergences on the same
    tickers in the same ~15-min window makes at most one upstream
    request per minute per ticker. Returns None on any error so the
    caller can keep going with a partial score.
    """

    def __init__(self) -> None:
        from dotenv import load_dotenv
        load_dotenv()
        from alpaca.data.historical.stock import StockHistoricalDataClient
        key = os.environ.get("ALPACA_API_KEY") or os.environ.get("ALPACA_KEY_ID")
        secret = (os.environ.get("ALPACA_API_SECRET")
                  or os.environ.get("ALPACA_SECRET_KEY"))
        if not (key and secret):
            raise RuntimeError("ALPACA_API_KEY / _SECRET missing in env")
        self._client = StockHistoricalDataClient(api_key=key, secret_key=secret)
        self._cache: dict[tuple[str, datetime], float | None] = {}

    def close_at(self, ticker: str, ts: datetime) -> float | None:
        """Last 1-min bar close at or just before `ts` (US/Eastern-aware
        timestamps the log uses). Returns None on miss or error."""
        from datetime import timezone
        # Normalize to the minute - Alpaca 1Min bars start on the minute.
        minute = ts.replace(second=0, microsecond=0)
        key = (ticker, minute)
        if key in self._cache:
            return self._cache[key]
        try:
            from alpaca.data.requests import StockBarsRequest
            from alpaca.data.timeframe import TimeFrame
            # Pull a window around the minute to tolerate missing bars
            # (sparse trades at open / halts).
            req = StockBarsRequest(
                symbol_or_symbols=ticker,
                timeframe=TimeFrame.Minute,
                start=(minute - timedelta(minutes=2)).replace(tzinfo=timezone.utc),
                end=(minute + timedelta(minutes=2)).replace(tzinfo=timezone.utc),
                feed="iex",
            )
            resp = self._client.get_stock_bars(req)
            bars = resp.data.get(ticker, []) if hasattr(resp, "data") else []
            if not bars:
                self._cache[key] = None
                return None
            # Pick the latest bar with timestamp <= minute+1 (so the bar
            # that COVERS our target). Fall back to earliest if the
            # exact bar is missing.
            match = None
            for b in bars:
                b_min = b.timestamp.replace(tzinfo=timezone.utc)
                if b_min <= minute.replace(tzinfo=timezone.utc) + timedelta(minutes=1):
                    match = b
            price = float(match.close) if match is not None else None
            self._cache[key] = price
            return price
        except Exception:  # noqa: BLE001
            self._cache[key] = None
            return None


def _score_divergence(
    divergence: dict,
    orders: list[dict],
    fetcher: _BarFetcher | None,
) -> dict:
    """Attach `acting_arm_tickers`, `acting_arm_order_count`,
    `acting_arm_notional_usd` and `acting_arm_pnl_15min_usd` to the
    divergence row. `pnl_15min_usd` is None if we can't price the
    forward window (no orders, no fetcher, or Alpaca said no)."""
    arm = divergence["acting_arm"]
    gate_ts = datetime.fromisoformat(divergence["ts"])
    end = gate_ts + timedelta(seconds=ORDER_WINDOW_SEC)
    fired = [o for o in orders
             if o["arm"] == arm and gate_ts <= o["ts"] <= end]
    out = dict(divergence)
    out["acting_arm_order_count"] = len(fired)
    out["acting_arm_tickers"] = sorted({o["ticker"] for o in fired})
    out["acting_arm_notional_usd"] = 0.0
    out["acting_arm_pnl_15min_usd"] = None
    if not fired or fetcher is None:
        return out
    forward_ts = gate_ts + timedelta(minutes=FORWARD_WINDOW_MIN)
    pnl = 0.0
    notional = 0.0
    priced = 0
    for o in fired:
        entry = fetcher.close_at(o["ticker"], gate_ts)
        exit_ = fetcher.close_at(o["ticker"], forward_ts)
        if entry is None or exit_ is None:
            continue
        # Buy wins when price rises (long exposure); sell wins when price
        # falls (short exposure or avoided drawdown on an exited long).
        direction = 1.0 if o["side"] == "buy" else -1.0
        pnl += direction * (exit_ - entry) * o["qty"]
        notional += entry * o["qty"]
        priced += 1
    out["acting_arm_notional_usd"] = round(notional, 2)
    out["acting_arm_priced_orders"] = priced
    out["acting_arm_pnl_15min_usd"] = round(pnl, 2) if priced else None
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=None,
                    help="YYYY-MM-DD, default today (UTC)")
    ap.add_argument("--stdout", action="store_true",
                    help="print JSONL to stdout instead of writing a file")
    ap.add_argument("--with-forward-pnl", action="store_true",
                    help="also mark each divergence to T+15min using "
                         "Alpaca 1-min bars (needs ALPACA_API_KEY env)")
    args = ap.parse_args()

    day = args.date or date.today().isoformat()
    events = _parse_events(LOG, day)
    if not events:
        print(f"no gate events for {day}")
        return 1

    divergent, total, agree = _pair(events)
    print(f"date: {day}")
    print(f"gates: {len(events)} "
          f"({sum(1 for e in events if e['arm']=='B')} B, "
          f"{sum(1 for e in events if e['arm']=='C')} C)")
    print(f"paired: {total}  agree: {agree}  "
          f"diverge: {len(divergent)}  "
          f"({100*len(divergent)/max(1,total):.1f}%)")

    if args.with_forward_pnl and divergent:
        orders = _parse_orders(LOG, day)
        fetcher: _BarFetcher | None
        try:
            fetcher = _BarFetcher()
        except Exception as exc:  # noqa: BLE001
            print(f"forward-pnl: fetcher init failed ({exc}); scoring "
                  "attribution-only", file=sys.stderr)
            fetcher = None
        divergent = [_score_divergence(d, orders, fetcher) for d in divergent]
        wins = sum(1 for d in divergent
                   if (d.get("acting_arm_pnl_15min_usd") or 0) > 0)
        losses = sum(1 for d in divergent
                     if (d.get("acting_arm_pnl_15min_usd") or 0) < 0)
        unpriced = sum(1 for d in divergent
                       if d.get("acting_arm_pnl_15min_usd") is None)
        total_pnl = sum(d.get("acting_arm_pnl_15min_usd") or 0.0
                        for d in divergent)
        print(f"forward-pnl: {wins} wins, {losses} losses, "
              f"{unpriced} unpriced  net=${total_pnl:+.2f}")

    lines = [json.dumps(d, default=str) for d in divergent]
    if args.stdout:
        for line in lines:
            print(line)
    else:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        out_path = OUT_DIR / f"gate_divergences_{day}.jsonl"
        out_path.write_text("\n".join(lines) + ("\n" if lines else ""),
                            encoding="utf-8")
        print(f"wrote {len(lines)} divergences to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
