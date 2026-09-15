"""Find moments where 2+ arms took opposite-direction trades on the same
ticker within a short window. Useful for the A/B interview narrative:
'here's a name where cooldown-arm A bought while ensemble-arm C sold,
here's who was right at t+15/30/60m and now.'

Also captures same-direction size divergence (both bought, but one at 3x
the other's size — still a conviction gap worth studying).

Idempotent. Writes to
`out/experiments/reasoning-quality/arm_contradictions.jsonl`.
"""

import json
import os
import pathlib
import re
from collections import defaultdict
from datetime import datetime, timedelta, timezone

from dotenv import load_dotenv

load_dotenv()

BASE = pathlib.Path("out/experiments/reasoning-quality")
OUT = BASE / "arm_contradictions.jsonl"
ET_OFFSET = timedelta(hours=-4)
WINDOW_SEC = 300  # 5 min: two trades count as "colliding" if within this

ORDER_RE = re.compile(
    r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}).*"
    r"\[order_submitted\] ticker=(\S+) side=(\S+) qty=([\d.]+)"
)


def _parse_orders(arm: str) -> list[dict]:
    p = BASE / f"{arm}.log"
    if not p.exists():
        return []
    rows = []
    for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
        m = ORDER_RE.search(line)
        if m:
            rows.append({
                "arm": arm,
                "ts_local": m.group(1),
                "ticker": m.group(2).strip(),
                "side": m.group(3).strip(),
                "qty": float(m.group(4)),
            })
    return rows


def _to_utc(ts_local: str) -> datetime:
    return (
        datetime.strptime(ts_local, "%Y-%m-%d %H:%M:%S") - ET_OFFSET
    ).replace(tzinfo=timezone.utc)


def find_contradictions(all_orders: list[dict]) -> list[dict]:
    """Group orders by ticker; find pairs across arms within WINDOW_SEC where
    sides oppose. Also captures same-side pairs with meaningful size divergence
    (>= 2x quantity ratio)."""
    by_ticker = defaultdict(list)
    for o in all_orders:
        by_ticker[o["ticker"]].append(o)

    events = []
    for ticker, orders in by_ticker.items():
        orders.sort(key=lambda x: x["ts_local"])
        for i, o in enumerate(orders):
            t = _to_utc(o["ts_local"])
            for j in range(i + 1, len(orders)):
                other = orders[j]
                if other["arm"] == o["arm"]:
                    continue
                dt = (_to_utc(other["ts_local"]) - t).total_seconds()
                if dt > WINDOW_SEC:
                    break
                if o["side"] != other["side"]:
                    kind = "opposite_direction"
                elif max(o["qty"], other["qty"]) / max(0.01, min(o["qty"], other["qty"])) >= 2.0:
                    kind = "size_divergence_2x"
                else:
                    continue
                events.append({
                    "ticker": ticker,
                    "kind": kind,
                    "gap_sec": round(dt, 1),
                    "first": {
                        "arm": o["arm"], "ts_local": o["ts_local"],
                        "side": o["side"], "qty": o["qty"],
                    },
                    "second": {
                        "arm": other["arm"], "ts_local": other["ts_local"],
                        "side": other["side"], "qty": other["qty"],
                    },
                })
    events.sort(key=lambda e: e["first"]["ts_local"])
    return events


def _price_lookup(tickers: set[str], earliest: datetime, latest: datetime):
    if not tickers:
        return {}
    try:
        from alpaca.data.enums import DataFeed
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame
    except ImportError:
        return {}
    cli = StockHistoricalDataClient(
        os.environ["ALPACA_API_KEY"], os.environ["ALPACA_API_SECRET"],
    )
    try:
        df = cli.get_stock_bars(StockBarsRequest(
            symbol_or_symbols=sorted(tickers),
            timeframe=TimeFrame.Minute,
            start=earliest - timedelta(minutes=2),
            end=latest + timedelta(minutes=1),
            feed=DataFeed.IEX,
        )).df
    except Exception as e:
        print(f"warn: bar fetch failed: {e}")
        return {}
    out = {}
    for tk in tickers:
        if tk in df.index.get_level_values(0):
            out[tk] = df.loc[tk].reset_index()
    return out


def main() -> None:
    all_orders = []
    for arm in ("A", "B", "C"):
        all_orders.extend(_parse_orders(arm))
    if not all_orders:
        print("no orders found")
        return
    events = find_contradictions(all_orders)
    if not events:
        print("no cross-arm contradictions found yet")
        return

    tickers = {e["ticker"] for e in events}
    earliest = min(_to_utc(e["first"]["ts_local"]) for e in events)
    now_utc = datetime.now(timezone.utc)
    bars = _price_lookup(tickers, earliest, now_utc)

    def _price(tk: str, at: datetime, offset_min: int = 0) -> float | None:
        if tk not in bars:
            return None
        t = at + timedelta(minutes=offset_min)
        diffs = (bars[tk].timestamp - t).abs()
        if diffs.min() > timedelta(minutes=5):
            return None
        return float(bars[tk].iloc[diffs.argmin()].close)

    for e in events:
        t0 = _to_utc(e["first"]["ts_local"])
        e["price_at_first"] = _price(e["ticker"], t0, 0)
        e["price_at_second"] = _price(e["ticker"], _to_utc(e["second"]["ts_local"]), 0)
        e["price_15m"] = _price(e["ticker"], t0, 15)
        e["price_30m"] = _price(e["ticker"], t0, 30)
        e["price_60m"] = _price(e["ticker"], t0, 60)
        if e["ticker"] in bars and len(bars[e["ticker"]]) > 0:
            e["price_now"] = float(bars[e["ticker"]].iloc[-1].close)
            if e["price_at_first"]:
                e["move_since_first_pct"] = round(
                    (e["price_now"] - e["price_at_first"]) / e["price_at_first"] * 100, 3,
                )

    with OUT.open("w", encoding="utf-8") as f:
        for e in events:
            f.write(json.dumps(e) + "\n")

    from collections import Counter
    print(f"wrote {len(events)} contradiction events to {OUT}")
    kinds = Counter((e["kind"],) for e in events)
    for (k,), n in kinds.items():
        print(f"  {k}: {n}")

    print(f"\nlast 10 events:")
    print(f"  {'time':<20} {'ticker':<6} {'kind':<20} "
          f"{'A vs B (arm,side,qty)':<40} gap {'move_now':>8}")
    for e in events[-10:]:
        arm1 = f"{e['first']['arm']} {e['first']['side'][:4]}/{e['first']['qty']:.2f}"
        arm2 = f"{e['second']['arm']} {e['second']['side'][:4]}/{e['second']['qty']:.2f}"
        move = e.get("move_since_first_pct")
        move_s = f"{move:+.2f}%" if move is not None else "-"
        print(f"  {e['first']['ts_local']:<20} {e['ticker']:<6} "
              f"{e['kind']:<20} {arm1:<15} vs  {arm2:<15}  "
              f"{e['gap_sec']:>4.0f}s  {move_s:>8}")


if __name__ == "__main__":
    main()
