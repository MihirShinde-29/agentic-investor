"""Snapshot CRM (or any ticker) prices at each opinion_drift_skip event.

Usage: python scripts/crm_drift_capture.py [TICKER]

For each drift-skip in any arm log where max_delta_ticker matches the target,
records timestamp + arm + drift magnitude + CRM price at that moment (fetched
from Alpaca historical 1-min bars).

Writes to out/experiments/reasoning-quality/{ticker}_drift_prices.jsonl
so re-running appends nothing new; safe to run repeatedly through the day.

Interpretation later: cross-reference drift-skip timestamps vs the CRM chart.
The bigger the drift block, the more the LLM was thrashing on CRM. Whether
its "wanted trades" would have been right is checked by seeing what CRM did
in the 30-60 min after each skip.
"""

import json
import os
import pathlib
import re
import sys
from datetime import datetime, timedelta, timezone

from dotenv import load_dotenv

load_dotenv()

TICKER = (sys.argv[1] if len(sys.argv) > 1 else "CRM").upper()
BASE = pathlib.Path("out/experiments/reasoning-quality")
OUT = BASE / f"{TICKER.lower()}_drift_prices.jsonl"

DRIFT_RE = re.compile(
    r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}).*"
    r"\[opinion_drift_skip\].*avg_drift_pp=([\d.]+).*"
    r"max_delta_pp=([\d.]+).*max_delta_ticker=(\S+)"
)


def parse_events(arm: str) -> list[dict]:
    p = BASE / f"{arm}.log"
    events = []
    if not p.exists():
        return events
    for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
        m = DRIFT_RE.search(line)
        if m and m.group(4).strip() == TICKER:
            events.append({
                "arm": arm,
                "ts_local": m.group(1),
                "avg_drift_pp": float(m.group(2)),
                "max_delta_pp": float(m.group(3)),
            })
    return events


all_events = []
for arm in ("A", "B", "C"):
    all_events.extend(parse_events(arm))

if not all_events:
    print(f"no {TICKER} opinion_drift_skip events found across A/B/C logs")
    sys.exit(0)

# Sort by time so the jsonl is chronological
all_events.sort(key=lambda e: e["ts_local"])
print(f"found {len(all_events)} {TICKER} drift-skip events across arms:")
for e in all_events:
    print(f"  {e['ts_local']}  {e['arm']}  avg={e['avg_drift_pp']:.1f}pp  max={e['max_delta_pp']:.1f}pp")

# Fetch CRM price at each event ts. Alpaca 1-min bars are the highest
# resolution we get in the paper-data plan; interpolate to bar close.
try:
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    from alpaca.data.enums import DataFeed
except ImportError:
    print("alpaca-py not installed; can't fetch prices, wrote timestamps only")
    with OUT.open("w", encoding="utf-8") as f:
        for e in all_events:
            f.write(json.dumps(e) + "\n")
    sys.exit(0)

# Local times in logs are ET; convert to UTC for Alpaca
ET_OFFSET = timedelta(hours=-4)  # EDT (14 Sep is DST-on)
for e in all_events:
    dt_local = datetime.strptime(e["ts_local"], "%Y-%m-%d %H:%M:%S")
    e["ts_utc"] = (dt_local - ET_OFFSET).replace(tzinfo=timezone.utc).isoformat()

# Batch fetch: earliest to now+5min so we have "later prices" for the write-up too
earliest = min(datetime.fromisoformat(e["ts_utc"]) for e in all_events)
now_utc = datetime.now(timezone.utc)
data_cli = StockHistoricalDataClient(
    os.environ["ALPACA_API_KEY"], os.environ["ALPACA_API_SECRET"],
)
req = StockBarsRequest(
    symbol_or_symbols=[TICKER],
    timeframe=TimeFrame.Minute,
    start=earliest - timedelta(minutes=2),
    end=now_utc,
    feed=DataFeed.IEX,  # paper plan is IEX-only
)
bars = data_cli.get_stock_bars(req).df
if bars.empty:
    print(f"no {TICKER} bars in requested window")
    with OUT.open("w", encoding="utf-8") as f:
        for e in all_events:
            f.write(json.dumps(e) + "\n")
    sys.exit(0)

# bars is a MultiIndex df on (symbol, timestamp); slice out the ticker
bars = bars.loc[TICKER].reset_index()
print(f"\nfetched {len(bars)} 1-min {TICKER} bars from {bars.iloc[0].timestamp} to {bars.iloc[-1].timestamp}")


def nearest_bar_price(target_utc: datetime) -> float | None:
    # Find the 1-min bar whose window contains target_utc
    for _, row in bars.iterrows():
        bar_start = row.timestamp.to_pydatetime()
        if bar_start <= target_utc < bar_start + timedelta(minutes=1):
            return float(row.close)
    # Fallback: closest bar in time
    if len(bars) == 0:
        return None
    diffs = (bars.timestamp - target_utc).abs()
    return float(bars.iloc[diffs.argmin()].close)


for e in all_events:
    e["crm_price"] = nearest_bar_price(datetime.fromisoformat(e["ts_utc"]))

# "Now" price for the transformation-later question
now_price = float(bars.iloc[-1].close)
print(f"\ncurrent {TICKER} price: ${now_price:.2f}")

with OUT.open("w", encoding="utf-8") as f:
    for e in all_events:
        f.write(json.dumps(e) + "\n")

print(f"\nwrote {len(all_events)} rows to {OUT}")
print()
print(f"{'ts_local':<20}  {'arm':<3}  {'drift_pp':>9}  {'price':>8}  vs_now")
for e in all_events:
    p = e.get("crm_price")
    if p is None:
        print(f"  {e['ts_local']}  {e['arm']}  {e['max_delta_pp']:>7.1f}pp  no-bar")
    else:
        move_pct = (now_price - p) / p * 100
        print(f"  {e['ts_local']}  {e['arm']}  {e['max_delta_pp']:>7.1f}pp  ${p:>6.2f}  {move_pct:+.2f}%")
