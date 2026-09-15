"""Capture every filter-drop event with the market price at that moment.

Scans A/B/C session logs for:
  - opinion_drift_skip (whole regen dropped)
  - materiality_skip (news batch dropped)
  - cite_to_trade blocked_N (some plans dropped)
  - pre_market_hold (plans held to open)
  - finbert_skip (sentiment-flat)

For events with an identified ticker, fetches Alpaca IEX 1-min close at the
skip timestamp plus t+15 / t+30 / t+60 min follow-ups so we can score "was
the LLM's aborted trade right?" retrospectively.

Idempotent: writes JSONL keyed by (arm, ts_local, skip_type). Re-run to
refresh follow-up prices as more time passes.

Feeds the future prompt-injection loop that will let the LLM see its own
recent aborted trades and their subsequent price action.
"""

import json
import os
import pathlib
import re
from datetime import UTC, datetime, timedelta

from dotenv import load_dotenv

load_dotenv()

BASE = pathlib.Path("out/experiments/reasoning-quality")
OUT = BASE / "skip_events.jsonl"
ET_OFFSET = timedelta(hours=-4)  # EDT

# One regex per skip flavor. Each captures whatever fields the event emits.
RE_DRIFT = re.compile(
    r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}).*"
    r"\[opinion_drift_skip\].*avg_drift_pp=([\d.]+).*"
    r"max_delta_pp=([\d.]+).*max_delta_ticker=(\S+)"
)
RE_MATERIALITY = re.compile(
    r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}).*"
    r"\[materiality_skip\].*batch_tickers=\[(\d+)\].*material_count=(\d+)"
)
RE_CITE = re.compile(
    r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}).*"
    r"\[knob_fired\] name=cite_to_trade reason=blocked_(\d+)"
)
RE_PREMKT = re.compile(
    r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}).*"
    r"\[pre_market_hold\] rec_id=(\d+) plan_count=(\d+)"
)
RE_FINBERT = re.compile(
    r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}).*"
    r"\[finbert_skip\].*score=([-\d.]+).*delta=([\d.]+)"
)


def _parse_arm(arm: str) -> list[dict]:
    p = BASE / f"{arm}.log"
    if not p.exists():
        return []
    out = []
    for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
        m = RE_DRIFT.search(line)
        if m:
            out.append({
                "arm": arm, "ts_local": m.group(1), "skip_type": "opinion_drift",
                "avg_drift_pp": float(m.group(2)),
                "max_delta_pp": float(m.group(3)),
                "ticker": m.group(4).strip(),
            })
            continue
        m = RE_MATERIALITY.search(line)
        if m:
            out.append({
                "arm": arm, "ts_local": m.group(1), "skip_type": "materiality",
                "batch_tickers": int(m.group(2)),
                "material_count": int(m.group(3)),
                "ticker": None,
            })
            continue
        m = RE_CITE.search(line)
        if m:
            out.append({
                "arm": arm, "ts_local": m.group(1), "skip_type": "cite_to_trade",
                "blocked_count": int(m.group(2)),
                "ticker": None,  # cite_to_trade doesn't emit which ticker(s)
            })
            continue
        m = RE_PREMKT.search(line)
        if m:
            out.append({
                "arm": arm, "ts_local": m.group(1), "skip_type": "pre_market_hold",
                "rec_id": int(m.group(2)),
                "plan_count": int(m.group(3)),
                "ticker": None,
            })
            continue
        m = RE_FINBERT.search(line)
        if m:
            out.append({
                "arm": arm, "ts_local": m.group(1), "skip_type": "finbert",
                "sentiment_score": float(m.group(2)),
                "sentiment_delta": float(m.group(3)),
                "ticker": None,
            })
    return out


def _load_existing() -> dict:
    """Load existing rows keyed by (arm, ts_local, skip_type). Lets re-runs
    update follow-up prices without duplicating."""
    if not OUT.exists():
        return {}
    idx = {}
    with OUT.open("r", encoding="utf-8") as f:
        for line in f:
            e = json.loads(line)
            idx[(e["arm"], e["ts_local"], e["skip_type"])] = e
    return idx


def main() -> None:
    all_events = []
    for arm in ("A", "B", "C"):
        all_events.extend(_parse_arm(arm))

    if not all_events:
        print("no skip events found across arms")
        return

    all_events.sort(key=lambda e: e["ts_local"])

    # Batch fetch bars per ticker we care about (opinion_drift is the only
    # skip type that names a ticker; others we skip pricing on).
    tickers_needed = {e["ticker"] for e in all_events if e.get("ticker")}
    tickers_needed.discard(None)
    tickers_needed.discard("?")

    bars_by_ticker = {}
    if tickers_needed:
        try:
            from alpaca.data.enums import DataFeed
            from alpaca.data.historical import StockHistoricalDataClient
            from alpaca.data.requests import StockBarsRequest
            from alpaca.data.timeframe import TimeFrame

            earliest = min(
                datetime.strptime(e["ts_local"], "%Y-%m-%d %H:%M:%S") - ET_OFFSET
                for e in all_events if e.get("ticker") in tickers_needed
            ).replace(tzinfo=UTC)
            now_utc = datetime.now(UTC)

            data_cli = StockHistoricalDataClient(
                os.environ["ALPACA_API_KEY"], os.environ["ALPACA_API_SECRET"],
            )
            req = StockBarsRequest(
                symbol_or_symbols=sorted(tickers_needed),
                timeframe=TimeFrame.Minute,
                start=earliest - timedelta(minutes=2),
                end=now_utc + timedelta(minutes=1),
                feed=DataFeed.IEX,
            )
            df = data_cli.get_stock_bars(req).df
            if not df.empty:
                for tk in tickers_needed:
                    if tk in df.index.get_level_values(0):
                        bars_by_ticker[tk] = df.loc[tk].reset_index()
        except Exception as e:
            print(f"warn: bar fetch failed: {e}")

    def _price_at(tk: str, target_utc: datetime, offset_min: int = 0) -> float | None:
        if tk not in bars_by_ticker:
            return None
        bars = bars_by_ticker[tk]
        t = target_utc + timedelta(minutes=offset_min)
        # Nearest bar in time. IEX has gaps outside market hours.
        diffs = (bars.timestamp - t).abs()
        if diffs.min() > timedelta(minutes=5):
            return None
        return float(bars.iloc[diffs.argmin()].close)

    existing = _load_existing()

    # Attach prices to each event and merge with existing
    for e in all_events:
        tk = e.get("ticker")
        if tk and tk in bars_by_ticker:
            dt_local = datetime.strptime(e["ts_local"], "%Y-%m-%d %H:%M:%S")
            dt_utc = (dt_local - ET_OFFSET).replace(tzinfo=UTC)
            e["ts_utc"] = dt_utc.isoformat()
            e["price_at_skip"] = _price_at(tk, dt_utc, 0)
            e["price_15m"] = _price_at(tk, dt_utc, 15)
            e["price_30m"] = _price_at(tk, dt_utc, 30)
            e["price_60m"] = _price_at(tk, dt_utc, 60)
            if e.get("price_at_skip") and bars_by_ticker[tk] is not None:
                now_close = float(bars_by_ticker[tk].iloc[-1].close)
                e["price_now"] = now_close
                e["move_since_skip_pct"] = round(
                    (now_close - e["price_at_skip"]) / e["price_at_skip"] * 100, 3,
                )
        existing[(e["arm"], e["ts_local"], e["skip_type"])] = e

    # Sort chronologically for a clean file
    merged = sorted(existing.values(), key=lambda x: x["ts_local"])
    with OUT.open("w", encoding="utf-8") as f:
        for e in merged:
            f.write(json.dumps(e) + "\n")

    # Summary
    print(f"wrote {len(merged)} events to {OUT}")
    from collections import Counter
    kinds = Counter((e["arm"], e["skip_type"]) for e in merged)
    print("\nby arm x skip_type:")
    for (arm, kind), n in sorted(kinds.items()):
        print(f"  {arm}  {kind:20}  n={n}")

    with_prices = [e for e in merged if e.get("price_at_skip")]
    if with_prices:
        print(f"\n{len(with_prices)} events priced. sample:")
        print(f"  {'arm':<3} {'ts_local':<20} {'ticker':<6} {'drift':>7} "
              f"{'price':>8} {'move':>8}")
        for e in with_prices[-10:]:
            move = e.get("move_since_skip_pct")
            move_s = f"{move:+.2f}%" if move is not None else "-"
            drift = e.get("max_delta_pp", 0)
            print(f"  {e['arm']:<3} {e['ts_local']:<20} {e.get('ticker','?'):<6} "
                  f"{drift:>5.1f}pp  ${e['price_at_skip']:>6.2f} {move_s:>8}")


if __name__ == "__main__":
    main()
