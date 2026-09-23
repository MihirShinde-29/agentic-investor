"""Weekly 5-day postmortem composer for the Jev A/B experiment.

Runs close_postmortem for each trading day in the requested week,
then adds three cross-day lenses on top:

1. Cumulative per-arm P&L across the 5 days + baseline compare
   (Sept 14-18 M14.5 baseline hardcoded: A -$70, B -$75, C -$168).
2. Jev block-rate stability + per-headline probability distribution
   aggregated across the whole week (bucketed histogram).
3. Verdict-feedback regret consistency on arm B: how often did the
   loser-chasing pattern (whipsaw drops correlate with per-ticker
   losses) show up day over day.

Output: `out/postmortems/<week_end>_week.txt`, plus per-day files
already produced by close_postmortem.

Usage:
    python scripts/week_postmortem.py              # this week (Mon..Fri)
    python scripts/week_postmortem.py --end 2026-09-26
    python scripts/week_postmortem.py --end 2026-09-26 --json
"""

from __future__ import annotations

import argparse
import glob
import json
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import UTC, datetime, timedelta
from pathlib import Path

BASELINE_LAST_WEEK = {"A": -70.0, "B": -75.0, "C": -168.0}


def _week_dates(end_date: str) -> list[str]:
    """Return 5 consecutive dates ending on `end_date` inclusive.
    Caller supplies the target Friday; we do not snap to the closest
    weekday because the manifest fixes Mon..Fri per experiment week
    and we don't want the script silently shifting the window.
    """
    d = datetime.strptime(end_date, "%Y-%m-%d")
    return [(d - timedelta(days=4 - i)).strftime("%Y-%m-%d")
            for i in range(5)]


def _load_session(arm: str, date: str) -> list[dict]:
    rows: list[dict] = []
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
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    rows.sort(key=lambda r: r.get("ts", ""))
    return rows


def _run_day_postmortem(date: str) -> str:
    """Shell to close_postmortem for one day. Its side effect writes
    the per-day file; we capture stdout to embed in the week report.
    """
    try:
        out = subprocess.run(
            [sys.executable, "scripts/close_postmortem.py", "--date", date],
            capture_output=True, text=True, timeout=240, check=False,
        )
        return out.stdout
    except Exception as e:  # noqa: BLE001
        return f"[close_postmortem for {date} failed: {e}]"


def _extract_day_pnl(day_report: str) -> dict[str, float]:
    """Grep the "day P&L: $..." line out of one day's close report.
    Cheaper than re-computing; the composed report already has it.
    """
    out: dict[str, float] = {}
    current_arm = None
    for line in day_report.splitlines():
        s = line.strip()
        if s.startswith("== arm ") and s.endswith("=="):
            current_arm = s.replace("== arm ", "").replace(" ==", "").strip()
        elif current_arm and "day P&L: $" in s:
            try:
                token = s.split("day P&L: $", 1)[1].split()[0]
                out[current_arm] = float(token.replace("+", ""))
                current_arm = None
            except (ValueError, IndexError):
                pass
    return out


def _week_jev_histogram(dates: list[str]) -> tuple[dict[str, int], int]:
    """Per-headline probability histogram aggregated across the week
    from arm C's session.jsonl. Same buckets as day-1 analysis so
    week-over-week compare stays honest.
    """
    buckets = {
        "0.00-0.20": 0, "0.20-0.40": 0, "0.40-0.60": 0,
        "0.60-0.80": 0, "0.80-1.00": 0,
    }
    total = 0
    for date in dates:
        for r in _load_session("C", date):
            if r.get("event") != "jev_materiality_gate":
                continue
            for ph in (r.get("per_headline") or []):
                p = ph.get("prob", 0)
                total += 1
                if p < 0.2:
                    buckets["0.00-0.20"] += 1
                elif p < 0.4:
                    buckets["0.20-0.40"] += 1
                elif p < 0.6:
                    buckets["0.40-0.60"] += 1
                elif p < 0.8:
                    buckets["0.60-0.80"] += 1
                else:
                    buckets["0.80-1.00"] += 1
    return buckets, total


def _week_whipsaw_pattern(dates: list[str]) -> dict:
    """Arm B whipsaw drops per ticker across the week + which tickers
    kept whipsawing multiple days in a row (the strongest signal
    that verdict-feedback induces persistent loser-chasing).
    """
    per_day_per_ticker: dict[str, Counter] = defaultdict(Counter)
    for date in dates:
        for r in _load_session("B", date):
            if (r.get("event") == "knob_fired"
                    and r.get("name") == "whipsaw_guard"
                    and r.get("ticker")):
                per_day_per_ticker[date][r["ticker"]] += 1
    total_per_ticker: Counter = Counter()
    for c in per_day_per_ticker.values():
        for tk, n in c.items():
            total_per_ticker[tk] += n
    days_with_drops_per_ticker: Counter = Counter()
    for c in per_day_per_ticker.values():
        for tk in c:
            days_with_drops_per_ticker[tk] += 1
    return {
        "total_per_ticker": dict(total_per_ticker.most_common(15)),
        "days_active_per_ticker": dict(
            days_with_drops_per_ticker.most_common(15)
        ),
        "persistent_tickers": [
            tk for tk, days in days_with_drops_per_ticker.items()
            if days >= 3
        ],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--end", default=None,
                    help="Fri of the target week (YYYY-MM-DD); default this Fri")
    ap.add_argument("--json", action="store_true",
                    help="emit JSON instead of text")
    args = ap.parse_args()

    if args.end is None:
        today = datetime.now(UTC).date()
        offset = (today.weekday() - 4) % 7
        end = (today - timedelta(days=offset)).strftime("%Y-%m-%d")
    else:
        end = args.end
    dates = _week_dates(end)

    per_day: dict[str, str] = {}
    per_day_pnl: dict[str, dict[str, float]] = {}
    for date in dates:
        day_report = _run_day_postmortem(date)
        per_day[date] = day_report
        per_day_pnl[date] = _extract_day_pnl(day_report)

    cum: dict[str, float] = {"A": 0, "B": 0, "C": 0}
    for pnl in per_day_pnl.values():
        for arm in "ABC":
            cum[arm] += pnl.get(arm, 0)

    histogram, total_hist = _week_jev_histogram(dates)
    whipsaws = _week_whipsaw_pattern(dates)

    if args.json:
        json.dump({
            "week_end": end,
            "dates": dates,
            "per_day_pnl": per_day_pnl,
            "cumulative": cum,
            "baseline_last_week": BASELINE_LAST_WEEK,
            "jev_histogram": histogram,
            "jev_hist_total": total_hist,
            "whipsaws": whipsaws,
        }, sys.stdout, indent=2)
        print()
        return 0

    lines: list[str] = [f"# week postmortem - week ending {end}", ""]
    lines.append("## 1. per-day P&L")
    lines.append(f"  {'date':<12}{'A':>10}{'B':>10}{'C':>10}"
                 f"{'combined':>12}")
    for date in dates:
        p = per_day_pnl.get(date, {})
        a, b, c = p.get("A", 0), p.get("B", 0), p.get("C", 0)
        lines.append(f"  {date:<12}${a:>+9.2f}${b:>+9.2f}${c:>+9.2f}"
                     f"${a+b+c:>+11.2f}")
    lines.append("")
    lines.append("## 2. cumulative vs last week (M14.5 baseline)")
    for arm in "ABC":
        base = BASELINE_LAST_WEEK[arm]
        this_wk = cum[arm]
        delta = this_wk - base
        arrow = "up" if delta > 0 else "down"
        lines.append(
            f"  arm {arm}: this week ${this_wk:+.2f} vs baseline ${base:+.2f} "
            f"({arrow} ${abs(delta):.2f})"
        )
    lines.append("")
    lines.append("## 3. Jev per-headline probability histogram (whole week)")
    for b, n in histogram.items():
        pct = 100 * n / max(total_hist, 1)
        bar = "#" * int(pct / 2)
        lines.append(f"  {b}: {n:5d}  ({pct:5.1f}%)  {bar}")
    lines.append(f"  total per-headline calls: {total_hist}")
    lines.append("")
    lines.append("## 4. arm B whipsaw persistence")
    lines.append(f"  top tickers by total drops: "
                 f"{whipsaws['total_per_ticker']}")
    lines.append(f"  days active (out of 5): "
                 f"{whipsaws['days_active_per_ticker']}")
    lines.append(
        f"  persistent tickers (whipsawed >=3 of 5 days): "
        f"{whipsaws['persistent_tickers']}"
    )
    lines.append("")
    lines.append("## 5. per-day reports (embedded)")
    for date in dates:
        lines.append(f"\n### {date}\n")
        lines.append(per_day.get(date, "(missing)"))

    report = "\n".join(lines)
    out_dir = Path("out/postmortems")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{end}_week.txt"
    out_path.write_text(report, encoding="utf-8")
    print(report[:4000])
    print(f"\n[full report written to {out_path}]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
