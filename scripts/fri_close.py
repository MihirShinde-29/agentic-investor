"""Friday close: one-shot end-of-week close for the Jev A/B.

Chains today's close postmortem + the 5-day week postmortem +
a per-day head-to-head hypothesis check ("does C systematically
beat A/B on shared tickers, or is C's aggregate win concentrated
in exclusive tickers + outlier position sizes?") into one text
report at `out/postmortems/<fri>_fri_close_summary.txt`.

Only new logic here is the head-to-head; the two postmortems are
already shipped, we just call them. The head-to-head parses each
day's close_postmortem stdout for the "top winners"/"top losers"
per-ticker rows, cross-references across A/B/C, and tallies where
C landed for every shared-ticker head-to-head.

Usage:
    python scripts/fri_close.py                   # this Fri
    python scripts/fri_close.py --end 2026-09-26  # specific Fri
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path


TICKER_ROW = re.compile(
    r"^\s*(?P<tk>[A-Z][A-Z0-9.\-]+)\s+\$"
    r"(?P<sign>[+-])(?P<amt>[0-9]+\.[0-9]+)\s+\(",
)


def _run(argv: list[str]) -> str:
    try:
        r = subprocess.run(
            argv, capture_output=True, text=True, check=False, timeout=300,
        )
        return r.stdout + (
            f"\n[stderr]\n{r.stderr}" if r.stderr.strip() else ""
        )
    except Exception as e:  # noqa: BLE001
        return f"[{argv[0]} failed: {e}]"


def _parse_day_pnl(day_text: str) -> dict[str, dict[str, float]]:
    """Parse close_postmortem output for one day.
    Returns {arm: {ticker: pnl_usd}} covering both winners + losers.
    """
    out: dict[str, dict[str, float]] = {}
    current_arm = None
    in_pnl_section = False
    for line in day_text.splitlines():
        s = line.strip()
        if s.startswith("== arm ") and s.endswith("=="):
            current_arm = s.replace("== arm ", "").replace(" ==", "").strip()
            out.setdefault(current_arm, {})
            in_pnl_section = False
            continue
        if s.startswith("top P&L winners:") or s.startswith("top P&L losers:"):
            in_pnl_section = True
            continue
        if not in_pnl_section or current_arm is None:
            continue
        m = TICKER_ROW.match(line)
        if m:
            tk = m.group("tk")
            amt = float(m.group("amt"))
            if m.group("sign") == "-":
                amt = -amt
            out[current_arm][tk] = amt
        elif s and not line.startswith(" "):
            # left the pnl block
            in_pnl_section = False
    return out


def _head_to_head_one_day(day_pnl: dict[str, dict[str, float]]) -> dict:
    """For one day's per-arm/per-ticker table, find shared tickers
    (all 3 arms traded them) and score C's rank + edge vs mean(A,B).
    Excludes tickers only 1-2 arms touched -- those are picker
    structural effects, not head-to-head signal quality.
    """
    a, b, c = day_pnl.get("A", {}), day_pnl.get("B", {}), day_pnl.get("C", {})
    shared = sorted(set(a) & set(b) & set(c))
    rows = []
    c_best = c_mid = c_worst = 0
    c_edge_sum = 0.0
    for tk in shared:
        av, bv, cv = a[tk], b[tk], c[tk]
        top, bot = max(av, bv, cv), min(av, bv, cv)
        if cv == top:
            verdict = "C best"
            c_best += 1
        elif cv == bot:
            verdict = "C worst"
            c_worst += 1
        else:
            verdict = "C middle"
            c_mid += 1
        mean_ab = (av + bv) / 2.0
        edge = cv - mean_ab
        c_edge_sum += edge
        rows.append({"ticker": tk, "A": av, "B": bv, "C": cv,
                     "verdict": verdict, "c_vs_mean_ab": edge})
    return {
        "n_shared": len(shared),
        "rows": rows,
        "c_best": c_best,
        "c_middle": c_mid,
        "c_worst": c_worst,
        "c_edge_vs_mean_ab": round(c_edge_sum, 2),
    }


def _fmt_head_to_head(date: str, h2h: dict) -> str:
    lines = [f"\n### {date} head-to-head ({h2h['n_shared']} shared tickers)"]
    if not h2h["rows"]:
        lines.append("  (no shared tickers today)")
        return "\n".join(lines)
    lines.append(f"  {'ticker':<8}{'A':>10}{'B':>10}{'C':>10}"
                 f"   {'verdict':<10}{'C vs mean(A,B)':>18}")
    for r in h2h["rows"]:
        lines.append(
            f"  {r['ticker']:<8}${r['A']:>+9.2f}${r['B']:>+9.2f}"
            f"${r['C']:>+9.2f}   {r['verdict']:<10}${r['c_vs_mean_ab']:>+15.2f}"
        )
    lines.append(f"  counts: C-best {h2h['c_best']}/{h2h['n_shared']}, "
                 f"C-middle {h2h['c_middle']}/{h2h['n_shared']}, "
                 f"C-worst {h2h['c_worst']}/{h2h['n_shared']}")
    lines.append(f"  aggregate C edge vs mean(A,B): "
                 f"${h2h['c_edge_vs_mean_ab']:+.2f}")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--end", default=None,
                    help="Friday date YYYY-MM-DD (default: today)")
    args = ap.parse_args()

    end = args.end or datetime.now(UTC).strftime("%Y-%m-%d")
    end_dt = datetime.strptime(end, "%Y-%m-%d")
    week_dates = [(end_dt - timedelta(days=4 - i)).strftime("%Y-%m-%d")
                  for i in range(5)]

    py = sys.executable
    sections: list[str] = []
    sections.append(f"# Fri close summary - week ending {end}\n")

    # 1. Today's close postmortem (Fri only)
    sections.append("## 1. Fri per-arm close\n")
    fri_report = _run([py, "scripts/close_postmortem.py", "--date", end])
    sections.append(fri_report)

    # 2. 5-day week postmortem
    sections.append("\n## 2. 5-day week postmortem\n")
    week_report = _run([py, "scripts/week_postmortem.py", "--end", end])
    sections.append(week_report)

    # 3. Per-day + cumulative head-to-head hypothesis test
    sections.append("\n## 3. head-to-head hypothesis (Jev makes LLM smarter?)\n")
    sections.append("null hypothesis: C's per-ticker P&L is drawn from the "
                    "same distribution as A/B on shared tickers.")
    sections.append("expected under null: C-best ~= C-middle ~= C-worst ~= n/3, "
                    "aggregate C-edge ~0.")
    sections.append("supported (hypothesis wrong = null holds) if C's counts "
                    "distribute uniformly and edge is ~0.")

    total_best = total_mid = total_worst = 0
    total_edge = 0.0
    total_shared = 0
    for date in week_dates:
        day_text = _run([py, "scripts/close_postmortem.py", "--date", date])
        day_pnl = _parse_day_pnl(day_text)
        h2h = _head_to_head_one_day(day_pnl)
        sections.append(_fmt_head_to_head(date, h2h))
        total_best += h2h["c_best"]
        total_mid += h2h["c_middle"]
        total_worst += h2h["c_worst"]
        total_edge += h2h["c_edge_vs_mean_ab"]
        total_shared += h2h["n_shared"]

    sections.append(f"\n### cumulative across all 5 days ({total_shared} shared heads)")
    sections.append(f"  C-best: {total_best}/{total_shared}")
    sections.append(f"  C-middle: {total_mid}/{total_shared}")
    sections.append(f"  C-worst: {total_worst}/{total_shared}")
    sections.append(f"  aggregate C edge vs mean(A,B): ${total_edge:+.2f}")
    if total_shared:
        ratio = total_best / total_shared
        if ratio >= 0.55:
            verdict = ("HYPOTHESIS SUPPORTED (C-best rate above uniform-3 "
                       "baseline of 0.33)")
        elif ratio <= 0.25:
            verdict = ("HYPOTHESIS REJECTED (C-best rate below uniform-3 "
                       "baseline; C systematically worse on shared)")
        else:
            verdict = ("HYPOTHESIS NOT SUPPORTED (C-best rate ~ uniform-3; "
                       "no evidence Jev cleans signal for the LLM)")
        sections.append(f"  verdict: {verdict}")

    report = "\n".join(sections)
    out_dir = Path("out/postmortems")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{end}_fri_close_summary.txt"
    out_path.write_text(report, encoding="utf-8")
    print(report[-3000:])
    print(f"\n[full report written to {out_path}]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
