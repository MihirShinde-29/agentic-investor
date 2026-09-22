"""End-of-day postmortem composer for the 3-arm A/B experiment.

Chains together three lenses on the day's run so I can write the
Fri wrap-up without hand-copying numbers:

1. Per-arm equity + fills + trigger mix (day_pnl_report).
2. Cross-arm gate audit for the finBERT-vs-Jev question
   (audit_jev_vs_finbert).
3. A-vs-B same-ticker divergence: for each ticker where BOTH arms
   traded today, list side/qty/count so I can see churn asymmetry
   at a glance (this file's own logic).

Writes a plain-text report to
`out/postmortems/<date>_close.txt` and prints the same to stdout.

Usage:
    python scripts/close_postmortem.py               # today
    python scripts/close_postmortem.py --date 2026-09-22
"""

from __future__ import annotations

import argparse
import glob
import json
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path


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


def _same_ticker_divergence(date: str) -> str:
    """For each ticker where BOTH A and B submitted orders today,
    show the side breakdown per arm. Highlights arms that acted on
    the same ticker in opposite directions, which is where the
    verdict-feedback effect should show up loudest.
    """
    by_arm_ticker: dict[str, dict[str, Counter]] = {
        "A": defaultdict(Counter),
        "B": defaultdict(Counter),
        "C": defaultdict(Counter),
    }
    for arm in "ABC":
        for r in _load_session(arm, date):
            if r.get("event") != "order_submitted":
                continue
            tk = r.get("ticker")
            side = r.get("side")
            if not tk or not side:
                continue
            by_arm_ticker[arm][tk][side] += 1

    tickers_ab = sorted(
        set(by_arm_ticker["A"].keys()) & set(by_arm_ticker["B"].keys())
    )

    lines = ["", "=== A vs B same-ticker action ==="]
    lines.append(f"{'ticker':<8}{'A buys':>8}{'A sells':>9}"
                 f"{'B buys':>8}{'B sells':>9}{'note':>18}")
    for tk in tickers_ab:
        a = by_arm_ticker["A"][tk]
        b = by_arm_ticker["B"][tk]
        a_b = a.get("buy", 0)
        a_s = a.get("sell", 0)
        b_b = b.get("buy", 0)
        b_s = b.get("sell", 0)
        note = ""
        if (a_b > 0 and b_s > 0 and a_s == 0 and b_b == 0):
            note = "OPPOSITE"
        elif (a_s > 0 and b_b > 0 and a_b == 0 and b_s == 0):
            note = "OPPOSITE"
        elif b_b + b_s > 2 * (a_b + a_s) and (a_b + a_s) > 0:
            note = "B churns more"
        elif a_b + a_s > 2 * (b_b + b_s) and (b_b + b_s) > 0:
            note = "A churns more"
        lines.append(f"{tk:<8}{a_b:>8}{a_s:>9}{b_b:>8}{b_s:>9}{note:>18}")
    return "\n".join(lines)


def _whipsaw_summary(date: str) -> str:
    """Whipsaw-guard activity on B (the arm that has it enabled).
    Shows which tickers the LLM keeps trying to flip within the
    15-min window - the fingerprint of verdict-feedback-induced
    reactivity.
    """
    rows = _load_session("B", date)
    per_ticker: Counter = Counter()
    for r in rows:
        if r.get("event") != "knob_fired":
            continue
        if r.get("name") != "whipsaw_guard":
            continue
        tk = r.get("ticker")
        if tk:
            per_ticker[tk] += 1
    lines = ["", "=== B whipsaw-guard drops (per ticker) ==="]
    if not per_ticker:
        lines.append("  (no whipsaws blocked today)")
        return "\n".join(lines)
    lines.append(f"  total drops: {sum(per_ticker.values())}")
    for tk, n in per_ticker.most_common(15):
        lines.append(f"  {tk:<8} {n}")
    return "\n".join(lines)


def _run_sub(argv: list[str]) -> str:
    try:
        out = subprocess.run(
            argv, capture_output=True, text=True, check=False, timeout=180,
        )
        return out.stdout + (
            f"\n[stderr]\n{out.stderr}" if out.stderr.strip() else ""
        )
    except Exception as e:  # noqa: BLE001
        return f"[{argv[0]} failed: {e}]"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date",
                    default=datetime.now(UTC).strftime("%Y-%m-%d"))
    args = ap.parse_args()

    py = sys.executable
    sections: list[str] = []
    sections.append(f"# close-time postmortem - {args.date}\n")
    sections.append("## 1. per-arm day P&L\n")
    sections.append(_run_sub([py, "scripts/day_pnl_report.py",
                              "--date", args.date]))
    sections.append("\n## 2. jev-vs-finbert cross-arm audit\n")
    sections.append(_run_sub([py, "scripts/audit_jev_vs_finbert.py",
                              "--date", args.date]))
    sections.append("\n## 3. A-vs-B same-ticker divergence\n")
    sections.append(_same_ticker_divergence(args.date))
    sections.append("\n## 4. whipsaw-guard fingerprint on B\n")
    sections.append(_whipsaw_summary(args.date))

    report = "\n".join(sections)

    out_dir = Path("out/postmortems")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.date}_close.txt"
    out_path.write_text(report, encoding="utf-8")

    print(report)
    print(f"\n[written to {out_path}]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
