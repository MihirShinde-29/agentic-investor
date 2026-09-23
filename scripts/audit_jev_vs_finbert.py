"""Cross-arm audit: what did each arm's news-materiality gate decide?

Compares C (Jev per-headline typed decision) against A + B (finBERT +
ticker-overlap heuristic) on the same news window. Produces per-arm
gate counts, agreement / disagreement breakdown, and lists the news
batches where the arms diverged so we can eyeball whether Jev's
extra skips were signal or over-blocking.

Data sources per arm (all under out/sessions/<date>T*_<arm>/):
- news_received: every headline the streamer emitted into that arm's
  event queue
- regen_start: batch-window-closed / force-regen / cooked-news-ready
  triggers that actually fired an LLM regen
- materiality_skip / materiality_bypass_promoted: A/B side, ticker
  overlap says no held ticker was mentioned (and no high-signal
  keyword bypass either)
- jev_materiality_gate: C only, per-headline Jev decision + prob

Usage:
  python scripts/audit_jev_vs_finbert.py             # today
  python scripts/audit_jev_vs_finbert.py --date 2026-09-22
  python scripts/audit_jev_vs_finbert.py --json      # machine output
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path


def _load_session(arm: str, date: str) -> list[dict]:
    """All session.jsonl rows across every session dir this arm had
    today (concatenated + sorted by ts). Multiple launches per day
    still count as one arm's story.
    """
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


def _arm_gate_stats(rows: list[dict], arm: str) -> dict:
    """Per-arm counts of gate-relevant events.

    A/B: n_news, n_regens_fired, n_material_skips (finBERT+overlap
         verdict of not-material), n_bypass_promotes (high-signal
         keyword bypass to on-deck)
    C:   same plus n_jev_blocks / n_jev_passes / avg_conf per decision.
    """
    c = Counter(r.get("event") for r in rows)
    out = {
        "arm": arm,
        "n_news_received": c.get("news_received", 0),
        "n_regen_start": c.get("regen_start", 0),
        "n_regen_done": c.get("regen_done", 0),
        "n_materiality_skip": c.get("materiality_skip", 0),
        "n_materiality_bypass_promoted": c.get("materiality_bypass_promoted", 0),
        "n_opinion_drift_skip": c.get("opinion_drift_skip", 0),
        "n_order_submitted": c.get("order_submitted", 0),
        "n_jev_gate": c.get("jev_materiality_gate", 0),
    }
    if arm == "C":
        jev_rows = [r for r in rows if r.get("event") == "jev_materiality_gate"]
        blocks = [r for r in jev_rows if not r.get("material")]
        passes = [r for r in jev_rows if r.get("material")]
        out["n_jev_blocks"] = len(blocks)
        out["n_jev_passes"] = len(passes)
        out["avg_jev_conf_blocks"] = round(
            sum(r.get("confidence", 0) for r in blocks) / max(len(blocks), 1), 3,
        )
        out["avg_jev_conf_passes"] = round(
            sum(r.get("confidence", 0) for r in passes) / max(len(passes), 1), 3,
        )
        out["n_jev_headlines_evaluated"] = sum(
            len(r.get("per_headline") or []) for r in jev_rows
        )
        out["n_jev_headlines_material"] = sum(
            1 for r in jev_rows
            for ph in (r.get("per_headline") or [])
            if ph.get("material")
        )
        ms_totals = [r.get("jev_ms_total", 0) for r in jev_rows
                     if r.get("jev_ms_total")]
        if ms_totals:
            out["avg_gate_ms"] = round(sum(ms_totals) / len(ms_totals), 1)
            out["max_gate_ms"] = round(max(ms_totals), 1)
    return out


def _arm_cost_estimate(rows: list[dict], arm: str) -> dict:
    """Rough cost estimate.

    - LLM cost from tick_cost events (already-summed by the loop).
    - Jev cost from headlines evaluated: ~50 tokens each * $0.042/M in.
    - Skips-saved-LLM: n_jev_blocks * ~$0.005 (avg LLM regen cost).
    """
    llm_cost = 0.0
    for r in rows:
        if r.get("event") == "tick_cost":
            raw = str(r.get("cost_usd", "$0")).replace("$", "")
            try:
                llm_cost += float(raw)
            except ValueError:
                pass
    jev_headlines = sum(
        len(r.get("per_headline") or [])
        for r in rows
        if r.get("event") == "jev_materiality_gate"
    )
    jev_cost = jev_headlines * 50 * 0.042 / 1_000_000
    jev_saved = 0.0
    if arm == "C":
        n_blocks = sum(
            1 for r in rows
            if r.get("event") == "jev_materiality_gate" and not r.get("material")
        )
        jev_saved = n_blocks * 0.005
    return {
        "arm": arm,
        "llm_cost_usd": round(llm_cost, 4),
        "jev_cost_usd": round(jev_cost, 6),
        "estimated_llm_saved_usd": round(jev_saved, 4) if arm == "C" else 0,
        "n_jev_calls": jev_headlines,
    }


def _headline_agreement(
    rows_a: list[dict], rows_b: list[dict], rows_c: list[dict],
) -> dict:
    """For each headline C evaluated via Jev, did A/B also see + act on it?

    Simplification: match by headline text (first 100 chars). If C
    blocked and A/B fired regens in a window covering that headline's
    receipt time, that's a disagreement worth logging.
    """
    def _ts(s: str) -> float:
        try:
            return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()
        except (ValueError, AttributeError):
            return 0.0

    def _regen_windows(rows):
        starts = [_ts(r["ts"]) for r in rows if r.get("event") == "regen_start"]
        return sorted(starts)

    a_starts = _regen_windows(rows_a)
    b_starts = _regen_windows(rows_b)

    def _fired_within(t: float, starts: list[float], window: float = 90.0) -> bool:
        return any(s - window <= t <= s + window for s in starts)

    c_jev_rows = [r for r in rows_c if r.get("event") == "jev_materiality_gate"]
    disagreements: list[dict] = []
    for gate in c_jev_rows:
        if gate.get("material"):
            continue  # Only care about C blocks
        g_ts = _ts(gate["ts"])
        if not _fired_within(g_ts, a_starts) and not _fired_within(g_ts, b_starts):
            continue  # Neither A nor B fired near this either - agreement
        per = gate.get("per_headline") or []
        disagreements.append({
            "ts": gate["ts"][:19],
            "n_headlines": gate.get("n_headlines"),
            "confidence": gate.get("confidence"),
            "a_fired_nearby": _fired_within(g_ts, a_starts),
            "b_fired_nearby": _fired_within(g_ts, b_starts),
            "sample_headlines": [
                (ph.get("prob"), (ph.get("headline") or "")[:80])
                for ph in per[:5]
            ],
        })
    return {
        "n_c_blocks": sum(1 for r in c_jev_rows if not r.get("material")),
        "n_disagreements": len(disagreements),
        "disagreements": disagreements,
    }


def _print_report(reports: list[dict], costs: list[dict], agreement: dict) -> None:
    print(f"{'arm':<4}{'news':>8}{'regens':>9}{'orders':>9}"
          f"{'skips':>8}{'jev_gates':>11}{'jev_block':>11}")
    for r in reports:
        n_jev = r.get("n_jev_gate", 0)
        n_blk = r.get("n_jev_blocks", "-")
        print(
            f"{r['arm']:<4}{r['n_news_received']:>8}"
            f"{r['n_regen_done']:>9}{r['n_order_submitted']:>9}"
            f"{r['n_materiality_skip']:>8}"
            f"{n_jev:>11}{n_blk if n_jev else '-':>11}"
        )

    print()
    print(f"{'arm':<4}{'llm_cost':>12}{'jev_cost':>12}"
          f"{'llm_saved':>12}{'jev_calls':>11}")
    for c in costs:
        print(
            f"{c['arm']:<4}${c['llm_cost_usd']:>10.4f}"
            f"${c['jev_cost_usd']:>10.6f}"
            f"${c['estimated_llm_saved_usd']:>10.4f}"
            f"{c['n_jev_calls']:>11}"
        )

    print()
    print(f"=== C-blocked-but-A-or-B-fired disagreements: "
          f"{agreement['n_disagreements']} of {agreement['n_c_blocks']} C blocks ===")
    for d in agreement["disagreements"][:20]:
        marks = []
        if d["a_fired_nearby"]:
            marks.append("A")
        if d["b_fired_nearby"]:
            marks.append("B")
        print(f"\n{d['ts']}  n={d['n_headlines']}  conf={d['confidence']:.2f}"
              f"  fired_by=[{','.join(marks)}]")
        for prob, h in d["sample_headlines"]:
            print(f"   {prob:.2f}  {h}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date",
                    default=datetime.now(UTC).strftime("%Y-%m-%d"),
                    help="YYYY-MM-DD (UTC); default today")
    ap.add_argument("--json", action="store_true",
                    help="emit JSON instead of the text report")
    args = ap.parse_args()

    rows_by_arm: dict[str, list[dict]] = {}
    for arm in "ABC":
        rows = _load_session(arm, args.date)
        if not rows:
            print(f"arm {arm}: no session data for {args.date}",
                  file=sys.stderr)
            continue
        rows_by_arm[arm] = rows
    if not rows_by_arm:
        return 1

    reports = [_arm_gate_stats(rows, arm) for arm, rows in rows_by_arm.items()]
    costs = [_arm_cost_estimate(rows, arm) for arm, rows in rows_by_arm.items()]
    agreement = _headline_agreement(
        rows_by_arm.get("A", []),
        rows_by_arm.get("B", []),
        rows_by_arm.get("C", []),
    )

    if args.json:
        json.dump({
            "date": args.date,
            "stats": reports,
            "costs": costs,
            "agreement": agreement,
        }, sys.stdout, indent=2, default=str)
        print()
        return 0

    print(f"jev-vs-finbert audit — {args.date}")
    _print_report(reports, costs, agreement)
    return 0


if __name__ == "__main__":
    sys.exit(main())
