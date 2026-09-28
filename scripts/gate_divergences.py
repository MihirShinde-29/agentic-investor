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
    python scripts/gate_divergences.py                   # today
    python scripts/gate_divergences.py --date 2026-09-28
    python scripts/gate_divergences.py --stdout          # skip file
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from datetime import date, datetime
from pathlib import Path

LOG = Path("out/logs/experiment.out")
OUT_DIR = Path("out/analytics")

GATE_RE = re.compile(
    r"^\[([BC])\] (\d{4}-\d{2}-\d{2}) (\d{2}:\d{2}:\d{2}),\d{3} .*"
    r"\[(laya_materiality_gate|jev_materiality_gate)\] "
    r"material=(True|False) confidence=([0-9.]+) "
    r"from_(?:laya|jev)=(?:True|False) n_headlines=(\d+)"
)

# Batches close within a few seconds across arms; 15s is generous
# enough to survive news-bus jitter but tight enough that we don't
# accidentally pair adjacent unrelated batches.
PAIR_WINDOW_SEC = 15.0


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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=None,
                    help="YYYY-MM-DD, default today (UTC)")
    ap.add_argument("--stdout", action="store_true",
                    help="print JSONL to stdout instead of writing a file")
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

    lines = [json.dumps(d) for d in divergent]
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
