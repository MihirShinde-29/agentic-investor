"""Post-hoc benchmark: Laya vs Jev on this week's real headlines.

Reads every jev_materiality_gate event in arm C's session.jsonl for
the requested date range, extracts (headline, portfolio_tickers,
jev_prob, jev_material) tuples, and (once Laya is installed) runs
Laya on the same headline+portfolio and produces:

- Per-headline agreement rate (both material / both not-material)
- Confusion matrix: Jev-material vs Laya-material
- Correlation of prob outputs (Pearson on the two probability streams)
- Latency compare: Jev's jev_ms_max/jev_ms_total from the log vs
  Laya's wall-clock from a local run

Guarded so the Jev-extraction half runs standalone today (no Laya
dependency needed) — call with --extract-only to write the input
corpus to a JSON file that a later Laya run can consume.

Usage (extract-only, safe to run now):
    python scripts/benchmark_laya_vs_jev.py --from 2026-09-22 \\
        --to 2026-09-26 --extract-only \\
        --out out/benchmarks/jev_corpus.jsonl

Usage (full compare, needs `pip install laya` post-install):
    python scripts/benchmark_laya_vs_jev.py --from 2026-09-22 \\
        --to 2026-09-26 --corpus out/benchmarks/jev_corpus.jsonl
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path


def _iter_dates(from_date: str, to_date: str):
    a = datetime.strptime(from_date, "%Y-%m-%d")
    b = datetime.strptime(to_date, "%Y-%m-%d")
    d = a
    while d <= b:
        yield d.strftime("%Y-%m-%d")
        d += timedelta(days=1)


def _load_c_jev_events(dates: list[str]) -> list[dict]:
    """Every jev_materiality_gate event across arm C's sessions
    for the requested dates. Deduped by (ts, first_headline_100)
    to survive session-dir rollovers on relaunches.
    """
    events: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for date in dates:
        for d in sorted(glob.glob(f"out/sessions/{date}T*_C/")):
            p = Path(d) / "session.jsonl"
            if not p.exists():
                continue
            with p.open("r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        r = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if r.get("event") != "jev_materiality_gate":
                        continue
                    per = r.get("per_headline") or []
                    if not per:
                        continue
                    key = (r.get("ts", ""), (per[0].get("headline") or "")[:100])
                    if key in seen:
                        continue
                    seen.add(key)
                    events.append(r)
    return events


def _events_to_corpus(events: list[dict]) -> list[dict]:
    """Flatten each gate event into per-headline rows: one row per
    (headline, jev_prob, jev_material, batch_ts, n_portfolio).
    n_portfolio is Jev's context ('this holding is size K'); the
    exact ticker list isn't in the session event so we log the
    count only — good enough for benchmark input, precise enough
    for a Laya run since Laya's model doesn't need the ticker set.
    """
    rows = []
    for r in events:
        ts = r.get("ts", "")
        n_port = r.get("n_portfolio", 0)
        for ph in (r.get("per_headline") or []):
            rows.append({
                "batch_ts": ts,
                "n_portfolio": n_port,
                "headline": ph.get("headline") or "",
                "jev_prob": ph.get("prob"),
                "jev_material": ph.get("material"),
            })
    return rows


def _write_corpus(rows: list[dict], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def _read_corpus(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _run_laya(rows: list[dict]) -> list[dict]:
    """Attach Laya's decision to each row. Same instruction as Jev's
    per-headline gate so the two are comparable head-to-head.
    """
    try:
        import laya  # type: ignore
    except ImportError:
        print("laya not installed - pip install laya", file=sys.stderr)
        print("(use --extract-only to just dump the Jev corpus)",
              file=sys.stderr)
        sys.exit(2)
    import time as _time

    import torch  # type: ignore
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"laya device: {device}", file=sys.stderr)
    agent = laya.Agent("convaiinnovations/laya", device=device)
    instructions = (
        "This single headline would change the risk or return outlook "
        "for at least one ticker in the current portfolio at a "
        "magnitude worth re-evaluating positions for"
    )
    out: list[dict] = []
    for i, r in enumerate(rows):
        state = (
            f"portfolio: (holds {r.get('n_portfolio', 0)} tickers)\n"
            f"headline: {r['headline'][:400]}"
        )
        t0 = _time.perf_counter()
        try:
            resp = agent.system_one(
                state=state,
                questions={"material": {
                    "type": "noul", "instructions": instructions,
                }},
            )
            ms = (_time.perf_counter() - t0) * 1000.0
            prob = float(resp["answers"]["material"].get("noul", 0.5))
        except Exception as e:  # noqa: BLE001
            print(f"laya call {i} failed: {e}", file=sys.stderr)
            prob = 0.5
            ms = 0.0
        out.append({
            **r,
            "laya_prob": prob,
            "laya_material": prob >= 0.5,
            "laya_ms": round(ms, 1),
        })
        if (i + 1) % 100 == 0:
            print(f"  laya progress: {i+1}/{len(rows)}", file=sys.stderr)
    return out


def _compare(rows: list[dict]) -> dict:
    """Confusion matrix + agreement stats + prob correlation."""
    n = len(rows)
    if not n:
        return {"n": 0}
    agree_both_yes = sum(
        1 for r in rows if r["jev_material"] and r.get("laya_material")
    )
    agree_both_no = sum(
        1 for r in rows if not r["jev_material"] and not r.get("laya_material")
    )
    jev_yes_laya_no = sum(
        1 for r in rows if r["jev_material"] and not r.get("laya_material")
    )
    jev_no_laya_yes = sum(
        1 for r in rows if not r["jev_material"] and r.get("laya_material")
    )
    agreement = (agree_both_yes + agree_both_no) / n

    # Pearson on the two probability streams (no numpy dep - hand-rolled).
    xs = [float(r["jev_prob"] or 0) for r in rows]
    ys = [float(r.get("laya_prob") or 0) for r in rows]
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True))
    dx = sum((x - mx) ** 2 for x in xs) ** 0.5
    dy = sum((y - my) ** 2 for y in ys) ** 0.5
    corr = num / (dx * dy) if dx and dy else 0.0

    return {
        "n": n,
        "agreement_rate": round(agreement, 3),
        "both_material": agree_both_yes,
        "both_non_material": agree_both_no,
        "jev_yes_laya_no": jev_yes_laya_no,
        "jev_no_laya_yes": jev_no_laya_yes,
        "prob_correlation": round(corr, 3),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="from_date", required=True,
                    help="YYYY-MM-DD inclusive")
    ap.add_argument("--to", dest="to_date", required=True,
                    help="YYYY-MM-DD inclusive")
    ap.add_argument("--extract-only", action="store_true",
                    help="dump the Jev corpus to --out and exit")
    ap.add_argument("--out", default="out/benchmarks/jev_corpus.jsonl")
    ap.add_argument("--corpus", default=None,
                    help="skip extraction, load pre-dumped corpus")
    args = ap.parse_args()

    if args.corpus:
        rows = _read_corpus(Path(args.corpus))
        print(f"loaded {len(rows)} pre-extracted rows from {args.corpus}")
    else:
        dates = list(_iter_dates(args.from_date, args.to_date))
        events = _load_c_jev_events(dates)
        rows = _events_to_corpus(events)
        print(f"extracted {len(rows)} per-headline decisions from "
              f"{len(events)} gate events across {len(dates)} dates")

    if args.extract_only:
        _write_corpus(rows, Path(args.out))
        print(f"wrote corpus to {args.out}")
        return 0

    rows = _run_laya(rows)
    result = _compare(rows)

    print()
    print("=== Jev vs Laya compare ===")
    for k, v in result.items():
        print(f"  {k:<22} {v}")

    # Latency roll-up (post-hoc, from the rows themselves).
    laya_ms = [r["laya_ms"] for r in rows if "laya_ms" in r]
    if laya_ms:
        print()
        print(f"  laya avg ms: {sum(laya_ms)/len(laya_ms):.1f}")
        print(f"  laya max ms: {max(laya_ms):.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
