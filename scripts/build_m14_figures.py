"""Generate the M14 write-up figures from run artifacts.

Reads paper_snapshots (per-arm SQLite), session.jsonl (post-restart run),
and the shared price_bus. Emits 4 PNGs under docs/figures/.

Idempotent: safe to re-run. Overwrites existing PNGs.
"""

from __future__ import annotations

import json
import re
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
EXP_DIR = ROOT / "out" / "experiments" / "reasoning-quality"
SESSIONS_DIR = ROOT / "out" / "sessions"
OUT_DIR = ROOT / "docs" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ARMS = ("A", "B", "C")
ARM_COLOR = {"A": "#2b7bba", "B": "#e07a3f", "C": "#5a9c50"}
ARM_LABEL = {
    "A": "A — baseline",
    "B": "B — self-consistency N=3",
    "C": "C — cross-tier ensemble",
}


def _iso(ts: str) -> datetime:
    return datetime.fromisoformat(ts.replace("Z", "+00:00"))


def load_equity_series() -> dict[str, list[tuple[datetime, float]]]:
    out: dict[str, list[tuple[datetime, float]]] = {}
    for arm in ARMS:
        db = EXP_DIR / f"{arm}.db"
        conn = sqlite3.connect(str(db))
        cur = conn.cursor()
        cur.execute(
            "SELECT captured_at, account_json FROM paper_snapshots ORDER BY captured_at"
        )
        pts = []
        for ts, aj in cur.fetchall():
            acct = json.loads(aj)
            pts.append((_iso(ts), float(acct.get("equity", 0.0))))
        out[arm] = pts
        conn.close()
    return out


def latest_session_paths() -> dict[str, Path]:
    """Return the latest post-restart session.jsonl per arm."""
    out = {}
    for arm in ARMS:
        candidates = sorted(
            SESSIONS_DIR.glob(f"2026-09-11T15-22-*_{arm}/session.jsonl"),
            key=lambda p: p.stat().st_mtime,
        )
        if candidates:
            out[arm] = candidates[-1]
    return out


def parse_agreement_events() -> dict[str, list[float]]:
    """Pull sc + cross_model agreement values from each arm's log file."""
    # Agreement values appear in the paper-loop log, not session.jsonl.
    # Log format sample:
    #   ...graph: self_consistency: n=3 agreement=1.00 meta={...}
    #   ...graph: cross_model: n=2 agreement=0.80 meta={...}
    pat = re.compile(r"agreement=([\d.]+)")
    out: dict[str, list[float]] = {"B": [], "C": []}
    for arm in ("B", "C"):
        log = EXP_DIR / f"{arm}.log"
        if not log.exists():
            continue
        with open(log, encoding="utf-8", errors="replace") as f:
            for line in f:
                if "self_consistency" in line or "cross_model" in line:
                    m = pat.search(line)
                    if m:
                        try:
                            out[arm].append(float(m.group(1)))
                        except ValueError:
                            pass
    return out


def load_filter_counts() -> dict[str, dict[str, int]]:
    """Per-arm knob-firing counts from the paper-ab-report."""
    # Hardcoded from the paper-ab-report run captured in the write-up.
    # These are the cumulative Sept 9-11 numbers. Regenerate by re-running
    # `python -m agentic_investor.cli paper-ab-report reasoning-quality`.
    return {
        "A": {
            "opinion_drift": 93,
            "materiality": 377,
            "self_consistency": 0,
            "cross_model": 0,
            "on_deck_purge": 0,
        },
        "B": {
            "opinion_drift": 72,
            "materiality": 310,
            "self_consistency": 58,
            "cross_model": 0,
            "on_deck_purge": 2,
        },
        "C": {
            "opinion_drift": 77,
            "materiality": 365,
            "self_consistency": 0,
            "cross_model": 42,
            "on_deck_purge": 4,
        },
    }


def load_hpq_data() -> tuple[
    list[tuple[datetime, float]],
    dict[str, list[tuple[datetime, str, float]]],
]:
    """HPQ price ticks + per-arm HPQ trades (ts, side, qty)."""
    conn = sqlite3.connect(str(EXP_DIR / "price_bus.db"))
    cur = conn.cursor()
    cur.execute(
        "SELECT ts_event, price FROM price_ticks WHERE ticker='HPQ' "
        "ORDER BY ts_event"
    )
    price = []
    for ts, p in cur.fetchall():
        try:
            dt = datetime.strptime(
                ts.split("+")[0], "%Y-%m-%d %H:%M:%S.%f",
            ).replace(tzinfo=UTC)
        except ValueError:
            continue
        price.append((dt, float(p)))
    conn.close()

    trades: dict[str, list[tuple[datetime, str, float]]] = {a: [] for a in ARMS}
    for arm, jsonl in latest_session_paths().items():
        with open(jsonl, encoding="utf-8", errors="replace") as f:
            for line in f:
                # Fast pre-filter (JSON emits ": " with a space in this run).
                if "order_submitted" not in line or "HPQ" not in line:
                    continue
                try:
                    o = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if o.get("event") != "order_submitted":
                    continue
                if o.get("ticker") != "HPQ":
                    continue
                trades[arm].append((_iso(o["ts"]), o["side"], float(o["qty"])))
    return price, trades


# --------- plots -----------


def plot_equity_curves(series):
    fig, ax = plt.subplots(figsize=(10, 5))
    for arm in ARMS:
        pts = series.get(arm, [])
        if not pts:
            continue
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ax.plot(xs, ys, label=ARM_LABEL[arm], color=ARM_COLOR[arm], linewidth=1.5)
    ax.axhline(100_000, color="grey", linestyle="--", alpha=0.5, linewidth=0.8)
    ax.set_title("Equity — reasoning-quality A/B (Sept 9-11, 2026)")
    ax.set_ylabel("Account equity ($)")
    ax.set_xlabel("UTC")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
    ax.grid(alpha=0.3)
    ax.legend(loc="lower left")
    fig.autofmt_xdate()
    fig.tight_layout()
    out = OUT_DIR / "m14_equity_curves.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"wrote {out}")


def plot_filter_firings(counts):
    fig, ax = plt.subplots(figsize=(9, 5))
    knobs = ["opinion_drift", "materiality", "self_consistency", "cross_model"]
    knob_labels = {
        "opinion_drift": "opinion_drift\n(over-limit skips)",
        "materiality": "materiality\n(non-material skips)",
        "self_consistency": "self_consistency\n(N=3 sampling)",
        "cross_model": "cross_model\n(ensemble votes)",
    }
    x = range(len(knobs))
    bar_w = 0.26
    for i, arm in enumerate(ARMS):
        vals = [counts[arm][k] for k in knobs]
        offsets = [xi + (i - 1) * bar_w for xi in x]
        bars = ax.bar(
            offsets, vals, bar_w, label=ARM_LABEL[arm], color=ARM_COLOR[arm],
        )
        for bar, v in zip(bars, vals, strict=False):
            if v > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    v + 4,
                    str(v),
                    ha="center", fontsize=8,
                )
    ax.set_xticks(list(x))
    ax.set_xticklabels([knob_labels[k] for k in knobs])
    ax.set_ylabel("Firings across 3-day run")
    ax.set_title("Filter + sampling firings per arm")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out = OUT_DIR / "m14_filter_firings.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"wrote {out}")


def plot_agreement_hist(agreements):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    bins = [i / 10 for i in range(11)]
    for ax, arm, mech in (
        (axes[0], "B", "self-consistency N=3"),
        (axes[1], "C", "cross-tier ensemble"),
    ):
        vals = agreements.get(arm, [])
        ax.hist(
            vals, bins=bins, color=ARM_COLOR[arm], alpha=0.75, edgecolor="white",
        )
        ax.axvline(0.7, color="red", linestyle="--", alpha=0.6, linewidth=1)
        ax.text(
            0.71, ax.get_ylim()[1] * 0.9 if ax.get_ylim()[1] else 1,
            "trust-bypass ≥0.7", color="red", fontsize=8, rotation=0,
        )
        ax.set_title(f"Arm {arm} — {mech}  (n={len(vals)})")
        ax.set_xlabel("agreement")
        ax.set_xlim(0, 1)
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("Regens")
    fig.suptitle("Distribution of sampling-agreement scores")
    fig.tight_layout()
    out = OUT_DIR / "m14_agreement_hist.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"wrote {out}")


def plot_hpq_journey(price, trades):
    fig, ax = plt.subplots(figsize=(11, 5))
    xs = [p[0] for p in price]
    ys = [p[1] for p in price]
    ax.plot(xs, ys, color="#444", linewidth=1.2, label="HPQ price")

    side_marker = {"buy": "^", "sell": "v"}
    seen_labels = set()
    for arm in ARMS:
        for ts, side, qty in trades.get(arm, []):
            # Find closest price for marker y-position
            y_at_ts = min(
                (p for p in price if p[0] <= ts),
                key=lambda p: abs((ts - p[0]).total_seconds()),
                default=(ts, ys[-1] if ys else 35),
            )[1]
            lbl = f"{arm} {side}"
            if lbl in seen_labels:
                lbl = None
            else:
                seen_labels.add(f"{arm} {side}")
            ax.scatter(
                ts, y_at_ts,
                s=40 + qty * 3, marker=side_marker.get(side, "o"),
                color=ARM_COLOR[arm], edgecolor="black", linewidth=0.5,
                zorder=5, label=lbl,
            )
    ax.set_title("HPQ — price + arm trades (post-restart run, Sept 11 2026)")
    ax.set_ylabel("HPQ price ($)")
    ax.set_xlabel("UTC")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=8, ncol=2)
    fig.autofmt_xdate()
    fig.tight_layout()
    out = OUT_DIR / "m14_hpq_journey.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"wrote {out}")


def main():
    print("loading data...")
    equity = load_equity_series()
    filters = load_filter_counts()
    agreements = parse_agreement_events()
    hpq_price, hpq_trades = load_hpq_data()

    print(f"equity points: {sum(len(v) for v in equity.values())}")
    print(f"agreement samples: B={len(agreements.get('B', []))} "
          f"C={len(agreements.get('C', []))}")
    print(f"HPQ ticks: {len(hpq_price)}  trades: "
          f"{sum(len(v) for v in hpq_trades.values())}")

    plot_equity_curves(equity)
    plot_filter_firings(filters)
    plot_agreement_hist(agreements)
    plot_hpq_journey(hpq_price, hpq_trades)


if __name__ == "__main__":
    main()
