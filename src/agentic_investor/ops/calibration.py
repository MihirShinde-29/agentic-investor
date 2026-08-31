"""Post-trade attribution + LLM-confidence calibration curve.

Answers the question: does confidence=0.8 actually win 80% of the time?

For every filled order with a linked rec_id, we can recover the LLM's
per-position confidence at decision time (Position.confidence). Combined
with an outcome measured N minutes / hours after fill, this produces a
calibration table + reliability diagram.

- Wins: BUY -> price went up, or SELL -> price went down
- Losses: the reverse
- Ties (~0.0% move) are counted as half-wins so buckets with all-flat
  trades don't blow up to 0% or 100%

Outputs (paper-calibration CLI):
- calibration.png   reliability diagram (confidence bucket vs actual win rate)
- calibration.md    markdown table + honesty notes
- calibration.csv   raw per-trade rows for offline analysis
"""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

logger = logging.getLogger(__name__)


@dataclass
class TradeOutcome:
    ticker: str
    side: str
    submitted_at: str
    filled_price: float
    confidence: float
    rec_id: int
    later_price: float
    horizon_minutes: int
    return_pct: float  # signed % change from filled_price to later_price
    win: float  # 1.0 if directionally correct, 0.5 if flat, 0.0 if wrong


@dataclass
class CalibrationBucket:
    lo: float
    hi: float
    n_trades: int = 0
    n_wins: float = 0.0  # sum of win values (0/0.5/1)
    confidences: list[float] = field(default_factory=list)

    @property
    def win_rate(self) -> float:
        return self.n_wins / self.n_trades if self.n_trades else 0.0

    @property
    def mean_confidence(self) -> float:
        return sum(self.confidences) / len(self.confidences) if self.confidences else 0.0


def _later_price(ticker: str, submitted_at: str, horizon_minutes: int) -> float | None:
    """Fetch the ticker's price `horizon_minutes` after submitted_at.

    Uses 1-minute bars via yfinance (7-day retention on the free tier).
    Returns None when the horizon lies outside the fetchable window
    (e.g. after market close on the same day, or older than 7 days).
    """
    try:
        import pandas as pd

        from agentic_investor.tools.market import fetch_ohlcv

        df = fetch_ohlcv(ticker, period="7d", interval="1m")
        if df.empty:
            return None
        target = datetime.fromisoformat(submitted_at.replace("Z", "+00:00"))
        target += timedelta(minutes=horizon_minutes)
        idx = df.index if df.index.tz else df.index.tz_localize(UTC)
        df = df.copy()
        df.index = idx
        # First bar at-or-after the horizon timestamp.
        mask = idx >= pd.Timestamp(target)
        if not mask.any():
            return None
        return float(df.loc[mask, "Close"].iloc[0])
    except Exception as e:  # noqa: BLE001 - price lookup is best-effort
        logger.debug("later_price failed for %s: %s", ticker, e)
        return None


def _win_for(side: str, filled_price: float, later_price: float) -> tuple[float, float]:
    """Return (return_pct, win_value) where win_value is 1/0.5/0."""
    if filled_price <= 0:
        return 0.0, 0.5
    ret = (later_price / filled_price - 1) * 100
    if side == "buy":
        directional = 1.0 if ret > 0.05 else (0.0 if ret < -0.05 else 0.5)
    else:
        directional = 1.0 if ret < -0.05 else (0.0 if ret > 0.05 else 0.5)
    return ret, directional


def compute_trade_outcomes(
    *,
    horizon_minutes: int = 60,
    order_limit: int = 500,
    url: str | None = None,
) -> list[TradeOutcome]:
    """Join paper_orders + recs + later prices into per-trade outcomes."""
    from agentic_investor.orchestrator.store import load_recommendation
    from agentic_investor.tools.paper_store import list_orders

    outcomes: list[TradeOutcome] = []
    orders = list_orders(limit=order_limit, url=url)
    rec_cache: dict[int, dict[str, float]] = {}

    for o in orders:
        if o.get("status") not in {"filled", "partially_filled"}:
            continue
        filled_price = o.get("filled_avg_price")
        rec_id = o.get("rec_id")
        if not filled_price or not rec_id:
            continue
        if rec_id not in rec_cache:
            try:
                rec = load_recommendation(int(rec_id))
                rec_cache[rec_id] = (
                    {p.ticker.upper(): p.confidence for p in rec.allocation.positions}
                    if rec else {}
                )
            except Exception:  # noqa: BLE001 - legacy recs may fail schema
                rec_cache[rec_id] = {}
        conf = rec_cache[rec_id].get(o["ticker"].upper())
        if conf is None:
            continue
        later = _later_price(o["ticker"], o["submitted_at"], horizon_minutes)
        if later is None:
            continue
        ret, win = _win_for(o["side"], float(filled_price), later)
        outcomes.append(TradeOutcome(
            ticker=o["ticker"], side=o["side"],
            submitted_at=o["submitted_at"], filled_price=float(filled_price),
            confidence=float(conf), rec_id=int(rec_id),
            later_price=later, horizon_minutes=horizon_minutes,
            return_pct=ret, win=win,
        ))
    return outcomes


def bucket_outcomes(
    outcomes: list[TradeOutcome], *, n_buckets: int = 5
) -> list[CalibrationBucket]:
    """Group outcomes into confidence buckets in [0, 1]."""
    step = 1.0 / n_buckets
    buckets = [CalibrationBucket(lo=i * step, hi=(i + 1) * step) for i in range(n_buckets)]
    for o in outcomes:
        idx = min(int(o.confidence * n_buckets), n_buckets - 1)
        b = buckets[idx]
        b.n_trades += 1
        b.n_wins += o.win
        b.confidences.append(o.confidence)
    return buckets


def render_calibration_plot(buckets: list[CalibrationBucket], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], "--", color="#888", label="Perfect calibration")
    xs = [b.mean_confidence for b in buckets if b.n_trades]
    ys = [b.win_rate for b in buckets if b.n_trades]
    sizes = [max(30, b.n_trades * 20) for b in buckets if b.n_trades]
    if xs:
        ax.scatter(xs, ys, s=sizes, color="#2E86AB", zorder=5,
                   label="Observed (marker size = n_trades)")
        for b in buckets:
            if b.n_trades:
                ax.annotate(f"{b.n_trades}", (b.mean_confidence, b.win_rate),
                            textcoords="offset points", xytext=(6, 6), fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("LLM confidence at decision")
    ax.set_ylabel("Actual directional win rate")
    ax.set_title("Confidence calibration (post-trade)")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def render_calibration_report(
    outcomes: list[TradeOutcome], buckets: list[CalibrationBucket], horizon: int
) -> str:
    total = len(outcomes)
    lines = [
        f"# Confidence calibration — {horizon}m horizon",
        "",
        f"**Trades analyzed:** {total}",
        "",
    ]
    if not outcomes:
        lines.append("_no trades with resolvable outcomes yet — need "
                     "filled orders + rec_id + prices within the "
                     "7-day intraday-bar window_")
        return "\n".join(lines)
    wins = sum(o.win for o in outcomes)
    lines.append(f"**Overall directional win rate:** {wins / total:.1%}")
    mean_conf = sum(o.confidence for o in outcomes) / total
    lines.append(f"**Mean LLM confidence:** {mean_conf:.2f}")
    lines.append("")
    lines.append("## Bucketed calibration")
    lines.append("")
    lines.append("| Confidence | n_trades | mean_conf | win_rate | gap |")
    lines.append("|---|---:|---:|---:|---:|")
    for b in buckets:
        if not b.n_trades:
            continue
        gap = b.win_rate - b.mean_confidence
        lines.append(
            f"| [{b.lo:.1f}, {b.hi:.1f}) | {b.n_trades} | "
            f"{b.mean_confidence:.2f} | {b.win_rate:.1%} | {gap:+.1%} |"
        )
    lines.append("")
    lines.append("`gap = win_rate - mean_confidence`. Negative = overconfident, "
                 "positive = underconfident.")
    lines.append("")
    lines.append("## Honesty notes")
    lines.append("")
    lines.append(
        f"- **Horizon is arbitrary**: {horizon}m is one plausible measurement "
        "window; longer horizons capture different dynamics."
    )
    lines.append(
        "- **Directional only**: this checks sign correctness, not magnitude. "
        "A 0.9-confidence BUY that gained 0.1% counts the same as one that "
        "gained 5%."
    )
    lines.append(
        "- **Selection bias**: only trades with resolvable prices show up. "
        "Trades near market close on the current day may be missing later data."
    )
    lines.append(
        "- **Sample size**: buckets with <10 trades are noisy; treat as "
        "indicative only until the loop has run for several sessions."
    )
    return "\n".join(lines)


def write_outcomes_csv(outcomes: list[TradeOutcome], out_path: Path) -> None:
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "submitted_at", "ticker", "side", "confidence", "filled_price",
            "later_price", "horizon_minutes", "return_pct", "win", "rec_id",
        ])
        for o in outcomes:
            writer.writerow([
                o.submitted_at, o.ticker, o.side, f"{o.confidence:.2f}",
                f"{o.filled_price:.4f}", f"{o.later_price:.4f}",
                o.horizon_minutes, f"{o.return_pct:+.4f}",
                f"{o.win:.1f}", o.rec_id,
            ])


def build_calibration_report(
    *,
    horizon_minutes: int = 60,
    out_dir: str = "out/calibration",
    url: str | None = None,
) -> tuple[Path, Path, Path]:
    """Compute outcomes, write PNG + MD + CSV. Returns (md, png, csv) paths."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    outcomes = compute_trade_outcomes(horizon_minutes=horizon_minutes, url=url)
    buckets = bucket_outcomes(outcomes)
    md_path = out / "calibration.md"
    png_path = out / "calibration.png"
    csv_path = out / "calibration.csv"
    render_calibration_plot(buckets, png_path)
    md_path.write_text(
        render_calibration_report(outcomes, buckets, horizon_minutes),
        encoding="utf-8",
    )
    write_outcomes_csv(outcomes, csv_path)
    return md_path, png_path, csv_path
