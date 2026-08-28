"""Post-market session report generator.

Reads paper_snapshots + paper_orders from SQLite and the session JSONL,
produces:

- equity_curve.png     matplotlib chart of portfolio value over the session
- trades.csv           one row per submitted order (grep/spreadsheet-friendly)
- REPORT.md            human-readable markdown with key metrics + narrative

Designed as the single artifact you screenshot for interview demos.
"""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402


def _load_session_events(session_dir: Path) -> list[dict]:
    """Parse the session's JSONL into a list of dicts, in order."""
    jsonl = session_dir / "session.jsonl"
    if not jsonl.exists():
        return []
    lines = jsonl.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def _load_snapshots(session_started_iso: str) -> list[dict]:
    """Pull paper_snapshots after the session's start timestamp."""
    from agentic_investor.tools.paper_store import list_snapshots

    all_snaps = list_snapshots(limit=1000)
    started = datetime.fromisoformat(session_started_iso.replace("Z", "+00:00"))
    kept = []
    for s in all_snaps:
        ts = datetime.fromisoformat(s["captured_at"].replace("Z", "+00:00"))
        if ts >= started:
            kept.append(s)
    kept.reverse()  # oldest first for time-series plots
    return kept


def _load_orders(session_started_iso: str) -> list[dict]:
    """Pull paper_orders submitted after the session's start."""
    from agentic_investor.tools.paper_store import list_orders

    all_orders = list_orders(limit=1000)
    started = datetime.fromisoformat(session_started_iso.replace("Z", "+00:00"))
    kept = []
    for o in all_orders:
        try:
            ts = datetime.fromisoformat(o["submitted_at"].replace("Z", "+00:00"))
        except (ValueError, KeyError):
            continue
        if ts >= started:
            kept.append(o)
    kept.reverse()
    return kept


def write_trades_csv(orders: list[dict], out_path: Path) -> None:
    if not orders:
        out_path.write_text("(no orders submitted this session)\n", encoding="utf-8")
        return
    cols = ["submitted_at", "ticker", "side", "qty", "order_type",
            "status", "filled_at", "filled_avg_price", "rec_id", "source"]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(orders)


def write_equity_curve(
    snapshots: list[dict], orders: list[dict], out_path: Path, benchmark_prices=None
) -> None:
    if len(snapshots) < 2:
        return
    times = [datetime.fromisoformat(s["captured_at"].replace("Z", "+00:00"))
             for s in snapshots]
    equity = [float(s["account"]["equity"]) for s in snapshots]
    start_equity = equity[0]

    fig, ax = plt.subplots(figsize=(11, 5.5))
    # Portfolio pct-return curve.
    pct = [(e / start_equity - 1) * 100 for e in equity]
    ax.plot(times, pct, linewidth=1.8, color="#2E86AB", label="Portfolio (%)")

    # Benchmark overlay if provided.
    if benchmark_prices is not None and len(benchmark_prices) >= 2:
        b_times = list(benchmark_prices.index)
        b_start = float(benchmark_prices.iloc[0])
        b_pct = [(float(p) / b_start - 1) * 100 for p in benchmark_prices]
        ax.plot(b_times, b_pct, linewidth=1.2, color="#888888",
                linestyle="--", label="SPY (%)")

    # Overlay trade markers.
    for o in orders:
        try:
            t = datetime.fromisoformat(o["submitted_at"].replace("Z", "+00:00"))
        except (ValueError, KeyError):
            continue
        color = "#2ECC71" if o["side"] == "buy" else "#E74C3C"
        # Find nearest snapshot pct for y-position; if none, use 0.
        y = 0
        for i, ts in enumerate(times):
            if ts >= t:
                y = pct[i]
                break
        ax.scatter([t], [y], color=color, s=40, zorder=5,
                   marker="^" if o["side"] == "buy" else "v")

    ax.axhline(0, color="#cccccc", linewidth=0.8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.set_xlabel("Time (UTC)")
    ax.set_ylabel("Return %")
    ax.set_title("Session equity curve with trade markers")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _bench_prices_for_window(start_iso: str, end_iso: str):
    """1m SPY bars for the session window; None on failure."""
    from agentic_investor.tools.market import fetch_ohlcv

    try:
        df = fetch_ohlcv("SPY", period="7d", interval="1m")
        start = datetime.fromisoformat(start_iso.replace("Z", "+00:00"))
        end = datetime.fromisoformat(end_iso.replace("Z", "+00:00"))
        # yfinance 1m bars are tz-aware; align sides.
        idx = df.index if df.index.tz else df.index.tz_localize("UTC")
        df = df.copy()
        df.index = idx
        window = df[(idx >= pd.Timestamp(start)) & (idx <= pd.Timestamp(end))]
        return window["Close"] if len(window) >= 2 else None
    except Exception:  # noqa: BLE001
        return None


def build_report(session_dir: Path) -> Path:
    """Generate REPORT.md + equity_curve.png + trades.csv for a session."""
    if not session_dir.exists():
        raise FileNotFoundError(session_dir)
    events = _load_session_events(session_dir)
    if not events:
        raise RuntimeError(f"no session events in {session_dir}")

    started = next((e["ts"] for e in events if e["event"] == "session_start"),
                   events[0]["ts"])
    ended = next((e["ts"] for e in reversed(events) if e["event"] == "session_end"),
                 events[-1]["ts"])

    snapshots = _load_snapshots(started)
    orders = _load_orders(started)

    # Metrics
    equity_series = [float(s["account"]["equity"]) for s in snapshots]
    equity_open = equity_series[0] if equity_series else 0.0
    equity_close = equity_series[-1] if equity_series else 0.0
    total_return_pct = (
        (equity_close / equity_open - 1) * 100 if equity_open > 0 else 0.0
    )
    peak = max(equity_series) if equity_series else 0.0
    trough = min(equity_series) if equity_series else 0.0
    max_drawdown_pct = (
        ((trough - peak) / peak * 100) if peak > 0 else 0.0
    )

    regen_events = [e for e in events if e["event"] == "regen_done"]
    tick_costs = [e for e in events if e["event"] == "tick_cost"]
    news_events = [e for e in events if e["event"] == "news_received"]
    total_llm_calls = sum(int(e.get("llm_calls", 0)) for e in tick_costs)
    # cost_usd landed as either "$0.0024" string or float.
    total_cost = 0.0
    for e in tick_costs:
        c = e.get("cost_usd", 0)
        if isinstance(c, str):
            c = float(c.lstrip("$"))
        total_cost += float(c)

    filled = [o for o in orders if o.get("status") in {"filled", "partially_filled"}]
    total_notional = 0.0
    for o in filled:
        price = o.get("filled_avg_price") or 0
        qty = o.get("qty") or 0
        total_notional += float(price) * float(qty)

    # Benchmark for the same window
    bench = _bench_prices_for_window(started, ended)
    spy_return_pct = None
    if bench is not None:
        spy_return_pct = (float(bench.iloc[-1]) / float(bench.iloc[0]) - 1) * 100

    # Write artifacts
    trades_csv = session_dir / "trades.csv"
    write_trades_csv(orders, trades_csv)
    equity_png = session_dir / "equity_curve.png"
    write_equity_curve(snapshots, orders, equity_png, benchmark_prices=bench)

    # Markdown report
    md = _render_markdown(
        started=started, ended=ended,
        equity_open=equity_open, equity_close=equity_close,
        total_return_pct=total_return_pct,
        max_drawdown_pct=max_drawdown_pct,
        spy_return_pct=spy_return_pct,
        n_regens=len(regen_events),
        n_orders=len(orders),
        n_filled=len(filled),
        total_notional=total_notional,
        total_llm_calls=total_llm_calls,
        total_cost=total_cost,
        n_news=len(news_events),
        regen_events=regen_events,
        orders=orders,
        equity_png=equity_png.name,
        trades_csv=trades_csv.name,
    )
    report_path = session_dir / "REPORT.md"
    report_path.write_text(md, encoding="utf-8")
    return report_path


def _render_markdown(*, started, ended, equity_open, equity_close,
                     total_return_pct, max_drawdown_pct, spy_return_pct,
                     n_regens, n_orders, n_filled, total_notional,
                     total_llm_calls, total_cost, n_news,
                     regen_events, orders, equity_png, trades_csv) -> str:
    lines: list[str] = []
    lines.append(f"# Session report — {started[:10]}")
    lines.append("")
    lines.append(f"**Session window:** `{started}` -> `{ended}`  ")
    lines.append("")
    lines.append("## Headline numbers")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|---|---:|")
    lines.append(f"| Equity open | ${equity_open:,.2f} |")
    lines.append(f"| Equity close | ${equity_close:,.2f} |")
    lines.append(f"| **Total return** | **{total_return_pct:+.2f}%** |")
    if spy_return_pct is not None:
        alpha = total_return_pct - spy_return_pct
        lines.append(f"| SPY return (same window) | {spy_return_pct:+.2f}% |")
        lines.append(f"| Alpha vs SPY | {alpha:+.2f}pp |")
    lines.append(f"| Max drawdown | {max_drawdown_pct:.2f}% |")
    lines.append(f"| LLM decisions (regens) | {n_regens} |")
    lines.append(f"| Orders submitted | {n_orders} ({n_filled} filled) |")
    lines.append(f"| Notional traded | ${total_notional:,.2f} |")
    lines.append(f"| Total LLM calls | {total_llm_calls} |")
    lines.append(f"| Total LLM cost | ${total_cost:.4f} |")
    lines.append(f"| News events received | {n_news} |")
    lines.append("")
    lines.append(f"![equity curve]({equity_png})")
    lines.append("")
    lines.append(f"Full trade log: [`{trades_csv}`]({trades_csv})")
    lines.append("")
    lines.append("## Decision timeline")
    lines.append("")
    if not regen_events:
        lines.append("_no LLM regens this session_")
    else:
        lines.append("| Time | Rec | Trigger | Cash % | Top target |")
        lines.append("|---|---:|---|---:|---|")
        for r in regen_events:
            targets = r.get("targets", {})
            top = max(targets.items(), key=lambda kv: kv[1]) if targets else ("-", 0)
            lines.append(
                f"| {r['ts'][11:19]} | #{r.get('rec_id','?')} | "
                f"{r.get('trigger','?')} | {r.get('cash_pct','?')}% | "
                f"{top[0]} {top[1]}% |"
            )
    lines.append("")
    lines.append("## All orders")
    lines.append("")
    if not orders:
        lines.append("_no orders submitted_")
    else:
        lines.append("| Time | Ticker | Side | Qty | Status | Filled @ |")
        lines.append("|---|---|---|---:|---|---:|")
        for o in orders:
            fill = o.get("filled_avg_price")
            fill_str = f"${fill:.2f}" if fill else "-"
            lines.append(
                f"| {o.get('submitted_at','')[11:19]} | {o.get('ticker')} | "
                f"{o.get('side')} | {o.get('qty')} | {o.get('status')} | {fill_str} |"
            )
    return "\n".join(lines)


def most_recent_session_dir(base: str = "out/sessions") -> Path:
    root = Path(base)
    if not root.exists():
        raise FileNotFoundError(root)
    candidates = sorted(
        (p for p in root.iterdir() if p.is_dir()),
        key=lambda p: p.stat().st_mtime, reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"no session directories in {root}")
    return candidates[0]
