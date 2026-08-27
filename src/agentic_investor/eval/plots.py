"""Plotting helpers for backtests.

matplotlib import is lazy so pytest and CLI paths that never plot stay fast.
The Agg backend is forced so this works headless (no display required).
"""

from datetime import date
from pathlib import Path

from agentic_investor.eval.backtest import run_backtest
from agentic_investor.orchestrator.state import Recommendation


def plot_equity_curve(
    rec: Recommendation,
    *,
    start: str | date | None = None,
    end: str | date | None = None,
    benchmark: str = "SPY",
    out_path: str | Path = "out/equity_curve.png",
    rec_id: int | None = None,
) -> Path:
    """Save a portfolio-vs-benchmark equity curve PNG and return its path."""
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    result, portfolio, bench = run_backtest(rec, start=start, end=end, benchmark=benchmark)

    fig, ax = plt.subplots(figsize=(11, 6), dpi=110)
    ax.plot(portfolio.index, portfolio.values, label="Portfolio", linewidth=2)
    ax.plot(bench.index, bench.values, label=benchmark, linewidth=2, alpha=0.85)
    ax.axhline(rec.request.amount, color="grey", linestyle="--", alpha=0.4, label="Initial")

    tickers = ", ".join(p.ticker for p in rec.allocation.positions)
    tag = f"#{rec_id} " if rec_id is not None else ""
    ax.set_title(
        f"Recommendation {tag}({tickers}) vs {benchmark}   "
        f"{result.start} to {result.end}\n"
        f"Portfolio {result.portfolio.total_return_pct:+.1f}%  "
        f"(Sharpe {result.portfolio.sharpe:.2f}, DD {result.portfolio.max_drawdown_pct:.1f}%)   "
        f"vs {benchmark} {result.benchmark.total_return_pct:+.1f}%  "
        f"(Sharpe {result.benchmark.sharpe:.2f}, DD {result.benchmark.max_drawdown_pct:.1f}%)"
    )
    ax.set_xlabel("Date")
    ax.set_ylabel("Portfolio value ($)")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.25)
    fig.autofmt_xdate()
    fig.tight_layout()

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    return out
