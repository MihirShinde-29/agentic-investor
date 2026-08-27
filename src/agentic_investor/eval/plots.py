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
    rebalance: str = "never",
    band_abs_pct: float = 5.0,
    band_rel_pct: float = 20.0,
    cash_yield_annual: float = 0.0,
    tcost_bps: float = 0.0,
    slippage_bps: float = 0.0,
) -> Path:
    """Save a portfolio-vs-benchmark equity curve PNG and return its path.

    If the backtest produced trades (rebalance != 'never'), buy/sell markers
    are overlaid on the portfolio curve.
    """
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    run = run_backtest(
        rec,
        start=start,
        end=end,
        benchmark=benchmark,
        rebalance=rebalance,
        band_abs_pct=band_abs_pct,
        band_rel_pct=band_rel_pct,
        cash_yield_annual=cash_yield_annual,
        tcost_bps=tcost_bps,
        slippage_bps=slippage_bps,
    )
    result, portfolio, bench, trades = (
        run.result, run.portfolio_value, run.benchmark_value, run.trades
    )

    fig, ax = plt.subplots(figsize=(11, 6), dpi=110)
    ax.plot(portfolio.index, portfolio.values, label="Portfolio", linewidth=2)
    ax.plot(bench.index, bench.values, label=benchmark, linewidth=2, alpha=0.85)
    ax.axhline(rec.request.amount, color="grey", linestyle="--", alpha=0.4, label="Initial")

    # Overlay buy/sell markers at portfolio value on each trade date.
    if not trades.empty:
        for side, marker, color, label in [
            ("BUY", "^", "green", "Buy"),
            ("SELL", "v", "red", "Sell"),
        ]:
            side_trades = trades[trades["side"] == side]
            if side_trades.empty:
                continue
            dates = side_trades["date"].tolist()
            # Look up portfolio value at each trade date.
            ys = [float(portfolio.loc[d]) for d in dates if d in portfolio.index]
            xs = [d for d in dates if d in portfolio.index]
            if xs:
                ax.scatter(xs, ys, marker=marker, color=color, s=45, alpha=0.7,
                           edgecolors="black", linewidths=0.5, label=label, zorder=5)

    tickers = ", ".join(p.ticker for p in rec.allocation.positions)
    tag = f"#{rec_id} " if rec_id is not None else ""
    trade_note = f", {len(trades)} trades" if not trades.empty else ""
    ax.set_title(
        f"Recommendation {tag}({tickers}) vs {benchmark}   "
        f"{result.start} to {result.end}   rebalance={rebalance}{trade_note}\n"
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
