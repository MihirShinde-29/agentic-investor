"""Backtest a saved Recommendation vs SPY buy-and-hold over a fixed window.

Deterministic and offline once prices are fetched. Cash is held flat at 0%
return (the honest, conservative choice; risk-free-rate accrual is a future
refinement). Look-ahead is bounded to the signal-generation moment because
the allocation is buy-and-hold from prices.index[0], not rebalanced.
"""

from datetime import date

import pandas as pd
import yfinance as yf
from pydantic import BaseModel

from agentic_investor.orchestrator.state import Recommendation

TRADING_DAYS = 252


class BacktestMetrics(BaseModel):
    total_return_pct: float
    cagr_pct: float
    sharpe: float
    max_drawdown_pct: float
    volatility_annual_pct: float


class BacktestResult(BaseModel):
    start: str
    end: str
    n_days: int
    init_cash: float
    portfolio: BacktestMetrics
    benchmark: BacktestMetrics
    alpha_annual_pct: float
    beta: float
    portfolio_final_value: float
    benchmark_final_value: float


def fetch_prices(
    tickers: list[str],
    start: str | date | None = None,
    end: str | date | None = None,
) -> pd.DataFrame:
    """Return auto-adjusted close prices as DataFrame indexed by date, columns = tickers."""
    raw = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        group_by="ticker",
    )
    if isinstance(raw.columns, pd.MultiIndex):
        close = pd.concat(
            {t: raw[t]["Close"] for t in tickers if t in raw.columns.get_level_values(0)},
            axis=1,
        )
    else:
        close = raw[["Close"]].rename(columns={"Close": tickers[0]})
    return close.dropna(how="all")


def simulate_portfolio(
    prices: pd.DataFrame,
    weights: dict[str, float],
    init_cash: float,
) -> pd.Series:
    """Return portfolio value over time under buy-and-hold from prices.index[0].

    weights values are fractions (0-1); their sum plus cash_weight = 1.
    """
    equity_weight = sum(weights.values())
    if equity_weight > 1.0 + 1e-9:
        raise ValueError(f"weights sum to more than 1 ({equity_weight:.4f})")
    cash_dollars = init_cash * (1.0 - equity_weight)

    p0 = prices.iloc[0]
    shares = {t: (w * init_cash) / p0[t] for t, w in weights.items() if t in prices.columns}
    equity_value = sum(shares[t] * prices[t] for t in shares)
    return cash_dollars + equity_value


def compute_metrics(value: pd.Series) -> BacktestMetrics:
    returns = value.pct_change().dropna()
    if len(returns) < 1:
        raise ValueError("need at least 2 observations to compute metrics")

    total_return = value.iloc[-1] / value.iloc[0] - 1
    cagr = (value.iloc[-1] / value.iloc[0]) ** (TRADING_DAYS / len(returns)) - 1
    vol_annual = returns.std() * (TRADING_DAYS**0.5)
    sharpe = (returns.mean() / returns.std()) * (TRADING_DAYS**0.5) if returns.std() > 0 else 0.0

    running_max = value.cummax()
    drawdown = (value - running_max) / running_max
    max_dd = drawdown.min()

    return BacktestMetrics(
        total_return_pct=round(total_return * 100, 2),
        cagr_pct=round(cagr * 100, 2),
        sharpe=round(float(sharpe), 2),
        max_drawdown_pct=round(float(max_dd) * 100, 2),
        volatility_annual_pct=round(float(vol_annual) * 100, 2),
    )


def alpha_beta(portfolio_returns: pd.Series, benchmark_returns: pd.Series) -> tuple[float, float]:
    """Regress portfolio returns on benchmark: r_p = alpha + beta * r_b.

    Returns (alpha_annualized, beta). Alpha is the intercept scaled by TRADING_DAYS;
    beta is the slope (unitless).
    """
    aligned = pd.concat([portfolio_returns, benchmark_returns], axis=1, join="inner").dropna()
    if len(aligned) < 2:
        return 0.0, 0.0
    y = aligned.iloc[:, 0]
    x = aligned.iloc[:, 1]
    var_x = ((x - x.mean()) ** 2).sum()
    if var_x == 0:
        return 0.0, 0.0
    beta = ((x - x.mean()) * (y - y.mean())).sum() / var_x
    alpha_daily = y.mean() - beta * x.mean()
    return alpha_daily * TRADING_DAYS, beta


def backtest_recommendation(
    rec: Recommendation,
    *,
    start: str | date | None = None,
    end: str | date | None = None,
    benchmark: str = "SPY",
) -> BacktestResult:
    """Fetch prices, simulate the recommended allocation vs SPY, return metrics."""
    weights = {p.ticker: p.weight_pct / 100.0 for p in rec.allocation.positions}
    tickers = list(weights.keys()) + [benchmark]

    prices = fetch_prices(tickers, start=start, end=end)
    if prices.empty:
        raise ValueError("no prices returned for backtest window")
    # Drop rows where any ticker is missing so the portfolio and benchmark
    # share the exact same time index (fair comparison).
    prices = prices.dropna(how="any")

    live_weights = {t: w for t, w in weights.items() if t in prices.columns}
    portfolio_value = simulate_portfolio(
        prices[list(live_weights.keys())], live_weights, init_cash=rec.request.amount
    )
    benchmark_value = simulate_portfolio(
        prices[[benchmark]], {benchmark: 1.0}, init_cash=rec.request.amount
    )

    alpha, beta = alpha_beta(
        portfolio_value.pct_change().dropna(),
        benchmark_value.pct_change().dropna(),
    )

    return BacktestResult(
        start=str(prices.index[0].date()),
        end=str(prices.index[-1].date()),
        n_days=len(prices),
        init_cash=rec.request.amount,
        portfolio=compute_metrics(portfolio_value),
        benchmark=compute_metrics(benchmark_value),
        alpha_annual_pct=round(float(alpha) * 100, 2),
        beta=round(float(beta), 3),
        portfolio_final_value=round(float(portfolio_value.iloc[-1]), 2),
        benchmark_final_value=round(float(benchmark_value.iloc[-1]), 2),
    )
