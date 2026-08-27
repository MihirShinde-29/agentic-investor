"""Backtest a saved Recommendation vs SPY over a fixed window.

Deterministic and offline once prices are fetched. Supports rebalancing
(never / calendar monthly or quarterly / drift bands), daily-compounded
risk-free cash yield, and per-trade transaction cost + slippage. Look-ahead
is bounded to the signal-generation moment; every trade uses that day's
close and pays a realistic friction.
"""

from dataclasses import dataclass, field
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
    n_trades: int = 0
    total_costs: float = 0.0


@dataclass
class BacktestRun:
    result: BacktestResult
    portfolio_value: pd.Series
    benchmark_value: pd.Series
    trades: pd.DataFrame = field(default_factory=pd.DataFrame)


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


def _is_rebalance_day(
    date_: pd.Timestamp,
    prev_date: pd.Timestamp | None,
    mode: str,
    current_weights: dict[str, float],
    target_weights: dict[str, float],
    band_abs_pct: float,
    band_rel_pct: float,
) -> bool:
    # Day 0 always rebalances to establish the initial allocation.
    if prev_date is None:
        return True
    if mode == "never":
        return False
    if mode == "monthly":
        return date_.month != prev_date.month
    if mode == "quarterly":
        return date_.quarter != prev_date.quarter or date_.year != prev_date.year
    if mode == "bands":
        # Trigger if ANY position drifts more than either threshold (pp or relative).
        for t, target in target_weights.items():
            current = current_weights.get(t, 0.0)
            abs_drift_pp = abs(current - target) * 100
            rel_drift_pct = (abs_drift_pp / (target * 100) * 100) if target > 0 else 0.0
            if abs_drift_pp > band_abs_pct or rel_drift_pct > band_rel_pct:
                return True
        return False
    raise ValueError(f"unknown rebalance mode {mode!r}")


def _execute_rebalance(
    shares: dict[str, float],
    cash: float,
    target_weights: dict[str, float],
    prices_today: pd.Series,
    trades: list[dict],
    date_: pd.Timestamp,
    tcost_bps: float,
    slippage_bps: float,
) -> float:
    """Rebalance in place to target weights. Returns new cash."""
    equity_value = sum(shares.get(t, 0.0) * prices_today[t] for t in target_weights)
    total_value = cash + equity_value

    for t, target_w in target_weights.items():
        target_dollars = total_value * target_w
        current_dollars = shares.get(t, 0.0) * prices_today[t]
        delta_dollars = target_dollars - current_dollars
        if abs(delta_dollars) < 1.0:  # ignore trivial adjustments
            continue

        side = "BUY" if delta_dollars > 0 else "SELL"
        # Slippage worsens the fill for us: buys fill above spot, sells below.
        slip_factor = 1 + slippage_bps / 10_000 if side == "BUY" else 1 - slippage_bps / 10_000
        fill_price = float(prices_today[t]) * slip_factor
        delta_shares = delta_dollars / fill_price
        gross_value = abs(delta_shares) * fill_price
        commission = gross_value * tcost_bps / 10_000

        shares[t] = shares.get(t, 0.0) + delta_shares
        cash -= delta_shares * fill_price + commission

        trades.append(
            {
                "date": date_,
                "ticker": t,
                "side": side,
                "shares": round(abs(delta_shares), 4),
                "price": round(fill_price, 4),
                "value": round(gross_value, 2),
                "cost": round(commission, 4),
            }
        )
    return cash


def simulate_portfolio(
    prices: pd.DataFrame,
    weights: dict[str, float],
    init_cash: float,
    *,
    rebalance: str = "never",
    band_abs_pct: float = 5.0,
    band_rel_pct: float = 20.0,
    cash_yield_annual: float = 0.0,
    tcost_bps: float = 0.0,
    slippage_bps: float = 0.0,
) -> tuple[pd.Series, pd.DataFrame]:
    """Return (portfolio value over time, trades DataFrame).

    weights are fractions in [0, 1]; their sum plus cash_weight = 1.
    rebalance modes: 'never' (buy-and-hold), 'monthly', 'quarterly', 'bands'.
    Cash grows daily at (1 + cash_yield_annual)**(1/252) - 1.
    """
    if sum(weights.values()) > 1.0 + 1e-9:
        raise ValueError(f"weights sum to more than 1 ({sum(weights.values()):.4f})")

    tickers = [t for t in weights if t in prices.columns]
    target_weights = {t: weights[t] for t in tickers}
    daily_cash_rate = (
        (1 + cash_yield_annual) ** (1 / TRADING_DAYS) - 1 if cash_yield_annual > 0 else 0.0
    )

    shares: dict[str, float] = dict.fromkeys(tickers, 0.0)
    cash = init_cash
    trades: list[dict] = []
    values: list[float] = []
    prev_date: pd.Timestamp | None = None

    for date_, row in prices.iterrows():
        cash *= 1 + daily_cash_rate

        equity_val_before = sum(shares[t] * row[t] for t in tickers)
        total_before = cash + equity_val_before
        current_weights = (
            {t: (shares[t] * row[t]) / total_before for t in tickers}
            if total_before > 0
            else {t: 0.0 for t in tickers}
        )

        if _is_rebalance_day(
            date_, prev_date, rebalance, current_weights, target_weights,
            band_abs_pct, band_rel_pct,
        ):
            cash = _execute_rebalance(
                shares, cash, target_weights, row, trades, date_, tcost_bps, slippage_bps
            )

        values.append(cash + sum(shares[t] * row[t] for t in tickers))
        prev_date = date_

    value_series = pd.Series(values, index=prices.index, dtype="float64")
    trades_df = pd.DataFrame(trades) if trades else pd.DataFrame(
        columns=["date", "ticker", "side", "shares", "price", "value", "cost"]
    )
    return value_series, trades_df


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
    """Regress portfolio returns on benchmark: r_p = alpha + beta * r_b."""
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


def run_backtest(
    rec: Recommendation,
    *,
    start: str | date | None = None,
    end: str | date | None = None,
    benchmark: str = "SPY",
    rebalance: str = "never",
    band_abs_pct: float = 5.0,
    band_rel_pct: float = 20.0,
    cash_yield_annual: float = 0.0,
    tcost_bps: float = 0.0,
    slippage_bps: float = 0.0,
) -> BacktestRun:
    """Full backtest: metrics + raw series + trades. Callers wanting only metrics
    can use backtest_recommendation which wraps this.
    """
    weights = {p.ticker: p.weight_pct / 100.0 for p in rec.allocation.positions}
    tickers = list(weights.keys()) + [benchmark]

    prices = fetch_prices(tickers, start=start, end=end)
    if prices.empty:
        raise ValueError("no prices returned for backtest window")
    # Drop rows missing any ticker so portfolio and benchmark share one index.
    prices = prices.dropna(how="any")

    live_weights = {t: w for t, w in weights.items() if t in prices.columns}
    portfolio_value, portfolio_trades = simulate_portfolio(
        prices[list(live_weights.keys())],
        live_weights,
        init_cash=rec.request.amount,
        rebalance=rebalance,
        band_abs_pct=band_abs_pct,
        band_rel_pct=band_rel_pct,
        cash_yield_annual=cash_yield_annual,
        tcost_bps=tcost_bps,
        slippage_bps=slippage_bps,
    )
    # Benchmark is 100% one asset; rebalance/cash-yield/friction still applied
    # so it faces the same trading assumptions and comparison stays fair.
    benchmark_value, _ = simulate_portfolio(
        prices[[benchmark]],
        {benchmark: 1.0},
        init_cash=rec.request.amount,
        rebalance="never",
        tcost_bps=tcost_bps,
        slippage_bps=slippage_bps,
    )

    alpha, beta = alpha_beta(
        portfolio_value.pct_change().dropna(),
        benchmark_value.pct_change().dropna(),
    )

    result = BacktestResult(
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
        n_trades=len(portfolio_trades),
        total_costs=round(
            float(portfolio_trades["cost"].sum()) if not portfolio_trades.empty else 0.0, 2
        ),
    )
    return BacktestRun(
        result=result,
        portfolio_value=portfolio_value,
        benchmark_value=benchmark_value,
        trades=portfolio_trades,
    )


def backtest_recommendation(
    rec: Recommendation,
    *,
    start: str | date | None = None,
    end: str | date | None = None,
    benchmark: str = "SPY",
    rebalance: str = "never",
    band_abs_pct: float = 5.0,
    band_rel_pct: float = 20.0,
    cash_yield_annual: float = 0.0,
    tcost_bps: float = 0.0,
    slippage_bps: float = 0.0,
) -> BacktestResult:
    """Backtest metrics only. Use run_backtest if you also want the series + trades."""
    return run_backtest(
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
    ).result


def multi_window_backtest(
    rec: Recommendation,
    windows: list[tuple[str, str]],
    *,
    benchmark: str = "SPY",
    **kwargs,
) -> list[BacktestResult]:
    """Run the same recommendation across multiple non-overlapping date windows."""
    return [
        backtest_recommendation(rec, start=s, end=e, benchmark=benchmark, **kwargs)
        for s, e in windows
    ]
