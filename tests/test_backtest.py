"""Unit tests for eval/backtest.py. Prices are synthetic so tests stay offline."""

import numpy as np
import pandas as pd
import pytest

from agentic_investor.eval import backtest
from agentic_investor.eval.backtest import (
    BacktestMetrics,
    alpha_beta,
    backtest_recommendation,
    compute_metrics,
    simulate_portfolio,
)
from agentic_investor.orchestrator.state import (
    Allocation,
    OrchestratorRequest,
    Position,
    Recommendation,
)


def _dates(n: int) -> pd.DatetimeIndex:
    return pd.date_range("2024-01-01", periods=n, freq="B")


def _prices(**cols) -> pd.DataFrame:
    n = len(next(iter(cols.values())))
    return pd.DataFrame(cols, index=_dates(n))


# simulate_portfolio


def test_simulate_flat_prices_leaves_value_unchanged():
    prices = _prices(AAPL=[100.0] * 5)
    v = simulate_portfolio(prices, {"AAPL": 1.0}, init_cash=1000)
    assert v.iloc[0] == pytest.approx(1000)
    assert v.iloc[-1] == pytest.approx(1000)


def test_simulate_100pct_equity_doubles_when_price_doubles():
    prices = _prices(AAPL=[100.0, 200.0])
    v = simulate_portfolio(prices, {"AAPL": 1.0}, init_cash=1000)
    assert v.iloc[-1] == pytest.approx(2000)


def test_simulate_cash_portion_is_flat():
    # 50% AAPL, 50% cash. AAPL doubles -> portfolio = 500*2 + 500 = 1500.
    prices = _prices(AAPL=[100.0, 200.0])
    v = simulate_portfolio(prices, {"AAPL": 0.5}, init_cash=1000)
    assert v.iloc[-1] == pytest.approx(1500)


def test_simulate_rejects_overweight():
    prices = _prices(AAPL=[100.0])
    with pytest.raises(ValueError):
        simulate_portfolio(prices, {"AAPL": 1.5}, init_cash=1000)


# compute_metrics


def test_total_return_and_cagr_for_known_series():
    value = pd.Series([100.0, 100.0, 100.0, 110.0], index=_dates(4))
    m = compute_metrics(value)
    assert m.total_return_pct == pytest.approx(10.0)
    # 10% over 3 daily returns, annualized to 252 -> should be very high.
    assert m.cagr_pct > 100


def test_max_drawdown_captures_worst_peak_to_trough():
    value = pd.Series([100, 120, 60, 80, 90], index=_dates(5), dtype="float64")
    m = compute_metrics(value)
    # Peak 120 -> trough 60 = -50% drawdown.
    assert m.max_drawdown_pct == pytest.approx(-50.0)


def test_flat_series_has_zero_sharpe_and_zero_vol():
    value = pd.Series([100.0] * 5, index=_dates(5))
    m = compute_metrics(value)
    assert m.sharpe == 0.0
    assert m.volatility_annual_pct == 0.0


# alpha_beta


def test_beta_is_1_when_portfolio_equals_benchmark():
    r = pd.Series([0.01, -0.02, 0.03, 0.005], index=_dates(4))
    alpha, beta = alpha_beta(r, r)
    assert beta == pytest.approx(1.0)
    assert alpha == pytest.approx(0.0, abs=1e-9)


def test_beta_is_2_when_portfolio_moves_twice_as_much():
    r_bench = pd.Series([0.01, -0.02, 0.03, 0.005], index=_dates(4))
    r_port = r_bench * 2
    _, beta = alpha_beta(r_port, r_bench)
    assert beta == pytest.approx(2.0)


def test_zero_variance_benchmark_returns_flat_alpha_beta():
    r_bench = pd.Series([0.0, 0.0, 0.0], index=_dates(3))
    r_port = pd.Series([0.01, 0.02, -0.01], index=_dates(3))
    alpha, beta = alpha_beta(r_port, r_bench)
    assert alpha == 0.0
    assert beta == 0.0


# End-to-end


def _rec(amount: float = 10_000) -> Recommendation:
    return Recommendation(
        request=OrchestratorRequest(
            tickers=["AAPL", "NVDA"], amount=amount, risk="moderate"
        ),
        allocation=Allocation(
            positions=[
                Position(ticker="AAPL", weight_pct=30, dollars=3000, rationale="r"),
                Position(ticker="NVDA", weight_pct=30, dollars=3000, rationale="r"),
            ],
            cash_pct=40,
            cash_dollars=4000,
            portfolio_rationale="r",
        ),
    )


def test_backtest_recommendation_end_to_end(monkeypatch):
    n = 100
    idx = _dates(n)
    # AAPL and NVDA rise linearly to 2x; SPY rises linearly to 1.5x.
    prices = pd.DataFrame(
        {
            "AAPL": np.linspace(100, 200, n),
            "NVDA": np.linspace(100, 200, n),
            "SPY": np.linspace(100, 150, n),
        },
        index=idx,
    )
    monkeypatch.setattr(backtest, "fetch_prices", lambda *a, **k: prices)

    result = backtest_recommendation(_rec())
    assert isinstance(result.portfolio, BacktestMetrics)
    # Portfolio: 60% equity doubles + 40% cash flat -> 60%*2 + 40%*1 = 1.6x
    assert result.portfolio_final_value == pytest.approx(16_000, rel=1e-2)
    # Benchmark: 1.5x -> 15,000
    assert result.benchmark_final_value == pytest.approx(15_000, rel=1e-2)
    # Portfolio holds 60% high-beta equity + 40% cash. With linear-price series
    # the exact beta depends on how return magnitudes evolve (base grows over
    # time), so bound directionally: >1 (above market) and <2 (cash drags it).
    assert 1.0 < result.beta < 2.0
