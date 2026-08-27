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
    multi_window_backtest,
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


# simulate_portfolio: buy-and-hold basics


def test_simulate_flat_prices_leaves_value_unchanged():
    prices = _prices(AAPL=[100.0] * 5)
    v, trades = simulate_portfolio(prices, {"AAPL": 1.0}, init_cash=1000)
    assert v.iloc[0] == pytest.approx(1000)
    assert v.iloc[-1] == pytest.approx(1000)
    assert len(trades) == 1  # only day-0 buy


def test_simulate_100pct_equity_doubles_when_price_doubles():
    prices = _prices(AAPL=[100.0, 200.0])
    v, _ = simulate_portfolio(prices, {"AAPL": 1.0}, init_cash=1000)
    assert v.iloc[-1] == pytest.approx(2000)


def test_simulate_cash_portion_is_flat_by_default():
    # 50% AAPL, 50% cash. AAPL doubles -> portfolio = 500*2 + 500 = 1500.
    prices = _prices(AAPL=[100.0, 200.0])
    v, _ = simulate_portfolio(prices, {"AAPL": 0.5}, init_cash=1000)
    assert v.iloc[-1] == pytest.approx(1500)


def test_simulate_rejects_overweight():
    prices = _prices(AAPL=[100.0])
    with pytest.raises(ValueError):
        simulate_portfolio(prices, {"AAPL": 1.5}, init_cash=1000)


# simulate_portfolio: new features (M4b)


def test_cash_yield_grows_all_cash_series():
    # 100% cash (weights = {}), 10% annual yield over 252 days = ~10% growth.
    prices = _prices(AAPL=[100.0] * 252)
    v, trades = simulate_portfolio(
        prices, {}, init_cash=1000, cash_yield_annual=0.10
    )
    assert v.iloc[-1] == pytest.approx(1100, rel=0.005)
    assert trades.empty


def test_monthly_rebalance_triggers_on_month_change():
    # 60 business days spans ~3 months; expect rebalance on day 0 + 2 month boundaries.
    n = 60
    prices = _prices(AAPL=[100.0] * n, NVDA=list(np.linspace(100, 200, n)))
    _, trades = simulate_portfolio(
        prices, {"AAPL": 0.5, "NVDA": 0.5}, init_cash=1000, rebalance="monthly"
    )
    # Trades happen in batches (one per ticker per rebalance day); count unique dates.
    rebalance_days = trades["date"].nunique()
    assert 3 <= rebalance_days <= 4


def test_bands_rebalance_triggers_when_drift_exceeds_threshold():
    # Two tickers start balanced, one rockets 3x -> weight drifts far past ±5pp.
    n = 20
    prices = _prices(
        AAPL=[100.0] * n,
        NVDA=list(np.linspace(100, 300, n)),
    )
    _, trades = simulate_portfolio(
        prices,
        {"AAPL": 0.5, "NVDA": 0.5},
        init_cash=1000,
        rebalance="bands",
        band_abs_pct=5.0,
        band_rel_pct=20.0,
    )
    # Should rebalance at least once past the initial buy.
    assert trades["date"].nunique() > 1


def test_transaction_cost_reduces_final_value():
    prices = _prices(AAPL=[100.0, 100.0, 100.0])
    v_no_cost, _ = simulate_portfolio(prices, {"AAPL": 1.0}, init_cash=1000)
    v_with_cost, _ = simulate_portfolio(
        prices, {"AAPL": 1.0}, init_cash=1000, tcost_bps=50  # 0.5%
    )
    # Cost is deducted from cash on day 0 buy, so final value is lower.
    assert v_with_cost.iloc[-1] < v_no_cost.iloc[-1]


def test_slippage_worsens_buy_fill_price():
    prices = _prices(AAPL=[100.0, 100.0])
    _, trades_no_slip = simulate_portfolio(prices, {"AAPL": 1.0}, init_cash=1000)
    _, trades_slip = simulate_portfolio(
        prices, {"AAPL": 1.0}, init_cash=1000, slippage_bps=100  # 1%
    )
    # Buy fill with slippage should be above spot.
    assert trades_no_slip.iloc[0]["price"] == pytest.approx(100.0)
    assert trades_slip.iloc[0]["price"] == pytest.approx(101.0)


# compute_metrics


def test_total_return_and_cagr_for_known_series():
    value = pd.Series([100.0, 100.0, 100.0, 110.0], index=_dates(4))
    m = compute_metrics(value)
    assert m.total_return_pct == pytest.approx(10.0)
    assert m.cagr_pct > 100


def test_max_drawdown_captures_worst_peak_to_trough():
    value = pd.Series([100, 120, 60, 80, 90], index=_dates(5), dtype="float64")
    m = compute_metrics(value)
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
    # 60% equity doubles + 40% cash flat -> 60%*2 + 40%*1 = 1.6x
    assert result.portfolio_final_value == pytest.approx(16_000, rel=1e-2)
    assert result.benchmark_final_value == pytest.approx(15_000, rel=1e-2)
    # 60% high-beta equity + 40% cash: expect beta > 1 and < 2.
    assert 1.0 < result.beta < 2.0
    # No rebalancing means no trades after day 0.
    assert result.n_trades == 2  # one buy per ticker


def test_compare_strategies_produces_baseline_plus_three_presets(monkeypatch):
    from agentic_investor.eval.backtest import compare_strategies

    n = 100
    prices = pd.DataFrame(
        {"AAPL": np.linspace(100, 200, n), "NVDA": np.linspace(100, 200, n),
         "SPY": np.linspace(100, 150, n)},
        index=_dates(n),
    )
    monkeypatch.setattr(backtest, "fetch_prices", lambda *a, **k: prices)

    comp = compare_strategies(_rec(), rec_id=1)
    assert comp.rec_id == 1
    assert len(comp.entries) == 4
    labels = [e.label for e in comp.entries]
    assert "baseline (no friction, buy-and-hold)" in labels
    assert any("conservative" in lab for lab in labels)
    assert any("moderate" in lab for lab in labels)
    assert any("aggressive" in lab for lab in labels)
    # Preset entries carry their profile name; baseline does not.
    assert comp.entries[0].profile_name is None
    for e in comp.entries[1:]:
        assert e.profile_name in {"conservative", "moderate", "aggressive"}


def test_render_comparison_markdown_includes_every_row(monkeypatch):
    from agentic_investor.eval.backtest import (
        compare_strategies,
        render_comparison_markdown,
    )

    n = 60
    prices = pd.DataFrame(
        {"AAPL": np.linspace(100, 150, n), "NVDA": np.linspace(100, 150, n),
         "SPY": np.linspace(100, 130, n)},
        index=_dates(n),
    )
    monkeypatch.setattr(backtest, "fetch_prices", lambda *a, **k: prices)

    md = render_comparison_markdown(compare_strategies(_rec(), rec_id=42))
    assert "# Strategy comparison for Recommendation #42" in md
    assert "baseline (no friction, buy-and-hold)" in md
    assert "conservative preset" in md
    assert "moderate preset" in md
    assert "aggressive preset" in md
    assert "**SPY** (benchmark)" in md


def test_compare_allocators_regenerates_rec_per_preset(monkeypatch):
    from agentic_investor.eval import backtest as bt_mod
    from agentic_investor.eval.backtest import compare_allocators
    from agentic_investor.orchestrator import graph as graph_mod
    from agentic_investor.orchestrator.state import OrchestratorRequest

    n = 80
    prices = pd.DataFrame(
        {"AAPL": np.linspace(100, 180, n), "NVDA": np.linspace(100, 180, n),
         "TLT": np.linspace(100, 105, n), "GLD": np.linspace(100, 110, n),
         "SPY": np.linspace(100, 140, n)},
        index=_dates(n),
    )
    monkeypatch.setattr(bt_mod, "fetch_prices", lambda *a, **k: prices)

    calls: list[dict] = []

    def fake_run_orchestrator(request, profile=None):
        calls.append({
            "risk": request.risk,
            "tickers": list(request.tickers),
            "profile_name": (profile.name if profile else None),
        })
        return _rec(amount=request.amount)

    monkeypatch.setattr(graph_mod, "run_orchestrator", fake_run_orchestrator)
    # backtest.compare_allocators does `from agentic_investor.orchestrator.graph
    # import run_orchestrator` inside the function - patch that resolved name
    # by patching the module attribute where it will be looked up.
    monkeypatch.setattr(
        "agentic_investor.orchestrator.graph.run_orchestrator",
        fake_run_orchestrator,
    )

    req = OrchestratorRequest(tickers=["AAPL", "NVDA"], amount=10_000)
    comp = compare_allocators(req)

    assert len(comp.entries) == 4  # baseline + 3 presets
    # baseline call is the default (moderate) with no profile.
    assert calls[0]["profile_name"] is None
    # Preset calls in order: conservative (adds TLT+GLD), moderate, aggressive.
    assert calls[1]["risk"] == "conservative"
    assert "TLT" in calls[1]["tickers"] and "GLD" in calls[1]["tickers"]
    assert calls[2]["risk"] == "moderate"
    assert calls[3]["risk"] == "aggressive"
    # Labels expose the allocator so a reader knows why rows differ.
    assert "inverse_vol" in comp.entries[1].label
    assert "llm" in comp.entries[2].label
    assert "llm" in comp.entries[3].label


def test_multi_window_backtest_runs_each_window(monkeypatch):
    n = 50
    prices = pd.DataFrame(
        {"AAPL": np.linspace(100, 200, n), "NVDA": np.linspace(100, 200, n),
         "SPY": np.linspace(100, 150, n)},
        index=_dates(n),
    )
    monkeypatch.setattr(backtest, "fetch_prices", lambda *a, **k: prices)

    windows = [("2024-01-01", "2024-06-01"), ("2024-06-01", "2024-12-01")]
    results = multi_window_backtest(_rec(), windows)
    assert len(results) == 2
    for r in results:
        assert r.n_days > 0
