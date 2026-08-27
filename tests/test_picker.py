"""Tests for M5 autonomous stock picker + universe registry."""

import pytest

from agentic_investor.orchestrator.picker import (
    TickerScore,
    pick_top_n,
    score_snapshot,
)
from agentic_investor.tools.market import MarketSnapshot
from agentic_investor.universes import DOW_30, get_universe, list_universes


def _bullish_snapshot(ticker: str = "TEST") -> MarketSnapshot:
    return MarketSnapshot(
        ticker=ticker,
        as_of="2026-08-01",
        close=200.0,
        sma_20=195.0, sma_50=180.0, sma_200=150.0,
        above_sma_50=True, above_sma_200=True,
        adx_14=32.0, rsi_14=60.0,
        macd=5.0, macd_signal=3.0, macd_hist=2.0,
        atr_pct=2.5, bb_percent_b=0.7, bb_bandwidth=0.15,
        vol_vs_avg=1.4, obv_trend="rising",
        high_52w=205.0, low_52w=120.0, pct_from_52w_high=-2.4,
        ret_1m=8.0, ret_3m=15.0, ret_6m=25.0, ret_1y=40.0,
        signals=["golden_cross", "macd_bullish_cross", "strong_trend"],
        patterns=[],
    )


def _bearish_snapshot(ticker: str = "TEST") -> MarketSnapshot:
    return MarketSnapshot(
        ticker=ticker,
        as_of="2026-08-01",
        close=50.0,
        sma_20=55.0, sma_50=65.0, sma_200=90.0,
        above_sma_50=False, above_sma_200=False,
        adx_14=28.0, rsi_14=32.0,
        macd=-3.0, macd_signal=-2.0, macd_hist=-1.0,
        atr_pct=4.0, bb_percent_b=0.15, bb_bandwidth=0.25,
        vol_vs_avg=1.6, obv_trend="falling",
        high_52w=110.0, low_52w=48.0, pct_from_52w_high=-54.5,
        ret_1m=-12.0, ret_3m=-25.0, ret_6m=-30.0, ret_1y=-45.0,
        signals=["death_cross", "macd_bearish_cross", "strong_trend"],
        patterns=["bearish_engulfing"],
    )


# Scoring


def test_bullish_snapshot_scores_higher_than_bearish():
    bull_score, _ = score_snapshot(_bullish_snapshot())
    bear_score, _ = score_snapshot(_bearish_snapshot())
    assert bull_score > bear_score
    assert bear_score < 0  # death_cross + drawdown = negative


def test_score_includes_reasons_for_bullish():
    _, reasons = score_snapshot(_bullish_snapshot())
    assert "above SMA200" in reasons
    assert "golden_cross" in reasons


def test_score_flags_overbought_as_penalty():
    snap = _bullish_snapshot()
    snap.rsi_14 = 85.0
    _, reasons = score_snapshot(snap)
    assert any("overbought" in r for r in reasons)


# Picker


def test_pick_top_n_sorts_and_limits():
    def fake_fetch(t: str) -> MarketSnapshot:
        # AAPL bullish, XYZ bearish, MSFT bullish - so top 2 should be bullish ones.
        if t == "XYZ":
            return _bearish_snapshot(t)
        return _bullish_snapshot(t)

    picks = pick_top_n(["AAPL", "MSFT", "XYZ"], top_n=2, fetch=fake_fetch)
    assert len(picks) == 2
    tickers = [p.ticker for p in picks]
    assert "XYZ" not in tickers
    assert all(isinstance(p, TickerScore) for p in picks)
    # Sorted descending by score
    assert picks[0].score >= picks[1].score


def test_pick_top_n_applies_exclude():
    def fake_fetch(t):
        return _bullish_snapshot(t)

    picks = pick_top_n(
        ["AAPL", "MSFT", "NVDA"], top_n=5, exclude={"MSFT"}, fetch=fake_fetch
    )
    assert {p.ticker for p in picks} == {"AAPL", "NVDA"}


def test_pick_top_n_skips_failing_fetches():
    def flaky_fetch(t):
        if t == "BAD":
            raise ValueError("no data")
        return _bullish_snapshot(t)

    picks = pick_top_n(["AAPL", "BAD", "MSFT"], top_n=5, fetch=flaky_fetch)
    assert {p.ticker for p in picks} == {"AAPL", "MSFT"}


def test_pick_top_n_handles_empty_universe():
    picks = pick_top_n([], top_n=5, fetch=lambda t: _bullish_snapshot(t))
    assert picks == []


# Universes


def test_dow30_has_30_unique_tickers():
    assert len(DOW_30) == 30
    assert len(set(DOW_30)) == 30
    # sanity: contains mega-caps
    assert {"AAPL", "MSFT", "NVDA"} <= set(DOW_30)


def test_get_universe_returns_copy():
    u1 = get_universe("dow30")
    u1.append("NEW")
    u2 = get_universe("dow30")
    assert "NEW" not in u2  # mutation should not leak


def test_get_universe_rejects_unknown():
    with pytest.raises(ValueError, match="unknown universe"):
        get_universe("bogus")


def test_list_universes_reports_counts():
    listing = list_universes()
    assert "dow30" in listing
    assert listing["dow30"] == 30
