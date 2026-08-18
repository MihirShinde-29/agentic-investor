"""Unit tests for the market tool. No network: fetch_ohlcv is mocked."""

import numpy as np
import pandas as pd
import pytest

from agentic_investor.tools import market


def _series(values):
    idx = pd.date_range("2024-01-01", periods=len(values), freq="D")
    return pd.Series(values, index=idx, dtype="float64")


def _ohlcv(opens, highs, lows, closes, volumes=None):
    n = len(closes)
    idx = pd.date_range("2023-01-01", periods=n, freq="D")
    return pd.DataFrame(
        {
            "Open": opens,
            "High": highs,
            "Low": lows,
            "Close": closes,
            "Volume": volumes if volumes is not None else [1000] * n,
        },
        index=idx,
    )


# Core indicators

def test_sma_matches_rolling_mean():
    s = _series([1, 2, 3, 4, 5])
    out = market.sma(s, 3)
    assert pd.isna(out.iloc[0])
    assert out.iloc[2] == pytest.approx(2.0)
    assert out.iloc[4] == pytest.approx(4.0)


def test_rsi_is_100_when_only_gains():
    assert market.rsi(_series(list(range(1, 40)))).iloc[-1] == pytest.approx(100.0)


def test_rsi_is_0_when_only_losses():
    assert market.rsi(_series(list(range(40, 1, -1)))).iloc[-1] == pytest.approx(0.0)


def test_macd_histogram_is_line_minus_signal():
    line, signal, hist = market.macd(_series(np.linspace(10, 50, 60)))
    assert hist.iloc[-1] == pytest.approx(line.iloc[-1] - signal.iloc[-1])


def test_atr_equals_constant_true_range():
    n = 30
    df = _ohlcv([100] * n, [101] * n, [99] * n, [100] * n)
    assert market.atr(df).iloc[-1] == pytest.approx(2.0)


def test_adx_is_bounded_0_to_100():
    closes = np.linspace(100, 200, 60)
    df = _ohlcv(closes, closes * 1.01, closes * 0.99, closes)
    adx_last = market.adx(df).iloc[-1]
    assert 0 <= adx_last <= 100


def test_bollinger_percent_b_within_bands_for_oscillating_series():
    closes = [100, 101, 99] * 10
    _, _, percent_b, bandwidth = market.bollinger(_series(closes))
    assert 0 <= percent_b.iloc[-1] <= 1
    assert bandwidth.iloc[-1] > 0


def test_obv_rises_with_price():
    closes = list(range(1, 30))
    df = _ohlcv(closes, closes, closes, closes)
    obv_s = market.obv(df)
    assert obv_s.iloc[-1] > obv_s.iloc[0]


# Crossovers and data hygiene

def test_drop_incomplete_removes_trailing_nan_bar():
    df = _ohlcv([100.0, 101.0], [101.0, 102.0], [99.0, 100.0], [100.0, np.nan])
    cleaned = market._drop_incomplete(df)
    assert len(cleaned) == 1
    assert not cleaned["Close"].isna().any()


def test_crossed_up_and_down():
    a = _series([1, 2, 3, 4, 5])
    b = _series([3, 3, 3, 3, 3])
    assert market._crossed_up(a, b) is True
    assert market._crossed_down(a, b) is False
    assert market._crossed_down(b, a) is True


# Candlestick patterns

def test_bullish_engulfing_detected():
    df = _ohlcv(
        opens=[10.0, 8.5], highs=[10.1, 10.6], lows=[8.9, 8.4], closes=[9.0, 10.5]
    )
    assert "bullish_engulfing" in market.detect_candlestick_patterns(df)


def test_doji_detected():
    df = _ohlcv(
        opens=[100.0, 100.0], highs=[100.5, 101.0], lows=[99.5, 99.0], closes=[100.1, 100.02]
    )
    assert "doji" in market.detect_candlestick_patterns(df)


# Full snapshot

def test_snapshot_populates_new_fields(monkeypatch):
    n = 260
    closes = np.linspace(100, 300, n)
    df = _ohlcv(closes * 0.995, closes * 1.01, closes * 0.99, closes, volumes=[1000] * n)
    monkeypatch.setattr(market, "fetch_ohlcv", lambda *a, **k: df)

    snap = market.get_market_snapshot("aapl")
    assert snap.ticker == "AAPL"
    assert snap.above_sma_200 is True
    assert snap.adx_14 is not None
    assert snap.atr_pct is not None and snap.atr_pct > 0
    assert snap.bb_percent_b is not None
    assert snap.ret_1y is not None
    assert isinstance(snap.signals, list)
    assert isinstance(snap.patterns, list)
    assert 0 <= snap.rsi_14 <= 100
