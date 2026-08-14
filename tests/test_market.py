"""Unit tests for the market tool. No network: fetch_ohlcv is mocked."""

import numpy as np
import pandas as pd
import pytest

from agentic_investor.tools import market


def _series(values):
    idx = pd.date_range("2024-01-01", periods=len(values), freq="D")
    return pd.Series(values, index=idx, dtype="float64")


def test_sma_matches_rolling_mean():
    s = _series([1, 2, 3, 4, 5])
    out = market.sma(s, 3)
    assert pd.isna(out.iloc[0])
    assert out.iloc[2] == pytest.approx(2.0)  # mean(1, 2, 3)
    assert out.iloc[4] == pytest.approx(4.0)  # mean(3, 4, 5)


def test_rsi_is_100_when_only_gains():
    s = _series(list(range(1, 40)))
    assert market.rsi(s).iloc[-1] == pytest.approx(100.0)


def test_rsi_is_0_when_only_losses():
    s = _series(list(range(40, 1, -1)))
    assert market.rsi(s).iloc[-1] == pytest.approx(0.0)


def test_macd_histogram_is_line_minus_signal():
    s = _series(np.linspace(10, 50, 60))
    line, signal, hist = market.macd(s)
    assert hist.iloc[-1] == pytest.approx(line.iloc[-1] - signal.iloc[-1])


def test_snapshot_reduces_prices_to_latest_values(monkeypatch):
    closes = np.linspace(100, 300, 250)  # 250 rising days: enough for sma_200
    idx = pd.date_range("2023-01-01", periods=len(closes), freq="D")
    df = pd.DataFrame(
        {"Open": closes, "High": closes, "Low": closes, "Close": closes, "Volume": 1000},
        index=idx,
    )
    monkeypatch.setattr(market, "fetch_ohlcv", lambda *a, **k: df)

    snap = market.get_market_snapshot("aapl")
    assert snap.ticker == "AAPL"
    assert snap.close == pytest.approx(300.0)
    assert snap.above_sma_50 is True
    assert snap.above_sma_200 is True
    assert snap.sma_200 is not None
    assert 0 <= snap.rsi_14 <= 100
