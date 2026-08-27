"""Market data tool: fetch OHLCV prices and compute technical features.

No LLM calls here. The tool computes indicators, detects strategy triggers and
candlestick patterns deterministically, and returns a structured snapshot. The
agent reasons over that snapshot; it does not do arithmetic.

Features span independent dimensions on purpose (trend, momentum, volatility,
volume, price structure, patterns) rather than piling up correlated momentum
indicators.
"""

import numpy as np
import pandas as pd
import yfinance as yf
from pydantic import BaseModel, Field


class MarketSnapshot(BaseModel):
    """Latest technical picture for one ticker, ready for an agent to read."""

    ticker: str
    as_of: str
    close: float

    # Trend
    sma_20: float | None = None
    sma_50: float | None = None
    sma_200: float | None = None
    above_sma_50: bool | None = None
    above_sma_200: bool | None = None
    adx_14: float | None = None  # trend strength, not direction

    # Momentum
    rsi_14: float | None = None
    macd: float | None = None
    macd_signal: float | None = None
    macd_hist: float | None = None

    # Volatility
    atr_pct: float | None = None  # ATR as % of price
    bb_percent_b: float | None = None  # 0 = lower band, 1 = upper band
    bb_bandwidth: float | None = None

    # Volume
    vol_vs_avg: float | None = None  # today's volume / 20-day average
    obv_trend: str | None = None  # rising / falling / flat

    # Price structure
    high_52w: float | None = None
    low_52w: float | None = None
    pct_from_52w_high: float | None = None
    ret_1m: float | None = None
    ret_3m: float | None = None
    ret_6m: float | None = None
    ret_1y: float | None = None

    # Deterministic detections
    signals: list[str] = Field(default_factory=list)
    patterns: list[str] = Field(default_factory=list)


def _drop_incomplete(df: pd.DataFrame) -> pd.DataFrame:
    """Drop bars missing any OHLC value.

    Data providers often return a placeholder row for the current, still-forming
    session with NaN prices. Left in, it poisons rolling-window indicators.
    """
    return df.dropna(subset=["Open", "High", "Low", "Close"])


def fetch_ohlcv(
    ticker: str,
    period: str = "2y",
    interval: str = "1d",
    end: str | None = None,
) -> pd.DataFrame:
    """Download OHLCV bars for one ticker as a clean, single-index DataFrame.

    When `end` is provided (point-in-time mode), fetches [end - period, end]
    instead of [today - period, today]. Used by the --as-of picker to eliminate
    look-ahead bias when backtesting a historical strategy.
    """
    if end is None:
        df = yf.Ticker(ticker).history(
            period=period, interval=interval, auto_adjust=True
        )
    else:
        # yfinance can't combine period + end, so translate the coarse period
        # keyword into a start-date offset from `end`.
        end_ts = pd.Timestamp(end)
        years = {"1y": 1, "2y": 2, "5y": 5, "10y": 10, "max": 20}.get(period, 2)
        start_ts = end_ts - pd.DateOffset(years=years)
        df = yf.Ticker(ticker).history(
            start=start_ts.strftime("%Y-%m-%d"),
            end=end_ts.strftime("%Y-%m-%d"),
            interval=interval,
            auto_adjust=True,
        )
    df = _drop_incomplete(df)
    if df.empty:
        raise ValueError(f"no price data for {ticker!r} (bad symbol or no history?)")
    return df


# Basic building blocks

def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n).mean()


def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def _wilder(s: pd.Series, n: int) -> pd.Series:
    """Wilder's smoothing (an EMA with alpha = 1/n), used by RSI/ATR/ADX."""
    return s.ewm(alpha=1 / n, adjust=False).mean()


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    rs = _wilder(gain, n) / _wilder(loss, n)
    return 100 - (100 / (1 + rs))


def macd(
    close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> tuple[pd.Series, pd.Series, pd.Series]:
    macd_line = ema(close, fast) - ema(close, slow)
    signal_line = ema(macd_line, signal)
    return macd_line, signal_line, macd_line - signal_line


def true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["Close"].shift(1)
    ranges = pd.concat(
        [df["High"] - df["Low"], (df["High"] - prev_close).abs(), (df["Low"] - prev_close).abs()],
        axis=1,
    )
    return ranges.max(axis=1)


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    return _wilder(true_range(df), n)


def adx(df: pd.DataFrame, n: int = 14) -> pd.Series:
    """Average Directional Index: how strong the trend is (0-100), not its direction."""
    up = df["High"].diff()
    down = -df["Low"].diff()
    plus_dm = up.where((up > down) & (up > 0), 0.0)
    minus_dm = down.where((down > up) & (down > 0), 0.0)
    atr_ = atr(df, n)
    plus_di = 100 * _wilder(plus_dm, n) / atr_
    minus_di = 100 * _wilder(minus_dm, n) / atr_
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    return _wilder(dx, n)


def bollinger(
    close: pd.Series, n: int = 20, k: float = 2.0
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Return (upper, lower, %B, bandwidth). Uses population std to match convention."""
    mid = close.rolling(n).mean()
    std = close.rolling(n).std(ddof=0)
    upper = mid + k * std
    lower = mid - k * std
    percent_b = (close - lower) / (upper - lower)
    bandwidth = (upper - lower) / mid
    return upper, lower, percent_b, bandwidth


def obv(df: pd.DataFrame) -> pd.Series:
    """On-balance volume: adds volume on up days, subtracts on down days."""
    direction = np.sign(df["Close"].diff().fillna(0))
    return (direction * df["Volume"]).cumsum()


# Deterministic detections

def _crossed_up(a: pd.Series, b: pd.Series, lookback: int = 3) -> bool:
    d = (a - b).dropna()
    if len(d) < lookback + 1:
        return False
    w = d.iloc[-(lookback + 1) :]
    return bool(w.iloc[-1] > 0 and (w.iloc[:-1] <= 0).any())


def _crossed_down(a: pd.Series, b: pd.Series, lookback: int = 3) -> bool:
    d = (a - b).dropna()
    if len(d) < lookback + 1:
        return False
    w = d.iloc[-(lookback + 1) :]
    return bool(w.iloc[-1] < 0 and (w.iloc[:-1] >= 0).any())


def detect_candlestick_patterns(df: pd.DataFrame) -> list[str]:
    """Detect a few high-signal patterns on the last one or two candles."""
    if len(df) < 2:
        return []
    prev, cur = df.iloc[-2], df.iloc[-1]
    o, h, low_, c = float(cur["Open"]), float(cur["High"]), float(cur["Low"]), float(cur["Close"])
    po, pc = float(prev["Open"]), float(prev["Close"])
    rng = h - low_
    if rng <= 0:
        return []

    body = abs(c - o)
    upper_wick = h - max(o, c)
    lower_wick = min(o, c) - low_
    out: list[str] = []

    if body <= 0.1 * rng:
        out.append("doji")
    if body > 0 and lower_wick >= 2 * body and upper_wick <= body:
        out.append("hammer")
    if body > 0 and upper_wick >= 2 * body and lower_wick <= body:
        out.append("shooting_star")

    cur_up, prev_up = c > o, pc > po
    if cur_up and not prev_up and c >= po and o <= pc:
        out.append("bullish_engulfing")
    if not cur_up and prev_up and o >= pc and c <= po:
        out.append("bearish_engulfing")
    return out


# Helpers

def _last(s: pd.Series) -> float | None:
    v = s.iloc[-1]
    return None if pd.isna(v) else float(v)


def _round(v: float | None, ndigits: int = 2) -> float | None:
    return None if v is None else round(v, ndigits)


def _return_over(close: pd.Series, days: int) -> float | None:
    if len(close) < days + 1:
        return None
    return round((close.iloc[-1] / close.iloc[-1 - days] - 1) * 100, 2)


def get_market_snapshot(
    ticker: str, period: str = "2y", end: str | None = None
) -> MarketSnapshot:
    """Fetch prices and reduce them to a single latest-values snapshot.

    Pass `end` to snapshot ticker state as of a historical date (point-in-time)
    rather than today - the required move for unbiased historical backtests.
    """
    df = fetch_ohlcv(ticker, period=period, end=end)
    close = df["Close"]
    last_close = float(close.iloc[-1])

    sma_20s, sma_50s, sma_200s = sma(close, 20), sma(close, 50), sma(close, 200)
    macd_line, signal_line, hist = macd(close)
    _, _, percent_b, bandwidth = bollinger(close)
    atr_s = atr(df)
    adx_s = adx(df)
    obv_s = obv(df)
    rsi_last = _last(rsi(close))

    sma_50, sma_200 = _last(sma_50s), _last(sma_200s)
    window_52w = min(len(df), 252)
    high_52w = float(df["High"].tail(window_52w).max())
    low_52w = float(df["Low"].tail(window_52w).min())
    vol_avg = sma(df["Volume"], 20).iloc[-1]
    vol_vs_avg = (
        None if pd.isna(vol_avg) or vol_avg == 0 else float(df["Volume"].iloc[-1] / vol_avg)
    )

    obv_delta = obv_s.iloc[-1] - obv_s.iloc[-min(len(obv_s), 20)]
    obv_trend = "rising" if obv_delta > 0 else "falling" if obv_delta < 0 else "flat"

    signals = _detect_signals(
        last_close, rsi_last, sma_50s, sma_200s, macd_line, signal_line,
        percent_b, adx_s, vol_vs_avg,
    )

    return MarketSnapshot(
        ticker=ticker.upper(),
        as_of=str(df.index[-1].date()),
        close=round(last_close, 2),
        sma_20=_round(_last(sma_20s)),
        sma_50=_round(sma_50),
        sma_200=_round(sma_200),
        above_sma_50=None if sma_50 is None else last_close > sma_50,
        above_sma_200=None if sma_200 is None else last_close > sma_200,
        adx_14=_round(_last(adx_s)),
        rsi_14=_round(rsi_last),
        macd=_round(_last(macd_line), 4),
        macd_signal=_round(_last(signal_line), 4),
        macd_hist=_round(_last(hist), 4),
        atr_pct=_round(None if _last(atr_s) is None else _last(atr_s) / last_close * 100),
        bb_percent_b=_round(_last(percent_b), 3),
        bb_bandwidth=_round(_last(bandwidth), 4),
        vol_vs_avg=_round(vol_vs_avg),
        obv_trend=obv_trend,
        high_52w=round(high_52w, 2),
        low_52w=round(low_52w, 2),
        pct_from_52w_high=round((last_close / high_52w - 1) * 100, 2),
        ret_1m=_return_over(close, 21),
        ret_3m=_return_over(close, 63),
        ret_6m=_return_over(close, 126),
        ret_1y=_return_over(close, 252),
        signals=signals,
        patterns=detect_candlestick_patterns(df),
    )


def _detect_signals(
    last_close, rsi_last, sma_50s, sma_200s, macd_line, signal_line,
    percent_b, adx_s, vol_vs_avg,
) -> list[str]:
    out: list[str] = []

    if _crossed_up(sma_50s, sma_200s, lookback=5):
        out.append("golden_cross")
    elif _crossed_down(sma_50s, sma_200s, lookback=5):
        out.append("death_cross")

    if _crossed_up(macd_line, signal_line):
        out.append("macd_bullish_cross")
    elif _crossed_down(macd_line, signal_line):
        out.append("macd_bearish_cross")

    if rsi_last is not None:
        if rsi_last < 30:
            out.append("rsi_oversold")
        elif rsi_last > 70:
            out.append("rsi_overbought")

    pb = _last(percent_b)
    if pb is not None:
        if pb > 1:
            out.append("bollinger_breakout_up")
        elif pb < 0:
            out.append("bollinger_breakout_down")

    adx_last = _last(adx_s)
    if adx_last is not None:
        out.append("strong_trend" if adx_last >= 25 else "weak_or_ranging")

    if vol_vs_avg is not None and vol_vs_avg >= 1.5:
        out.append("volume_spike")

    return out
