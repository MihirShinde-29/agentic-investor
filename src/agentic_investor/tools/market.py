"""Market data tool: fetch OHLCV prices and compute technical indicators.

No LLM calls live here. The tool does the math and returns a structured
snapshot; the agent reasons over that snapshot.
"""

import pandas as pd
import yfinance as yf
from pydantic import BaseModel


class MarketSnapshot(BaseModel):
    """Latest technical picture for one ticker, ready for an agent to read."""

    ticker: str
    as_of: str
    close: float
    sma_20: float | None = None
    sma_50: float | None = None
    sma_200: float | None = None
    rsi_14: float | None = None
    macd: float | None = None
    macd_signal: float | None = None
    macd_hist: float | None = None
    above_sma_50: bool | None = None
    above_sma_200: bool | None = None


def fetch_ohlcv(ticker: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
    """Download OHLCV bars for one ticker as a clean, single-index DataFrame."""
    df = yf.Ticker(ticker).history(period=period, interval=interval, auto_adjust=True)
    if df.empty:
        raise ValueError(f"no price data for {ticker!r} (bad symbol or no history?)")
    return df


def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n).mean()


def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    """Wilder's RSI: gains/losses smoothed with an EMA of alpha = 1/n."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / n, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / n, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def macd(
    close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Return (macd_line, signal_line, histogram)."""
    macd_line = ema(close, fast) - ema(close, slow)
    signal_line = ema(macd_line, signal)
    return macd_line, signal_line, macd_line - signal_line


def _last(s: pd.Series) -> float | None:
    """Last value as a float, or None if it isn't there yet (short history)."""
    v = s.iloc[-1]
    return None if pd.isna(v) else float(v)


def _round(v: float | None, ndigits: int = 2) -> float | None:
    return None if v is None else round(v, ndigits)


def get_market_snapshot(ticker: str, period: str = "1y") -> MarketSnapshot:
    """Fetch prices and reduce them to a single latest-values snapshot."""
    df = fetch_ohlcv(ticker, period=period)
    close = df["Close"]

    macd_line, signal_line, hist = macd(close)
    sma_50 = _last(sma(close, 50))
    sma_200 = _last(sma(close, 200))
    last_close = float(close.iloc[-1])

    return MarketSnapshot(
        ticker=ticker.upper(),
        as_of=str(df.index[-1].date()),
        close=round(last_close, 2),
        sma_20=_round(_last(sma(close, 20))),
        sma_50=_round(sma_50),
        sma_200=_round(sma_200),
        rsi_14=_round(_last(rsi(close))),
        macd=_round(_last(macd_line), 4),
        macd_signal=_round(_last(signal_line), 4),
        macd_hist=_round(_last(hist), 4),
        above_sma_50=None if sma_50 is None else last_close > sma_50,
        above_sma_200=None if sma_200 is None else last_close > sma_200,
    )
