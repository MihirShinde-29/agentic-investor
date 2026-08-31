"""Market-regime detector (lite): one label for the allocator to condition on.

Combines three off-the-shelf signals into a single regime bucket:

- **VIX** (^VIX): implied vol. Above 25 = elevated fear, above 30 = panic.
- **SPY vs 200-day SMA**: bull tape when SPY > SMA200, bear when below.
- **SPY 20-day momentum**: rising when +2%, falling when -2%.

Buckets: bull, bear, sideways, high_vol. The allocator prompt gets the
label + a one-line justification so it can push aggressive in bull, defensive
in bear/high_vol.

Fetches are best-effort: if yfinance chokes, returns "unknown" so the
allocator falls back to the profile default.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

RegimeLabel = str  # "bull" | "bear" | "sideways" | "high_vol" | "unknown"


@dataclass
class MarketRegime:
    label: RegimeLabel
    vix: float | None
    spy_close: float | None
    spy_sma200: float | None
    spy_20d_return_pct: float | None
    justification: str

    def prompt_block(self) -> str:
        parts = [f"regime = {self.label}"]
        if self.vix is not None:
            parts.append(f"VIX={self.vix:.1f}")
        if self.spy_close is not None and self.spy_sma200 is not None:
            above = "above" if self.spy_close > self.spy_sma200 else "below"
            parts.append(f"SPY {above} 200-day")
        if self.spy_20d_return_pct is not None:
            parts.append(f"SPY 20d={self.spy_20d_return_pct:+.1f}%")
        return " · ".join(parts) + f"\n  {self.justification}"


def _last_close(ticker: str, period: str = "1y") -> tuple[float, float, float] | None:
    """Return (last_close, sma200, pct_20d) or None on failure."""
    try:
        from agentic_investor.tools.market import fetch_ohlcv

        df = fetch_ohlcv(ticker, period=period, interval="1d")
        if df is None or df.empty or len(df) < 20:
            return None
        close = df["Close"].astype(float)
        last = float(close.iloc[-1])
        sma200 = float(close.tail(200).mean()) if len(close) >= 200 else float("nan")
        past = float(close.iloc[-min(len(close), 20)])
        pct_20d = (last / past - 1) * 100.0 if past > 0 else 0.0
        return last, sma200, pct_20d
    except Exception as e:  # noqa: BLE001
        logger.debug("regime: fetch failed for %s: %s", ticker, e)
        return None


def _last_vix() -> float | None:
    try:
        from agentic_investor.tools.market import fetch_ohlcv

        df = fetch_ohlcv("^VIX", period="1mo", interval="1d")
        if df is None or df.empty:
            return None
        return float(df["Close"].iloc[-1])
    except Exception as e:  # noqa: BLE001
        logger.debug("regime: VIX fetch failed: %s", e)
        return None


def detect_regime() -> MarketRegime:
    """Compute the current regime label. Safe to call each regen."""
    spy = _last_close("SPY")
    vix = _last_vix()

    spy_close = spy[0] if spy else None
    sma200 = spy[1] if spy else None
    pct_20d = spy[2] if spy else None

    # Panic wins first: elevated VIX always means high_vol regardless of trend.
    if vix is not None and vix >= 25:
        return MarketRegime(
            label="high_vol",
            vix=vix, spy_close=spy_close, spy_sma200=sma200,
            spy_20d_return_pct=pct_20d,
            justification=(
                f"VIX {vix:.1f} elevated - trim risk, prefer cash and hedges"
            ),
        )

    if spy_close is None or sma200 is None or pct_20d is None:
        return MarketRegime(
            label="unknown", vix=vix, spy_close=spy_close,
            spy_sma200=sma200, spy_20d_return_pct=pct_20d,
            justification="regime signals unavailable - use profile default",
        )

    above_200 = spy_close > sma200
    if above_200 and pct_20d > 2:
        label, note = "bull", "SPY above 200d + 20d momentum > +2% - lean risk-on"
    elif not above_200 and pct_20d < -2:
        label, note = "bear", "SPY below 200d + 20d momentum < -2% - lean defensive"
    else:
        label, note = "sideways", "no clear directional trend - stick to profile default"

    return MarketRegime(
        label=label, vix=vix, spy_close=spy_close, spy_sma200=sma200,
        spy_20d_return_pct=pct_20d, justification=note,
    )
