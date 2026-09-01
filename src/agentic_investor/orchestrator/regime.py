"""Market-regime detector: label + supporting macro signals for the allocator.

Signals combined into a single label bucket the allocator can condition on:

- **VIX** (^VIX): implied vol. >=25 flips to high_vol regardless of trend.
- **SPY vs 200-day SMA + 20-day momentum**: bull tape when SPY > SMA200 and
  20d > +2%; bear when the opposite; sideways otherwise.
- **Yield curve** (10y - 3mo, ^TNX - ^IRX): inversion is a classic
  late-cycle warning; noted alongside the label.
- **Credit spread** (HYG / LQD ratio, 20d change): falling ratio = credit
  stress rising; flags as a recession-watch modifier.
- **Dollar** (UUP 20d change): rising dollar signals global risk-off.

Buckets: bull, bear, sideways, high_vol, unknown. When VIX or credit stress
is loud, high_vol wins outright. Yield-curve and dollar mostly inform the
prompt narrative rather than swinging the label.

Fetches are best-effort: yfinance issues degrade to `None` fields and the
label falls back to "unknown".
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
    # Broader macro signals. Any of these may be None if yfinance chokes on
    # the symbol; label logic degrades gracefully.
    yield_curve_bps: float | None = None  # 10y - 3mo in basis points
    credit_stress_pct: float | None = None  # 20d change in HYG/LQD ratio, %
    dollar_20d_pct: float | None = None  # 20d change in UUP, %
    justification: str = ""

    def prompt_block(self) -> str:
        parts = [f"regime = {self.label}"]
        if self.vix is not None:
            parts.append(f"VIX={self.vix:.1f}")
        if self.spy_close is not None and self.spy_sma200 is not None:
            above = "above" if self.spy_close > self.spy_sma200 else "below"
            parts.append(f"SPY {above} 200-day")
        if self.spy_20d_return_pct is not None:
            parts.append(f"SPY 20d={self.spy_20d_return_pct:+.1f}%")
        if self.yield_curve_bps is not None:
            state = "inverted" if self.yield_curve_bps < 0 else "positive"
            parts.append(f"10y-3mo={self.yield_curve_bps:+.0f}bps ({state})")
        if self.credit_stress_pct is not None:
            parts.append(f"HYG/LQD 20d={self.credit_stress_pct:+.1f}%")
        if self.dollar_20d_pct is not None:
            parts.append(f"UUP 20d={self.dollar_20d_pct:+.1f}%")
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


def _last_price(ticker: str, period: str = "1mo") -> float | None:
    try:
        from agentic_investor.tools.market import fetch_ohlcv

        df = fetch_ohlcv(ticker, period=period, interval="1d")
        if df is None or df.empty:
            return None
        return float(df["Close"].iloc[-1])
    except Exception as e:  # noqa: BLE001
        logger.debug("regime: %s fetch failed: %s", ticker, e)
        return None


def _yield_curve_bps() -> float | None:
    """10-year Treasury yield minus 3-month T-bill, in basis points.

    Negative = inverted (classic late-cycle / recession warning).
    yfinance reports both as percent points (e.g. 4.25 = 4.25%), so the
    subtraction * 100 yields basis points.
    """
    ten = _last_price("^TNX")
    three_mo = _last_price("^IRX")
    if ten is None or three_mo is None:
        return None
    return (ten - three_mo) * 100.0


def _credit_stress_pct() -> float | None:
    """20-day change in the HYG/LQD ratio.

    HYG is junk debt, LQD is investment grade; when their price ratio drops,
    junk is underperforming and credit stress is rising. Positive number =
    credit conditions easing; negative = tightening.
    """
    try:
        from agentic_investor.tools.market import fetch_ohlcv

        h = fetch_ohlcv("HYG", period="2mo", interval="1d")
        l_ = fetch_ohlcv("LQD", period="2mo", interval="1d")
        if h is None or l_ is None or h.empty or l_.empty or len(h) < 20 or len(l_) < 20:
            return None
        ratio = (h["Close"].astype(float) / l_["Close"].astype(float)).dropna()
        if len(ratio) < 20:
            return None
        return float((ratio.iloc[-1] / ratio.iloc[-20] - 1.0) * 100.0)
    except Exception as e:  # noqa: BLE001
        logger.debug("regime: credit stress fetch failed: %s", e)
        return None


def _dollar_20d_pct() -> float | None:
    """20-day return on the UUP dollar-index ETF (positive = strengthening)."""
    try:
        from agentic_investor.tools.market import fetch_ohlcv

        df = fetch_ohlcv("UUP", period="2mo", interval="1d")
        if df is None or df.empty or len(df) < 20:
            return None
        close = df["Close"].astype(float)
        return float((close.iloc[-1] / close.iloc[-20] - 1.0) * 100.0)
    except Exception as e:  # noqa: BLE001
        logger.debug("regime: UUP fetch failed: %s", e)
        return None


def detect_regime() -> MarketRegime:
    """Compute the current regime label. Safe to call each regen."""
    spy = _last_close("SPY")
    vix = _last_price("^VIX")
    yc = _yield_curve_bps()
    credit = _credit_stress_pct()
    dollar = _dollar_20d_pct()

    spy_close = spy[0] if spy else None
    sma200 = spy[1] if spy else None
    pct_20d = spy[2] if spy else None

    # Panic wins first: elevated VIX or a big credit-stress drop always
    # flips to high_vol regardless of trend.
    if vix is not None and vix >= 25:
        return MarketRegime(
            label="high_vol",
            vix=vix, spy_close=spy_close, spy_sma200=sma200,
            spy_20d_return_pct=pct_20d,
            yield_curve_bps=yc, credit_stress_pct=credit, dollar_20d_pct=dollar,
            justification=(
                f"VIX {vix:.1f} elevated - trim risk, prefer cash and hedges"
            ),
        )
    if credit is not None and credit < -3.0:
        return MarketRegime(
            label="high_vol",
            vix=vix, spy_close=spy_close, spy_sma200=sma200,
            spy_20d_return_pct=pct_20d,
            yield_curve_bps=yc, credit_stress_pct=credit, dollar_20d_pct=dollar,
            justification=(
                f"HYG/LQD down {credit:.1f}% in 20d - credit stress, trim risk"
            ),
        )

    if spy_close is None or sma200 is None or pct_20d is None:
        return MarketRegime(
            label="unknown", vix=vix, spy_close=spy_close,
            spy_sma200=sma200, spy_20d_return_pct=pct_20d,
            yield_curve_bps=yc, credit_stress_pct=credit, dollar_20d_pct=dollar,
            justification="regime signals unavailable - use profile default",
        )

    above_200 = spy_close > sma200
    if above_200 and pct_20d > 2:
        label, note = "bull", "SPY above 200d + 20d momentum > +2% - lean risk-on"
    elif not above_200 and pct_20d < -2:
        label, note = "bear", "SPY below 200d + 20d momentum < -2% - lean defensive"
    else:
        label, note = "sideways", "no clear directional trend - stick to profile default"

    # Recession-watch modifier: inverted curve doesn't flip the label but is
    # worth flagging in the narrative so the allocator can bias defensive.
    if yc is not None and yc < 0 and label in {"bull", "sideways"}:
        note += f"; note yield curve inverted {yc:.0f}bps - recession watch"

    return MarketRegime(
        label=label, vix=vix, spy_close=spy_close, spy_sma200=sma200,
        spy_20d_return_pct=pct_20d,
        yield_curve_bps=yc, credit_stress_pct=credit, dollar_20d_pct=dollar,
        justification=note,
    )
