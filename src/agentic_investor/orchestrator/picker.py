"""Two-tier funnel: rules-based Selector for --auto stock picking.

Scans a universe of candidate tickers, fetches each MarketSnapshot (parallel
via ThreadPoolExecutor to keep wall-clock reasonable), scores with pure
deterministic rules over the snapshot fields, and returns the top-N. No LLM
calls in this layer - keeps cost O(1) per ticker regardless of LLM prices.

The M3 stack (technical + news + LLM allocator) then runs only on the top-N.
"""

import logging
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed

from pydantic import BaseModel, Field

from agentic_investor.tools.market import MarketSnapshot, get_market_snapshot

logger = logging.getLogger(__name__)


class TickerScore(BaseModel):
    ticker: str
    score: float
    reasons: list[str] = Field(default_factory=list)


def score_snapshot(snap: MarketSnapshot) -> tuple[float, list[str]]:
    """Rules-based score for a single MarketSnapshot. Higher = more attractive.

    Multi-dimension confluence: trend (SMA stack + ADX signals), momentum
    (returns, MACD, RSI zone), volume conviction, distance from 52-week high,
    named strategy signals. Symmetric penalties on obvious warnings.
    """
    score = 0.0
    reasons: list[str] = []

    # Momentum - recent returns
    if snap.ret_1m is not None:
        score += snap.ret_1m * 0.5
    if snap.ret_3m is not None:
        score += snap.ret_3m * 0.3

    # Trend confirmation
    if snap.above_sma_200:
        score += 10
        reasons.append("above SMA200")
    if snap.above_sma_50:
        score += 5

    # RSI zone: 45-65 is trend-following sweet spot; extremes penalized
    if snap.rsi_14 is not None:
        if 45 <= snap.rsi_14 <= 65:
            score += 5
            reasons.append(f"RSI {snap.rsi_14:.0f} (trend zone)")
        elif snap.rsi_14 > 75:
            score -= 5
            reasons.append(f"RSI {snap.rsi_14:.0f} overbought")
        elif snap.rsi_14 < 25:
            score -= 5
            reasons.append(f"RSI {snap.rsi_14:.0f} oversold")

    # Volume conviction
    if snap.vol_vs_avg is not None and snap.vol_vs_avg > 1.3:
        score += 3
        reasons.append(f"vol {snap.vol_vs_avg:.1f}x avg")

    # Distance from 52-week high (momentum retention)
    if snap.pct_from_52w_high is not None and snap.pct_from_52w_high > -8:
        score += 5
        reasons.append("near 52w high")

    # Named deterministic signals from tools/market
    if "golden_cross" in snap.signals:
        score += 5
        reasons.append("golden_cross")
    if "death_cross" in snap.signals:
        score -= 15
        reasons.append("death_cross")
    if "macd_bullish_cross" in snap.signals:
        score += 3
    if "macd_bearish_cross" in snap.signals:
        score -= 3
    if "strong_trend" in snap.signals:
        score += 2
    if "weak_or_ranging" in snap.signals:
        score -= 3

    return round(score, 2), reasons


def pick_top_n(
    tickers: list[str],
    *,
    top_n: int = 10,
    exclude: set[str] | None = None,
    max_workers: int = 10,
    period: str = "1y",
    fetch: Callable[[str], MarketSnapshot] | None = None,
) -> list[TickerScore]:
    """Score every ticker in parallel and return the top_n by score.

    `fetch` defaults to fetching a real snapshot from yfinance; tests inject a
    deterministic fake so they stay offline.
    """
    exclude = exclude or set()
    candidates = [t for t in tickers if t not in exclude]
    real_fetch = fetch or (lambda t: get_market_snapshot(t, period=period))

    def _fetch_one(t: str) -> tuple[str, MarketSnapshot | None]:
        try:
            return t, real_fetch(t)
        except Exception as e:  # noqa: BLE001 - one bad ticker mustn't sink the run
            logger.warning("picker: failed to fetch %s: %s", t, e)
            return t, None

    scored: list[TickerScore] = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(_fetch_one, t) for t in candidates]
        for fut in as_completed(futures):
            t, snap = fut.result()
            if snap is None:
                continue
            score, reasons = score_snapshot(snap)
            scored.append(TickerScore(ticker=t, score=score, reasons=reasons))

    scored.sort(key=lambda x: x.score, reverse=True)
    return scored[:top_n]
