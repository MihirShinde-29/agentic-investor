"""Rolling-correlation utilities for concentration control.

Two-name allocator that ignores covariance can accidentally build a very
concentrated position: e.g. NVDA + MSFT both bullish -> 25% + 25% weight into
what is really one "mega-cap tech beta" bet at ~70% correlation. This module:

- Fetches N-day daily closes for a ticker list
- Returns the pairwise correlation matrix
- Finds pairs above a threshold whose combined allocation weight exceeds a cap

Fetches are best-effort: if any ticker fails, its correlations return None and
the constraint is silently skipped for that pair (never crashes the run).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class CorrelatedPair:
    """A high-correlation pair whose combined weight breaches the cap."""

    a: str
    b: str
    correlation: float
    joint_weight_pct: float
    cap_pct: float

    def as_violation(self) -> str:
        return (
            f"{self.a}+{self.b} joint weight {self.joint_weight_pct:.1f}% "
            f"exceeds correlated-pair cap {self.cap_pct:.1f}% "
            f"(corr={self.correlation:.2f})"
        )


def _fetch_daily_closes(
    tickers: list[str], window_days: int
) -> pd.DataFrame | None:
    """Fetch daily closes over the last N days. Returns None on total failure.

    Reads from the paper_store daily_bars cache first; only hits the provider
    when the cache doesn't already cover the requested window with enough
    rows. Past-day rows are immutable so caching them saves hundreds of
    provider calls per session (correlation recomputes on every regen).
    """
    if not tickers:
        return None
    try:
        from agentic_investor.tools.market import fetch_ohlcv
        from agentic_investor.tools.paper_store import (
            load_cached_daily_closes,
            save_daily_closes,
        )
    except ImportError:
        return None

    # Pull a bit more than window_days worth of trading days.
    period = f"{max(window_days + 10, 30)}d"
    today = pd.Timestamp.now(tz="UTC").normalize()
    # Cache lookup window: reach back enough calendar days to cover ~window_days
    # trading days plus a buffer for weekends/holidays.
    lookup_start = (today - pd.Timedelta(days=window_days + 20)).strftime("%Y-%m-%d")
    lookup_end = today.strftime("%Y-%m-%d")
    # A ticker's cache is considered "warm" when it has at least this many rows
    # in the range. Correlation on 55-of-60 days is indistinguishable from the
    # true 60-day correlation for our threshold-crossing use case.
    min_cached_rows = max(window_days - 5, 20)

    frames: dict[str, pd.Series] = {}
    for t in tickers:
        tkr = t.upper()
        cached: dict[str, float] = {}
        try:
            cached = load_cached_daily_closes(tkr, lookup_start, lookup_end)
        except Exception as e:  # noqa: BLE001 - cache miss shouldn't sink the run
            logger.debug("correlation: cache read failed for %s: %s", tkr, e)
        if len(cached) >= min_cached_rows:
            frames[tkr] = pd.Series(
                {pd.Timestamp(d): v for d, v in cached.items()},
                dtype=float,
            ).sort_index()
            continue
        try:
            df = fetch_ohlcv(t, period=period, interval="1d")
            if df is None or df.empty:
                continue
            closes = df["Close"].astype(float)
            frames[tkr] = closes
            try:
                save_daily_closes(
                    tkr,
                    {ts.strftime("%Y-%m-%d"): float(v) for ts, v in closes.items()},
                )
            except Exception as e:  # noqa: BLE001 - cache write shouldn't sink the run
                logger.debug("correlation: cache write failed for %s: %s", tkr, e)
        except Exception as e:  # noqa: BLE001 - one bad ticker mustn't sink the run
            logger.debug("correlation: fetch failed for %s: %s", t, e)
            continue
    if not frames:
        return None
    joined = pd.DataFrame(frames).dropna(how="all")
    if joined.empty or len(joined) < 5:  # need at least a week of data
        return None
    # Trim to the requested window (in trading days).
    return joined.tail(window_days)


def compute_correlation_matrix(
    tickers: list[str], window_days: int = 60
) -> pd.DataFrame | None:
    """Pairwise correlation of daily returns. None if no usable data."""
    closes = _fetch_daily_closes(tickers, window_days)
    if closes is None:
        return None
    returns = closes.pct_change().dropna(how="all")
    if len(returns) < 5:
        return None
    return returns.corr()


def find_correlated_over_cap(
    weights_pct: dict[str, float],
    *,
    window_days: int = 60,
    max_pair_correlation: float = 0.7,
    max_joint_pct: float = 50.0,
) -> list[CorrelatedPair]:
    """Return every pair whose joint weight breaches the correlated-pair cap.

    Only considers positions with weight > 0. Silently skips pairs whose
    correlation can't be computed (missing data).
    """
    tickers = [t for t, w in weights_pct.items() if w > 0]
    if len(tickers) < 2:
        return []
    matrix = compute_correlation_matrix(tickers, window_days=window_days)
    if matrix is None:
        return []
    out: list[CorrelatedPair] = []
    seen: set[tuple[str, str]] = set()
    for a in matrix.columns:
        for b in matrix.columns:
            if a >= b:
                continue
            key = (a, b)
            if key in seen:
                continue
            seen.add(key)
            try:
                corr = float(matrix.loc[a, b])
            except Exception:  # noqa: BLE001
                continue
            if corr != corr:  # NaN
                continue
            if corr < max_pair_correlation:
                continue
            joint = weights_pct.get(a, 0.0) + weights_pct.get(b, 0.0)
            if joint > max_joint_pct:
                out.append(CorrelatedPair(
                    a=a, b=b, correlation=corr,
                    joint_weight_pct=joint, cap_pct=max_joint_pct,
                ))
    return out


def find_correlated_pairs_hint(
    tickers: list[str],
    *,
    window_days: int = 60,
    threshold: float = 0.7,
) -> list[tuple[str, str, float]]:
    """List high-correlation pairs among a candidate universe, for the
    allocator prompt hint. Returns [(a, b, corr), ...] sorted by |corr| desc.
    Silently returns [] on any data failure.
    """
    matrix = compute_correlation_matrix(tickers, window_days=window_days)
    if matrix is None:
        return []
    out: list[tuple[str, str, float]] = []
    for a in matrix.columns:
        for b in matrix.columns:
            if a >= b:
                continue
            try:
                corr = float(matrix.loc[a, b])
            except Exception:  # noqa: BLE001
                continue
            if corr != corr:
                continue
            if abs(corr) >= threshold:
                out.append((a, b, corr))
    out.sort(key=lambda p: abs(p[2]), reverse=True)
    return out
