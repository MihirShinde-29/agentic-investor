"""Tests for the paper_store daily_bars cache and its use by correlation."""

from __future__ import annotations

from unittest.mock import patch

import pandas as pd

from agentic_investor.tools.paper_store import (
    load_cached_daily_closes,
    save_daily_closes,
)


def test_load_returns_empty_when_no_rows_present():
    assert load_cached_daily_closes("AAPL", "2026-01-01", "2026-01-31") == {}


def test_save_then_load_roundtrips():
    save_daily_closes("AAPL", {"2026-01-02": 150.0, "2026-01-03": 151.5})
    got = load_cached_daily_closes("AAPL", "2026-01-01", "2026-01-31")
    assert got == {"2026-01-02": 150.0, "2026-01-03": 151.5}


def test_save_upserts_on_conflict():
    save_daily_closes("AAPL", {"2026-01-02": 150.0})
    save_daily_closes("AAPL", {"2026-01-02": 152.5})
    got = load_cached_daily_closes("AAPL", "2026-01-01", "2026-01-31")
    assert got == {"2026-01-02": 152.5}


def test_load_respects_date_range():
    save_daily_closes(
        "AAPL",
        {"2026-01-02": 150.0, "2026-02-15": 160.0, "2026-03-01": 170.0},
    )
    got = load_cached_daily_closes("AAPL", "2026-02-01", "2026-02-28")
    assert got == {"2026-02-15": 160.0}


def test_load_isolates_by_ticker():
    save_daily_closes("AAPL", {"2026-01-02": 150.0})
    save_daily_closes("MSFT", {"2026-01-02": 400.0})
    assert load_cached_daily_closes("AAPL", "2026-01-01", "2026-01-31") == {
        "2026-01-02": 150.0
    }
    assert load_cached_daily_closes("MSFT", "2026-01-01", "2026-01-31") == {
        "2026-01-02": 400.0
    }


def _synthetic_bars(days: int) -> pd.DataFrame:
    """A minimal fetch_ohlcv-shaped frame with `days` rows ending today."""
    idx = pd.date_range(
        end=pd.Timestamp.now(tz="UTC").normalize(),
        periods=days,
        freq="B",
    )
    return pd.DataFrame({"Close": [100.0 + i for i in range(days)]}, index=idx)


def test_correlation_second_call_uses_cache_and_skips_provider():
    from agentic_investor.orchestrator import correlation as corr_mod

    calls = {"n": 0}

    def _fake_fetch(ticker, *args, **kwargs):
        calls["n"] += 1
        return _synthetic_bars(65)

    # Correlation module imports fetch_ohlcv lazily inside the function, so
    # patch the source module rather than the re-export.
    with patch(
        "agentic_investor.tools.market.fetch_ohlcv", side_effect=_fake_fetch
    ):
        # First call: cache empty, provider hits once per ticker.
        first = corr_mod._fetch_daily_closes(["AAPL", "MSFT"], window_days=60)
        assert first is not None
        assert calls["n"] == 2
        # Second call same tickers: cache warm, provider not touched.
        second = corr_mod._fetch_daily_closes(["AAPL", "MSFT"], window_days=60)
        assert second is not None
        assert calls["n"] == 2  # no new fetches
    # And the shape is trimmed to at most the requested window.
    assert len(second) <= 60
    assert len(second) >= 55  # trading-day calendar has some slack
