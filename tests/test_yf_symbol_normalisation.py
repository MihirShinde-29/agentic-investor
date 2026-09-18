"""Regression tests for the yfinance symbol normaliser.

Every arm startup pre-fix spewed
    ERROR yfinance: $BRK: possibly delisted; no price data found
    ERROR yfinance: HTTP Error 404: Quote not found for symbol: SNCY
because Alpaca returns 'BRK' but yfinance indexes 'BRK-B', and SNCY
was actually delisted. Both showed up across all three arms on every
launch (12+ errors per restart, and there were 4 restarts today).

The normaliser rewrites known class-share tickers to yfinance's dash
form and returns None for known-delisted symbols so callers can skip
cleanly rather than logging the 404 stack.
"""

from __future__ import annotations

from agentic_investor.tools.market import _normalise_yf_symbol


def test_plain_ticker_passes_through():
    assert _normalise_yf_symbol("AAPL") == "AAPL"
    assert _normalise_yf_symbol("MSFT") == "MSFT"


def test_lowercase_upper_normalised():
    """Callers upstream upper-case already, but the helper should
    stand alone. Case-insensitive filter + rewrite.
    """
    assert _normalise_yf_symbol("aapl") == "AAPL"
    assert _normalise_yf_symbol("sncy") is None
    assert _normalise_yf_symbol("brk.b") == "BRK-B"


def test_berkshire_class_shares_rewritten():
    """Alpaca style -> yfinance style: BRK.B / BRK.A / BRK -> dash form.
    Bare 'BRK' rewrites to BRK-B (the tradable class-B).
    """
    assert _normalise_yf_symbol("BRK") == "BRK-B"
    assert _normalise_yf_symbol("BRK.A") == "BRK-A"
    assert _normalise_yf_symbol("BRK.B") == "BRK-B"


def test_other_class_shares_rewritten():
    assert _normalise_yf_symbol("BF.A") == "BF-A"
    assert _normalise_yf_symbol("BF.B") == "BF-B"
    assert _normalise_yf_symbol("RDS.A") == "RDS-A"
    assert _normalise_yf_symbol("RDS.B") == "RDS-B"


def test_known_delisted_returns_none():
    """SNCY was actually delisted; every startup universe-scan hit its
    404. Returning None lets fetch_ohlcv raise ValueError cleanly
    instead of eating a per-arm HTTP error.
    """
    assert _normalise_yf_symbol("SNCY") is None


def test_empty_and_none_return_none():
    assert _normalise_yf_symbol("") is None
    assert _normalise_yf_symbol(None) is None  # type: ignore[arg-type]


def test_unknown_symbol_passes_through_unchanged():
    """The helper is a normaliser, not a validator. Unknown tickers
    pass through so yfinance can attempt the fetch - the goal is to
    stop known-noise, not to become an allow-list.
    """
    assert _normalise_yf_symbol("NOTAREALTICKER") == "NOTAREALTICKER"
