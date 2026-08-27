"""Ticker universes: static lists of index membership for the autonomous picker.

Note on survivorship bias: these are current constituents. That is correct for
forward-looking recommendation (you can only buy what trades today) but wrong
for historical backtests, which need point-in-time constituent lookup. M4's
backtest layer runs on user-provided tickers only, so this is not an issue for
the current code but is worth understanding.
"""

# Dow Jones Industrial Average - current 30 members.
DOW_30 = [
    "AAPL", "AMGN", "AMZN", "AXP", "BA", "CAT", "CRM", "CSCO", "CVX", "DIS",
    "GS", "HD", "HON", "IBM", "JNJ", "JPM", "KO", "MCD", "MMM", "MRK",
    "MSFT", "NKE", "NVDA", "PG", "SHW", "TRV", "UNH", "V", "VZ", "WMT",
]

# SPDR Select Sector ETFs - one per major S&P 500 sector.
SECTOR_ETFS = [
    "XLK",   # Technology
    "XLF",   # Financials
    "XLV",   # Healthcare
    "XLE",   # Energy
    "XLI",   # Industrials
    "XLY",   # Consumer Discretionary
    "XLP",   # Consumer Staples
    "XLB",   # Materials
    "XLU",   # Utilities
    "XLRE",  # Real Estate
    "XLC",   # Communication Services
]

# Small AI/mega-cap tech basket for fast demos.
MEGA_TECH = ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA", "AVGO"]

_REGISTRY: dict[str, list[str]] = {
    "dow30": DOW_30,
    "sectors": SECTOR_ETFS,
    "mega_tech": MEGA_TECH,
}


def get_universe(name: str) -> list[str]:
    """Return the ticker list for a named universe."""
    if name not in _REGISTRY:
        raise ValueError(
            f"unknown universe {name!r}; available: {sorted(_REGISTRY)}"
        )
    return list(_REGISTRY[name])


def list_universes() -> dict[str, int]:
    """Return {name: count} for every available universe."""
    return {name: len(tickers) for name, tickers in _REGISTRY.items()}
