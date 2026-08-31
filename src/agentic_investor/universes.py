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

# S&P 500 large-cap whitelist used by the news-body promotion filter (0w).
# Any ticker extracted from a news headline that ISN'T in this set gets
# dropped before it reaches the allocator. Prevents the LLM from opening
# micro-cap punts on single news headlines (BRVE, SLI, GPRO, WBUY, ATTO,
# RCBC-style trades observed 2026-08-31 live session).
# ~200 largest by market cap; refresh occasionally against the current index.
SP500_LARGE_CAPS = [
    # Mega tech
    "AAPL", "MSFT", "NVDA", "GOOGL", "GOOG", "AMZN", "META", "TSLA", "AVGO",
    "ORCL", "CRM", "AMD", "ADBE", "CSCO", "ACN", "IBM", "TXN", "QCOM", "INTU",
    "NOW", "AMAT", "MU", "LRCX", "ADI", "PANW", "KLAC", "SNPS", "CDNS", "ANET",
    "CRWD", "MRVL", "FTNT", "APH", "MSI", "ROP", "ADSK", "WDAY", "MCHP", "TEAM",
    "PLTR", "DDOG", "SNOW", "NET", "ZS", "MDB", "SMCI", "DELL", "HPQ", "HPE",
    # Financials
    "BRK.B", "JPM", "V", "MA", "BAC", "WFC", "GS", "MS", "AXP", "SCHW",
    "BLK", "C", "SPGI", "CB", "PGR", "PYPL", "USB", "PNC", "TFC", "COF",
    "TROW", "AON", "ICE", "CME", "MMC", "MET", "PRU", "ALL", "TRV", "AFL",
    # Healthcare
    "UNH", "JNJ", "LLY", "PFE", "ABBV", "MRK", "TMO", "ABT", "DHR", "AMGN",
    "ISRG", "CVS", "SYK", "REGN", "MDT", "VRTX", "ELV", "GILD", "BMY", "CI",
    "HUM", "MCK", "COR", "BSX", "ZTS", "BDX", "DXCM", "IDXX", "IQV", "A",
    "ARGX",
    # Consumer
    "WMT", "PG", "KO", "PEP", "COST", "HD", "MCD", "NKE", "SBUX", "TJX",
    "LOW", "TGT", "CMG", "DIS", "NFLX", "BKNG", "MDLZ", "PM", "MO", "CL",
    "GIS", "K", "HSY", "STZ", "KHC", "KMB", "EL", "MNST", "LULU", "ROST",
    "AZO", "ORLY", "YUM", "MAR", "HLT",
    # Industrials
    "GE", "CAT", "BA", "HON", "UPS", "RTX", "LMT", "DE", "UNP", "MMM",
    "GD", "ETN", "EMR", "ITW", "NSC", "CSX", "WM", "FDX", "NOC", "PH",
    # Energy
    "XOM", "CVX", "COP", "SLB", "EOG", "PSX", "MPC", "VLO", "OKE", "WMB",
    "DVN", "HES", "OXY", "FANG",
    # Defense + government services
    "LDOS", "SAIC", "CACI", "BAH",
    # Utilities + REITs
    "NEE", "SO", "DUK", "AEP", "SRE", "EXC", "XEL", "PCG", "AMT", "PLD",
    "CCI", "EQIX", "PSA", "SPG", "O",
    # Materials
    "LIN", "APD", "SHW", "ECL", "FCX", "NEM", "DOW", "DD", "NUE",
    # Communication + Media
    "VZ", "T", "TMUS", "CMCSA", "CHTR", "EA", "TTWO", "WBD", "PARA",
    # Transportation
    "UBER", "LYFT",
    # Consumer discretionary + fintech
    "AFRM", "DLTR",
    # Major ADRs
    "TSM", "SONY", "TAK",
    # Biotech + specialty pharma
    "BIIB", "CYTK", "EXEL",
    # Telecom equipment
    "CIEN",
    # Digital-asset holding
    "MSTR",
]

_REGISTRY: dict[str, list[str]] = {
    "dow30": DOW_30,
    "sectors": SECTOR_ETFS,
    "mega_tech": MEGA_TECH,
    "sp500_large": SP500_LARGE_CAPS,
}


# Set for O(1) lookup by the 0w promotion filter.
SP500_LARGE_SET: set[str] = set(SP500_LARGE_CAPS)


def is_actionable_ticker(ticker: str) -> bool:
    """True when a ticker is in the large-cap whitelist for 0w promotions."""
    return ticker.upper() in SP500_LARGE_SET


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
