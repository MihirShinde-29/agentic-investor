"""Technical Agent: turn a MarketSnapshot into a structured stance.

The tool already did the math and detected strategy triggers and candlestick
patterns. Here the LLM only weighs those features across dimensions and must
return a validated TechnicalSignal.
"""

from typing import Literal

from pydantic import BaseModel, Field

from agentic_investor.llm.client import structured_complete
from agentic_investor.tools.market import MarketSnapshot, get_market_snapshot

Stance = Literal["bullish", "neutral", "bearish"]


class TechnicalSignal(BaseModel):
    ticker: str
    stance: Stance
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str
    key_drivers: list[str] = Field(default_factory=list)


SYSTEM = (
    "You are a disciplined technical-analysis agent. You are given precomputed "
    "indicators, strategy signals, and candlestick patterns for a single stock, "
    "and must classify its near-term stance. Weigh confluence across independent "
    "dimensions (trend, momentum, volatility, volume, price structure); do not let "
    "several correlated momentum readings count as separate evidence. Cite the "
    "specific values and named signals you used, never invent data, and prefer "
    "'neutral' when dimensions disagree. In key_drivers, list the few signals or "
    "values that most drove your call. Confidence is your calibrated probability "
    "the stance is correct, in [0, 1]."
)

GUIDE = (
    "How to read the fields:\n"
    "- Trend: price vs SMA50/SMA200 gives direction; ADX>=25 means a real trend "
    "exists (follow it), ADX<20 means choppy (momentum signals whipsaw).\n"
    "- Momentum: RSI>70 overbought, <30 oversold; MACD histogram sign is momentum.\n"
    "- Volatility: bb_percent_b near 1 rides the upper band, >1 is a breakout, <0 "
    "breaks down; low bb_bandwidth is a squeeze that can precede a move; atr_pct is "
    "volatility as a share of price.\n"
    "- Volume: vol_vs_avg>1.5 is a conviction spike; obv_trend confirms or diverges "
    "from price.\n"
    "- Structure: pct_from_52w_high and the return fields show position and drift.\n"
    "- signals and patterns are already-detected triggers; treat them as evidence to "
    "weigh, not orders to obey."
)


def _messages(snapshot: MarketSnapshot) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM},
        {
            "role": "user",
            "content": (
                f"{GUIDE}\n\nFeatures (JSON):\n{snapshot.model_dump_json(indent=2)}\n\n"
                "Classify the near-term stance for this ticker."
            ),
        },
    ]


def analyze_technical(snapshot: MarketSnapshot, *, model: str | None = None) -> TechnicalSignal:
    signal = structured_complete(TechnicalSignal, _messages(snapshot), model=model)
    signal.ticker = snapshot.ticker  # trust our data over the model's echo
    return signal


def analyze_ticker(
    ticker: str, *, period: str = "2y", model: str | None = None
) -> TechnicalSignal:
    """Fetch prices and analyze in one call."""
    return analyze_technical(get_market_snapshot(ticker, period=period), model=model)
