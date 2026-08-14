"""Technical Agent: turn a MarketSnapshot into a structured stance.

The tool already did the math. Here the LLM only exercises judgment over those
numbers and must return a validated TechnicalSignal.
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


SYSTEM = (
    "You are a disciplined technical-analysis agent. You are given precomputed "
    "indicators for a single stock and must classify its near-term stance. "
    "Cite the specific indicator values you used, never invent data, and prefer "
    "'neutral' when signals conflict. Confidence is your calibrated probability "
    "the stance is correct, in [0, 1]."
)

GUIDE = (
    "Interpretation: RSI > 70 overbought, < 30 oversold. Price above SMA200 is a "
    "long-term uptrend, above SMA50 a medium-term one. MACD histogram > 0 means "
    "upward momentum, < 0 downward."
)


def _messages(snapshot: MarketSnapshot) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM},
        {
            "role": "user",
            "content": (
                f"{GUIDE}\n\nIndicators (JSON):\n{snapshot.model_dump_json(indent=2)}\n\n"
                "Classify the near-term stance for this ticker."
            ),
        },
    ]


def analyze_technical(snapshot: MarketSnapshot, *, model: str | None = None) -> TechnicalSignal:
    signal = structured_complete(TechnicalSignal, _messages(snapshot), model=model)
    signal.ticker = snapshot.ticker  # trust our data over the model's echo
    return signal


def analyze_ticker(
    ticker: str, *, period: str = "1y", model: str | None = None
) -> TechnicalSignal:
    """Fetch prices and analyze in one call."""
    return analyze_technical(get_market_snapshot(ticker, period=period), model=model)
