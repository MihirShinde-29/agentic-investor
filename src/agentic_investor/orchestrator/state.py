"""Shared types for the orchestrator graph.

The orchestrator collects the per-ticker agent signals, hands them to an
allocator LLM, then validates the result. Two categories of type live here:

- Domain models (Pydantic): OrchestratorRequest, Position, Allocation,
  Recommendation. The Allocation model enforces the sum-to-100 and risk-band
  guardrails so a bad LLM output fails validation and instructor re-prompts.
- Graph state (TypedDict): GraphState, the shared bag the LangGraph nodes
  read/write. TypedDict is idiomatic for LangGraph.
"""

from typing import Literal, TypedDict

from pydantic import BaseModel, Field, model_validator

from agentic_investor.agents.news import NewsSignal
from agentic_investor.agents.technical import TechnicalSignal

RiskLevel = Literal["conservative", "moderate", "aggressive"]

# (max single-position weight, cash floor) per risk tier, in percent.
RISK_RULES: dict[RiskLevel, tuple[float, float]] = {
    "conservative": (20.0, 20.0),
    "moderate": (35.0, 10.0),
    "aggressive": (50.0, 0.0),
}


class OrchestratorRequest(BaseModel):
    tickers: list[str] = Field(min_length=1)
    amount: float = Field(gt=0)
    risk: RiskLevel = "moderate"
    target: str = "12-month growth"


class Position(BaseModel):
    ticker: str
    weight_pct: float = Field(ge=0.0, le=100.0)
    dollars: float = Field(ge=0.0)
    rationale: str


class Allocation(BaseModel):
    positions: list[Position]
    cash_pct: float = Field(ge=0.0, le=100.0)
    cash_dollars: float = Field(ge=0.0)
    portfolio_rationale: str

    @model_validator(mode="after")
    def _weights_sum_to_100(self):
        total = sum(p.weight_pct for p in self.positions) + self.cash_pct
        # Small tolerance for LLM float rounding; instructor retries on failure.
        if not (99.5 <= total <= 100.5):
            raise ValueError(
                f"weights must sum to 100 (got {total:.2f}); adjust positions or cash_pct"
            )
        return self


def check_risk_rules(allocation: Allocation, risk: RiskLevel) -> list[str]:
    """Return a list of guardrail violations, empty if the allocation is clean."""
    max_single, cash_floor = RISK_RULES[risk]
    violations: list[str] = []
    for p in allocation.positions:
        if p.weight_pct > max_single + 0.5:
            violations.append(
                f"{p.ticker} weight {p.weight_pct:.1f}% exceeds {risk} cap {max_single:.0f}%"
            )
    if allocation.cash_pct + 0.5 < cash_floor:
        violations.append(
            f"cash {allocation.cash_pct:.1f}% below {risk} floor {cash_floor:.0f}%"
        )
    return violations


def check_profile_rules(allocation: Allocation, profile) -> list[str]:
    """Profile-aware version of check_risk_rules. Accepts a StrategyProfile.

    Uses the profile's max_single_pct + cash_floor_pct rather than the fixed
    RISK_RULES table. This is the M6+ path; check_risk_rules stays for backward
    compatibility.
    """
    violations: list[str] = []
    for p in allocation.positions:
        if p.weight_pct > profile.max_single_pct + 0.5:
            violations.append(
                f"{p.ticker} weight {p.weight_pct:.1f}% exceeds "
                f"{profile.name} cap {profile.max_single_pct:.0f}%"
            )
    if allocation.cash_pct + 0.5 < profile.cash_floor_pct:
        violations.append(
            f"cash {allocation.cash_pct:.1f}% below "
            f"{profile.name} floor {profile.cash_floor_pct:.0f}%"
        )
    return violations


class Recommendation(BaseModel):
    request: OrchestratorRequest
    allocation: Allocation
    technical_signals: list[TechnicalSignal] = Field(default_factory=list)
    news_signals: list[NewsSignal] = Field(default_factory=list)
    violations: list[str] = Field(default_factory=list)


class GraphState(TypedDict, total=False):
    """Shared bag threaded through the LangGraph nodes."""

    request: OrchestratorRequest
    profile: object  # StrategyProfile, kept as object to avoid circular imports
    technical_signals: list[TechnicalSignal]
    news_signals: list[NewsSignal]
    market_snapshots: dict[str, object]  # dict[str, MarketSnapshot]
    allocation: Allocation
    violations: list[str]
