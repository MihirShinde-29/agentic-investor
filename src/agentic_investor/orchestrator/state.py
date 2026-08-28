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
    # LLM's stated conviction for this weight (0.0-1.0). Required as of M7+
    # so confidence-adaptive bands work consistently (was optional; live
    # observation 2026-08-28 rec #58 showed the LLM silently omitting it,
    # disabling the adaptive-band feature). Defaults to 0.5 (neutral) on
    # missing values for legacy recs loaded from store.
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class Allocation(BaseModel):
    positions: list[Position]
    cash_pct: float = Field(ge=0.0, le=100.0)
    cash_dollars: float = Field(ge=0.0)
    portfolio_rationale: str

    @model_validator(mode="before")
    @classmethod
    def _fold_cash_pseudo_position(cls, data):
        # gpt-4o-mini occasionally emits a Position with ticker "cash" AND a
        # cash_pct field, double-counting cash. Roll any such pseudo-positions
        # into cash_pct/cash_dollars so the real validators see a clean input.
        if not isinstance(data, dict):
            return data
        positions = data.get("positions") or []
        if not positions:
            return data
        def _field(p, name):
            return p.get(name) if isinstance(p, dict) else getattr(p, name, None)

        cash_labels = {"cash", "$", "usd"}
        real_positions = []
        pseudo_cash_pct = 0.0
        pseudo_cash_dollars = 0.0
        for p in positions:
            ticker = (_field(p, "ticker") or "").strip().lower()
            if ticker in cash_labels or ticker == "":
                pseudo_cash_pct += float(_field(p, "weight_pct") or 0.0)
                pseudo_cash_dollars += float(_field(p, "dollars") or 0.0)
                continue
            real_positions.append(p)
        if pseudo_cash_pct or pseudo_cash_dollars:
            existing_cash_pct = float(data.get("cash_pct") or 0.0)
            existing_cash_dollars = float(data.get("cash_dollars") or 0.0)
            # If BOTH fields are set the LLM duplicated cash - take the max
            # (the more conservative interpretation). If only one is set, sum
            # is trivially correct.
            merged_cash_pct = (
                max(existing_cash_pct, pseudo_cash_pct)
                if existing_cash_pct and pseudo_cash_pct
                else existing_cash_pct + pseudo_cash_pct
            )
            merged_cash_dollars = (
                max(existing_cash_dollars, pseudo_cash_dollars)
                if existing_cash_dollars and pseudo_cash_dollars
                else existing_cash_dollars + pseudo_cash_dollars
            )
            data = {
                **data,
                "positions": real_positions,
                "cash_pct": merged_cash_pct,
                "cash_dollars": merged_cash_dollars,
            }
        return data

    @model_validator(mode="after")
    def _weights_sum_to_100(self):
        total = sum(p.weight_pct for p in self.positions) + self.cash_pct
        # Perfect / near-perfect: no repair needed.
        if 99.5 <= total <= 100.5:
            return self
        # LLM arithmetic error inside a reasonable band: silently renormalize.
        # gpt-4o-mini reliably outputs sums like 105 or 110 with 8-10 tickers;
        # instructor retries eat cost without helping. Preserving ratios keeps
        # the LLM's relative conviction intact - only the absolute scale moves.
        if 90.0 <= total <= 115.0:
            factor = 100.0 / total
            for p in self.positions:
                p.weight_pct = round(p.weight_pct * factor, 2)
                p.dollars = round(p.dollars * factor, 2)
            self.cash_pct = round(self.cash_pct * factor, 2)
            self.cash_dollars = round(self.cash_dollars * factor, 2)
            # Absorb sub-percent residual into cash_pct so the total hits 100.
            residual = 100.0 - (sum(p.weight_pct for p in self.positions) + self.cash_pct)
            self.cash_pct = round(self.cash_pct + residual, 2)
            return self
        raise ValueError(
            f"weights sum to {total:.2f}, outside repair band [90, 115]; "
            f"regenerate allocation"
        )


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
    # Event-driven mode: fresh news events tagged HOT/COOKED with reaction_pct
    # rendered as a text block; None or empty when the daily orchestrator runs
    # without event context.
    news_batch_context: str
