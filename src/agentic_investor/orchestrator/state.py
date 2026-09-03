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
    weight_pct: float = Field(gt=0.0, le=100.0)
    dollars: float = Field(ge=0.0)
    rationale: str
    # LLM conviction for this weight; drives confidence-adaptive rebalance
    # bands. Default 0.5 keeps legacy recs (loaded from store) sane.
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class Allocation(BaseModel):
    positions: list[Position]
    cash_pct: float = Field(ge=0.0, le=100.0)
    cash_dollars: float = Field(ge=0.0)
    portfolio_rationale: str

    @model_validator(mode="before")
    @classmethod
    def _fold_cash_pseudo_position(cls, data):
        # LLMs occasionally emit a Position with ticker "cash" AND a cash_pct
        # field, double-counting. Fold pseudo-cash into cash_pct/cash_dollars.
        if not isinstance(data, dict):
            return data
        positions = data.get("positions") or []
        if not positions:
            return data

        def _field(p, name):
            return p.get(name) if isinstance(p, dict) else getattr(p, name, None)

        # LLM variants observed: bare "cash", "$", "usd", plus the mistake
        # where it names the Position after the sibling field "cash_pct" or
        # "cash_dollars" and stuffs the cash allocation there.
        cash_labels = {"cash", "$", "usd", "cash_pct", "cash_dollars"}
        real_positions = []
        pseudo_cash_pct = 0.0
        pseudo_cash_dollars = 0.0
        dropped_phantom = False
        for p in positions:
            ticker = (_field(p, "ticker") or "").strip().lower()
            weight = float(_field(p, "weight_pct") or 0.0)
            if ticker in cash_labels or ticker == "":
                pseudo_cash_pct += weight
                pseudo_cash_dollars += float(_field(p, "dollars") or 0.0)
                continue
            # Position.weight_pct is Field(gt=0). LLMs still emit 0pp
            # "phantom" positions (acknowledgement without conviction). Drop
            # them here so Instructor doesn't ValidationError on the whole
            # rec and lose the regen entirely - real allocations survive.
            if weight <= 0.0:
                dropped_phantom = True
                continue
            real_positions.append(p)
        if pseudo_cash_pct or pseudo_cash_dollars:
            existing_cash_pct = float(data.get("cash_pct") or 0.0)
            existing_cash_dollars = float(data.get("cash_dollars") or 0.0)
            # Both set = LLM double-counted; take max. Else sum.
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
        elif dropped_phantom:
            data = {**data, "positions": real_positions}
        return data

    @model_validator(mode="after")
    def _weights_sum_to_100(self):
        """Renormalize small LLM arithmetic errors; reject if wildly off.

        gpt-4o-mini reliably outputs sums like 105-115 with 8+ tickers;
        instructor retries eat cost without helping. Preserving ratios keeps
        relative conviction intact - only absolute scale shifts.
        """
        total = sum(p.weight_pct for p in self.positions) + self.cash_pct
        if 99.5 <= total <= 100.5:
            return self
        if 90.0 <= total <= 115.0:
            factor = 100.0 / total
            for p in self.positions:
                p.weight_pct = round(p.weight_pct * factor, 2)
                p.dollars = round(p.dollars * factor, 2)
            self.cash_pct = round(self.cash_pct * factor, 2)
            self.cash_dollars = round(self.cash_dollars * factor, 2)
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


def effective_max_weight(
    profile, atr_pct: float | None
) -> float:
    """Vol-scaled max weight for a single ticker.

    High-volatility names get a smaller cap so no single volatile bet
    dominates the portfolio. NVDA at 3.5% daily vol with 2% reference =
    35% * (2/3.5) = 20% max weight, vs base 35%.
    """
    base = profile.max_single_pct
    if not getattr(profile, "vol_scaling_enabled", False) or not atr_pct:
        return base
    ref = getattr(profile, "vol_reference_pct", 2.0)
    if atr_pct <= ref:
        return base  # low-vol tickers stay at base cap
    return base * (ref / atr_pct)


def check_profile_rules(
    allocation: Allocation,
    profile,
    snapshots: "dict | None" = None,
) -> list[str]:
    """Profile-aware version of check_risk_rules. Accepts a StrategyProfile.

    Uses the profile's max_single_pct + cash_floor_pct rather than the fixed
    RISK_RULES table. This is the M6+ path; check_risk_rules stays for backward
    compatibility.

    When `snapshots` is provided (dict[ticker, MarketSnapshot]), the max-weight
    check per position uses vol-scaled cap via effective_max_weight().
    """
    violations: list[str] = []
    for p in allocation.positions:
        atr = None
        if snapshots and p.ticker in snapshots:
            atr = getattr(snapshots[p.ticker], "atr_pct", None)
        cap = effective_max_weight(profile, atr)
        if p.weight_pct > cap + 0.5:
            violations.append(
                f"{p.ticker} weight {p.weight_pct:.1f}% exceeds "
                f"{profile.name} cap {cap:.1f}%"
                + (f" (vol-scaled from {profile.max_single_pct:.0f})"
                   if cap < profile.max_single_pct else "")
            )
    if allocation.cash_pct + 0.5 < profile.cash_floor_pct:
        violations.append(
            f"cash {allocation.cash_pct:.1f}% below "
            f"{profile.name} floor {profile.cash_floor_pct:.0f}%"
        )
    # Correlation constraint: two highly-correlated names shouldn't add up to
    # more than max_joint_correlated_weight_pct combined (one big bet
    # masquerading as diversification).
    if getattr(profile, "correlation_enabled", False):
        try:
            from agentic_investor.orchestrator.correlation import (
                find_correlated_over_cap,
            )
            weights = {
                p.ticker.upper(): p.weight_pct for p in allocation.positions
            }
            pairs = find_correlated_over_cap(
                weights,
                window_days=getattr(profile, "correlation_window_days", 60),
                max_pair_correlation=getattr(profile, "max_pair_correlation", 0.7),
                max_joint_pct=getattr(
                    profile, "max_joint_correlated_weight_pct", 50.0
                ),
            )
            for pair in pairs:
                violations.append(pair.as_violation())
        except Exception:  # noqa: BLE001 - correlation check must never crash
            pass
    return violations


def repair_allocation(allocation: Allocation, profile) -> tuple[Allocation, list[str]]:
    """Enforce position-count cap and cash floor after the LLM allocates.

    The prompt asks for both, but the LLM doesn't consistently respect either
    when a lot of tickers are in play. Drop smallest positions past the cap
    (weight -> cash), then trim proportionally if cash is still below floor.
    Returns (repaired, notes) so the loop can log what changed.
    """
    notes: list[str] = []
    positions = list(allocation.positions)
    cash_pct = allocation.cash_pct
    cash_dollars = allocation.cash_dollars
    max_positions = int(getattr(profile, "max_positions", 12))
    cash_floor = float(getattr(profile, "cash_floor_pct", 0.0))

    if len(positions) > max_positions:
        positions.sort(key=lambda p: p.weight_pct, reverse=True)
        dropped = positions[max_positions:]
        positions = positions[:max_positions]
        dropped_pct = sum(p.weight_pct for p in dropped)
        dropped_dollars = sum(p.dollars for p in dropped)
        cash_pct += dropped_pct
        cash_dollars += dropped_dollars
        notes.append(
            f"position-cap: dropped {len(dropped)} smallest "
            f"({', '.join(p.ticker for p in dropped)}); "
            f"{dropped_pct:.1f}pp -> cash"
        )

    if cash_pct + 0.05 < cash_floor and positions:
        gap_pp = cash_floor - cash_pct
        total_pos_pct = sum(p.weight_pct for p in positions)
        if total_pos_pct > 0:
            scale = max(0.0, 1.0 - gap_pp / total_pos_pct)
            trimmed_dollars = 0.0
            for p in positions:
                new_weight = round(p.weight_pct * scale, 2)
                new_dollars = round(p.dollars * scale, 2)
                trimmed_dollars += p.dollars - new_dollars
                p.weight_pct = new_weight
                p.dollars = new_dollars
            cash_pct = round(cash_floor, 2)
            cash_dollars = round(cash_dollars + trimmed_dollars, 2)
            notes.append(
                f"cash-floor: trimmed positions {(1 - scale) * 100:.1f}% "
                f"to lift cash from {allocation.cash_pct:.1f}% to "
                f"{cash_floor:.1f}%"
            )

    # Sweep any small residual into cash so weights still sum to 100.
    total = sum(p.weight_pct for p in positions) + cash_pct
    residual = round(100.0 - total, 2)
    if abs(residual) > 0.01:
        cash_pct = round(cash_pct + residual, 2)

    repaired = Allocation(
        positions=positions,
        cash_pct=cash_pct,
        cash_dollars=cash_dollars,
        portfolio_rationale=allocation.portfolio_rationale,
    )
    return repaired, notes


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
    # Previous rec's allocation, threaded in so the allocator can anchor on
    # existing weights and propose delta-form changes instead of re-conceiving
    # the portfolio from a blank slate. Prevents baseline churn / whipsaw.
    previous_allocation: Allocation
    # Pre-rendered macro/regime block for the allocator prompt.
    macro_prompt_block: str
    # Current regime label ("bull"/"bear"/"sideways"/"high_vol"/"unknown"),
    # used by repair/validate to know why the profile was tightened.
    macro_regime: str
