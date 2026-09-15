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
    # News IDs (e.g. "N3f9a2b1c") from the batch context that drove this
    # position. Empty list = no news-driven thesis (interval regen, drift,
    # technical-only). The paper_orders row copies this at submit time so
    # trades can be joined back to specific headlines in the compare view.
    triggering_news_ids: list[str] = Field(default_factory=list)


class AllocationReasoning(BaseModel):
    """CoT scratchpad emitted before weights. Logged by the loop for
    audit; nothing here drives sizing. Forcing an explicit bear_case
    before positions catches reflex trades the model would otherwise
    post without weighing disconfirming evidence.
    """

    bull_case: str = Field(
        description=(
            "1-2 sentences summarizing the strongest bullish evidence this "
            "tick (specific tickers + signals). If nothing bullish, say so."
        ),
    )
    bear_case: str = Field(
        description=(
            "1-2 sentences summarizing the strongest bearish evidence this "
            "tick (specific tickers + signals). If nothing bearish, say so."
        ),
    )
    disqualifiers: list[str] = Field(
        default_factory=list,
        description=(
            "Tickers you are DELIBERATELY not sizing up despite one bullish "
            "signal, each with a one-line reason (cooldown, correlation, "
            "cash floor, stale thesis)."
        ),
    )
    verdict: str = Field(
        description=(
            "1-2 sentences: what changed vs the previous allocation and why. "
            "If nothing meaningful changed, say so and expect the loop's "
            "drift filter to skip the regen."
        ),
    )
    no_material_change: bool = Field(
        default=False,
        description=(
            "Set True when the current tick's evidence does NOT justify any "
            "weight change from the previous allocation. When True, the loop "
            "compares your positions to the previous allocation; any actual "
            "delta > 5pp is logged as a self-inconsistency (you said no "
            "change but shipped one). Use this to short-circuit force-regens "
            "and stale-evidence re-thinks: if you're about to rationalize a "
            "rebalance with no fresh signal, set this True and keep the "
            "previous weights."
        ),
    )
    # Observability only. NOT gated on today - we log it and (after
    # enough ticks) correlate with realized 1d P/L to see if the
    # self-reported number is calibrated. Prior LLM-calibration work
    # (Kadavath 2022, Tian 2023) expects systematic overconfidence, so
    # treat this as a hypothesis to test, not a signal to trust yet.
    confidence: float = Field(
        default=0.5, ge=0.0, le=1.0,
        description=(
            "Your self-assessed probability that this allocation outperforms "
            "the previous one over the next 1-day horizon (0.0 = certain "
            "worse, 0.5 = coin flip, 1.0 = certain better). Consider your "
            "signal strength, how much of the thesis is already priced in, "
            "and whether disqualifiers apply. Nothing in the loop reads this "
            "yet - it is logged for post-hoc calibration analysis. Do NOT "
            "default to 0.5; give a real estimate."
        ),
    )


class Allocation(BaseModel):
    # Optional so pre-CoT DB rows still deserialize; new regens fill it.
    reasoning: AllocationReasoning | None = Field(
        default=None,
        description=(
            "Emit this BEFORE positions/cash_pct. Structured bull/bear/"
            "disqualifier/verdict trace. The loop logs it for post-hoc "
            "calibration; nothing in it drives sizing directly."
        ),
    )
    positions: list[Position]
    cash_pct: float = Field(ge=0.0, le=100.0)
    cash_dollars: float = Field(ge=0.0)
    portfolio_rationale: str
    # LLM-nominated tickers to drop from the on-deck watchlist. Populated
    # when the LLM judges a promoted-but-not-acted-on candidate no longer
    # worth watching (stale news, thesis moved, no follow-up flow). Loop
    # removes these from state.frozen_picker_tickers after honoring the
    # trade plan. Empty by default — no purge nominated.
    on_deck_purge: list[str] = Field(default_factory=list)

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


def repair_allocation(
    allocation: Allocation, profile
) -> tuple[Allocation, list[str], list[dict]]:
    """Enforce position-count cap and cash floor after the LLM allocates.

    The prompt asks for both, but the LLM doesn't consistently respect either
    when a lot of tickers are in play. Drop smallest positions past the cap
    (weight -> cash), then trim proportionally if cash is still below floor.
    Returns (repaired, notes, events). Notes are human-readable log lines
    for the loop. Events are structured dicts the loop forwards to the
    session recorder so we can grep phantom pressure across runs.
    """
    notes: list[str] = []
    events: list[dict] = []
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
        dropped_tickers = [p.ticker for p in dropped]
        notes.append(
            f"position-cap: dropped {len(dropped)} smallest "
            f"({', '.join(dropped_tickers)}); "
            f"{dropped_pct:.1f}pp -> cash"
        )
        events.append({
            "action": "position_cap_drop",
            "n_dropped": len(dropped),
            "tickers": dropped_tickers,
            "pp_to_cash": round(dropped_pct, 2),
            "cap": max_positions,
        })

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
            events.append({
                "action": "cash_floor_lift",
                "trim_pct": round((1 - scale) * 100, 2),
                "cash_before_pct": round(allocation.cash_pct, 2),
                "cash_after_pct": round(cash_floor, 2),
            })

    # Sweep any small residual into cash so weights still sum to 100.
    total = sum(p.weight_pct for p in positions) + cash_pct
    residual = round(100.0 - total, 2)
    if abs(residual) > 0.01:
        cash_pct = round(cash_pct + residual, 2)

    repaired = Allocation(
        reasoning=allocation.reasoning,
        positions=positions,
        cash_pct=cash_pct,
        cash_dollars=cash_dollars,
        portfolio_rationale=allocation.portfolio_rationale,
        on_deck_purge=allocation.on_deck_purge,
    )
    return repaired, notes, events


class Recommendation(BaseModel):
    request: OrchestratorRequest
    allocation: Allocation
    technical_signals: list[TechnicalSignal] = Field(default_factory=list)
    news_signals: list[NewsSignal] = Field(default_factory=list)
    violations: list[str] = Field(default_factory=list)
    # Structured events emitted by repair_allocation (position-cap drops,
    # cash-floor lifts). Loop forwards each to the session recorder so
    # phantom pressure can be counted across runs.
    repair_events: list[dict] = Field(default_factory=list)
    # Only populated when self-consistency or cross-model ensembling ran.
    ensemble_meta: dict | None = None
    # Frozen news batch as {news_id: {ticker, headline, source, published_at,
    # url}}. The LLM cited these IDs on Position.triggering_news_ids, and
    # downstream analyzers resolve them here after the streaming queue has
    # long-since recycled the raw NewsEvent objects.
    news_batch_snapshot: dict[str, dict] = Field(default_factory=dict)


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
    # Threaded up to Recommendation when ensemble sampling ran.
    ensemble_meta: dict
    # Loop-side hint that this regen was fired by force-regen / interval
    # (no news, no price move, no correlation shift). The prompt uses
    # this to push the LLM toward no_material_change=True on stale ticks.
    stale_evidence_hint: bool
