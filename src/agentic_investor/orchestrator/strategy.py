"""StrategyProfile: risk-driven strategy configuration with presets + overrides.

The risk parameter drives WHICH strategies run, not just guardrails. Users
either pick a preset (conservative / moderate / aggressive) with empirically-
validated defaults (from M4b multi-window backtests) or override individual
dimensions via CLI flags / API fields.
"""

from typing import Literal

from pydantic import BaseModel, Field

RiskLevel = Literal["conservative", "moderate", "aggressive"]
Allocator = Literal[
    "llm", "equal_weight", "inverse_vol", "risk_parity", "top_n_signal"
]
RebalanceMode = Literal["never", "monthly", "quarterly", "bands", "on_signal"]
SignalHandling = Literal["none", "trim_extremes", "full_dynamic"]


class StrategyProfile(BaseModel):
    """Complete strategy configuration; drives allocator + backtest defaults."""

    name: str = "custom"
    allocator: Allocator = "llm"
    rebalance: RebalanceMode = "never"

    # Bands mode parameters (asymmetric buy multiplier + DD circuit-breaker from M4b)
    band_abs_pct: float = 5.0
    band_rel_pct: float = 20.0
    band_buy_multiplier: float = 1.0
    dd_buy_pause_pct: float = 0.0

    # Cash + universe
    cash_yield_annual: float = 0.0
    universe_extras: list[str] = Field(default_factory=list)

    # Signal handling (full_dynamic is M8+ territory)
    signal_handling: SignalHandling = "none"

    # Guardrails enforced by the validator + orchestrator
    max_single_pct: float = Field(35.0, ge=0.0, le=100.0)
    cash_floor_pct: float = Field(10.0, ge=0.0, le=100.0)
    drawdown_stop_pct: float | None = None  # e.g. -25.0 halts on -25% DD

    # Backtest friction defaults
    tcost_bps: float = 0.0
    slippage_bps: float = 0.0


# Preset defaults empirically validated in M4b multi-window analysis:
# - conservative: inverse-vol allocator, quarterly rebalance (fewer trades),
#   asymmetric bands + DD pause + risk-free cash + bond/gold diversifiers.
# - moderate: LLM allocator, bands rebalance with 2x buy multiplier + 15% DD
#   pause (strict Pareto improvement over symmetric bands in bear windows).
# - aggressive: LLM allocator, monthly rebalance, higher caps, no cash floor.
CONSERVATIVE = StrategyProfile(
    name="conservative",
    allocator="inverse_vol",
    rebalance="quarterly",
    band_buy_multiplier=2.0,
    dd_buy_pause_pct=10.0,
    cash_yield_annual=0.045,
    universe_extras=["TLT", "GLD"],
    max_single_pct=20.0,
    cash_floor_pct=20.0,
    drawdown_stop_pct=-15.0,
    tcost_bps=10.0,
    slippage_bps=5.0,
)

MODERATE = StrategyProfile(
    name="moderate",
    allocator="llm",
    rebalance="bands",
    band_abs_pct=5.0,
    band_rel_pct=20.0,
    band_buy_multiplier=2.0,
    dd_buy_pause_pct=15.0,
    cash_yield_annual=0.045,
    max_single_pct=35.0,
    cash_floor_pct=10.0,
    drawdown_stop_pct=-25.0,
    tcost_bps=10.0,
    slippage_bps=5.0,
)

AGGRESSIVE = StrategyProfile(
    name="aggressive",
    allocator="llm",  # top_n_signal in a future upgrade (needs M5 picker integration)
    rebalance="monthly",
    band_buy_multiplier=1.5,
    dd_buy_pause_pct=0.0,
    cash_yield_annual=0.0,
    max_single_pct=50.0,
    cash_floor_pct=0.0,
    drawdown_stop_pct=-40.0,
    tcost_bps=10.0,
    slippage_bps=5.0,
)

PRESETS: dict[str, StrategyProfile] = {
    "conservative": CONSERVATIVE,
    "moderate": MODERATE,
    "aggressive": AGGRESSIVE,
}


def get_preset(risk: RiskLevel) -> StrategyProfile:
    """Return a fresh copy of the preset for the given risk level."""
    if risk not in PRESETS:
        raise ValueError(f"unknown risk level {risk!r}; available: {sorted(PRESETS)}")
    return PRESETS[risk].model_copy()


def apply_overrides(profile: StrategyProfile, **overrides) -> StrategyProfile:
    """Return a copy of profile with non-None override fields applied."""
    filtered = {k: v for k, v in overrides.items() if v is not None}
    return profile.model_copy(update=filtered)
