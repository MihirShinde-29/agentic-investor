"""StrategyProfile: risk-driven strategy configuration with presets + overrides.

The risk parameter drives WHICH strategies run, not just guardrails. Users
either pick a preset (conservative / moderate / aggressive) with empirically-
validated defaults (from M4b multi-window backtests), or override individual
dimensions via CLI flags, or supply a fully custom profile via TOML file
(stdlib tomllib, no extra dep).
"""

import tomllib
from pathlib import Path
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
    # Past ~15 the LLM drifts toward over-diversified books that just track
    # SPY; excess names get dropped smallest-first, weight goes to cash.
    max_positions: int = Field(12, ge=1, le=50)
    drawdown_stop_pct: float | None = None  # e.g. -25.0 halts on -25% DD
    # Vol-scaled max weight: effective cap = max_single * min(1, ref/atr).
    # High-vol tickers get proportionally smaller caps (Kelly-inspired).
    vol_scaling_enabled: bool = True
    vol_reference_pct: float = 2.0

    # Correlation constraint: when any two held names have rolling correlation
    # above max_pair_correlation, their combined weight can't exceed
    # max_joint_correlated_weight_pct. Prevents "two names, one bet" (e.g.
    # NVDA + MSFT both bullish -> allocator stacks both into concentrated
    # tech-mega exposure). Set correlation_enabled=False to disable entirely.
    correlation_enabled: bool = True
    max_pair_correlation: float = Field(0.7, ge=0.0, le=1.0)
    max_joint_correlated_weight_pct: float = Field(50.0, ge=0.0, le=100.0)
    correlation_window_days: int = Field(60, ge=10, le=252)

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


def regime_adjusted_profile(
    profile: StrategyProfile, regime_label: str
) -> tuple[StrategyProfile, list[str]]:
    """Return a copy of profile with knobs adjusted for the current regime.

    high_vol tightens across the board (more cash, fewer names, smaller
    single caps, wider drift bands). bear leans defensive. bull loosens the
    cash floor slightly. sideways/unknown leave the profile alone.

    Returns (adjusted, notes) so the caller can log what changed.
    """
    notes: list[str] = []
    updates: dict = {}
    if regime_label == "high_vol":
        new_cash = min(profile.cash_floor_pct + 10.0, 40.0)
        new_max_pos = max(int(profile.max_positions) - 3, 5)
        new_max_single = max(profile.max_single_pct - 10.0, 15.0)
        new_band = profile.band_abs_pct + 2.0
        updates = {
            "cash_floor_pct": new_cash,
            "max_positions": new_max_pos,
            "max_single_pct": new_max_single,
            "band_abs_pct": new_band,
        }
        notes.append(
            f"high_vol: cash_floor {profile.cash_floor_pct:.0f}->{new_cash:.0f}%, "
            f"max_positions {profile.max_positions}->{new_max_pos}, "
            f"max_single {profile.max_single_pct:.0f}->{new_max_single:.0f}%, "
            f"band_abs {profile.band_abs_pct:.0f}->{new_band:.0f}pp"
        )
    elif regime_label == "bear":
        new_cash = min(profile.cash_floor_pct + 5.0, 40.0)
        new_max_single = max(profile.max_single_pct - 5.0, 15.0)
        updates = {
            "cash_floor_pct": new_cash,
            "max_single_pct": new_max_single,
        }
        notes.append(
            f"bear: cash_floor {profile.cash_floor_pct:.0f}->{new_cash:.0f}%, "
            f"max_single {profile.max_single_pct:.0f}->{new_max_single:.0f}%"
        )
    elif regime_label == "bull":
        new_cash = max(profile.cash_floor_pct - 3.0, 0.0)
        if new_cash != profile.cash_floor_pct:
            updates = {"cash_floor_pct": new_cash}
            notes.append(
                f"bull: cash_floor {profile.cash_floor_pct:.0f}->{new_cash:.0f}%"
            )
    if not updates:
        return profile, notes
    return profile.model_copy(update=updates), notes


def load_profile(name_or_path: str) -> StrategyProfile:
    """Resolve a profile from a preset name OR a TOML file path.

    Examples:
      load_profile("conservative")     -> the preset
      load_profile("my_custom.toml")   -> loaded + validated from disk
    """
    if name_or_path in PRESETS:
        return get_preset(name_or_path)  # type: ignore[arg-type]
    path = Path(name_or_path)
    if not path.exists():
        raise FileNotFoundError(
            f"profile {name_or_path!r} not found: not a preset "
            f"({sorted(PRESETS)}) and no file at {path}"
        )
    with path.open("rb") as f:
        data = tomllib.load(f)
    return StrategyProfile.model_validate(data)
