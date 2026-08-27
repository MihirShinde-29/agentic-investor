"""Non-LLM allocators, pluggable via StrategyProfile.allocator.

Each allocator takes the request + collected signals + market snapshots +
profile and returns an Allocation that satisfies the profile's guardrails
(caps, cash floor, weights sum to 100). The LLM allocator stays inline in
graph.py to keep the LangGraph node self-contained; this module handles the
non-LLM options (equal_weight, inverse_vol, ...).
"""

from collections.abc import Callable

from agentic_investor.agents.news import NewsSignal
from agentic_investor.agents.technical import TechnicalSignal
from agentic_investor.orchestrator.state import (
    Allocation,
    OrchestratorRequest,
    Position,
)
from agentic_investor.orchestrator.strategy import StrategyProfile
from agentic_investor.tools.market import MarketSnapshot

AllocatorFn = Callable[
    [
        OrchestratorRequest,
        list[TechnicalSignal],
        list[NewsSignal],
        dict[str, MarketSnapshot],
        StrategyProfile,
    ],
    Allocation,
]


def _cap_and_scale(
    raw_weights: dict[str, float],
    max_single_pct: float,
    cash_floor_pct: float,
) -> tuple[dict[str, float], float]:
    """Fit raw equity weights into the profile budget: cash floor + max single cap.

    - Total budget for equity = 1 - cash_floor_pct/100.
    - Any single position exceeding max_single_pct/100 gets clipped; excess
      flows to cash.
    - Returns (equity_weights, cash_weight); all in fractions [0, 1], summing
      to exactly 1.0.
    """
    total_raw = sum(raw_weights.values())
    if total_raw == 0:
        return {}, 1.0

    cash_floor = cash_floor_pct / 100.0
    equity_budget = 1.0 - cash_floor
    scaled = {t: (w / total_raw) * equity_budget for t, w in raw_weights.items()}

    max_single = max_single_pct / 100.0
    excess = 0.0
    capped: dict[str, float] = {}
    for t, w in scaled.items():
        if w > max_single:
            excess += w - max_single
            capped[t] = max_single
        else:
            capped[t] = w

    return capped, cash_floor + excess


def _build_allocation(
    request: OrchestratorRequest,
    equity_weights: dict[str, float],
    cash_weight: float,
    per_position_rationale: dict[str, str],
    portfolio_rationale: str,
) -> Allocation:
    positions = [
        Position(
            ticker=t,
            weight_pct=round(w * 100, 2),
            dollars=round(w * request.amount, 2),
            rationale=per_position_rationale.get(t, ""),
        )
        for t, w in equity_weights.items()
    ]
    return Allocation(
        positions=positions,
        cash_pct=round(cash_weight * 100, 2),
        cash_dollars=round(cash_weight * request.amount, 2),
        portfolio_rationale=portfolio_rationale,
    )


def allocate_equal_weight(
    request: OrchestratorRequest,
    tech_signals: list[TechnicalSignal],
    news_signals: list[NewsSignal],
    snapshots: dict[str, MarketSnapshot],
    profile: StrategyProfile,
) -> Allocation:
    """Simplest allocator: identical raw weight per ticker; caps + cash floor applied."""
    raw = dict.fromkeys(request.tickers, 1.0)
    equity, cash = _cap_and_scale(raw, profile.max_single_pct, profile.cash_floor_pct)
    return _build_allocation(
        request,
        equity,
        cash,
        per_position_rationale={t: "equal-weight baseline" for t in equity},
        portfolio_rationale=(
            f"Equal-weight allocation across {len(equity)} tickers. "
            f"Applied {profile.name} caps (max {profile.max_single_pct:.0f}% single, "
            f"{profile.cash_floor_pct:.0f}% cash floor). Hard baseline to beat "
            f"per DeMiguel et al. 2009."
        ),
    )


def allocate_inverse_vol(
    request: OrchestratorRequest,
    tech_signals: list[TechnicalSignal],
    news_signals: list[NewsSignal],
    snapshots: dict[str, MarketSnapshot],
    profile: StrategyProfile,
) -> Allocation:
    """Weight inversely proportional to volatility (ATR%); dollar-risk equalized."""
    raw: dict[str, float] = {}
    for t in request.tickers:
        snap = snapshots.get(t)
        # Guard: missing atr_pct -> treat as median (weight 1); negative/zero -> ignore.
        atr = snap.atr_pct if snap is not None else None
        if atr is None or atr <= 0:
            raw[t] = 1.0
        else:
            raw[t] = 1.0 / atr

    equity, cash = _cap_and_scale(raw, profile.max_single_pct, profile.cash_floor_pct)

    per_pos = {}
    for t in equity:
        snap = snapshots.get(t)
        if snap and snap.atr_pct:
            per_pos[t] = (
                f"Inverse-vol weight: ATR% {snap.atr_pct:.2f} "
                f"(less volatile -> larger share of equity budget)"
            )
        else:
            per_pos[t] = "Inverse-vol weight (ATR unavailable, used median)"

    return _build_allocation(
        request,
        equity,
        cash,
        per_position_rationale=per_pos,
        portfolio_rationale=(
            f"Inverse-volatility allocation across {len(equity)} tickers using "
            f"ATR% as the risk measure. Weight proportional to 1/vol so dollar "
            f"risk is roughly equalized. Applied {profile.name} caps "
            f"(max {profile.max_single_pct:.0f}% single, "
            f"{profile.cash_floor_pct:.0f}% cash floor)."
        ),
    )


_REGISTRY: dict[str, AllocatorFn] = {
    "equal_weight": allocate_equal_weight,
    "inverse_vol": allocate_inverse_vol,
}


def get_allocator(name: str) -> AllocatorFn:
    """Return the non-LLM allocator function for a name. Raises for unknown."""
    if name not in _REGISTRY:
        raise ValueError(
            f"non-LLM allocator {name!r} not registered; available: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[name]


def list_non_llm_allocators() -> list[str]:
    return sorted(_REGISTRY)
