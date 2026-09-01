"""Macro Agent: wraps the market-regime detector as a signal-shaped agent.

Parallel in shape to TechnicalSignal / NewsSignal: exposes a MacroSignal
that downstream code (allocator prompt, profile-adjustment layer, dashboard)
can consume without knowing about the underlying regime.py plumbing.

No LLM call today — detection is deterministic. Keeping it as an agent
module lets us bolt on an LLM reasoning step later without touching callers.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from agentic_investor.orchestrator.regime import MarketRegime, detect_regime

Stance = Literal["risk_on", "risk_off", "neutral", "unknown"]


class MacroSignal(BaseModel):
    regime: str  # bull | bear | sideways | high_vol | unknown
    stance: Stance
    confidence: float = Field(ge=0.0, le=1.0)
    vix: float | None = None
    yield_curve_bps: float | None = None
    credit_stress_pct: float | None = None
    dollar_20d_pct: float | None = None
    spy_20d_return_pct: float | None = None
    justification: str
    prompt_block: str  # ready to paste into the allocator user message


def _stance_for(regime: MarketRegime) -> tuple[Stance, float]:
    """Derive a directional stance + confidence from the regime label.

    Confidence lifts when supporting signals concur (e.g. bear label with an
    inverted curve + credit stress = higher confidence risk-off).
    """
    if regime.label == "bull":
        conf = 0.6
        if regime.credit_stress_pct is not None and regime.credit_stress_pct > 1.0:
            conf += 0.1
        return "risk_on", min(conf, 0.9)
    if regime.label == "bear":
        conf = 0.6
        if regime.yield_curve_bps is not None and regime.yield_curve_bps < 0:
            conf += 0.1
        if regime.credit_stress_pct is not None and regime.credit_stress_pct < 0:
            conf += 0.1
        return "risk_off", min(conf, 0.9)
    if regime.label == "high_vol":
        return "risk_off", 0.8
    if regime.label == "sideways":
        return "neutral", 0.5
    return "unknown", 0.0


def analyze_macro() -> MacroSignal:
    """Compute the current macro signal. Safe to call each regen."""
    regime = detect_regime()
    stance, conf = _stance_for(regime)
    return MacroSignal(
        regime=regime.label,
        stance=stance,
        confidence=conf,
        vix=regime.vix,
        yield_curve_bps=regime.yield_curve_bps,
        credit_stress_pct=regime.credit_stress_pct,
        dollar_20d_pct=regime.dollar_20d_pct,
        spy_20d_return_pct=regime.spy_20d_return_pct,
        justification=regime.justification,
        prompt_block=regime.prompt_block(),
    )
