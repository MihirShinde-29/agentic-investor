"""Tests for orchestrator state models and risk guardrails."""

import pytest
from pydantic import ValidationError

from agentic_investor.orchestrator.state import (
    Allocation,
    OrchestratorRequest,
    Position,
    check_risk_rules,
)


def _pos(ticker: str, weight: float, dollars: float = 0.0) -> Position:
    return Position(ticker=ticker, weight_pct=weight, dollars=dollars, rationale="x")


def test_position_defaults_confidence_to_neutral():
    p = Position(ticker="AAPL", weight_pct=40, dollars=4000, rationale="x")
    assert p.confidence == 0.5


def test_position_accepts_explicit_confidence():
    p = Position(ticker="NVDA", weight_pct=25, dollars=2500, rationale="x", confidence=0.85)
    assert p.confidence == 0.85


def test_position_rejects_confidence_outside_range():
    with pytest.raises(ValidationError):
        Position(ticker="AAPL", weight_pct=40, dollars=4000, rationale="x", confidence=1.5)


def test_allocation_accepts_weights_summing_to_100():
    a = Allocation(
        positions=[_pos("AAPL", 40), _pos("NVDA", 40)],
        cash_pct=20,
        cash_dollars=0,
        portfolio_rationale="x",
    )
    assert a.cash_pct == 20


def test_allocation_rejects_weights_wildly_outside_repair_band():
    # 200% is well beyond the [90, 115] repair band; must reject hard.
    with pytest.raises(ValidationError, match="outside repair band"):
        Allocation(
            positions=[_pos("AAPL", 100), _pos("NVDA", 100)],
            cash_pct=0,
            cash_dollars=0,
            portfolio_rationale="x",
        )


def test_allocation_folds_cash_pseudo_position_deduping_with_cash_pct():
    # gpt-4o-mini failure: emitted a "cash" Position AND a cash_pct field,
    # double-counting cash so raw sum was 125. Dedup keeps max(cash_pct).
    a = Allocation.model_validate({
        "positions": [
            {"ticker": "MSFT", "weight_pct": 30, "dollars": 3000, "rationale": "x"},
            {"ticker": "AAPL", "weight_pct": 25, "dollars": 2500, "rationale": "x"},
            {"ticker": "MRK",  "weight_pct": 15, "dollars": 1500, "rationale": "x"},
            {"ticker": "WMT",  "weight_pct": 5,  "dollars": 500,  "rationale": "x"},
            {"ticker": "cash", "weight_pct": 25, "dollars": 2500, "rationale": "x"},
        ],
        "cash_pct": 25,
        "cash_dollars": 2500,
        "portfolio_rationale": "x",
    })
    assert [p.ticker for p in a.positions] == ["MSFT", "AAPL", "MRK", "WMT"]
    total = sum(p.weight_pct for p in a.positions) + a.cash_pct
    assert 99.5 <= total <= 100.5


def test_allocation_folds_cash_pseudo_position_summing_when_no_cash_pct():
    # Different failure: only the "cash" position set, cash_pct=0. Sum, not max.
    a = Allocation.model_validate({
        "positions": [
            {"ticker": "MSFT", "weight_pct": 70, "dollars": 7000, "rationale": "x"},
            {"ticker": "cash", "weight_pct": 30, "dollars": 3000, "rationale": "x"},
        ],
        "cash_pct": 0,
        "cash_dollars": 0,
        "portfolio_rationale": "x",
    })
    assert [p.ticker for p in a.positions] == ["MSFT"]
    assert a.cash_pct == 30


def test_allocation_renormalizes_small_llm_arithmetic_error():
    # Real gpt-4o-mini failure mode: positions sum to 100, cash adds 10 = 110.
    a = Allocation(
        positions=[_pos("AAPL", 60, dollars=6000), _pos("NVDA", 40, dollars=4000)],
        cash_pct=10,
        cash_dollars=1000,
        portfolio_rationale="x",
    )
    total = sum(p.weight_pct for p in a.positions) + a.cash_pct
    assert 99.5 <= total <= 100.5
    # Ratios preserved: AAPL was 60/110 of the raw sum, NVDA 40/110.
    aapl = next(p for p in a.positions if p.ticker == "AAPL")
    nvda = next(p for p in a.positions if p.ticker == "NVDA")
    assert aapl.weight_pct == pytest.approx(60 * 100 / 110, abs=0.1)
    assert nvda.weight_pct == pytest.approx(40 * 100 / 110, abs=0.1)


def test_allocation_tolerates_small_float_rounding():
    Allocation(
        positions=[_pos("AAPL", 33.33), _pos("NVDA", 33.33), _pos("MSFT", 33.34)],
        cash_pct=0,
        cash_dollars=0,
        portfolio_rationale="x",
    )


def test_risk_rules_flag_oversized_position_for_conservative():
    a = Allocation(
        positions=[_pos("AAPL", 40), _pos("NVDA", 40)],
        cash_pct=20,
        cash_dollars=0,
        portfolio_rationale="x",
    )
    violations = check_risk_rules(a, "conservative")
    assert any("AAPL" in v and "20%" in v for v in violations)


def test_risk_rules_flag_insufficient_cash_for_conservative():
    a = Allocation(
        positions=[_pos("AAPL", 50), _pos("NVDA", 50)],
        cash_pct=0,
        cash_dollars=0,
        portfolio_rationale="x",
    )
    violations = check_risk_rules(a, "conservative")
    assert any("cash" in v for v in violations)


def test_risk_rules_allow_aggressive_concentration():
    a = Allocation(
        positions=[_pos("NVDA", 50), _pos("AAPL", 30), _pos("MSFT", 20)],
        cash_pct=0,
        cash_dollars=0,
        portfolio_rationale="x",
    )
    assert check_risk_rules(a, "aggressive") == []


def test_request_requires_at_least_one_ticker():
    with pytest.raises(ValidationError):
        OrchestratorRequest(tickers=[], amount=10_000)
