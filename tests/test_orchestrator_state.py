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


def test_allocation_accepts_weights_summing_to_100():
    a = Allocation(
        positions=[_pos("AAPL", 40), _pos("NVDA", 40)],
        cash_pct=20,
        cash_dollars=0,
        portfolio_rationale="x",
    )
    assert a.cash_pct == 20


def test_allocation_rejects_weights_not_summing_to_100():
    with pytest.raises(ValidationError, match="sum to 100"):
        Allocation(
            positions=[_pos("AAPL", 60), _pos("NVDA", 30)],
            cash_pct=0,
            cash_dollars=0,
            portfolio_rationale="x",
        )


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
