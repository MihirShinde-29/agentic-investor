"""Tests for repair_allocation's structured event output.

The events feed session.log so we can grep phantom pressure across runs
instead of parsing log text.
"""

from __future__ import annotations

from types import SimpleNamespace

from agentic_investor.orchestrator.state import (
    Allocation,
    Position,
    repair_allocation,
)


def _pos(ticker: str, weight: float) -> Position:
    return Position(
        ticker=ticker, weight_pct=weight, dollars=weight * 100, rationale="x",
    )


def _profile(max_positions: int = 3, cash_floor: float = 10.0) -> SimpleNamespace:
    return SimpleNamespace(
        max_positions=max_positions,
        cash_floor_pct=cash_floor,
        max_single_pct=35.0,
    )


def test_no_repair_returns_empty_events():
    a = Allocation(
        positions=[_pos("AAPL", 40), _pos("NVDA", 40)],
        cash_pct=20, cash_dollars=2000, portfolio_rationale="x",
    )
    _repaired, notes, events = repair_allocation(a, _profile())
    assert notes == []
    assert events == []


def test_position_cap_drop_emits_structured_event():
    a = Allocation(
        positions=[
            _pos("AAPL", 30), _pos("NVDA", 25), _pos("MSFT", 20),
            _pos("GOOGL", 10), _pos("TSLA", 5),
        ],
        cash_pct=10, cash_dollars=1000, portfolio_rationale="x",
    )
    _repaired, notes, events = repair_allocation(a, _profile(max_positions=3))
    assert len(events) == 1
    ev = events[0]
    assert ev["action"] == "position_cap_drop"
    assert ev["n_dropped"] == 2
    assert set(ev["tickers"]) == {"GOOGL", "TSLA"}
    assert ev["cap"] == 3
    assert ev["pp_to_cash"] == 15.0  # 10 + 5


def test_cash_floor_lift_emits_structured_event():
    # Positions total 95%, cash only 5% -> below 10% floor. Trim to lift.
    a = Allocation(
        positions=[_pos("AAPL", 50), _pos("NVDA", 45)],
        cash_pct=5, cash_dollars=500, portfolio_rationale="x",
    )
    _repaired, notes, events = repair_allocation(a, _profile(cash_floor=10.0))
    lift_events = [e for e in events if e["action"] == "cash_floor_lift"]
    assert len(lift_events) == 1
    ev = lift_events[0]
    assert ev["cash_before_pct"] == 5.0
    assert ev["cash_after_pct"] == 10.0
    assert ev["trim_pct"] > 0


def test_both_repairs_emit_two_events_in_order():
    a = Allocation(
        positions=[
            _pos("AAPL", 30), _pos("NVDA", 25), _pos("MSFT", 20),
            _pos("GOOGL", 10), _pos("TSLA", 5),
        ],
        cash_pct=10, cash_dollars=1000, portfolio_rationale="x",
    )
    _repaired, notes, events = repair_allocation(
        a, _profile(max_positions=3, cash_floor=30.0),
    )
    # First: cap drop. Second: cash floor lift on the still-underfilled cash.
    assert len(events) == 2
    assert events[0]["action"] == "position_cap_drop"
    assert events[1]["action"] == "cash_floor_lift"
