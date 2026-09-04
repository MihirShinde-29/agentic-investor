"""Allocation.on_deck_purge lets the LLM nominate stale on-deck tickers
for removal from the frozen picker list. Loop honors the purge but never
drops held or actively-targeted tickers as a safety measure.
"""

from __future__ import annotations

from agentic_investor.orchestrator.state import Allocation, Position


def _pos(ticker: str, w: float = 20.0) -> Position:
    return Position(
        ticker=ticker, weight_pct=w, dollars=w * 100, rationale="x",
    )


def test_allocation_defaults_on_deck_purge_to_empty_list():
    a = Allocation(
        positions=[_pos("AAPL")],
        cash_pct=80, cash_dollars=8000, portfolio_rationale="x",
    )
    assert a.on_deck_purge == []


def test_allocation_accepts_llm_purge_nomination():
    a = Allocation(
        positions=[_pos("AAPL")],
        cash_pct=80, cash_dollars=8000, portfolio_rationale="x",
        on_deck_purge=["CQP", "DOCU"],
    )
    assert a.on_deck_purge == ["CQP", "DOCU"]


def test_purge_filter_never_drops_held_or_active():
    """Simulate the loop-side filter logic that keeps held and active
    tickers safe even if the LLM nominates them.
    """
    frozen = ["MSFT", "CVX", "CQP", "DOCU", "PATH"]
    held = {"MSFT", "CVX"}
    active_targets = {"CVX", "CQP"}  # LLM keeps CQP in positions
    requested_purge = {"MSFT", "CQP", "DOCU", "PATH"}  # LLM asks for these

    # Filter: subtract held + active from requested
    effective = requested_purge - held - active_targets
    result = [t for t in frozen if t.upper() not in effective]

    # MSFT (held) survives, CQP (active) survives, DOCU + PATH dropped
    assert result == ["MSFT", "CVX", "CQP"]
    assert effective == {"DOCU", "PATH"}


def test_purge_of_only_safe_tickers_is_noop():
    frozen = ["MSFT", "CVX", "CQP"]
    held = {"MSFT"}
    active_targets = {"CVX", "CQP"}
    requested_purge = {"MSFT", "CVX"}  # both protected

    effective = requested_purge - held - active_targets
    result = [t for t in frozen if t.upper() not in effective]

    assert effective == set()
    assert result == frozen
