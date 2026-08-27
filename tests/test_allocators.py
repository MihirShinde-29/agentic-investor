"""Tests for the non-LLM allocators (equal_weight, inverse_vol)."""

import pytest

from agentic_investor.orchestrator.allocators import (
    _cap_and_scale,
    allocate_equal_weight,
    allocate_inverse_vol,
    get_allocator,
    list_non_llm_allocators,
)
from agentic_investor.orchestrator.state import OrchestratorRequest
from agentic_investor.orchestrator.strategy import get_preset
from agentic_investor.tools.market import MarketSnapshot


def _snap(ticker: str, atr_pct: float = 2.0) -> MarketSnapshot:
    return MarketSnapshot(ticker=ticker, as_of="2026-08-27", close=100.0, atr_pct=atr_pct)


# _cap_and_scale


def test_cap_and_scale_respects_cash_floor():
    equity, cash = _cap_and_scale({"A": 1.0, "B": 1.0}, max_single_pct=100, cash_floor_pct=30)
    assert cash == pytest.approx(0.30)
    assert equity["A"] + equity["B"] == pytest.approx(0.70)
    # Equal raw weights -> equal scaled weights
    assert equity["A"] == pytest.approx(equity["B"])


def test_cap_and_scale_caps_single_position_and_redirects_to_cash():
    # A hugely overweighted raw weight -> capped at 20% single, excess to cash.
    equity, cash = _cap_and_scale(
        {"A": 100.0, "B": 1.0}, max_single_pct=20, cash_floor_pct=10,
    )
    assert equity["A"] == pytest.approx(0.20)
    # Cash floor 10% + excess from capping A
    assert cash > 0.10


def test_cap_and_scale_empty_input_gives_all_cash():
    equity, cash = _cap_and_scale({}, max_single_pct=50, cash_floor_pct=0)
    assert equity == {}
    assert cash == pytest.approx(1.0)


# Equal-weight allocator


def test_equal_weight_gives_identical_position_sizes():
    req = OrchestratorRequest(
        tickers=["AAPL", "NVDA", "MSFT"], amount=10_000, risk="moderate"
    )
    profile = get_preset("moderate")
    alloc = allocate_equal_weight(req, [], [], {}, profile)
    weights = [p.weight_pct for p in alloc.positions]
    assert len(alloc.positions) == 3
    # All equal (within Pydantic rounding)
    assert all(abs(w - weights[0]) < 0.1 for w in weights)


def test_equal_weight_respects_conservative_caps():
    req = OrchestratorRequest(tickers=["A", "B"], amount=10_000, risk="conservative")
    profile = get_preset("conservative")
    alloc = allocate_equal_weight(req, [], [], {}, profile)
    # Conservative: max_single 20%, cash_floor 20%.
    # 2 equal picks in an 80% budget -> 40% each, but capped to 20% each.
    # Excess flows to cash.
    for p in alloc.positions:
        assert p.weight_pct <= 20.5
    assert alloc.cash_pct >= 20.0


# Inverse-vol allocator


def test_inverse_vol_gives_larger_weight_to_less_volatile():
    # Use aggressive profile (max_single 50, cash_floor 0) so caps don't flatten
    # the raw 4:1 inverse-vol ratio.
    req = OrchestratorRequest(tickers=["LOW", "HI"], amount=10_000, risk="aggressive")
    profile = get_preset("aggressive")
    snapshots = {
        "LOW": _snap("LOW", atr_pct=1.0),   # low vol
        "HI": _snap("HI", atr_pct=4.0),     # high vol (4x more volatile)
    }
    alloc = allocate_inverse_vol(req, [], [], snapshots, profile)
    weights = {p.ticker: p.weight_pct for p in alloc.positions}
    assert weights["LOW"] > weights["HI"]
    # 1/1 vs 1/4 -> raw ratio 4:1; expect roughly that after cap-and-scale.
    assert weights["LOW"] > 2 * weights["HI"]


def test_inverse_vol_handles_missing_atr():
    req = OrchestratorRequest(tickers=["A", "B"], amount=10_000, risk="moderate")
    profile = get_preset("moderate")
    # A has atr, B doesn't -> B falls back to median weight
    alloc = allocate_inverse_vol(req, [], [], {"A": _snap("A", atr_pct=2.0)}, profile)
    assert {p.ticker for p in alloc.positions} == {"A", "B"}
    total = sum(p.weight_pct for p in alloc.positions) + alloc.cash_pct
    assert 99.5 <= total <= 100.5


# Registry


def test_registry_lists_expected_allocators():
    assert "equal_weight" in list_non_llm_allocators()
    assert "inverse_vol" in list_non_llm_allocators()


def test_get_allocator_rejects_unknown():
    with pytest.raises(ValueError, match="not registered"):
        get_allocator("black_magic")
