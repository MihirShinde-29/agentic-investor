"""Tests for the correlation-aware allocation constraint."""

from __future__ import annotations

import pandas as pd
import pytest

from agentic_investor.orchestrator import correlation as corr_mod
from agentic_investor.orchestrator.correlation import (
    CorrelatedPair,
    find_correlated_over_cap,
)
from agentic_investor.orchestrator.state import (
    Allocation,
    Position,
    check_profile_rules,
)
from agentic_investor.orchestrator.strategy import get_preset


def _stub_correlation_matrix(pairs: dict[tuple[str, str], float]) -> pd.DataFrame:
    """Build a symmetric correlation matrix from pair -> corr entries."""
    tickers = sorted({t for pair in pairs for t in pair})
    m = pd.DataFrame(1.0, index=tickers, columns=tickers)
    for (a, b), corr in pairs.items():
        m.loc[a, b] = corr
        m.loc[b, a] = corr
    return m


def test_find_correlated_over_cap_flags_pair(monkeypatch):
    """NVDA + MSFT at 0.75 corr with joint 55% weight breaches the 50% cap."""
    monkeypatch.setattr(
        corr_mod,
        "compute_correlation_matrix",
        lambda tickers, window_days=60: _stub_correlation_matrix(
            {("MSFT", "NVDA"): 0.75}
        ),
    )
    pairs = find_correlated_over_cap(
        {"NVDA": 30.0, "MSFT": 25.0},
        max_pair_correlation=0.7,
        max_joint_pct=50.0,
    )
    assert len(pairs) == 1
    p = pairs[0]
    assert {p.a, p.b} == {"NVDA", "MSFT"}
    assert p.joint_weight_pct == pytest.approx(55.0)
    assert p.cap_pct == 50.0
    assert p.correlation == pytest.approx(0.75)


def test_find_correlated_over_cap_ignores_low_correlation(monkeypatch):
    """AAPL + GLD at 0.3 corr never trips even at 80% joint weight."""
    monkeypatch.setattr(
        corr_mod,
        "compute_correlation_matrix",
        lambda tickers, window_days=60: _stub_correlation_matrix(
            {("AAPL", "GLD"): 0.30}
        ),
    )
    pairs = find_correlated_over_cap(
        {"AAPL": 40.0, "GLD": 40.0},
        max_pair_correlation=0.7,
        max_joint_pct=50.0,
    )
    assert pairs == []


def test_find_correlated_over_cap_high_corr_but_low_joint(monkeypatch):
    """MSFT + GOOGL at 0.8 with only 30% joint weight is allowed."""
    monkeypatch.setattr(
        corr_mod,
        "compute_correlation_matrix",
        lambda tickers, window_days=60: _stub_correlation_matrix(
            {("GOOGL", "MSFT"): 0.80}
        ),
    )
    pairs = find_correlated_over_cap(
        {"MSFT": 15.0, "GOOGL": 15.0},
        max_pair_correlation=0.7,
        max_joint_pct=50.0,
    )
    assert pairs == []


def test_find_correlated_over_cap_returns_empty_when_matrix_missing(monkeypatch):
    """Data-fetch failure must not crash - just skip the constraint."""
    monkeypatch.setattr(
        corr_mod,
        "compute_correlation_matrix",
        lambda tickers, window_days=60: None,
    )
    assert (
        find_correlated_over_cap(
            {"NVDA": 30.0, "MSFT": 30.0},
            max_pair_correlation=0.7,
            max_joint_pct=50.0,
        )
        == []
    )


def test_find_correlated_over_cap_skips_zero_weights(monkeypatch):
    """Positions with 0 weight are excluded before matrix computation."""
    called: list[list[str]] = []

    def _spy(tickers, window_days=60):
        called.append(list(tickers))
        return _stub_correlation_matrix({("MSFT", "NVDA"): 0.9})

    monkeypatch.setattr(corr_mod, "compute_correlation_matrix", _spy)
    # Three positions but only NVDA + AAPL have real weight; MSFT excluded.
    pairs = find_correlated_over_cap(
        {"NVDA": 30.0, "MSFT": 0.0, "AAPL": 25.0},
        max_pair_correlation=0.7,
        max_joint_pct=50.0,
    )
    assert pairs == []
    assert called and "MSFT" not in called[0]
    assert set(called[0]) == {"NVDA", "AAPL"}


def test_correlated_pair_violation_message():
    p = CorrelatedPair(
        a="NVDA", b="MSFT", correlation=0.78,
        joint_weight_pct=55.0, cap_pct=50.0,
    )
    msg = p.as_violation()
    assert "NVDA+MSFT" in msg
    assert "55.0%" in msg
    assert "50.0%" in msg
    assert "0.78" in msg


def test_check_profile_rules_surfaces_correlation_violation(monkeypatch):
    """End-to-end: correlated pair over-weighted -> violation string emitted."""
    monkeypatch.setattr(
        corr_mod,
        "compute_correlation_matrix",
        lambda tickers, window_days=60: _stub_correlation_matrix(
            {("MSFT", "NVDA"): 0.75}
        ),
    )
    profile = get_preset("moderate")
    # Bring the max-single cap up so we only trip the correlation rule.
    profile = profile.model_copy(update={
        "max_single_pct": 50.0,
        "vol_scaling_enabled": False,
        "cash_floor_pct": 0.0,
    })
    alloc = Allocation(
        positions=[
            Position(ticker="NVDA", weight_pct=30, dollars=3000,
                     rationale="x", confidence=0.7),
            Position(ticker="MSFT", weight_pct=30, dollars=3000,
                     rationale="x", confidence=0.7),
        ],
        cash_pct=40, cash_dollars=4000, portfolio_rationale="x",
    )
    violations = check_profile_rules(alloc, profile)
    joined = "; ".join(violations)
    assert "NVDA+MSFT" in joined or "MSFT+NVDA" in joined


def test_check_profile_rules_skips_when_correlation_disabled(monkeypatch):
    def _boom(*a, **k):
        raise RuntimeError("should not be called")

    monkeypatch.setattr(corr_mod, "compute_correlation_matrix", _boom)
    profile = get_preset("moderate").model_copy(update={
        "correlation_enabled": False,
        "max_single_pct": 50.0,
        "vol_scaling_enabled": False,
        "cash_floor_pct": 0.0,
    })
    alloc = Allocation(
        positions=[
            Position(ticker="NVDA", weight_pct=30, dollars=3000,
                     rationale="x", confidence=0.7),
            Position(ticker="MSFT", weight_pct=30, dollars=3000,
                     rationale="x", confidence=0.7),
        ],
        cash_pct=40, cash_dollars=4000, portfolio_rationale="x",
    )
    # No exception, no correlation violation surfaced.
    violations = check_profile_rules(alloc, profile)
    assert not any("NVDA+MSFT" in v or "MSFT+NVDA" in v for v in violations)
