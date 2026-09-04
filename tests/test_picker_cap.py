"""Frozen-picker hard cap: promotions can't grow the list unboundedly.

Backstop for #104's promote-to-on-deck pattern. Observed 2026-09-04: the
list grew 12 → 39 tickers in 30 min, doubling per-regen prompt tokens and
collapsing cache hits. Cap drops oldest-promoted first, keeping held
tickers regardless of position (they're the real book).
"""

from __future__ import annotations


def test_cap_drops_oldest_when_over_limit():
    """Simulate the append + cap-enforce logic from loop.py."""
    frozen = ["MSFT", "CVX", "AAPL", "NVDA", "TSLA"]  # 5 tickers, cap=5
    held = {"MSFT", "CVX"}
    cap = 5

    # Promote 2 new → 7 total, cap says drop 2 oldest non-held.
    frozen.extend(["CQP", "DOCU"])
    dropped = []
    while len(frozen) > cap:
        drop_idx = None
        for i, t in enumerate(frozen):
            if t.upper() not in held:
                drop_idx = i
                break
        if drop_idx is None:
            break
        dropped.append(frozen.pop(drop_idx))

    assert len(frozen) == cap
    # Held tickers must survive
    assert "MSFT" in frozen
    assert "CVX" in frozen
    # Newest additions survive
    assert "CQP" in frozen
    assert "DOCU" in frozen
    # Oldest non-held dropped first
    assert dropped == ["AAPL", "NVDA"]


def test_cap_preserves_held_even_when_at_front():
    """If all non-held are already gone, held tickers stay even over cap."""
    frozen = ["MSFT", "CVX", "AAPL"]  # 3 held
    held = {"MSFT", "CVX", "AAPL"}
    cap = 2

    dropped = []
    while len(frozen) > cap:
        drop_idx = None
        for i, t in enumerate(frozen):
            if t.upper() not in held:
                drop_idx = i
                break
        if drop_idx is None:
            break
        dropped.append(frozen.pop(drop_idx))

    # No drops — all held, cap exceeded but not enforceable
    assert dropped == []
    assert frozen == ["MSFT", "CVX", "AAPL"]


def test_cap_no_op_when_within_limit():
    frozen = ["MSFT", "CVX"]
    held = {"MSFT"}
    cap = 25

    dropped = []
    while len(frozen) > cap:
        drop_idx = None
        for i, t in enumerate(frozen):
            if t.upper() not in held:
                drop_idx = i
                break
        if drop_idx is None:
            break
        dropped.append(frozen.pop(drop_idx))

    assert dropped == []
    assert frozen == ["MSFT", "CVX"]
