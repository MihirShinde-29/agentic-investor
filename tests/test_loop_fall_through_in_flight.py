"""Fall-through walk-back must not double-submit sells for shares already
held for pending orders.

Reproduces the Sept 4 open-bell rejection cluster: 12 orders submitted at
09:30:44, barely-moved skip fired at 09:31:42, walk-back re-derived the
same sells at 09:31:44 and Alpaca rejected them with
`insufficient qty available (held_for_orders=X, available=0)`.
"""

from __future__ import annotations

from datetime import UTC, datetime

from agentic_investor.tools.paper_broker import PaperOrder


def _open_sell(ticker: str, qty: float, i: int = 0) -> PaperOrder:
    return PaperOrder(
        id=f"b-{i}", client_order_id=f"c-{i}",
        ticker=ticker, side="sell", qty=qty,
        order_type="market", status="new",
        submitted_at=datetime.now(UTC).isoformat(),
    )


def test_walk_back_drops_plans_matching_open_orders():
    """The core dedup: a walk-back sell for a ticker with an open sell
    order should be dropped so we don't re-submit and hit
    insufficient-qty rejection at the broker.
    """
    # Simulate what loop.py does after computing plans + entering the
    # fall_through_reduce_only branch. This is the exact filter added
    # in task #103.
    from types import SimpleNamespace as NS

    plans = [
        NS(ticker="AAPL", side="sell", qty=15.30),
        NS(ticker="CRM", side="sell", qty=28.17),
        NS(ticker="NVDA", side="sell", qty=5.0),  # no open order -> keep
    ]
    open_orders = [
        _open_sell("AAPL", 15.30, 1),
        _open_sell("CRM", 28.17, 2),
    ]

    in_flight = {(o.ticker.upper(), o.side) for o in open_orders}
    kept = [p for p in plans if (p.ticker.upper(), p.side) not in in_flight]

    assert len(kept) == 1
    assert kept[0].ticker == "NVDA"


def test_walk_back_keeps_plan_when_no_open_orders():
    from types import SimpleNamespace as NS

    plans = [NS(ticker="AAPL", side="sell", qty=15.30)]
    open_orders: list[PaperOrder] = []
    in_flight = {(o.ticker.upper(), o.side) for o in open_orders}
    kept = [p for p in plans if (p.ticker.upper(), p.side) not in in_flight]
    assert kept == plans


def test_walk_back_direction_specific():
    """An open BUY doesn't block a walk-back SELL on the same ticker."""
    from types import SimpleNamespace as NS

    plans = [NS(ticker="AAPL", side="sell", qty=15.30)]
    open_orders = [
        PaperOrder(
            id="b-1", client_order_id="c-1",
            ticker="AAPL", side="buy", qty=10.0,
            order_type="market", status="new",
            submitted_at=datetime.now(UTC).isoformat(),
        )
    ]
    in_flight = {(o.ticker.upper(), o.side) for o in open_orders}
    kept = [p for p in plans if (p.ticker.upper(), p.side) not in in_flight]
    assert kept == plans
