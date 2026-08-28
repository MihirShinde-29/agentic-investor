"""Rebalance-as-diff: turn a target Allocation into a minimal set of orders.

Given a saved Recommendation (target weights) and the broker's current positions
(actual dollar values), compute the delta trades that move actual toward target.
Skip trades whose dollar delta is below `min_trade_dollars` so we don't churn
$3 rebalances that pay more in friction than they capture in drift.

No LLM calls here - all deterministic arithmetic. Orders get a stable
client_order_id derived from (rec_id, ticker, submitted-day) so a retry of the
same rebalance is a no-op at the broker.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import UTC, datetime

from agentic_investor.orchestrator.state import Recommendation
from agentic_investor.tools.paper_broker import PaperBroker, PaperOrder


@dataclass
class TradePlan:
    ticker: str
    side: str  # "buy" | "sell"
    dollars: float
    qty: float
    target_pct: float
    current_pct: float
    reason: str


def compute_trade_plan(
    rec: Recommendation,
    current_positions: dict[str, float],  # ticker -> current $ market value
    total_equity: float,
    *,
    prices: dict[str, float],  # ticker -> latest price for qty sizing
    min_trade_dollars: float = 25.0,
) -> list[TradePlan]:
    """Diff target-weight allocation against current positions."""
    target_dollars = {
        p.ticker: total_equity * (p.weight_pct / 100.0) for p in rec.allocation.positions
    }
    tickers = set(target_dollars) | set(current_positions)
    plans: list[TradePlan] = []
    for t in sorted(tickers):
        cur_d = float(current_positions.get(t, 0.0))
        tgt_d = float(target_dollars.get(t, 0.0))
        delta = tgt_d - cur_d
        if abs(delta) < min_trade_dollars:
            continue
        price = prices.get(t)
        if not price or price <= 0:
            # Can't size a trade without a price; skip and let the next tick
            # try again with fresh data.
            continue
        qty = round(abs(delta) / price, 4)
        if qty <= 0:
            continue
        side = "buy" if delta > 0 else "sell"
        plans.append(
            TradePlan(
                ticker=t,
                side=side,
                dollars=round(abs(delta), 2),
                qty=qty,
                target_pct=round(tgt_d / total_equity * 100, 2) if total_equity else 0.0,
                current_pct=round(cur_d / total_equity * 100, 2) if total_equity else 0.0,
                reason=f"drift {(tgt_d - cur_d) / max(total_equity, 1) * 100:+.2f}pp",
            )
        )
    return plans


def _client_order_id(rec_id: int, ticker: str, side: str, day: str) -> str:
    # Stable per (rec_id, ticker, side, day) so retries within a day are no-ops.
    key = f"rec{rec_id}:{ticker}:{side}:{day}".encode()
    return "ai-" + hashlib.sha1(key).hexdigest()[:20]


def execute_trade_plan(
    plans: list[TradePlan],
    broker: PaperBroker,
    *,
    rec_id: int,
    stop_loss_pct: float | None = None,
    take_profit_pct: float | None = None,
    day: str | None = None,
) -> list[PaperOrder]:
    """Submit each plan through the broker. Idempotent by client_order_id."""
    day = day or datetime.now(UTC).strftime("%Y-%m-%d")
    submitted: list[PaperOrder] = []
    for p in plans:
        coid = _client_order_id(rec_id, p.ticker, p.side, day)
        order = broker.submit_market_order(
            p.ticker,
            p.side,
            p.qty,
            client_order_id=coid,
            stop_loss_pct=stop_loss_pct if p.side == "buy" else None,
            take_profit_pct=take_profit_pct if p.side == "buy" else None,
        )
        submitted.append(order)
    return submitted
