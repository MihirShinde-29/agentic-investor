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
    recent_trades: dict[str, tuple[str, datetime]] | None = None,
    cooldown_seconds: int = 900,
    now: datetime | None = None,
    news_batch_tickers: set[str] | None = None,
    news_bypass_cooldown: bool = True,
    ticker_recent_moves: dict[str, float] | None = None,
    adverse_move_threshold_pct: float = 1.0,
    avg_entry_prices: dict[str, float] | None = None,
    halt_buys_drawdown_pct: float = 5.0,
    small_drawdown_hold_pct: float = 3.0,
    force_loss_cut_pct: float = 8.0,
) -> list[TradePlan]:
    """Diff target-weight allocation against current positions.

    Discipline layer applies BUY-side and TRIM-side vetoes before returning
    plans:
      - Temporal cooldown: no reverse-side trade within cooldown_seconds.
        A ticker in the current news batch bypasses the cooldown by default;
        set news_bypass_cooldown=False for a strict block. The LLM is
        expected to self-restrain via the recent-trades context in its
        prompt.
      - Adverse-move veto (BUY): skip if the ticker moved beyond
        -adverse_move_threshold_pct recently (don't catch falling knives).
      - Halt-buys drawdown (BUY): skip if position already down more than
        halt_buys_drawdown_pct from entry (don't average down).
      - Small-drawdown hold (SELL): skip trim if position is between
        -small_drawdown_hold_pct and 0 AND bouncing (don't sell into
        intraday lows).
      - Force loss-cut (post-plan): any held position down more than
        force_loss_cut_pct gets a forced full SELL, overriding the LLM.
    """
    now = now or datetime.now(UTC)
    recent_trades = recent_trades or {}
    news_batch_tickers = news_batch_tickers or set()
    ticker_recent_moves = ticker_recent_moves or {}
    avg_entry_prices = avg_entry_prices or {}
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

        # Cooldown veto: no reverse-side trade within window. News bypass is
        # opt-in because the LLM will happily flip the same ticker every
        # news batch otherwise.
        recent = recent_trades.get(t.upper())
        if recent is not None:
            recent_side, recent_ts = recent
            age = (now - recent_ts).total_seconds()
            in_news = t.upper() in news_batch_tickers
            bypassed = news_bypass_cooldown and in_news
            if recent_side != side and age < cooldown_seconds and not bypassed:
                continue

        if side == "buy":
            # Adverse-move: don't buy into recent weakness.
            move = ticker_recent_moves.get(t.upper())
            if move is not None and move <= -adverse_move_threshold_pct:
                continue
            # Halt-buys: don't average down on losing positions.
            avg_entry = avg_entry_prices.get(t.upper())
            if avg_entry and avg_entry > 0:
                loss_pct = (price / avg_entry - 1) * 100
                if loss_pct <= -halt_buys_drawdown_pct:
                    continue
        else:  # sell
            # Small-drawdown hold: don't trim into a bounce on tiny drawdown.
            avg_entry = avg_entry_prices.get(t.upper())
            if avg_entry and avg_entry > 0:
                unrealized_pct = (price / avg_entry - 1) * 100
                if -small_drawdown_hold_pct < unrealized_pct < 0:
                    move = ticker_recent_moves.get(t.upper())
                    if move is not None and move > 0:
                        continue
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
    # Force loss-cut override: any position down > force_loss_cut_pct becomes
    # a forced full SELL, overriding any partial-trim plan the LLM generated.
    planned_by_ticker = {p.ticker.upper(): idx for idx, p in enumerate(plans)}
    for ticker_upper, cur_val in current_positions.items():
        if cur_val <= 0:
            continue
        entry = avg_entry_prices.get(ticker_upper.upper())
        if not entry or entry <= 0:
            continue
        cur_price = prices.get(ticker_upper.upper())
        if not cur_price:
            continue
        loss_pct = (cur_price / entry - 1) * 100
        if loss_pct <= -force_loss_cut_pct:
            qty = round(cur_val / cur_price, 4)
            if qty <= 0:
                continue
            forced_plan = TradePlan(
                ticker=ticker_upper.upper(),
                side="sell",
                dollars=round(cur_val, 2),
                qty=qty,
                target_pct=0.0,
                current_pct=(round(cur_val / total_equity * 100, 2)
                             if total_equity else 0.0),
                reason=f"force loss-cut ({loss_pct:.1f}% from entry)",
            )
            if ticker_upper.upper() in planned_by_ticker:
                plans[planned_by_ticker[ticker_upper.upper()]] = forced_plan
            else:
                plans.append(forced_plan)
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
    """Submit each plan through the broker. Idempotent by client_order_id.

    Full-exit SELLs (target_pct == 0) are routed via broker.close_position()
    so Alpaca liquidates exact holdings without leaving fractional-share dust.
    """
    day = day or datetime.now(UTC).strftime("%Y-%m-%d")
    submitted: list[PaperOrder] = []
    for p in plans:
        # Per-order try/except so one bad order (non-fractionable ticker,
        # trading halt, etc.) doesn't sink the rest of the batch. Log the
        # failure and continue with the next plan.
        try:
            if p.side == "sell" and p.target_pct == 0.0:
                try:
                    submitted.append(broker.close_position(p.ticker))
                    continue
                except Exception:  # noqa: BLE001 - fall back to computed-qty sell
                    pass
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
        except Exception as e:  # noqa: BLE001 - per-trade isolation
            import logging as _logging
            _logging.getLogger(__name__).warning(
                "order rejected for %s %s %s: %s", p.side, p.qty, p.ticker, e,
            )
            continue
    return submitted
