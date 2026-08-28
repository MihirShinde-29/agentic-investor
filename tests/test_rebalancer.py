"""Tests for the paper-trading rebalancer (deterministic diff)."""

from agentic_investor.orchestrator.rebalancer import (
    _client_order_id,
    compute_trade_plan,
    execute_trade_plan,
)
from agentic_investor.orchestrator.state import (
    Allocation,
    OrchestratorRequest,
    Position,
    Recommendation,
)
from agentic_investor.tools.paper_broker import PaperOrder


def _rec() -> Recommendation:
    return Recommendation(
        request=OrchestratorRequest(tickers=["AAPL", "NVDA"], amount=10_000),
        allocation=Allocation(
            positions=[
                Position(ticker="AAPL", weight_pct=40, dollars=4000, rationale="x"),
                Position(ticker="NVDA", weight_pct=40, dollars=4000, rationale="x"),
            ],
            cash_pct=20,
            cash_dollars=2000,
            portfolio_rationale="x",
        ),
    )


def test_empty_portfolio_produces_buys_for_every_target():
    plans = compute_trade_plan(
        _rec(), current_positions={}, total_equity=10_000,
        prices={"AAPL": 100, "NVDA": 200}, min_trade_dollars=1.0,
    )
    tickers = sorted(p.ticker for p in plans)
    assert tickers == ["AAPL", "NVDA"]
    assert all(p.side == "buy" for p in plans)
    aapl = next(p for p in plans if p.ticker == "AAPL")
    assert aapl.dollars == 4000
    assert aapl.qty == 40  # 4000 / 100


def test_overweight_position_generates_sell():
    # Currently 80% AAPL, target 40% -> SELL half.
    plans = compute_trade_plan(
        _rec(), current_positions={"AAPL": 8000}, total_equity=10_000,
        prices={"AAPL": 100, "NVDA": 200}, min_trade_dollars=1.0,
    )
    aapl = next(p for p in plans if p.ticker == "AAPL")
    assert aapl.side == "sell"
    assert aapl.dollars == 4000


def test_position_at_target_produces_no_trade():
    # Current == target for AAPL, NVDA still empty.
    plans = compute_trade_plan(
        _rec(), current_positions={"AAPL": 4000}, total_equity=10_000,
        prices={"AAPL": 100, "NVDA": 200}, min_trade_dollars=1.0,
    )
    assert not any(p.ticker == "AAPL" for p in plans)
    assert any(p.ticker == "NVDA" and p.side == "buy" for p in plans)


def test_min_trade_dollars_skips_micro_drift():
    # Drift is $10 - under $25 min - so no trade.
    plans = compute_trade_plan(
        _rec(), current_positions={"AAPL": 3990, "NVDA": 3990}, total_equity=10_000,
        prices={"AAPL": 100, "NVDA": 200}, min_trade_dollars=25.0,
    )
    assert plans == []


def test_position_no_longer_in_target_gets_fully_sold():
    # Rec has AAPL + NVDA; portfolio also holds TSLA - should sell TSLA to 0.
    plans = compute_trade_plan(
        _rec(),
        current_positions={"AAPL": 0, "NVDA": 0, "TSLA": 3000},
        total_equity=10_000,
        prices={"AAPL": 100, "NVDA": 200, "TSLA": 250},
        min_trade_dollars=1.0,
    )
    tsla = next(p for p in plans if p.ticker == "TSLA")
    assert tsla.side == "sell"
    assert tsla.dollars == 3000


def test_missing_price_skips_ticker_gracefully():
    plans = compute_trade_plan(
        _rec(), current_positions={}, total_equity=10_000,
        prices={"AAPL": 100},  # NVDA price missing
        min_trade_dollars=1.0,
    )
    assert [p.ticker for p in plans] == ["AAPL"]


def test_client_order_id_is_stable_across_same_day_retries():
    a = _client_order_id(1, "AAPL", "buy", "2026-08-27")
    b = _client_order_id(1, "AAPL", "buy", "2026-08-27")
    assert a == b
    c = _client_order_id(1, "AAPL", "buy", "2026-08-28")
    assert a != c


class _FakeBroker:
    def __init__(self):
        self.calls = []

    def submit_market_order(self, ticker, side, qty, *, client_order_id=None,
                            stop_loss_pct=None, take_profit_pct=None):
        self.calls.append({
            "ticker": ticker, "side": side, "qty": qty,
            "client_order_id": client_order_id,
            "stop_loss_pct": stop_loss_pct,
            "take_profit_pct": take_profit_pct,
        })
        return PaperOrder(
            id=f"broker-{len(self.calls)}", client_order_id=client_order_id,
            ticker=ticker, side=side, qty=qty,
            order_type="market", status="accepted",
            submitted_at="2026-08-27T10:00:00Z",
        )


def test_cooldown_vetoes_reversal_within_window():
    from datetime import UTC, datetime, timedelta

    now = datetime.now(UTC)
    # Recent trade: SOLD AAPL 10 min ago
    recent = {"AAPL": ("sell", now - timedelta(minutes=10))}
    # Current portfolio: AAPL 0, NVDA 0 (target 40% each)
    plans = compute_trade_plan(
        _rec(),
        current_positions={"AAPL": 0, "NVDA": 0},
        total_equity=10_000,
        prices={"AAPL": 100, "NVDA": 200},
        min_trade_dollars=1.0,
        recent_trades=recent,
        cooldown_seconds=900,  # 15 min
        now=now,
    )
    # AAPL BUY should be vetoed (opposite of recent SELL, still in cooldown).
    # NVDA BUY should go through (no recent trade).
    tickers = [p.ticker for p in plans]
    assert "AAPL" not in tickers
    assert "NVDA" in tickers


def test_cooldown_allows_same_side_repeat():
    from datetime import UTC, datetime, timedelta

    now = datetime.now(UTC)
    recent = {"AAPL": ("buy", now - timedelta(minutes=10))}
    # Position lags target, want to BUY more AAPL - same side, allowed.
    plans = compute_trade_plan(
        _rec(),
        current_positions={"AAPL": 1000},  # short of $4000 target
        total_equity=10_000,
        prices={"AAPL": 100, "NVDA": 200},
        min_trade_dollars=1.0,
        recent_trades=recent,
        cooldown_seconds=900,
        now=now,
    )
    aapl = next((p for p in plans if p.ticker == "AAPL"), None)
    assert aapl is not None
    assert aapl.side == "buy"


def test_cooldown_bypass_when_ticker_in_news_batch():
    from datetime import UTC, datetime, timedelta

    now = datetime.now(UTC)
    recent = {"AAPL": ("sell", now - timedelta(minutes=10))}
    plans = compute_trade_plan(
        _rec(),
        current_positions={"AAPL": 0, "NVDA": 0},
        total_equity=10_000,
        prices={"AAPL": 100, "NVDA": 200},
        min_trade_dollars=1.0,
        recent_trades=recent,
        cooldown_seconds=900,
        now=now,
        news_batch_tickers={"AAPL"},  # news for AAPL - bypass cooldown
    )
    # AAPL BUY allowed because AAPL is in news batch (fresh signal justifies).
    assert any(p.ticker == "AAPL" and p.side == "buy" for p in plans)


def test_cooldown_expired_allows_reversal():
    from datetime import UTC, datetime, timedelta

    now = datetime.now(UTC)
    recent = {"AAPL": ("sell", now - timedelta(minutes=20))}  # past 15min cooldown
    plans = compute_trade_plan(
        _rec(),
        current_positions={"AAPL": 0, "NVDA": 0},
        total_equity=10_000,
        prices={"AAPL": 100, "NVDA": 200},
        min_trade_dollars=1.0,
        recent_trades=recent,
        cooldown_seconds=900,  # 15 min
        now=now,
    )
    assert any(p.ticker == "AAPL" for p in plans)


def test_execute_only_attaches_stops_to_buy_side():
    plans = compute_trade_plan(
        _rec(),
        current_positions={"AAPL": 0, "NVDA": 0, "TSLA": 3000},
        total_equity=10_000,
        prices={"AAPL": 100, "NVDA": 200, "TSLA": 250},
        min_trade_dollars=1.0,
    )
    broker = _FakeBroker()
    execute_trade_plan(
        plans, broker, rec_id=42,
        stop_loss_pct=5.0, take_profit_pct=10.0, day="2026-08-27",
    )
    buys = [c for c in broker.calls if c["side"] == "buy"]
    sells = [c for c in broker.calls if c["side"] == "sell"]
    assert all(c["stop_loss_pct"] == 5.0 for c in buys)
    assert all(c["stop_loss_pct"] is None for c in sells)
