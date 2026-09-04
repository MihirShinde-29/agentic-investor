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


def test_add_floor_blocks_ladder_add_that_passes_legacy_floor():
    # Currently hold AAPL at $3900, target $4000. Delta = $100 (an ADD to an
    # existing position). Legacy threshold $50 would let it through, but the
    # add-floor at $500 blocks the nano-add. NVDA is at target (no trade).
    plans = compute_trade_plan(
        _rec(),
        current_positions={"AAPL": 3900, "NVDA": 4000},
        total_equity=10_000,
        prices={"AAPL": 100, "NVDA": 200},
        min_trade_dollars=50.0,
        min_add_dollars=500.0,
    )
    assert not any(p.ticker == "AAPL" for p in plans)


def test_open_floor_gates_first_entry_separately_from_add():
    # NVDA target $4000, currently 0. Delta = $4000 (an OPEN). Set the open
    # floor low to allow it but the add floor high to prove they're independent.
    plans = compute_trade_plan(
        _rec(),
        current_positions={"AAPL": 4000},  # AAPL already at target
        total_equity=10_000,
        prices={"AAPL": 100, "NVDA": 200},
        min_open_dollars=100.0,
        min_add_dollars=10_000.0,
    )
    tickers = {p.ticker for p in plans}
    assert "NVDA" in tickers
    assert "AAPL" not in tickers


def test_close_always_passes_regardless_of_trim_floor():
    # TSLA is held but not in the target rec, and the current value ($150) is
    # under the trim floor ($500). It should still be fully sold - we never
    # want to strand a partial position because the last chunk is small.
    plans = compute_trade_plan(
        _rec(),
        current_positions={"AAPL": 4000, "NVDA": 4000, "TSLA": 150},
        total_equity=10_000,
        prices={"AAPL": 100, "NVDA": 200, "TSLA": 250},
        min_trim_dollars=500.0,
    )
    tsla = next(p for p in plans if p.ticker == "TSLA")
    assert tsla.side == "sell"
    assert tsla.dollars == 150


def test_trim_floor_blocks_nano_trim_but_allows_bigger_reduction():
    # AAPL at $4700, target $4000 -> $700 trim (well above floor). NVDA at
    # $4050, target $4000 -> $50 trim (under $200 floor -> skipped).
    plans = compute_trade_plan(
        _rec(),
        current_positions={"AAPL": 4700, "NVDA": 4050},
        total_equity=10_000,
        prices={"AAPL": 100, "NVDA": 200},
        min_trim_dollars=200.0,
    )
    tickers = {p.ticker: p for p in plans}
    assert "AAPL" in tickers and tickers["AAPL"].side == "sell"
    assert "NVDA" not in tickers


def test_multi_source_bypass_requires_min_sources():
    # AAPL held at $8000, target $4000 (a REDUCE-side trade, direction is
    # sell). Cooldown says we just BOUGHT AAPL 60s ago - opposite side, so
    # cooldown blocks the sell... unless enough news sources converged. One
    # source alone (min=2) doesn't cut it.
    from datetime import UTC, datetime, timedelta

    now = datetime.now(UTC)
    recent = {"AAPL": ("buy", now - timedelta(seconds=60))}
    plans = compute_trade_plan(
        _rec(), current_positions={"AAPL": 8000}, total_equity=10_000,
        prices={"AAPL": 100, "NVDA": 200}, min_trade_dollars=1.0,
        recent_trades=recent, cooldown_seconds=900,
        news_batch_tickers={"AAPL"},
        news_source_counts={"AAPL": 1},
        min_bypass_sources=2,
        now=now,
    )
    assert not any(p.ticker == "AAPL" for p in plans)


def test_multi_source_bypass_fires_at_threshold():
    # Same setup, but now 2 distinct URLs on AAPL - meets convergence threshold.
    from datetime import UTC, datetime, timedelta

    now = datetime.now(UTC)
    recent = {"AAPL": ("buy", now - timedelta(seconds=60))}
    plans = compute_trade_plan(
        _rec(), current_positions={"AAPL": 8000}, total_equity=10_000,
        prices={"AAPL": 100, "NVDA": 200}, min_trade_dollars=1.0,
        recent_trades=recent, cooldown_seconds=900,
        news_batch_tickers={"AAPL"},
        news_source_counts={"AAPL": 2},
        min_bypass_sources=2,
        now=now,
    )
    aapl = next(p for p in plans if p.ticker == "AAPL")
    assert aapl.side == "sell"


def test_add_concentration_ceiling_blocks_buy_above_cap():
    # AAPL currently held at $2000 (20% of $10K equity). LLM wants it at $3500
    # (35%). Cap is 25%. The buy would push the position over the ceiling so
    # it must be skipped. NVDA at target (no trade) as a control.
    rec = Recommendation(
        request=OrchestratorRequest(tickers=["AAPL", "NVDA"], amount=10_000),
        allocation=Allocation(
            positions=[
                Position(ticker="AAPL", weight_pct=35, dollars=3500, rationale="x"),
                Position(ticker="NVDA", weight_pct=40, dollars=4000, rationale="x"),
            ],
            cash_pct=25, cash_dollars=2500, portfolio_rationale="x",
        ),
    )
    plans = compute_trade_plan(
        rec,
        current_positions={"AAPL": 2000, "NVDA": 4000},
        total_equity=10_000,
        prices={"AAPL": 100, "NVDA": 200},
        min_trade_dollars=1.0,
        max_add_concentration_pct=25.0,
    )
    assert not any(p.ticker == "AAPL" for p in plans)


def test_add_concentration_ceiling_allows_buy_within_cap():
    # AAPL held at $1000 (10%). Target $2400 (24%). Cap 25%. Under the cap,
    # buy proceeds.
    rec = Recommendation(
        request=OrchestratorRequest(tickers=["AAPL", "NVDA"], amount=10_000),
        allocation=Allocation(
            positions=[
                Position(ticker="AAPL", weight_pct=24, dollars=2400, rationale="x"),
                Position(ticker="NVDA", weight_pct=40, dollars=4000, rationale="x"),
            ],
            cash_pct=36, cash_dollars=3600, portfolio_rationale="x",
        ),
    )
    plans = compute_trade_plan(
        rec,
        current_positions={"AAPL": 1000, "NVDA": 4000},
        total_equity=10_000,
        prices={"AAPL": 100, "NVDA": 200},
        min_trade_dollars=1.0,
        max_add_concentration_pct=25.0,
    )
    aapl = next(p for p in plans if p.ticker == "AAPL")
    assert aapl.side == "buy"
    assert aapl.dollars == 1400


def test_add_concentration_ceiling_never_blocks_sells():
    # AAPL currently at $4000 (40%, above cap already from price appreciation).
    # LLM wants to trim to $2000. Even though position is above the ceiling,
    # the SELL must proceed - the ceiling only gates buys.
    rec = Recommendation(
        request=OrchestratorRequest(tickers=["AAPL", "NVDA"], amount=10_000),
        allocation=Allocation(
            positions=[
                Position(ticker="AAPL", weight_pct=20, dollars=2000, rationale="x"),
                Position(ticker="NVDA", weight_pct=40, dollars=4000, rationale="x"),
            ],
            cash_pct=40, cash_dollars=4000, portfolio_rationale="x",
        ),
    )
    plans = compute_trade_plan(
        rec,
        current_positions={"AAPL": 4000, "NVDA": 4000},
        total_equity=10_000,
        prices={"AAPL": 100, "NVDA": 200},
        min_trade_dollars=1.0,
        max_add_concentration_pct=25.0,
    )
    aapl = next(p for p in plans if p.ticker == "AAPL")
    assert aapl.side == "sell"
    assert aapl.dollars == 2000


def test_missing_price_skips_ticker_gracefully():
    plans = compute_trade_plan(
        _rec(), current_positions={}, total_equity=10_000,
        prices={"AAPL": 100},  # NVDA price missing
        min_trade_dollars=1.0,
    )
    assert [p.ticker for p in plans] == ["AAPL"]


def test_client_order_id_is_stable_across_same_day_retries():
    a = _client_order_id(1, "AAPL", "buy", "2026-08-27", 5.0)
    b = _client_order_id(1, "AAPL", "buy", "2026-08-27", 5.0)
    assert a == b
    c = _client_order_id(1, "AAPL", "buy", "2026-08-28", 5.0)
    assert a != c


def test_client_order_id_differs_when_fall_through_qty_differs():
    # Same rec, ticker, side, day but a different qty must produce a distinct
    # id, or Alpaca 409-rejects the second submission as a duplicate and the
    # walk-back silently disappears. This bit us when the mechanical
    # fall-through fired a second time against the same rec baseline.
    a = _client_order_id(1, "SNOW", "buy", "2026-09-03", 4.02)
    b = _client_order_id(1, "SNOW", "buy", "2026-09-03", 0.2243)
    assert a != b


class _FakeBroker:
    def __init__(self):
        self.calls = []
        self.close_calls = []

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

    def close_position(self, ticker):
        self.close_calls.append(ticker)
        return PaperOrder(
            id=f"close-{len(self.close_calls)}", client_order_id="close",
            ticker=ticker, side="sell", qty=0.0,
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
        news_batch_tickers={"AAPL"},
    )
    # Default news bypass: AAPL BUY allowed because AAPL is in the news batch.
    # LLM is expected to self-restrain via the recent-trades prompt block.
    assert any(p.ticker == "AAPL" and p.side == "buy" for p in plans)


def test_cooldown_strict_block_when_news_bypass_disabled():
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
        news_batch_tickers={"AAPL"},
        news_bypass_cooldown=False,
    )
    # Strict mode: mechanical cooldown holds even when news is present.
    assert not any(p.ticker == "AAPL" and p.side == "buy" for p in plans)


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


def test_force_loss_cut_overrides_llm_decision():
    from agentic_investor.orchestrator.state import (
        Allocation,
        OrchestratorRequest,
        Position,
        Recommendation,
    )
    # LLM says hold NVDA at 25%. Current NVDA is down 12% from entry.
    # Force loss-cut (threshold 8%) should override with full SELL.
    rec = Recommendation(
        request=OrchestratorRequest(tickers=["NVDA"], amount=10_000),
        allocation=Allocation(
            positions=[Position(ticker="NVDA", weight_pct=25, dollars=2500, rationale="x")],
            cash_pct=75, cash_dollars=7500, portfolio_rationale="x",
        ),
    )
    plans = compute_trade_plan(
        rec,
        current_positions={"NVDA": 2500},  # at target so no drift-based plan
        total_equity=10_000,
        prices={"NVDA": 88},  # avg entry 100 → -12%
        min_trade_dollars=1.0,
        avg_entry_prices={"NVDA": 100.0},
        force_loss_cut_pct=8.0,
    )
    nvda_plan = next((p for p in plans if p.ticker == "NVDA"), None)
    assert nvda_plan is not None
    assert nvda_plan.side == "sell"
    assert "force loss-cut" in nvda_plan.reason


def test_force_loss_cut_replaces_partial_trim_with_full_exit():
    from agentic_investor.orchestrator.state import (
        Allocation,
        OrchestratorRequest,
        Position,
        Recommendation,
    )
    # LLM wants to trim to 15% but position is down 12% - force full exit.
    rec = Recommendation(
        request=OrchestratorRequest(tickers=["NVDA"], amount=10_000),
        allocation=Allocation(
            positions=[Position(ticker="NVDA", weight_pct=15, dollars=1500, rationale="x")],
            cash_pct=85, cash_dollars=8500, portfolio_rationale="x",
        ),
    )
    plans = compute_trade_plan(
        rec,
        current_positions={"NVDA": 2500},
        total_equity=10_000,
        prices={"NVDA": 88},
        min_trade_dollars=1.0,
        avg_entry_prices={"NVDA": 100.0},
        force_loss_cut_pct=8.0,
    )
    nvda_plan = next(p for p in plans if p.ticker == "NVDA")
    assert nvda_plan.side == "sell"
    assert nvda_plan.dollars == 2500  # full exit, not partial
    assert "force loss-cut" in nvda_plan.reason


def test_full_exit_sell_routes_via_close_position():
    """target_pct == 0 should trigger broker.close_position (no fractional dust)."""
    # Rec doesn't include TSLA (dropped from portfolio); we hold $3000 of TSLA.
    plans = compute_trade_plan(
        _rec(),
        current_positions={"AAPL": 0, "NVDA": 0, "TSLA": 3000},
        total_equity=10_000,
        prices={"AAPL": 100, "NVDA": 200, "TSLA": 250},
        min_trade_dollars=1.0,
    )
    tsla_plan = next(p for p in plans if p.ticker == "TSLA")
    assert tsla_plan.target_pct == 0.0
    broker = _FakeBroker()
    execute_trade_plan(plans, broker, rec_id=1, day="2026-08-28")
    # TSLA should be closed via close_position, not submit_market_order
    assert "TSLA" in broker.close_calls
    assert not any(c["ticker"] == "TSLA" for c in broker.calls)


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
