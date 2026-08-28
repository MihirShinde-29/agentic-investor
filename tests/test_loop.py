"""Tests for the two-tier paper-loop. Broker + rec-store faked."""

from datetime import UTC, datetime, timedelta

from agentic_investor.orchestrator.loop import (
    LoopConfig,
    LoopState,
    _drift_exceeds_band,
    run_loop,
    run_tick,
)
from agentic_investor.orchestrator.state import (
    Allocation,
    OrchestratorRequest,
    Position,
    Recommendation,
)
from agentic_investor.tools.paper_broker import (
    PaperAccount,
    PaperClock,
    PaperOrder,
    PaperPosition,
)


def _rec(rec_amount: float = 10_000.0) -> Recommendation:
    return Recommendation(
        request=OrchestratorRequest(tickers=["AAPL", "NVDA"], amount=rec_amount),
        allocation=Allocation(
            positions=[
                Position(ticker="AAPL", weight_pct=40, dollars=4000, rationale="x"),
                Position(ticker="NVDA", weight_pct=40, dollars=4000, rationale="x"),
            ],
            cash_pct=20, cash_dollars=2000, portfolio_rationale="x",
        ),
    )


class FakeBroker:
    """Broker double: injectable clock + positions, records submitted orders."""

    def __init__(self, *, is_open=True, cash=100_000, equity=100_000, positions=None):
        self._is_open = is_open
        self._cash = cash
        self._equity = equity
        self._positions = positions or []
        self.submitted: list[PaperOrder] = []

    def get_clock(self) -> PaperClock:
        now = datetime.now(UTC).isoformat()
        nxt = (datetime.now(UTC) + timedelta(hours=1)).isoformat()
        return PaperClock(now=now, is_open=self._is_open, next_open=nxt, next_close=nxt)

    def get_account(self) -> PaperAccount:
        return PaperAccount(
            account_number="PA1", cash=self._cash, equity=self._equity,
            buying_power=self._equity * 2, portfolio_value=self._equity,
        )

    def get_positions(self) -> list[PaperPosition]:
        return list(self._positions)

    def list_orders(self, limit=50, status="all"):
        return list(self.submitted)

    def submit_market_order(self, ticker, side, qty, *,
                            client_order_id=None, stop_loss_pct=None, take_profit_pct=None):
        o = PaperOrder(
            id=f"b-{len(self.submitted)+1}",
            client_order_id=client_order_id or "",
            ticker=ticker, side=side, qty=qty,
            order_type="market", status="accepted",
            submitted_at=datetime.now(UTC).isoformat(),
        )
        self.submitted.append(o)
        return o

    def cancel_order(self, oid): pass


def test_drift_exceeds_band_true_when_position_far_from_target():
    rec = _rec()
    # Target AAPL=40, actual=60 (20pp drift) - triggers.
    assert _drift_exceeds_band(
        rec, {"AAPL": 6000, "NVDA": 4000}, total_equity=10_000, band_abs_pct=5.0
    ) is True


def test_drift_exceeds_band_false_when_within_tolerance():
    rec = _rec()
    # Target AAPL=40, actual=42 (2pp drift) - within a 5pp band.
    assert _drift_exceeds_band(
        rec, {"AAPL": 4200, "NVDA": 3800}, total_equity=10_000, band_abs_pct=5.0
    ) is False


def test_drift_true_when_position_no_longer_in_target():
    rec = _rec()
    # Portfolio still holds 30% TSLA which isn't in target - big drift.
    assert _drift_exceeds_band(
        rec, {"AAPL": 4000, "NVDA": 3000, "TSLA": 3000},
        total_equity=10_000, band_abs_pct=5.0
    ) is True


def test_tick_regenerates_rec_on_first_run_and_submits_orders(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "agentic_investor.orchestrator.loop._generate_recommendation",
        lambda cfg, as_of=None: _rec(cfg.amount),
    )
    monkeypatch.setattr(
        "agentic_investor.orchestrator.loop.record_snapshot",
        lambda *a, **k: 1,
    )
    monkeypatch.setattr(
        "agentic_investor.orchestrator.loop.record_order",
        lambda *a, **k: None,
    )

    cfg = LoopConfig(tickers=["AAPL", "NVDA"], band_abs_pct=5.0, min_trade_dollars=1.0)
    state = LoopState()
    broker = FakeBroker(cash=10_000, equity=10_000)

    saved = []
    def _save(rec):
        saved.append(rec)
        return len(saved)

    result = run_tick(cfg, state, broker, save_rec=_save,
                     price_fetcher=lambda t: {"AAPL": 100, "NVDA": 200}[t])

    assert result.regenerated_rec is True
    assert result.rec_id == 1
    assert state.last_rec_id == 1
    assert result.plan_count == 2  # buys for AAPL + NVDA
    assert len(result.submitted) == 2


def test_tick_reuses_rec_within_same_day_and_skips_when_no_drift(monkeypatch):
    monkeypatch.setattr(
        "agentic_investor.orchestrator.loop._generate_recommendation",
        lambda cfg, as_of=None: _rec(cfg.amount),
    )
    monkeypatch.setattr("agentic_investor.orchestrator.loop.record_snapshot", lambda *a, **k: 1)
    monkeypatch.setattr("agentic_investor.orchestrator.loop.record_order", lambda *a, **k: None)
    monkeypatch.setattr(
        "agentic_investor.orchestrator.store.load_recommendation",
        lambda rec_id, url=None: _rec(),
    )

    cfg = LoopConfig(tickers=["AAPL", "NVDA"], band_abs_pct=5.0, min_trade_dollars=1.0)
    today = datetime.now(UTC).strftime("%Y-%m-%d")
    state = LoopState(last_rec_id=42, last_rec_date=today)

    # Positions match target (40/40/20), so no drift, no trades.
    positions = [
        PaperPosition(ticker="AAPL", qty=40, avg_entry_price=100, market_value=4000,
                      unrealized_pl=0, unrealized_pl_pct=0),
        PaperPosition(ticker="NVDA", qty=20, avg_entry_price=200, market_value=4000,
                      unrealized_pl=0, unrealized_pl_pct=0),
    ]
    broker = FakeBroker(cash=2000, equity=10_000, positions=positions)

    result = run_tick(cfg, state, broker,
                     save_rec=lambda rec: 999,
                     price_fetcher=lambda t: {"AAPL": 100, "NVDA": 200}[t])

    assert result.regenerated_rec is False
    assert result.plan_count == 0
    assert result.submitted == []


def test_dry_run_computes_plan_but_submits_no_orders(monkeypatch):
    monkeypatch.setattr(
        "agentic_investor.orchestrator.loop._generate_recommendation",
        lambda cfg, as_of=None: _rec(cfg.amount),
    )
    monkeypatch.setattr("agentic_investor.orchestrator.loop.record_snapshot", lambda *a, **k: 1)

    cfg = LoopConfig(tickers=["AAPL", "NVDA"], dry_run=True, min_trade_dollars=1.0)
    state = LoopState()
    broker = FakeBroker(cash=10_000, equity=10_000)

    result = run_tick(cfg, state, broker,
                     save_rec=lambda rec: 1,
                     price_fetcher=lambda t: {"AAPL": 100, "NVDA": 200}[t])

    assert result.plan_count == 2
    assert result.submitted == []  # dry_run short-circuits execute
    assert broker.submitted == []


def test_loop_exits_when_market_closed_and_once_flag(monkeypatch):
    monkeypatch.setattr(
        "agentic_investor.orchestrator.loop._generate_recommendation",
        lambda cfg, as_of=None: _rec(cfg.amount),
    )
    cfg = LoopConfig(once=True, tickers=["AAPL"])
    broker = FakeBroker(is_open=False)

    state = run_loop(cfg, broker, sleep_fn=lambda s: None)
    assert state.ticks_run == 0
    assert state.orders_submitted == 0


def test_loop_runs_one_tick_then_exits_with_once(monkeypatch):
    monkeypatch.setattr(
        "agentic_investor.orchestrator.loop._generate_recommendation",
        lambda cfg, as_of=None: _rec(cfg.amount),
    )
    monkeypatch.setattr("agentic_investor.orchestrator.loop.record_snapshot", lambda *a, **k: 1)
    monkeypatch.setattr("agentic_investor.orchestrator.loop.record_order", lambda *a, **k: None)
    from agentic_investor.orchestrator import loop as loop_mod
    monkeypatch.setattr(
        loop_mod, "compute_trade_plan",
        lambda *a, **k: [],  # no trades needed; just prove the tick ran
    )

    cfg = LoopConfig(once=True, dry_run=True, tickers=["AAPL", "NVDA"])
    broker = FakeBroker(cash=10_000, equity=10_000)

    state = run_loop(cfg, broker, sleep_fn=lambda s: None)
    assert state.ticks_run == 1
