"""Tests for the two-tier paper-loop. Broker + rec-store faked."""

from datetime import UTC, datetime, timedelta

import pytest

from agentic_investor.orchestrator.loop import (
    LoopConfig,
    LoopState,
    _drift_exceeds_band,
    _effective_band,
    _filter_should_skip,
    _opinion_barely_moved,
    _price_move_trigger,
    _technical_stance_changed,
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


def _fake_generate(cfg, as_of=None, news_batch_context=None,
                   pre_picked_tickers=None, previous_rec=None,
                   extra_tickers=None):
    return _rec(cfg.amount), list(cfg.tickers) or ["AAPL", "NVDA"]


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


def test_effective_band_narrows_with_high_confidence():
    # confidence=0.9 -> factor 0.6 -> 3.0
    assert _effective_band(5.0, confidence=0.9) < 5.0
    assert _effective_band(5.0, confidence=0.9) == 5.0 * 0.6


def test_effective_band_widens_with_low_confidence():
    # confidence=0.2 -> factor 1.3 -> 6.5
    assert _effective_band(5.0, confidence=0.2) > 5.0
    assert _effective_band(5.0, confidence=0.2) == 5.0 * 1.3


def test_effective_band_neutral_at_half_confidence():
    assert _effective_band(5.0, confidence=0.5) == 5.0


def test_effective_band_defaults_to_base_when_confidence_missing():
    assert _effective_band(5.0, confidence=None) == 5.0


def test_effective_band_tightens_for_small_positions():
    # 5% target * 20% rel = 1pp rel band; less than 5pp abs -> use 1pp.
    band = _effective_band(5.0, confidence=None, target_pct=5.0, band_rel_pct=20.0)
    assert band == 1.0


def test_effective_band_keeps_abs_for_large_positions():
    # 30% target * 20% rel = 6pp rel band; more than 5pp abs -> use 5pp cap.
    band = _effective_band(5.0, confidence=None, target_pct=30.0, band_rel_pct=20.0)
    assert band == 5.0


def test_effective_band_composes_size_and_confidence():
    # 10% target * 20% rel = 2pp. High confidence 0.9 -> 2 * 0.6 = 1.2pp.
    band = _effective_band(5.0, confidence=0.9, target_pct=10.0, band_rel_pct=20.0)
    assert band == pytest.approx(1.2)


def test_effective_band_rel_disabled_when_zero():
    band = _effective_band(5.0, confidence=None, target_pct=5.0, band_rel_pct=0.0)
    assert band == 5.0


def test_drift_exceeds_band_triggers_on_small_position_when_size_aware():
    # 5% target, 8% current = 3pp drift. Without size-aware: 5pp band ->
    # no trigger. With band_rel_pct=20 -> band=1pp -> trigger.
    small_rec = Recommendation(
        request=OrchestratorRequest(tickers=["AAPL"], amount=10_000),
        allocation=Allocation(
            positions=[Position(
                ticker="AAPL", weight_pct=5, dollars=500,
                rationale="x", confidence=0.5,
            )],
            cash_pct=95, cash_dollars=9500, portfolio_rationale="x",
        ),
    )
    # Without size-aware: no trigger.
    assert _drift_exceeds_band(
        small_rec, {"AAPL": 800}, total_equity=10_000,
        band_abs_pct=5.0, band_rel_pct=0.0,
    ) is False
    # With size-aware: trigger.
    assert _drift_exceeds_band(
        small_rec, {"AAPL": 800}, total_equity=10_000,
        band_abs_pct=5.0, band_rel_pct=20.0,
    ) is True


def test_low_confidence_widens_band_and_suppresses_drift_trigger():
    # 3pp drift; base band=5pp normally does NOT trigger anyway.
    # But test at 5.5pp drift with different confidence:
    #   confidence=0.2 -> effective band 6.5 -> 5.5 NOT trigger.
    #   confidence=0.9 -> effective band 3.0 -> 5.5 DOES trigger.
    rec_low = Recommendation(
        request=OrchestratorRequest(tickers=["AAPL"], amount=10_000),
        allocation=Allocation(
            positions=[Position(
                ticker="AAPL", weight_pct=40, dollars=4000,
                rationale="x", confidence=0.2,
            )],
            cash_pct=60, cash_dollars=6000, portfolio_rationale="x",
        ),
    )
    assert _drift_exceeds_band(
        rec_low, {"AAPL": 4550}, total_equity=10_000, band_abs_pct=5.0
    ) is False

    rec_high = Recommendation(
        request=OrchestratorRequest(tickers=["AAPL"], amount=10_000),
        allocation=Allocation(
            positions=[Position(
                ticker="AAPL", weight_pct=40, dollars=4000,
                rationale="x", confidence=0.9,
            )],
            cash_pct=60, cash_dollars=6000, portfolio_rationale="x",
        ),
    )
    assert _drift_exceeds_band(
        rec_high, {"AAPL": 4550}, total_equity=10_000, band_abs_pct=5.0
    ) is True


def test_opinion_barely_moved_true_when_all_deltas_below_threshold():
    prev = Recommendation(
        request=OrchestratorRequest(tickers=["AAPL", "NVDA"], amount=10_000),
        allocation=Allocation(
            positions=[
                Position(ticker="AAPL", weight_pct=40, dollars=4000, rationale="x"),
                Position(ticker="NVDA", weight_pct=40, dollars=4000, rationale="x"),
            ],
            cash_pct=20, cash_dollars=2000, portfolio_rationale="x",
        ),
    )
    # New rec: AAPL 41 (delta 1), NVDA 39 (delta 1), cash 20 - all < 3pp
    new = Recommendation(
        request=OrchestratorRequest(tickers=["AAPL", "NVDA"], amount=10_000),
        allocation=Allocation(
            positions=[
                Position(ticker="AAPL", weight_pct=41, dollars=4100, rationale="x"),
                Position(ticker="NVDA", weight_pct=39, dollars=3900, rationale="x"),
            ],
            cash_pct=20, cash_dollars=2000, portfolio_rationale="x",
        ),
    )
    barely, deltas = _opinion_barely_moved(new, prev, threshold_pct=3.0)
    assert barely is True
    assert deltas == {"AAPL": 1.0, "NVDA": 1.0}


def test_opinion_barely_moved_false_when_any_delta_exceeds_threshold():
    prev = Recommendation(
        request=OrchestratorRequest(tickers=["AAPL", "NVDA"], amount=10_000),
        allocation=Allocation(
            positions=[
                Position(ticker="AAPL", weight_pct=40, dollars=4000, rationale="x"),
                Position(ticker="NVDA", weight_pct=40, dollars=4000, rationale="x"),
            ],
            cash_pct=20, cash_dollars=2000, portfolio_rationale="x",
        ),
    )
    # AAPL delta = 5pp (exceeds 3pp threshold) - opinion DID move
    new = Recommendation(
        request=OrchestratorRequest(tickers=["AAPL", "NVDA"], amount=10_000),
        allocation=Allocation(
            positions=[
                Position(ticker="AAPL", weight_pct=45, dollars=4500, rationale="x"),
                Position(ticker="NVDA", weight_pct=35, dollars=3500, rationale="x"),
            ],
            cash_pct=20, cash_dollars=2000, portfolio_rationale="x",
        ),
    )
    barely, _ = _opinion_barely_moved(new, prev, threshold_pct=3.0)
    assert barely is False


def test_opinion_barely_moved_treats_added_position_as_moved():
    prev = Recommendation(
        request=OrchestratorRequest(tickers=["AAPL"], amount=10_000),
        allocation=Allocation(
            positions=[
                Position(ticker="AAPL", weight_pct=80, dollars=8000, rationale="x"),
            ],
            cash_pct=20, cash_dollars=2000, portfolio_rationale="x",
        ),
    )
    # Added NVDA position - delta from 0 -> 30 counts as moved.
    new = Recommendation(
        request=OrchestratorRequest(tickers=["AAPL", "NVDA"], amount=10_000),
        allocation=Allocation(
            positions=[
                Position(ticker="AAPL", weight_pct=50, dollars=5000, rationale="x"),
                Position(ticker="NVDA", weight_pct=30, dollars=3000, rationale="x"),
            ],
            cash_pct=20, cash_dollars=2000, portfolio_rationale="x",
        ),
    )
    barely, _ = _opinion_barely_moved(new, prev, threshold_pct=3.0)
    assert barely is False


def test_opinion_barely_moved_false_when_no_previous_rec():
    new = Recommendation(
        request=OrchestratorRequest(tickers=["AAPL"], amount=10_000),
        allocation=Allocation(
            positions=[Position(ticker="AAPL", weight_pct=80, dollars=8000, rationale="x")],
            cash_pct=20, cash_dollars=2000, portfolio_rationale="x",
        ),
    )
    barely, _ = _opinion_barely_moved(new, None, threshold_pct=3.0)
    assert barely is False


def test_price_move_trigger_fires_when_ticker_moves_beyond_threshold():
    baseline = {"NVDA": 200.0, "AAPL": 150.0}
    current = {"NVDA": 204.5, "AAPL": 148.0}  # NVDA +2.25% (over 2), AAPL -1.33%
    fire, moves = _price_move_trigger(baseline, current, threshold_pct=2.0)
    assert fire is True
    assert moves["NVDA"] == 2.25
    assert moves["AAPL"] == -1.33


def test_price_move_trigger_silent_when_all_within_threshold():
    baseline = {"NVDA": 200.0, "AAPL": 150.0}
    current = {"NVDA": 201.5, "AAPL": 149.5}  # both < 1% move
    fire, moves = _price_move_trigger(baseline, current, threshold_pct=2.0)
    assert fire is False
    assert len(moves) == 2


def test_price_move_trigger_silent_when_baseline_empty():
    fire, moves = _price_move_trigger({}, {"NVDA": 200}, threshold_pct=2.0)
    assert fire is False
    assert moves == {}


def test_technical_stance_change_detects_flips():
    prev = {"NVDA": "bullish", "AAPL": "neutral", "MSFT": "bearish"}
    cur = {"NVDA": "neutral", "AAPL": "neutral", "MSFT": "bearish"}
    fire, changes = _technical_stance_changed(prev, cur)
    assert fire is True
    assert changes == {"NVDA": "bullish -> neutral"}


def test_technical_stance_no_change_returns_false():
    prev = {"NVDA": "bullish", "AAPL": "neutral"}
    cur = {"NVDA": "bullish", "AAPL": "neutral"}
    fire, changes = _technical_stance_changed(prev, cur)
    assert fire is False
    assert changes == {}


def _rec_with_weights(**kw) -> Recommendation:
    positions = [
        Position(ticker=t, weight_pct=w, dollars=w * 100, rationale="x",
                 confidence=kw.get("confidences", {}).get(t, 0.5))
        for t, w in kw.get("weights", {}).items()
    ]
    return Recommendation(
        request=OrchestratorRequest(tickers=list(kw.get("weights", {})),
                                    amount=10_000),
        allocation=Allocation(
            positions=positions,
            cash_pct=kw.get("cash", 100 - sum(kw.get("weights", {}).values())),
            cash_dollars=0, portfolio_rationale="x",
        ),
    )


def test_opinion_barely_moved_now_includes_cash_delta():
    prev = _rec_with_weights(weights={"AAPL": 40, "NVDA": 40}, cash=20)
    # Cash rotates 5pp - filter now catches this in deltas
    new = _rec_with_weights(weights={"AAPL": 40, "NVDA": 45}, cash=15)
    barely, deltas = _opinion_barely_moved(new, prev, threshold_pct=3.0)
    assert "__cash__" in deltas
    assert deltas["__cash__"] == 5.0
    assert barely is False  # cash delta 5pp exceeds 3pp threshold


def test_filter_skips_when_avg_drift_high():
    prev = _rec_with_weights(weights={"AAPL": 30, "NVDA": 30, "MSFT": 30}, cash=10)
    # All 3 positions moved 6pp - avg drift 6pp, above 5pp default
    new = _rec_with_weights(weights={"AAPL": 36, "NVDA": 24, "MSFT": 36}, cash=4)
    skip, reason, deltas, stats = _filter_should_skip(
        new, prev,
        opinion_drift_threshold_pct=3.0,
        max_avg_drift_pct=5.0,
        max_single_delta_pct=15.0,
    )
    assert skip is True
    assert reason == "avg-drift-too-high"
    assert stats["avg_drift"] == 6.0


def test_filter_allows_max_delta_when_news_covers_ticker():
    prev = _rec_with_weights(weights={"AAPL": 20, "NVDA": 60}, cash=20)
    new = _rec_with_weights(weights={"AAPL": 20, "NVDA": 40}, cash=40,
                            confidences={"NVDA": 0.4})
    # NVDA moved 20pp (over 15pp threshold), low confidence.
    # But news specifically for NVDA - allowed.
    skip, reason, _, _ = _filter_should_skip(
        new, prev,
        opinion_drift_threshold_pct=3.0,
        max_avg_drift_pct=25.0,  # keep avg check relaxed
        max_single_delta_pct=15.0,
        news_batch_tickers={"NVDA"},
    )
    assert skip is False


def test_filter_blocks_max_delta_when_no_news_and_low_confidence():
    prev = _rec_with_weights(weights={"AAPL": 20, "NVDA": 60}, cash=20)
    new = _rec_with_weights(weights={"AAPL": 20, "NVDA": 40}, cash=40,
                            confidences={"NVDA": 0.4})
    # NVDA moved 20pp, no news for NVDA, low confidence -> blocked.
    skip, reason, _, _ = _filter_should_skip(
        new, prev,
        opinion_drift_threshold_pct=3.0,
        max_avg_drift_pct=25.0,
        max_single_delta_pct=15.0,
        news_batch_tickers=set(),
    )
    assert skip is True
    assert reason == "max-delta-unjustified"


def test_filter_allows_max_delta_when_high_confidence():
    prev = _rec_with_weights(weights={"AAPL": 20, "NVDA": 60}, cash=20)
    new = _rec_with_weights(weights={"AAPL": 20, "NVDA": 40}, cash=40,
                            confidences={"NVDA": 0.85})
    # NVDA moved 20pp, no news, but HIGH confidence -> allowed.
    skip, _, _, _ = _filter_should_skip(
        new, prev,
        opinion_drift_threshold_pct=3.0,
        max_avg_drift_pct=25.0,
        max_single_delta_pct=15.0,
        confidence_by_ticker={"NVDA": 0.85},
    )
    assert skip is False


def test_filter_blocks_max_delta_when_no_news_low_conf_from_lookup():
    prev = _rec_with_weights(weights={"AAPL": 20, "NVDA": 60}, cash=20)
    new = _rec_with_weights(weights={"AAPL": 20, "NVDA": 40}, cash=40)
    skip, reason, _, _ = _filter_should_skip(
        new, prev,
        opinion_drift_threshold_pct=3.0,
        max_avg_drift_pct=25.0,
        max_single_delta_pct=15.0,
        confidence_by_ticker={"NVDA": 0.4},
    )
    assert skip is True
    assert reason == "max-delta-unjustified"


def test_loop_state_round_trips_through_dict():
    from datetime import UTC as _UTC
    from datetime import datetime as _dt
    state = LoopState(
        last_rec_id=42,
        last_rec_date="2026-08-28",
        ticks_run=7,
        orders_submitted=3,
        frozen_picker_tickers=["NVDA", "MSFT"],
        baseline_prices={"NVDA": 200.5, "MSFT": 515.0},
        last_regen_at=_dt.now(_UTC),
        last_stances={"NVDA": "bullish", "MSFT": "neutral"},
    )
    d = state.to_dict()
    rehydrated = LoopState.from_dict(d)
    assert rehydrated.last_rec_id == 42
    assert rehydrated.last_rec_date == "2026-08-28"
    assert rehydrated.ticks_run == 7
    assert rehydrated.frozen_picker_tickers == ["NVDA", "MSFT"]
    assert rehydrated.baseline_prices == {"NVDA": 200.5, "MSFT": 515.0}
    assert rehydrated.last_stances == {"NVDA": "bullish", "MSFT": "neutral"}
    assert rehydrated.last_regen_at == state.last_regen_at


def test_loop_state_from_empty_dict_returns_defaults():
    state = LoopState.from_dict({})
    assert state.last_rec_id is None
    assert state.ticks_run == 0
    assert state.baseline_prices == {}


def test_drift_true_when_position_no_longer_in_target():
    rec = _rec()
    # Portfolio still holds 30% TSLA which isn't in target - big drift.
    assert _drift_exceeds_band(
        rec, {"AAPL": 4000, "NVDA": 3000, "TSLA": 3000},
        total_equity=10_000, band_abs_pct=5.0
    ) is True


def test_promotion_extracts_beneficiaries_from_batch_and_caps(monkeypatch):
    """0w: news-body tickers get promoted into the LLM's ticker set, bounded."""
    captured = {}

    def _capture_generate(cfg, as_of=None, news_batch_context=None,
                          pre_picked_tickers=None, previous_rec=None,
                          extra_tickers=None):
        captured["extra_tickers"] = extra_tickers
        return _rec(cfg.amount), list(cfg.tickers) or ["AAPL"]

    monkeypatch.setattr(
        "agentic_investor.orchestrator.loop._generate_recommendation",
        _capture_generate,
    )
    monkeypatch.setattr("agentic_investor.orchestrator.loop.record_snapshot", lambda *a, **k: 1)
    monkeypatch.setattr("agentic_investor.orchestrator.loop.record_order", lambda *a, **k: None)

    batch_ctx = (
        "- [HOT] BE  age=1m: Bloom Energy could benefit\n"
        "- [HOT] TSM  age=1m: TSMC ramps capacity\n"
        "- [HOT] AVGO  age=2m: Broadcom secures deal\n"
        "- [HOT] PLTR  age=2m: Palantir expands contract\n"
        "- [HOT] SNOW  age=3m: Snowflake beats"
    )
    cfg = LoopConfig(
        tickers=["AAPL"], band_abs_pct=5.0,
        min_open_dollars=1.0, min_add_dollars=1.0, min_trim_dollars=1.0,
        max_promotions_per_regen=3,
    )
    state = LoopState(pending_news_context=batch_ctx)
    broker = FakeBroker(cash=10_000, equity=10_000)

    run_tick(cfg, state, broker, save_rec=lambda r: 1,
             price_fetcher=lambda t: 100.0)

    # AAPL is in cfg.tickers; other 5 are candidates. Capped at 3.
    assert captured["extra_tickers"] is not None
    assert len(captured["extra_tickers"]) == 3
    assert "AAPL" not in captured["extra_tickers"]
    assert set(captured["extra_tickers"]).issubset({"BE", "TSM", "AVGO", "PLTR", "SNOW"})


def test_tick_regenerates_rec_on_first_run_and_submits_orders(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "agentic_investor.orchestrator.loop._generate_recommendation",
        _fake_generate,
    )
    monkeypatch.setattr(
        "agentic_investor.orchestrator.loop.record_snapshot",
        lambda *a, **k: 1,
    )
    monkeypatch.setattr(
        "agentic_investor.orchestrator.loop.record_order",
        lambda *a, **k: None,
    )

    # Fake rec puts AAPL + NVDA at 40% each - above the production ceiling of
    # 25%. Disable the ceiling here so this test can focus on the tick pipeline.
    cfg = LoopConfig(
        tickers=["AAPL", "NVDA"], band_abs_pct=5.0,
        min_open_dollars=1.0, min_add_dollars=1.0, min_trim_dollars=1.0,
        max_add_concentration_pct=100.0,
    )
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
        _fake_generate,
    )
    monkeypatch.setattr("agentic_investor.orchestrator.loop.record_snapshot", lambda *a, **k: 1)
    monkeypatch.setattr("agentic_investor.orchestrator.loop.record_order", lambda *a, **k: None)
    monkeypatch.setattr(
        "agentic_investor.orchestrator.store.load_recommendation",
        lambda rec_id, url=None: _rec(),
    )

    cfg = LoopConfig(
        tickers=["AAPL", "NVDA"], band_abs_pct=5.0,
        min_open_dollars=1.0, min_add_dollars=1.0, min_trim_dollars=1.0,
    )
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
        _fake_generate,
    )
    monkeypatch.setattr("agentic_investor.orchestrator.loop.record_snapshot", lambda *a, **k: 1)

    cfg = LoopConfig(
        tickers=["AAPL", "NVDA"], dry_run=True,
        min_open_dollars=1.0, min_add_dollars=1.0, min_trim_dollars=1.0,
        max_add_concentration_pct=100.0,
    )
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
        _fake_generate,
    )
    cfg = LoopConfig(once=True, tickers=["AAPL"])
    broker = FakeBroker(is_open=False)

    state = run_loop(cfg, broker, sleep_fn=lambda s: None)
    assert state.ticks_run == 0
    assert state.orders_submitted == 0


def test_loop_runs_one_tick_then_exits_with_once(monkeypatch):
    monkeypatch.setattr(
        "agentic_investor.orchestrator.loop._generate_recommendation",
        _fake_generate,
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
