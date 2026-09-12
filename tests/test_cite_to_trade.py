"""Cite-to-trade gate: block plans whose ticker is not in the rec's CoT.

Structural replacement for the wall-clock per-ticker cooldown. Any
planned trade must be justifiable by the ticker appearing in
bull_case, bear_case, verdict, or disqualifiers - or be a drift
rebalance (target unchanged from prev rec) or a force loss-cut.
"""

from __future__ import annotations

from dataclasses import dataclass

from agentic_investor.orchestrator.loop import (
    _apply_cite_to_trade,
    _direction_of_citation,
    _ticker_cited,
)
from agentic_investor.orchestrator.rebalancer import TradePlan


class _Reasoning:
    def __init__(
        self, bull_case="", bear_case="", verdict="", disqualifiers=None,
    ):
        self.bull_case = bull_case
        self.bear_case = bear_case
        self.verdict = verdict
        self.disqualifiers = list(disqualifiers or [])


@dataclass
class _Alloc:
    reasoning: object
    positions: list


@dataclass
class _Pos:
    ticker: str
    weight_pct: float


@dataclass
class _Rec:
    allocation: _Alloc


class _RecorderSession:
    def __init__(self):
        self.events = []

    def log(self, event_name, payload):
        self.events.append((event_name, dict(payload)))

    def events_of(self, name):
        return [e for k, e in self.events if k == name]


def _plan(ticker, side="buy", qty=1.0, reason="drift"):
    return TradePlan(
        ticker=ticker, side=side, dollars=100.0, qty=qty,
        target_pct=10.0, current_pct=5.0, reason=reason,
    )


def _rec(bull="", bear="", verdict="", disq=None, positions=None):
    return _Rec(_Alloc(
        _Reasoning(bull_case=bull, bear_case=bear, verdict=verdict,
                   disqualifiers=disq),
        [_Pos(ticker=t, weight_pct=w) for t, w in (positions or [])],
    ))


# --- _ticker_cited ---


def test_cited_when_named_in_bull_case():
    r = _Reasoning(bull_case="MRK breakout on volume")
    assert _ticker_cited("MRK", r) is True


def test_cited_is_case_insensitive():
    r = _Reasoning(bull_case="mrk holds above SMA")
    assert _ticker_cited("MRK", r) is True


def test_not_cited_when_absent():
    r = _Reasoning(bull_case="MRK breakout", bear_case="TJX weakening")
    assert _ticker_cited("VZ", r) is False


def test_word_boundary_avoids_substring_match():
    # "A" should NOT match inside "AMD".
    r = _Reasoning(bull_case="AMD holding trend")
    assert _ticker_cited("A", r) is False
    assert _ticker_cited("AMD", r) is True


def test_cited_in_disqualifiers():
    r = _Reasoning(disqualifiers=["CRM - cooldown from earlier trim"])
    assert _ticker_cited("CRM", r) is True


# --- _direction_of_citation ---


def test_direction_bull_only():
    r = _Reasoning(bull_case="MRK positive", bear_case="TJX bearish")
    assert _direction_of_citation("MRK", r) == "bull"


def test_direction_bear_only():
    r = _Reasoning(bull_case="MRK positive", bear_case="TJX bearish")
    assert _direction_of_citation("TJX", r) == "bear"


def test_direction_none_when_both_or_neither():
    both = _Reasoning(bull_case="MRK positive", bear_case="MRK also risky")
    assert _direction_of_citation("MRK", both) is None
    neither = _Reasoning(bull_case="ORCL", bear_case="AAPL")
    assert _direction_of_citation("VZ", neither) is None


# --- _apply_cite_to_trade ---


def test_uncited_ticker_gets_blocked():
    rec = _rec(bull="MRK strong", bear="TJX weak",
              verdict="Increase MRK weight",
              positions=[("MRK", 25), ("TJX", 5), ("VZ", 10)])
    plans = [_plan("VZ", "sell"), _plan("MRK", "buy")]
    session = _RecorderSession()
    kept = _apply_cite_to_trade(plans, rec, prev_rec=None,
                                session=session, rec_id=42)
    kept_tickers = [p.ticker for p in kept]
    assert "MRK" in kept_tickers  # cited
    assert "VZ" not in kept_tickers  # not cited
    violations = session.events_of("cite_violation")
    assert len(violations) == 1
    assert violations[0]["ticker"] == "VZ"


def test_drift_rebalance_exempt_when_target_unchanged():
    # MRK target unchanged between prev and curr rec -> drift-rebalance,
    # cite-to-trade must NOT require it in CoT.
    prev = _rec(positions=[("MRK", 20)])
    curr = _rec(bull="Only talking about NVDA today",
                positions=[("MRK", 20), ("NVDA", 15)])
    plans = [_plan("MRK", "buy", reason="drift +2pp")]
    session = _RecorderSession()
    kept = _apply_cite_to_trade(plans, curr, prev_rec=prev,
                                session=session, rec_id=1)
    # MRK trade preserved because target didn't change.
    assert any(p.ticker == "MRK" for p in kept)
    assert session.events_of("cite_violation") == []


def test_changed_target_still_needs_cite():
    # Target changed from 20 -> 25 on MRK, and MRK is NOT in CoT -> blocked.
    prev = _rec(positions=[("MRK", 20)])
    curr = _rec(bull="NVDA breakout", positions=[("MRK", 25), ("NVDA", 10)])
    plans = [_plan("MRK", "buy")]
    session = _RecorderSession()
    kept = _apply_cite_to_trade(plans, curr, prev_rec=prev,
                                session=session, rec_id=1)
    assert not any(p.ticker == "MRK" for p in kept)


def test_force_loss_cut_bypasses_cite():
    rec = _rec(bull="MRK strong",
               positions=[("MRK", 30), ("BADCO", 5)])
    plans = [_plan("BADCO", "sell", reason="force loss-cut (-9.2%)")]
    session = _RecorderSession()
    kept = _apply_cite_to_trade(plans, rec, prev_rec=None,
                                session=session, rec_id=7)
    assert any(p.ticker == "BADCO" for p in kept)
    assert session.events_of("cite_violation") == []


def test_bull_cited_sell_logs_direction_violation():
    rec = _rec(bull="MRK strong breakout",
               positions=[("MRK", 20)])
    plans = [_plan("MRK", "sell")]
    session = _RecorderSession()
    kept = _apply_cite_to_trade(plans, rec, prev_rec=None,
                                session=session, rec_id=3)
    # Trade allowed (ticker is cited), but violation logged.
    assert len(kept) == 1
    violations = session.events_of("direction_violation")
    assert len(violations) == 1
    assert violations[0]["cited_as"] == "bull"
    assert violations[0]["side"] == "sell"


def test_bear_cited_buy_logs_direction_violation():
    rec = _rec(bear="TSLA overbought, avoid adding",
               positions=[("TSLA", 0)])
    plans = [_plan("TSLA", "buy")]
    session = _RecorderSession()
    kept = _apply_cite_to_trade(plans, rec, prev_rec=None,
                                session=session, rec_id=4)
    assert len(kept) == 1
    violations = session.events_of("direction_violation")
    assert violations and violations[0]["cited_as"] == "bear"


def test_missing_reasoning_falls_through():
    # Rec with no CoT -> can't gate, let all plans through.
    rec_no_reasoning = _Rec(_Alloc(None, [_Pos("MRK", 20)]))
    plans = [_plan("MRK", "buy"), _plan("VZ", "sell")]
    session = _RecorderSession()
    kept = _apply_cite_to_trade(plans, rec_no_reasoning, prev_rec=None,
                                session=session, rec_id=5)
    assert len(kept) == 2
    assert session.events_of("cite_violation") == []
