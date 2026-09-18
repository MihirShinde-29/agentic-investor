"""Tests for the whipsaw guard.

Motivating pattern from the 2026-09-14/18 A/B (5 days across arms
A/B/C): arm C flipped CRM 6+ times in a 20-min window and arm B had
best hit-rate 3/5 days but net-lost from asymmetric single-tick
reversals. Neither the opinion-drift filter nor cite-to-trade caught
this - each individual leg was small enough to pass the delta band
and the LLM cited *something* every time.

The guard blocks a plan whose side is opposite the last recorded
trade on the same ticker within
AGENTIC_WHIPSAW_GUARD_WINDOW_MIN minutes, unless the ticker is in
the current news batch (fresh news exempts).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

from agentic_investor.orchestrator.loop import _apply_whipsaw_guard


@dataclass
class _Plan:
    ticker: str
    side: str  # "buy" | "sell"
    qty: float = 1.0


@dataclass
class _State:
    recent_trades: dict[str, tuple[str, str]] = field(default_factory=dict)


class _Session:
    def __init__(self):
        self.logs: list[tuple[str, dict]] = []

    def log(self, event: str, payload: dict) -> None:
        self.logs.append((event, payload))


def _now() -> datetime:
    return datetime.now(UTC)


def test_no_op_when_flag_zero(monkeypatch):
    monkeypatch.setenv("AGENTIC_WHIPSAW_GUARD_WINDOW_MIN", "0")
    plans = [_Plan("AAPL", "sell")]
    state = _State(recent_trades={
        "AAPL": ("buy", (_now() - timedelta(minutes=1)).isoformat()),
    })
    session = _Session()
    out = _apply_whipsaw_guard(plans, state, session, set(), _now())
    assert out == plans
    assert session.logs == []


def test_blocks_opposite_side_within_window(monkeypatch):
    """Buy 2 min ago -> sell now, no news for the ticker -> dropped."""
    monkeypatch.setenv("AGENTIC_WHIPSAW_GUARD_WINDOW_MIN", "15")
    now = _now()
    plans = [_Plan("CRM", "sell", qty=3.29)]
    state = _State(recent_trades={
        "CRM": ("buy", (now - timedelta(minutes=2)).isoformat()),
    })
    session = _Session()
    out = _apply_whipsaw_guard(plans, state, session,
                                news_batch_tickers=set(), now=now)
    assert out == []
    # First event = per-plan reason, second = aggregate blocked_N.
    events = [e for e, _ in session.logs]
    assert "knob_fired" in events
    assert any(
        p.get("name") == "whipsaw_guard" and p.get("ticker") == "CRM"
        for _, p in session.logs
    )
    assert any(
        p.get("reason", "").startswith("blocked_")
        for _, p in session.logs
    )


def test_allows_same_side(monkeypatch):
    """Doubling down (buy after buy) is not whipsaw - passes through."""
    monkeypatch.setenv("AGENTIC_WHIPSAW_GUARD_WINDOW_MIN", "15")
    now = _now()
    plans = [_Plan("MSTR", "buy", qty=5.0)]
    state = _State(recent_trades={
        "MSTR": ("buy", (now - timedelta(minutes=3)).isoformat()),
    })
    session = _Session()
    out = _apply_whipsaw_guard(plans, state, session, set(), now)
    assert out == plans
    assert session.logs == []


def test_allows_reversal_when_ticker_in_news_batch(monkeypatch):
    """Fresh news on the ticker justifies the reversal. This is the
    'decisive exit on news' pattern we DO want to allow.
    """
    monkeypatch.setenv("AGENTIC_WHIPSAW_GUARD_WINDOW_MIN", "15")
    now = _now()
    plans = [_Plan("NVDA", "sell", qty=2.0)]
    state = _State(recent_trades={
        "NVDA": ("buy", (now - timedelta(minutes=1)).isoformat()),
    })
    session = _Session()
    out = _apply_whipsaw_guard(plans, state, session,
                                news_batch_tickers={"NVDA"}, now=now)
    assert out == plans
    assert session.logs == []


def test_allows_reversal_outside_window(monkeypatch):
    """After the window expires, a reversal is a legitimate re-decision."""
    monkeypatch.setenv("AGENTIC_WHIPSAW_GUARD_WINDOW_MIN", "15")
    now = _now()
    plans = [_Plan("AAPL", "sell", qty=1.0)]
    state = _State(recent_trades={
        "AAPL": ("buy", (now - timedelta(minutes=30)).isoformat()),
    })
    session = _Session()
    out = _apply_whipsaw_guard(plans, state, session, set(), now)
    assert out == plans


def test_no_recent_trade_passes_through(monkeypatch):
    """First trade on a ticker can't be a whipsaw - nothing to reverse."""
    monkeypatch.setenv("AGENTIC_WHIPSAW_GUARD_WINDOW_MIN", "15")
    plans = [_Plan("GOOG", "buy", qty=2.0)]
    state = _State(recent_trades={})
    session = _Session()
    out = _apply_whipsaw_guard(plans, state, session, set(), _now())
    assert out == plans


def test_mixed_plans_partial_block(monkeypatch):
    """One whipsaw + one legitimate = one dropped, one kept."""
    monkeypatch.setenv("AGENTIC_WHIPSAW_GUARD_WINDOW_MIN", "15")
    now = _now()
    plans = [
        _Plan("CRM", "sell", qty=3.0),   # opposite recent buy, no news -> drop
        _Plan("MSFT", "buy", qty=1.0),   # first trade, no recent -> keep
    ]
    state = _State(recent_trades={
        "CRM": ("buy", (now - timedelta(minutes=1)).isoformat()),
    })
    session = _Session()
    out = _apply_whipsaw_guard(plans, state, session, set(), now)
    assert [p.ticker for p in out] == ["MSFT"]


def test_unparseable_timestamp_fails_open(monkeypatch):
    """Bad ts in state = fail open (keep the plan). The guard is a
    safety filter, not a hard block; corrupt state shouldn't wedge
    the loop.
    """
    monkeypatch.setenv("AGENTIC_WHIPSAW_GUARD_WINDOW_MIN", "15")
    plans = [_Plan("AAPL", "sell", qty=1.0)]
    state = _State(recent_trades={"AAPL": ("buy", "not-a-date")})
    session = _Session()
    out = _apply_whipsaw_guard(plans, state, session, set(), _now())
    assert out == plans


def test_empty_plans_short_circuit(monkeypatch):
    monkeypatch.setenv("AGENTIC_WHIPSAW_GUARD_WINDOW_MIN", "15")
    out = _apply_whipsaw_guard(
        [], _State(), _Session(), set(), _now(),
    )
    assert out == []
