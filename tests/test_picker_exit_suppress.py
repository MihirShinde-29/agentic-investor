"""Same-session exits are suppressed from the picker's on-deck offer.

Root cause of the ZS/TRV whipsaws on 2026-09-04: the picker's frozen
top-N kept offering back tickers the LLM had already exited that day.
This test locks in the filter that subtracts recent-exit tickers from
the pre_picked list before it's handed to the LLM.
"""

from __future__ import annotations

from datetime import UTC, datetime

from agentic_investor.orchestrator.loop import _same_session_exits
from agentic_investor.tools.paper_broker import PaperPosition


class _FakeBroker:
    def __init__(self, positions: list[PaperPosition]):
        self._positions = positions

    def get_positions(self):
        return list(self._positions)


def test_same_session_exits_excludes_held(monkeypatch):
    """A ticker recently sold BUT still held (partial sell) shouldn't be
    marked as an exit — we still own shares.
    """
    monkeypatch.setattr(
        "agentic_investor.tools.paper_store.recent_sold_tickers",
        lambda since_iso: ["AAPL", "NVDA"],
    )
    held = [PaperPosition(
        ticker="AAPL", qty=10.0, avg_entry_price=100.0,
        market_value=1000.0, unrealized_pl=0.0, unrealized_pl_pct=0.0,
    )]
    broker = _FakeBroker(held)
    exits = _same_session_exits(broker, since_minutes=60)
    assert "AAPL" not in exits  # still held
    assert "NVDA" in exits      # sold to zero


def test_same_session_exits_empty_when_no_recent_sells(monkeypatch):
    monkeypatch.setattr(
        "agentic_investor.tools.paper_store.recent_sold_tickers",
        lambda since_iso: [],
    )
    broker = _FakeBroker([])
    assert _same_session_exits(broker, since_minutes=60) == set()


def test_same_session_exits_uppercases(monkeypatch):
    """Broker returns lowercase; helper must normalize."""
    monkeypatch.setattr(
        "agentic_investor.tools.paper_store.recent_sold_tickers",
        lambda since_iso: ["trv"],
    )
    broker = _FakeBroker([])
    assert _same_session_exits(broker, since_minutes=60) == {"TRV"}


def test_same_session_exits_swallows_lookup_errors(monkeypatch):
    """If the paper_store query blows up, we degrade to 'no suppression'
    rather than blocking the whole regen path.
    """
    def _boom(since_iso):
        raise RuntimeError("db down")

    monkeypatch.setattr(
        "agentic_investor.tools.paper_store.recent_sold_tickers", _boom,
    )
    broker = _FakeBroker([])
    assert _same_session_exits(broker, since_minutes=60) == set()


def test_same_session_exits_window_is_passed_through(monkeypatch):
    captured: list[str] = []

    def _capture(since_iso):
        captured.append(since_iso)
        return []

    monkeypatch.setattr(
        "agentic_investor.tools.paper_store.recent_sold_tickers", _capture,
    )
    broker = _FakeBroker([])
    _same_session_exits(broker, since_minutes=30)
    assert captured, "recent_sold_tickers should be called with an ISO"
    since = datetime.fromisoformat(captured[0].replace("Z", "+00:00"))
    delta = (datetime.now(UTC) - since).total_seconds() / 60.0
    assert 29.0 <= delta <= 31.0
