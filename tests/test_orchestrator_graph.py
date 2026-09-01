"""Tests for the LangGraph orchestrator. Snapshot fetch, agents, and LLM are mocked."""

from agentic_investor.agents.news import NewsSignal
from agentic_investor.agents.technical import TechnicalSignal
from agentic_investor.orchestrator import graph as g
from agentic_investor.orchestrator.state import (
    Allocation,
    OrchestratorRequest,
    Position,
)
from agentic_investor.tools.market import MarketSnapshot


def _tech(ticker: str, stance: str = "bullish", conf: float = 0.7) -> TechnicalSignal:
    return TechnicalSignal(
        ticker=ticker, stance=stance, confidence=conf, reasoning="rr", key_drivers=[]
    )


def _news(ticker: str, stance: str = "bullish", conf: float = 0.6) -> NewsSignal:
    return NewsSignal(
        ticker=ticker, stance=stance, confidence=conf, reasoning="rr", citations=[]
    )


def _snap(ticker: str) -> MarketSnapshot:
    return MarketSnapshot(ticker=ticker, as_of="2026-08-27", close=100.0, atr_pct=2.0)


def _fake_alloc(*, aapl: float = 35, nvda: float = 35, cash: float = 30) -> Allocation:
    return Allocation(
        positions=[
            Position(ticker="AAPL", weight_pct=aapl, dollars=100 * aapl, rationale="x"),
            Position(ticker="NVDA", weight_pct=nvda, dollars=100 * nvda, rationale="x"),
        ],
        cash_pct=cash,
        cash_dollars=100 * cash,
        portfolio_rationale="x",
    )


def _stub_agents_and_llm(monkeypatch, alloc: Allocation):
    # gather_signals now uses get_market_snapshot + analyze_technical instead of analyze_ticker.
    monkeypatch.setattr(g, "get_market_snapshot", lambda t, period="1y": _snap(t.upper()))
    monkeypatch.setattr(g, "analyze_technical", lambda snap: _tech(snap.ticker))
    monkeypatch.setattr(g, "analyze_news", lambda t: _news(t.upper()))
    monkeypatch.setattr(g, "structured_complete", lambda *a, **k: alloc)


def test_run_orchestrator_end_to_end(monkeypatch):
    _stub_agents_and_llm(monkeypatch, _fake_alloc())
    req = OrchestratorRequest(tickers=["AAPL", "NVDA"], amount=10_000, risk="moderate")

    rec = g.run_orchestrator(req)

    assert rec.request == req
    assert rec.allocation.cash_pct == 30
    assert rec.violations == []
    assert {s.ticker for s in rec.technical_signals} == {"AAPL", "NVDA"}
    assert {s.ticker for s in rec.news_signals} == {"AAPL", "NVDA"}


def test_run_orchestrator_flags_conservative_violation(monkeypatch):
    # 35% single position violates conservative's 20% cap. Override the
    # preset allocator to "llm" so the fake bad allocation flows through
    # validate (the real inverse_vol allocator would satisfy caps naturally).
    from agentic_investor.orchestrator.strategy import apply_overrides, get_preset

    _stub_agents_and_llm(monkeypatch, _fake_alloc(aapl=35, nvda=35, cash=30))
    req = OrchestratorRequest(tickers=["AAPL", "NVDA"], amount=10_000, risk="conservative")
    profile = apply_overrides(get_preset("conservative"), allocator="llm")

    rec = g.run_orchestrator(req, profile=profile)

    assert any("AAPL" in v for v in rec.violations)


def test_gather_signals_skips_failing_agent(monkeypatch):
    monkeypatch.setattr(g, "get_market_snapshot", lambda t, period="1y": _snap(t.upper()))
    monkeypatch.setattr(g, "analyze_technical", lambda snap: _tech(snap.ticker))

    def _broken_news(_t):
        raise RuntimeError("provider down")

    monkeypatch.setattr(g, "analyze_news", _broken_news)

    out = g.gather_signals({"request": OrchestratorRequest(tickers=["AAPL"], amount=1000)})

    assert len(out["technical_signals"]) == 1
    assert out["news_signals"] == []
    # New: gather_signals also emits snapshots.
    assert "AAPL" in out["market_snapshots"]


def test_allocator_prompt_includes_profile_caps_and_signals(monkeypatch):
    _stub_agents_and_llm(monkeypatch, _fake_alloc())
    state = {
        "request": OrchestratorRequest(tickers=["AAPL"], amount=1000, risk="conservative"),
        "technical_signals": [_tech("AAPL")],
        "news_signals": [_news("AAPL")],
    }
    msgs = g._messages(state)
    user = "".join(b["text"] for b in msgs[1]["content"])
    assert "conservative" in user
    assert "20%" in user  # conservative caps: max_single 20%, cash floor 20%
    assert "AAPL" in user
