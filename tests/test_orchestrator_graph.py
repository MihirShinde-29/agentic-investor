"""Tests for the LangGraph orchestrator. Agents and LLM are mocked at seams."""

from agentic_investor.agents.news import NewsSignal
from agentic_investor.agents.technical import TechnicalSignal
from agentic_investor.orchestrator import graph as g
from agentic_investor.orchestrator.state import (
    Allocation,
    OrchestratorRequest,
    Position,
)


def _tech(ticker: str, stance: str = "bullish", conf: float = 0.7) -> TechnicalSignal:
    return TechnicalSignal(
        ticker=ticker, stance=stance, confidence=conf, reasoning="rr", key_drivers=[]
    )


def _news(ticker: str, stance: str = "bullish", conf: float = 0.6) -> NewsSignal:
    return NewsSignal(
        ticker=ticker, stance=stance, confidence=conf, reasoning="rr", citations=[]
    )


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
    monkeypatch.setattr(g, "analyze_ticker", lambda t: _tech(t.upper()))
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
    # 35% single position and 30% cash pass moderate but violate conservative
    # caps (20% single, 20% cash floor is met but the 35% single fails).
    _stub_agents_and_llm(monkeypatch, _fake_alloc(aapl=35, nvda=35, cash=30))
    req = OrchestratorRequest(tickers=["AAPL", "NVDA"], amount=10_000, risk="conservative")

    rec = g.run_orchestrator(req)

    assert any("AAPL" in v for v in rec.violations)


def test_gather_signals_skips_failing_agent(monkeypatch):
    monkeypatch.setattr(g, "analyze_ticker", lambda t: _tech(t.upper()))

    def _broken_news(_t):
        raise RuntimeError("provider down")

    monkeypatch.setattr(g, "analyze_news", _broken_news)

    out = g.gather_signals({"request": OrchestratorRequest(tickers=["AAPL"], amount=1000)})

    assert len(out["technical_signals"]) == 1
    assert out["news_signals"] == []


def test_allocator_prompt_includes_risk_caps_and_signals(monkeypatch):
    _stub_agents_and_llm(monkeypatch, _fake_alloc())
    state = {
        "request": OrchestratorRequest(tickers=["AAPL"], amount=1000, risk="conservative"),
        "technical_signals": [_tech("AAPL")],
        "news_signals": [_news("AAPL")],
    }
    msgs = g._messages(state)
    user = msgs[1]["content"]
    assert "conservative" in user
    assert "max single 20%" in user
    assert "cash floor 20%" in user
    assert "AAPL" in user
