"""Unit tests for the Technical Agent. The LLM seam is mocked, so no calls."""

import pytest
from pydantic import ValidationError

from agentic_investor.agents import technical
from agentic_investor.agents.technical import TechnicalSignal
from agentic_investor.tools.market import MarketSnapshot


def _snap(ticker="AAPL"):
    return MarketSnapshot(
        ticker=ticker, as_of="2026-08-13", close=100.0, rsi_14=65.0, above_sma_200=True
    )


def test_messages_carry_the_indicator_data():
    msgs = technical._messages(_snap())
    assert msgs[0]["role"] == "system"
    assert "AAPL" in msgs[1]["content"]
    assert "rsi_14" in msgs[1]["content"]


def test_analyze_overrides_ticker_from_snapshot(monkeypatch):
    fake = TechnicalSignal(ticker="WRONG", stance="bullish", confidence=0.7, reasoning="x")
    monkeypatch.setattr(technical, "structured_complete", lambda *a, **k: fake)

    out = technical.analyze_technical(_snap("MSFT"))
    assert out.ticker == "MSFT"
    assert out.stance == "bullish"


def test_confidence_is_bounded_0_to_1():
    with pytest.raises(ValidationError):
        TechnicalSignal(ticker="AAPL", stance="bullish", confidence=1.5, reasoning="x")


def test_stance_enum_is_enforced():
    with pytest.raises(ValidationError):
        TechnicalSignal(ticker="AAPL", stance="mooning", confidence=0.5, reasoning="x")
