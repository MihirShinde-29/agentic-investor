"""Tests for the Jev wrapper.

Two axes: (a) flag off vs on, (b) Jev call succeeds vs fails.
Both public functions must always return a valid result — never raise —
and must fall back to the deterministic path whenever Jev isn't
available (flag off, SDK unimportable, TYPESAFE_API_KEY unset,
client init raises, or API call raises).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from agentic_investor.llm import jev_client
from agentic_investor.llm.jev_client import (
    VERDICT_LABELS,
    MaterialityResult,
    VerdictResult,
    _deterministic_materiality,
    _deterministic_verdict,
    materiality_check,
    verdict_for_trade,
)


@pytest.fixture(autouse=True)
def _reset_client():
    jev_client._reset_client_for_tests()
    yield
    jev_client._reset_client_for_tests()


# --- deterministic verdict ------------------------------------------------

def test_deterministic_too_early_when_recent():
    r = _deterministic_verdict(
        {"side": "buy", "age_min": 10}, [{"minutes": 10, "pnl_pct": 0.3}],
    )
    assert r.label == "TOO_EARLY"
    assert r.from_jev is False


def test_deterministic_working_when_buy_up_after_60m():
    r = _deterministic_verdict(
        {"side": "buy", "age_min": 65},
        [{"minutes": 30, "pnl_pct": 0.4}, {"minutes": 60, "pnl_pct": 1.2}],
    )
    assert r.label == "WORKING"


def test_deterministic_wrong_when_buy_down_after_60m():
    r = _deterministic_verdict(
        {"side": "buy", "age_min": 65},
        [{"minutes": 60, "pnl_pct": -0.9}],
    )
    assert r.label == "WRONG"


def test_deterministic_unclear_when_chop():
    r = _deterministic_verdict(
        {"side": "buy", "age_min": 65},
        [{"minutes": 60, "pnl_pct": 0.2}],
    )
    assert r.label == "UNCLEAR"


def test_deterministic_sell_inverts_sign():
    """A SELL that's followed by a price DROP is WORKING (we exited
    into a fall). A SELL followed by a rally is WRONG (sold too early).
    """
    working = _deterministic_verdict(
        {"side": "sell", "age_min": 65},
        [{"minutes": 60, "pnl_pct": -0.9}],
    )
    assert working.label == "WORKING"
    wrong = _deterministic_verdict(
        {"side": "sell", "age_min": 65},
        [{"minutes": 60, "pnl_pct": 1.3}],
    )
    assert wrong.label == "WRONG"


def test_deterministic_reversal_after_60m():
    """Trajectory goes green then turns red by the 60m point. The
    60m sample is what dominates - matches how humans judge.
    """
    r = _deterministic_verdict(
        {"side": "buy", "age_min": 70},
        [{"minutes": 15, "pnl_pct": 1.5}, {"minutes": 60, "pnl_pct": -0.7}],
    )
    assert r.label == "WRONG"


def test_deterministic_empty_trajectory_is_unclear():
    r = _deterministic_verdict({"side": "buy", "age_min": 65}, [])
    assert r.label == "UNCLEAR"


# --- flag-gated dispatch --------------------------------------------------

def test_verdict_uses_deterministic_when_flag_off(monkeypatch):
    monkeypatch.setenv("AGENTIC_JEV_VERDICT_ENABLED", "0")
    r = verdict_for_trade(
        {"side": "buy", "age_min": 65},
        [{"minutes": 60, "pnl_pct": 1.0}],
    )
    assert r.from_jev is False
    assert r.label == "WORKING"


def test_materiality_uses_deterministic_when_flag_off(monkeypatch):
    monkeypatch.setenv("AGENTIC_JEV_MATERIALITY_ENABLED", "0")
    r = materiality_check(["NVDA beats earnings"], {"NVDA"})
    assert r.from_jev is False
    assert r.material is True


def test_materiality_deterministic_no_match():
    r = _deterministic_materiality(["Weather update"], {"AAPL"})
    assert r.material is False


# --- Jev happy path (mocked SDK) ------------------------------------------

class _FakeAnswer:
    def __init__(self, choice=None, noul=None, probability=0.9):
        self.choice = choice
        self.noul = noul
        self.probability = probability


class _FakeResponse:
    def __init__(self, answers: dict):
        self.answers = answers


class _FakeClient:
    def __init__(self, answers: dict):
        self._answers = answers
        self.calls: list[tuple[str, dict]] = []

    def system_one(self, *, state: str, questions: dict) -> _FakeResponse:
        self.calls.append((state, questions))
        return _FakeResponse(self._answers)


def test_verdict_uses_jev_when_flag_on_and_client_returns(monkeypatch):
    monkeypatch.setenv("AGENTIC_JEV_VERDICT_ENABLED", "1")
    monkeypatch.setenv("TYPESAFE_API_KEY", "fake-key")
    monkeypatch.setattr(
        jev_client, "_get_client",
        lambda: _FakeClient({"verdict": _FakeAnswer(choice="WORKING",
                                                      probability=0.83)}),
    )
    r = verdict_for_trade(
        {"side": "buy", "age_min": 65},
        [{"minutes": 60, "pnl_pct": 0.2}],  # deterministic would say UNCLEAR
    )
    assert r.from_jev is True
    assert r.label == "WORKING"
    assert 0.82 < r.confidence < 0.84


def test_verdict_unknown_label_falls_back(monkeypatch):
    """A Jev response with an out-of-vocabulary label must not
    corrupt the downstream prompt block."""
    monkeypatch.setenv("AGENTIC_JEV_VERDICT_ENABLED", "1")
    monkeypatch.setenv("TYPESAFE_API_KEY", "fake-key")
    monkeypatch.setattr(
        jev_client, "_get_client",
        lambda: _FakeClient({"verdict": _FakeAnswer(choice="WEIRD_LABEL")}),
    )
    r = verdict_for_trade(
        {"side": "buy", "age_min": 65},
        [{"minutes": 60, "pnl_pct": 1.0}],
    )
    assert r.from_jev is False
    assert r.label in VERDICT_LABELS


def test_verdict_jev_exception_falls_back(monkeypatch):
    class _BoomClient:
        def system_one(self, **_kw):
            raise RuntimeError("simulated jev outage")
    monkeypatch.setenv("AGENTIC_JEV_VERDICT_ENABLED", "1")
    monkeypatch.setenv("TYPESAFE_API_KEY", "fake-key")
    monkeypatch.setattr(jev_client, "_get_client", lambda: _BoomClient())
    r = verdict_for_trade(
        {"side": "buy", "age_min": 65},
        [{"minutes": 60, "pnl_pct": 1.0}],
    )
    assert r.from_jev is False
    assert r.label == "WORKING"


def test_materiality_uses_jev_when_flag_on(monkeypatch):
    monkeypatch.setenv("AGENTIC_JEV_MATERIALITY_ENABLED", "1")
    monkeypatch.setenv("TYPESAFE_API_KEY", "fake-key")
    monkeypatch.setattr(
        jev_client, "_get_client",
        lambda: _FakeClient(
            {"material": _FakeAnswer(noul=True, probability=0.71)},
        ),
    )
    r = materiality_check(
        ["Fed hikes 25bp", "SPY reaches new high"], {"AAPL", "MSFT"},
    )
    assert r.from_jev is True
    assert r.material is True
    assert 0.70 < r.confidence < 0.72


def test_materiality_empty_headlines_short_circuits(monkeypatch):
    """Never call Jev for an empty batch — the answer is always False.
    Preserves the free-tier request quota on quiet-market windows."""
    monkeypatch.setenv("AGENTIC_JEV_MATERIALITY_ENABLED", "1")
    monkeypatch.setenv("TYPESAFE_API_KEY", "fake-key")
    called = []
    class _WatchClient:
        def system_one(self, **_kw):
            called.append(1)
            return _FakeResponse({"material": _FakeAnswer(noul=True)})
    monkeypatch.setattr(jev_client, "_get_client", lambda: _WatchClient())
    r = materiality_check([], {"AAPL"})
    assert r.material is False
    assert called == []


def test_confidence_extractor_handles_scores_dict():
    """Older SDK responses returned `scores: dict` instead of a
    probability scalar. Extractor should pick the max score."""
    answer = SimpleNamespace(choice="WORKING", scores={
        "WORKING": 0.65, "WRONG": 0.20, "UNCLEAR": 0.10, "TOO_EARLY": 0.05,
    })
    assert 0.64 < jev_client._extract_confidence(answer) < 0.66


def test_confidence_extractor_returns_neutral_when_missing():
    answer = SimpleNamespace(choice="WORKING")
    assert jev_client._extract_confidence(answer) == 0.5


def test_get_client_returns_none_when_key_unset(monkeypatch):
    """Explicit auth guard — a missing key at process start should
    not lead to an SDK-side error surfaced at first call."""
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    jev_client._reset_client_for_tests()
    assert jev_client._get_client() is None


def test_return_types():
    r1 = _deterministic_verdict({"side": "buy", "age_min": 65}, [])
    assert isinstance(r1, VerdictResult)
    r2 = _deterministic_materiality([], set())
    assert isinstance(r2, MaterialityResult)
