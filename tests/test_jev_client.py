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


def test_materiality_uses_jev_per_headline_when_flag_on(monkeypatch):
    """Per-headline mode: every headline gets its own Jev call. High
    probability on any single headline -> batch material=True with
    confidence = the max prob across the batch. Uses a stateful
    _FakeClient that returns different noul values per call.
    """
    monkeypatch.setenv("AGENTIC_JEV_MATERIALITY_ENABLED", "1")
    monkeypatch.setenv("TYPESAFE_API_KEY", "fake-key")

    call_count = {"n": 0}
    probs = [0.85, 0.30]  # first headline material, second not

    class _StatefulClient:
        def system_one(self, *, state, questions):
            i = call_count["n"]
            call_count["n"] += 1
            noul = probs[i] if i < len(probs) else 0.5
            return _FakeResponse({"material": _FakeAnswer(noul=noul)})

    monkeypatch.setattr(jev_client, "_get_client", lambda: _StatefulClient())
    r = materiality_check(
        ["Fed hikes 25bp", "SPY reaches new high"], {"AAPL", "MSFT"},
    )
    assert r.from_jev is True
    assert r.material is True             # 0.85 flips it
    assert call_count["n"] == 2           # one call per headline
    assert 0.84 < r.confidence < 0.86     # max prob when material
    # per-headline breakdown carries the audit trail
    assert len(r.per_headline) == 2
    assert r.per_headline[0][1] == 0.85
    assert r.per_headline[0][2] is True
    assert r.per_headline[1][1] == 0.30
    assert r.per_headline[1][2] is False


def test_materiality_per_headline_any_one_material_flips_batch(monkeypatch):
    """The core motivation for per-headline mode: 1 good signal
    hiding in a batch of noise MUST flip the batch to material.
    Pre-fix batch mode would return False on such a batch.
    """
    monkeypatch.setenv("AGENTIC_JEV_MATERIALITY_ENABLED", "1")
    monkeypatch.setenv("TYPESAFE_API_KEY", "fake-key")

    probs = [0.05, 0.10, 0.15, 0.92, 0.08]  # single material headline

    class _StatefulClient:
        def __init__(self):
            self.i = 0
        def system_one(self, *, state, questions):
            noul = probs[self.i]
            self.i += 1
            return _FakeResponse({"material": _FakeAnswer(noul=noul)})

    monkeypatch.setattr(jev_client, "_get_client", lambda: _StatefulClient())
    r = materiality_check(
        ["a"*20, "b"*20, "c"*20, "d"*20, "e"*20], {"NVDA"},
    )
    assert r.material is True
    assert r.confidence == 0.92


def test_materiality_low_prob_all_returns_false(monkeypatch):
    """Every headline low-prob -> batch False. Confidence reports
    the distance-from-ambiguity of the max prob (so we know how
    confidently non-material it was)."""
    monkeypatch.setenv("AGENTIC_JEV_MATERIALITY_ENABLED", "1")
    monkeypatch.setenv("TYPESAFE_API_KEY", "fake-key")

    probs = [0.02, 0.08, 0.12]

    class _StatefulClient:
        def __init__(self):
            self.i = 0
        def system_one(self, *, state, questions):
            noul = probs[self.i]
            self.i += 1
            return _FakeResponse({"material": _FakeAnswer(noul=noul)})

    monkeypatch.setattr(jev_client, "_get_client", lambda: _StatefulClient())
    r = materiality_check(["bakery", "weather", "local"], {"AAPL"})
    assert r.from_jev is True
    assert r.material is False
    # max prob = 0.12, distance from 0.5 = 0.38, *2 = 0.76
    assert 0.75 < r.confidence < 0.77


def test_materiality_per_headline_fallback_on_single_call_error(monkeypatch):
    """If one Jev call in the loop errors, that headline falls back
    to deterministic ticker-mention while other headlines still get
    their Jev decision. Batch shouldn't abort entirely."""
    monkeypatch.setenv("AGENTIC_JEV_MATERIALITY_ENABLED", "1")
    monkeypatch.setenv("TYPESAFE_API_KEY", "fake-key")

    class _FlakeyClient:
        def __init__(self): self.i = 0
        def system_one(self, *, state, questions):
            self.i += 1
            if self.i == 2:
                raise RuntimeError("simulated Jev flake")
            return _FakeResponse({"material": _FakeAnswer(noul=0.30)})

    monkeypatch.setattr(jev_client, "_get_client", lambda: _FlakeyClient())
    r = materiality_check(
        ["headline A", "headline B mentions AAPL directly"], {"AAPL"},
    )
    # H1: Jev returned 0.30 -> not material
    # H2: Jev raised -> deterministic path -> mentioned=True -> material
    assert r.material is True
    assert r.from_jev is True  # overall still counts as Jev path


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
