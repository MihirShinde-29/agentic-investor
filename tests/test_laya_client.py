"""Tests for laya_client.materiality_check.

Mirror the jev_client test shape so behaviour parity is easy to
verify. Uses a fake Agent that lets us script the noul-per-headline
sequence without actually loading the Laya weights.
"""

from __future__ import annotations

import pytest

from agentic_investor.llm import laya_client
from agentic_investor.llm.laya_client import materiality_check


class _FakeAgent:
    """Stand-in for laya.Agent(). Yields scripted noul values for
    each system_one() call in order. Raises after the script is
    exhausted so an over-call bug shows up clearly in a failing test.
    """
    def __init__(self, probs: list[float]) -> None:
        self._probs = list(probs)
        self._i = 0

    def system_one(self, *, state, questions):  # noqa: ARG002
        if self._i >= len(self._probs):
            raise RuntimeError("_FakeAgent script exhausted")
        p = self._probs[self._i]
        self._i += 1
        return {
            "answers": {
                "material": {
                    "type": "noul", "noul": p, "confidence": p,
                },
            },
        }


@pytest.fixture(autouse=True)
def _reset_module_cache(monkeypatch):
    """Blank the module-level Agent cache before each test so the
    monkeypatched _get_agent doesn't inherit state from the last test."""
    monkeypatch.setattr(laya_client, "_AGENT_CACHE", None)
    monkeypatch.setattr(laya_client, "_AGENT_LOAD_FAILED", False)


def test_materiality_deterministic_when_flag_off(monkeypatch):
    """Flag off -> ticker-mention deterministic path, ms fields zero.
    Mirrors the parity commitment: an arm not on Laya degrades
    exactly like an arm not on Jev."""
    monkeypatch.setenv("AGENTIC_LAYA_MATERIALITY_ENABLED", "0")
    r = materiality_check(["headline mentions AAPL"], {"AAPL"})
    assert r.material is True
    assert r.from_laya is False
    assert r.laya_ms_total == 0.0
    assert r.laya_ms_max == 0.0


def test_materiality_deterministic_no_mention(monkeypatch):
    monkeypatch.setenv("AGENTIC_LAYA_MATERIALITY_ENABLED", "0")
    r = materiality_check(["unrelated news"], {"AAPL"})
    assert r.material is False
    assert r.from_laya is False


def test_materiality_uses_laya_per_headline_when_flag_on(monkeypatch):
    """Flag on, one headline scores > 0.5 -> batch material.
    Confidence is the max per-headline prob when material=True."""
    monkeypatch.setenv("AGENTIC_LAYA_MATERIALITY_ENABLED", "1")
    monkeypatch.setattr(laya_client, "_get_agent",
                        lambda: _FakeAgent([0.10, 0.85, 0.05]))
    r = materiality_check(["h1", "h2", "h3"], {"AAPL"})
    assert r.from_laya is True
    assert r.material is True
    assert r.confidence == pytest.approx(0.85, abs=0.01)
    # per_headline captures the (headline, prob, material) triples
    assert len(r.per_headline) == 3
    assert r.per_headline[1][2] is True  # h2 is material


def test_materiality_all_low_returns_false(monkeypatch):
    """All headlines low-prob -> batch False. Confidence reports
    the distance-from-ambiguity of the max prob."""
    monkeypatch.setenv("AGENTIC_LAYA_MATERIALITY_ENABLED", "1")
    monkeypatch.setattr(laya_client, "_get_agent",
                        lambda: _FakeAgent([0.05, 0.10, 0.15]))
    r = materiality_check(["a", "b", "c"], {"AAPL"})
    assert r.from_laya is True
    assert r.material is False
    # max prob 0.15, distance from 0.5 = 0.35, *2 = 0.7
    assert 0.69 < r.confidence < 0.71


def test_materiality_populates_ms_timing(monkeypatch):
    """jev_ms_total >= jev_ms_max invariant (same for laya)."""
    monkeypatch.setenv("AGENTIC_LAYA_MATERIALITY_ENABLED", "1")

    # Fake agent adds a small sleep per call so perf_counter records
    # non-zero deltas -- avoids the "0/0 ratio" case where forgetting
    # to record ms would ship green
    import time as _time

    class _SlowAgent(_FakeAgent):
        def system_one(self, *, state, questions):  # noqa: ARG002
            _time.sleep(0.003)
            return super().system_one(state=state, questions=questions)

    monkeypatch.setattr(laya_client, "_get_agent",
                        lambda: _SlowAgent([0.05, 0.10, 0.15]))
    r = materiality_check(["a", "b", "c"], {"AAPL"})
    assert r.laya_ms_max > 0.0
    assert r.laya_ms_total >= r.laya_ms_max
    assert r.laya_ms_total < 5000.0  # loose upper bound


def test_materiality_empty_headlines_short_circuits(monkeypatch):
    """No headlines -> no laya calls."""
    monkeypatch.setenv("AGENTIC_LAYA_MATERIALITY_ENABLED", "1")
    called: list[int] = []

    class _WatchAgent:
        def system_one(self, **_kw):
            called.append(1)
            return {"answers": {"material": {"noul": 0.9}}}

    monkeypatch.setattr(laya_client, "_get_agent", lambda: _WatchAgent())
    r = materiality_check([], {"AAPL"})
    assert r.material is False
    assert called == []


def test_materiality_per_headline_fallback_on_call_error(monkeypatch):
    """One system_one failure falls back deterministically for that
    headline only; other headlines still get Laya's decision."""
    monkeypatch.setenv("AGENTIC_LAYA_MATERIALITY_ENABLED", "1")

    class _FlakeyAgent:
        def __init__(self): self.i = 0
        def system_one(self, **_kw):
            self.i += 1
            if self.i == 2:
                raise RuntimeError("simulated Laya flake")
            return {"answers": {"material": {"noul": 0.30}}}

    monkeypatch.setattr(laya_client, "_get_agent", lambda: _FlakeyAgent())
    r = materiality_check(
        ["headline A", "headline B mentions AAPL directly"], {"AAPL"},
    )
    # H1: Laya returned 0.30 -> not material
    # H2: Laya raised -> deterministic path -> mentioned=True -> material
    assert r.material is True
    assert r.from_laya is True  # overall still counts as Laya path


def test_materiality_falls_back_when_agent_load_fails(monkeypatch):
    """No laya SDK / model load failure -> deterministic path, doesn't
    raise. Same failure semantics as Jev so an arm's behaviour on
    outage matches whichever gate it runs."""
    monkeypatch.setenv("AGENTIC_LAYA_MATERIALITY_ENABLED", "1")
    monkeypatch.setattr(laya_client, "_get_agent", lambda: None)
    r = materiality_check(["headline mentions AAPL"], {"AAPL"})
    assert r.material is True   # deterministic fallback fired
    assert r.from_laya is False
