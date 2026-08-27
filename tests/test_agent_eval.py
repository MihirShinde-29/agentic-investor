"""Tests for L2 agent-output eval runner. Uses an injected fake analyzer."""

from agentic_investor.agents.technical import TechnicalSignal
from agentic_investor.eval.agents import (
    AgentCase,
    _run_one_case,
    load_cases,
    run_agent_eval,
)
from agentic_investor.tools.market import MarketSnapshot


def _snap(ticker="AAPL"):
    return MarketSnapshot(ticker=ticker, as_of="2026-08-01", close=100.0)


def _case(stances, min_c=0.0, max_c=1.0):
    return AgentCase(
        id="t",
        description="test",
        snapshot=_snap(),
        acceptable_stances=stances,
        min_confidence=min_c,
        max_confidence=max_c,
    )


def _make_analyzer(stance="bullish", confidence=0.7, raise_after=None):
    calls = [0]

    def analyze(snap):
        calls[0] += 1
        if raise_after is not None and calls[0] > raise_after:
            raise RuntimeError("boom")
        return TechnicalSignal(
            ticker=snap.ticker, stance=stance, confidence=confidence, reasoning="r"
        )

    return analyze


def test_all_pass_when_stance_and_confidence_match():
    case = _case(["bullish"], 0.5, 0.9)
    analyze = _make_analyzer("bullish", 0.7)
    result = _run_one_case(case, n_samples=3, analyze=analyze)

    assert result.schema_validity == 1.0
    assert result.stance_pass_rate == 1.0
    assert result.confidence_in_range is True
    assert len(result.errors) == 0


def test_stance_pass_rate_when_stance_wrong():
    case = _case(["bearish"])
    analyze = _make_analyzer("bullish", 0.7)
    result = _run_one_case(case, n_samples=3, analyze=analyze)

    assert result.stance_pass_rate == 0.0
    assert result.schema_validity == 1.0  # schema still valid, stance just wrong


def test_confidence_out_of_range_flagged():
    case = _case(["bullish"], min_c=0.8, max_c=1.0)  # need high confidence
    analyze = _make_analyzer("bullish", 0.5)  # too low
    result = _run_one_case(case, n_samples=3, analyze=analyze)

    assert result.stance_pass_rate == 1.0
    assert result.confidence_in_range is False


def test_schema_validity_drops_when_analyzer_errors():
    case = _case(["bullish"])
    analyze = _make_analyzer("bullish", 0.7, raise_after=2)  # 2 valid, then raises
    result = _run_one_case(case, n_samples=5, analyze=analyze)

    assert result.schema_validity == 2 / 5
    assert len(result.errors) == 3
    # Stance pass rate is over the VALID returns only, not total.
    assert result.stance_pass_rate == 1.0


def test_load_cases_reads_bundled_golden_set():
    cases = load_cases()
    assert len(cases) >= 5
    ids = {c.id for c in cases}
    assert "clear-bullish-momentum" in ids
    assert "clear-bearish-downtrend" in ids


def test_run_agent_eval_end_to_end_with_fake():
    analyze = _make_analyzer("bullish", 0.7)
    report = run_agent_eval(n_samples=2, analyze=analyze)

    assert report.n_cases >= 5
    assert report.n_samples_per_case == 2
    # Fake always says bullish 0.7 - only cases with "bullish" in acceptable_stances pass.
    assert report.schema_validity == 1.0
    assert 0.0 < report.stance_pass_rate <= 1.0
    assert 0 <= report.fully_passing_cases <= report.n_cases
