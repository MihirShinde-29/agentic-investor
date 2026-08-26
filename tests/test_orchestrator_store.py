"""Tests for the SQLite portfolio store. tmp_path isolates each test to its own DB."""

from agentic_investor.orchestrator.state import (
    Allocation,
    OrchestratorRequest,
    Position,
    Recommendation,
)
from agentic_investor.orchestrator.store import (
    list_recommendations,
    load_recommendation,
    save_recommendation,
)


def _url(tmp_path) -> str:
    return f"sqlite:///{tmp_path / 'test.db'}"


def _sample_recommendation() -> Recommendation:
    return Recommendation(
        request=OrchestratorRequest(
            tickers=["AAPL", "NVDA"], amount=10_000, risk="moderate"
        ),
        allocation=Allocation(
            positions=[
                Position(ticker="AAPL", weight_pct=45, dollars=4500, rationale="strong"),
                Position(ticker="NVDA", weight_pct=45, dollars=4500, rationale="momentum"),
            ],
            cash_pct=10,
            cash_dollars=1000,
            portfolio_rationale="balanced moderate",
        ),
    )


def test_save_and_load_round_trips(tmp_path):
    rec = _sample_recommendation()
    url = _url(tmp_path)
    rec_id = save_recommendation(rec, url=url)
    got = load_recommendation(rec_id, url=url)
    assert got == rec


def test_load_missing_returns_none(tmp_path):
    assert load_recommendation(999, url=_url(tmp_path)) is None


def test_list_returns_newest_first(tmp_path):
    url = _url(tmp_path)
    save_recommendation(_sample_recommendation(), url=url)
    save_recommendation(_sample_recommendation(), url=url)
    rows = list_recommendations(url=url)
    ids = [r[0] for r in rows]
    assert ids == sorted(ids, reverse=True)
    assert len(rows) == 2


def test_creates_parent_directory(tmp_path):
    url = f"sqlite:///{tmp_path / 'nested' / 'sub' / 'test.db'}"
    rec_id = save_recommendation(_sample_recommendation(), url=url)
    assert load_recommendation(rec_id, url=url) is not None
