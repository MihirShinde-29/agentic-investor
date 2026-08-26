"""Unit tests for the News-Sentiment agent. All I/O and LLM seams are mocked."""

from agentic_investor.agents import news as agent
from agentic_investor.agents.news import Citation, NewsSignal, _NewsSignalDraft
from agentic_investor.tools.news import NewsArticle


def _art(art_id: str, headline: str, url: str, source: str = "wire") -> NewsArticle:
    return NewsArticle(
        id=art_id,
        ticker="AAPL",
        headline=headline,
        summary="s",
        source=source,
        url=url,
        published_at="2026-08-18T12:00:00+00:00",
    )


def test_format_context_labels_articles_1_indexed():
    ctx = agent._format_context(
        [_art("1", "Earnings beat", "https://x/1"), _art("2", "Lawsuit filed", "https://x/2")]
    )
    assert "[1]" in ctx and "[2]" in ctx
    assert "Earnings beat" in ctx and "Lawsuit filed" in ctx


def test_format_context_handles_empty():
    assert "no recent articles" in agent._format_context([])


def test_resolve_citations_maps_ids_to_urls():
    articles = [
        _art("a", "Earnings", "https://x/1"),
        _art("b", "Lawsuit", "https://x/2"),
        _art("c", "iPhone", "https://x/3"),
    ]
    cites = agent._resolve_citations([1, 3], articles)
    assert [c.url for c in cites] == ["https://x/1", "https://x/3"]


def test_resolve_citations_ignores_out_of_range():
    articles = [_art("a", "Earnings", "https://x/1")]
    cites = agent._resolve_citations([0, 1, 99], articles)
    assert [c.url for c in cites] == ["https://x/1"]


def test_resolve_citations_dedupes_by_url():
    articles = [
        _art("a", "Earnings", "https://x/1"),
        _art("b", "Earnings again", "https://x/1"),
    ]
    cites = agent._resolve_citations([1, 2], articles)
    assert len(cites) == 1


def test_analyze_news_pipes_retrieval_into_llm_and_resolves_citations(monkeypatch):
    articles = [
        _art("a1", "Earnings crush", "https://x/1", "Reuters"),
        _art("a2", "Lawsuit filed", "https://x/2", "Bloomberg"),
        _art("a3", "New iPhone", "https://x/3", "Verge"),
    ]

    monkeypatch.setattr(agent, "fetch_company_news", lambda *a, **k: articles)
    monkeypatch.setattr(agent, "upsert_news_articles", lambda *a, **k: len(articles))
    monkeypatch.setattr(agent, "retrieve_news", lambda *a, **k: articles)

    draft = _NewsSignalDraft(
        stance="bullish", confidence=0.72, reasoning="Strong quarter.", citation_ids=[1]
    )
    monkeypatch.setattr(agent, "structured_complete", lambda *a, **k: draft)

    out = agent.analyze_news("aapl")
    assert isinstance(out, NewsSignal)
    assert out.ticker == "AAPL"  # ticker guardrail overrides any model echo
    assert out.stance == "bullish"
    assert out.confidence == 0.72
    assert out.citations == [
        Citation(headline="Earnings crush", url="https://x/1", source="Reuters")
    ]


def test_analyze_news_skips_refresh_when_disabled(monkeypatch):
    called = {"fetch": 0, "upsert": 0, "retrieve": 0}

    def _bump(name):
        def _f(*a, **k):
            called[name] += 1
            return []

        return _f

    monkeypatch.setattr(agent, "fetch_company_news", _bump("fetch"))
    monkeypatch.setattr(agent, "upsert_news_articles", _bump("upsert"))
    monkeypatch.setattr(agent, "retrieve_news", _bump("retrieve"))
    monkeypatch.setattr(
        agent,
        "structured_complete",
        lambda *a, **k: _NewsSignalDraft(
            stance="neutral", confidence=0.5, reasoning="", citation_ids=[]
        ),
    )

    agent.analyze_news("aapl", refresh=False)
    assert called["fetch"] == 0
    assert called["upsert"] == 0
    assert called["retrieve"] == 1
