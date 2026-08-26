"""Unit tests for the news tool.

No network (finnhub is mocked), no real embedder (fake maps keywords to distinct
vectors), and Chroma runs ephemeral (in-memory) so nothing hits the disk.
"""

import uuid

import chromadb

from agentic_investor.tools import news
from agentic_investor.tools.news import NewsArticle


def _fake_embed(texts: list[str]) -> list[list[float]]:
    # Each topic gets its own axis so cosine similarity is deterministic.
    out: list[list[float]] = []
    for t in texts:
        low = t.lower()
        if "earnings" in low:
            out.append([1.0, 0.0, 0.0])
        elif "lawsuit" in low:
            out.append([0.0, 1.0, 0.0])
        else:
            out.append([0.0, 0.0, 1.0])
    return out


def _fresh_collection():
    # EphemeralClient is a per-process singleton in Chroma 1.x, so a fixed
    # collection name leaks state between tests. Unique name = real isolation.
    return chromadb.EphemeralClient().get_or_create_collection(
        name=f"test_{uuid.uuid4().hex}", metadata={"hnsw:space": "cosine"}
    )


def _sample(ticker, art_id, headline, summary=""):
    return NewsArticle(
        id=art_id,
        ticker=ticker,
        headline=headline,
        summary=summary,
        source="wire",
        url="https://example.com",
        published_at="2026-08-18T12:00:00+00:00",
    )


def test_fetch_company_news_parses_finnhub_response(monkeypatch):
    class FakeClient:
        def company_news(self, symbol, _from, to):
            return [
                {
                    "id": 1,
                    "headline": "NVDA beats estimates",
                    "summary": "strong quarter",
                    "source": "Reuters",
                    "url": "https://r.com/1",
                    "datetime": 1_723_000_000,
                },
                {"headline": "", "datetime": 1_723_000_100},  # skipped: empty headline
            ]

    monkeypatch.setattr(news, "_finnhub_client", lambda: FakeClient())
    arts = news.fetch_company_news("nvda", days=3)
    assert len(arts) == 1
    assert arts[0].ticker == "NVDA"
    assert arts[0].headline.startswith("NVDA beats")
    assert arts[0].url == "https://r.com/1"


def test_upsert_and_retrieve_ranks_by_semantic_similarity():
    coll = _fresh_collection()
    articles = [
        _sample("AAPL", "a1", "Apple earnings crushed it"),
        _sample("AAPL", "a2", "Apple faces new lawsuit"),
        _sample("AAPL", "a3", "Apple ships new iPhone"),
    ]
    news.upsert_news_articles(articles, collection=coll, embedder=_fake_embed)

    top = news.retrieve_news(
        "AAPL", "quarterly earnings", k=1, collection=coll, embedder=_fake_embed
    )
    assert top[0].id == "a1"


def test_retrieve_filters_by_ticker():
    coll = _fresh_collection()
    news.upsert_news_articles(
        [
            _sample("AAPL", "a1", "Apple earnings"),
            _sample("MSFT", "m1", "Microsoft earnings"),
        ],
        collection=coll,
        embedder=_fake_embed,
    )

    got = news.retrieve_news(
        "MSFT", "earnings", k=5, collection=coll, embedder=_fake_embed
    )
    assert len(got) == 1
    assert got[0].ticker == "MSFT"


def test_upsert_is_idempotent_by_id():
    coll = _fresh_collection()
    art = _sample("AAPL", "same-id", "Apple earnings")
    news.upsert_news_articles([art], collection=coll, embedder=_fake_embed)
    news.upsert_news_articles([art], collection=coll, embedder=_fake_embed)

    got = news.retrieve_news(
        "AAPL", "earnings", k=5, collection=coll, embedder=_fake_embed
    )
    assert len(got) == 1
