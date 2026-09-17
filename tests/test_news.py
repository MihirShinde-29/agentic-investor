"""Unit tests for the news tool (sqlite-vec backed).

No network (alpaca news is mocked), no real embedder (fake maps keywords
to distinct 384-dim vectors), store is an in-memory sqlite-vec conn so
nothing hits disk.
"""

from agentic_investor.tools import news
from agentic_investor.tools.news import NewsArticle
from agentic_investor.tools.news_store import EMBED_DIM, open_memory_conn


def _fake_embed(texts: list[str]) -> list[list[float]]:
    """Each topic gets its own axis (dim 0/1/2, others zero) so cosine /
    L2 similarity is deterministic against the query embeddings.
    Pads to EMBED_DIM = 384 to match the vec_news schema.
    """
    out: list[list[float]] = []
    for t in texts:
        low = t.lower()
        vec = [0.0] * EMBED_DIM
        if "earnings" in low:
            vec[0] = 1.0
        elif "lawsuit" in low:
            vec[1] = 1.0
        else:
            vec[2] = 1.0
        out.append(vec)
    return out


def _fresh_conn():
    return open_memory_conn()


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


def test_fetch_company_news_parses_alpaca_response(monkeypatch):
    # Fake the module-level cached function directly to bypass alpaca-py.
    monkeypatch.setattr(
        news, "_cached_alpaca_news",
        lambda ticker, frm, to: (
            {
                "id": "1",
                "headline": "NVDA beats estimates",
                "summary": "strong quarter",
                "source": "Reuters",
                "url": "https://r.com/1",
                "created_at": "2026-08-18T12:00:00+00:00",
            },
            {"headline": "", "created_at": "2026-08-18T12:01:00+00:00"},  # skipped
        ),
    )
    arts = news.fetch_company_news("nvda", days=3)
    assert len(arts) == 1
    assert arts[0].ticker == "NVDA"
    assert arts[0].headline.startswith("NVDA beats")
    assert arts[0].url == "https://r.com/1"


def test_upsert_and_retrieve_ranks_by_semantic_similarity():
    conn = _fresh_conn()
    articles = [
        _sample("AAPL", "a1", "Apple earnings crushed it"),
        _sample("AAPL", "a2", "Apple faces new lawsuit"),
        _sample("AAPL", "a3", "Apple ships new iPhone"),
    ]
    news.upsert_news_articles(articles, conn=conn, embedder=_fake_embed)

    top = news.retrieve_news(
        "AAPL", "quarterly earnings", k=1, conn=conn, embedder=_fake_embed,
    )
    assert top[0].id == "a1"


def test_retrieve_filters_by_ticker():
    conn = _fresh_conn()
    news.upsert_news_articles(
        [
            _sample("AAPL", "a1", "Apple earnings"),
            _sample("MSFT", "m1", "Microsoft earnings"),
        ],
        conn=conn,
        embedder=_fake_embed,
    )

    got = news.retrieve_news(
        "MSFT", "earnings", k=5, conn=conn, embedder=_fake_embed,
    )
    assert len(got) == 1
    assert got[0].ticker == "MSFT"


def test_upsert_is_idempotent_by_id():
    conn = _fresh_conn()
    art = _sample("AAPL", "same-id", "Apple earnings")
    news.upsert_news_articles([art], conn=conn, embedder=_fake_embed)
    news.upsert_news_articles([art], conn=conn, embedder=_fake_embed)

    got = news.retrieve_news(
        "AAPL", "earnings", k=5, conn=conn, embedder=_fake_embed,
    )
    assert len(got) == 1
    # Both news_articles + vec_news should have exactly one row.
    assert conn.execute("SELECT COUNT(*) FROM news_articles").fetchone()[0] == 1
    assert conn.execute("SELECT COUNT(*) FROM vec_news").fetchone()[0] == 1


def test_upsert_updates_fields_on_conflict():
    """A re-upsert with same id but different headline should overwrite."""
    conn = _fresh_conn()
    art = _sample("AAPL", "same", "Original headline")
    news.upsert_news_articles([art], conn=conn, embedder=_fake_embed)

    updated = _sample("AAPL", "same", "Updated headline")
    news.upsert_news_articles([updated], conn=conn, embedder=_fake_embed)

    row = conn.execute(
        "SELECT headline FROM news_articles WHERE id = ?", ("same",),
    ).fetchone()
    assert row[0] == "Updated headline"
