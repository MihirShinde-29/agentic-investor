"""News tool: fetch company news, embed locally, store in Chroma, retrieve by similarity.

No LLM calls here. Embeddings come from a local sentence-transformers model.
The vector store is Chroma, persistent under settings.chroma_dir. The heavy
embedding model is imported lazily so tests that mock the embedder never load it.
"""

import logging
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from functools import lru_cache

import chromadb
from pydantic import BaseModel

from agentic_investor.config import get_settings

logger = logging.getLogger(__name__)


class NewsArticle(BaseModel):
    id: str
    ticker: str
    headline: str
    summary: str
    source: str
    url: str
    published_at: str  # ISO 8601 UTC


# Alpaca News (unified with paper trading account; no separate key needed)

@lru_cache(maxsize=1)
def _alpaca_news_client():
    from alpaca.data.historical.news import NewsClient

    s = get_settings()
    if not s.alpaca_api_key or not s.alpaca_api_secret:
        raise RuntimeError(
            "ALPACA_API_KEY/SECRET must be set for news "
            "(get free paper keys at alpaca.markets)"
        )
    return NewsClient(api_key=s.alpaca_api_key, secret_key=s.alpaca_api_secret)


@lru_cache(maxsize=512)
def _cached_alpaca_news(ticker: str, frm_iso: str, to_iso: str) -> tuple[dict, ...]:
    """In-process cache keyed on (ticker, from, to). Same call reuses result."""
    from alpaca.data.requests import NewsRequest

    client = _alpaca_news_client()
    # alpaca-py's NewsRequest.symbols is a comma-separated string, not a list.
    req = NewsRequest(
        symbols=ticker.upper(),
        start=datetime.fromisoformat(frm_iso),
        end=datetime.fromisoformat(to_iso),
        limit=50,
    )
    resp = client.get_news(req)
    # alpaca-py returns a NewsSet; .data is dict[symbol, list[News]]
    articles = []
    for arts in resp.data.values():
        for a in arts:
            articles.append({
                "id": str(getattr(a, "id", "")),
                "headline": str(getattr(a, "headline", "")),
                "summary": str(getattr(a, "summary", "")),
                "source": str(getattr(a, "source", "")),
                "url": str(getattr(a, "url", "")),
                "created_at": getattr(a, "created_at", None),
            })
    return tuple(articles)


def fetch_company_news(ticker: str, days: int = 7) -> list[NewsArticle]:
    """Pull recent company news for a ticker from Alpaca (Benzinga feed)."""
    now = datetime.now(UTC)
    frm = now - timedelta(days=days)
    raw = list(_cached_alpaca_news(ticker.upper(), frm.isoformat(), now.isoformat()))

    out: list[NewsArticle] = []
    for item in raw:
        headline = item.get("headline", "")
        if not headline:
            continue
        ts = item.get("created_at")
        published_at = str(ts) if ts else now.isoformat()
        out.append(
            NewsArticle(
                id=item.get("id") or f"{ticker.upper()}-{published_at}",
                ticker=ticker.upper(),
                headline=headline,
                summary=item.get("summary", "").strip(),
                source=item.get("source", "").strip(),
                url=item.get("url", "").strip(),
                published_at=published_at,
            )
        )
    return out


# Embeddings (local sentence-transformers; import kept lazy so tests skip it)

@lru_cache(maxsize=1)
def _get_embed_model():
    from sentence_transformers import SentenceTransformer

    s = get_settings()
    return SentenceTransformer(s.embedding_model)


def _embed_text(texts: list[str]) -> list[list[float]]:
    model = _get_embed_model()
    vecs = model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
    return vecs.tolist()


# Vector store (Chroma)

@lru_cache(maxsize=1)
def _get_client():
    s = get_settings()
    return chromadb.PersistentClient(path=s.chroma_dir)


def get_collection(name: str = "company_news"):
    # Cosine is the canonical distance for normalized sentence embeddings.
    return _get_client().get_or_create_collection(
        name=name, metadata={"hnsw:space": "cosine"}
    )


def upsert_news_articles(
    articles: list[NewsArticle],
    *,
    collection=None,
    embedder: Callable[[list[str]], list[list[float]]] = _embed_text,
) -> int:
    """Store articles in the vector store. Idempotent by id."""
    if not articles:
        return 0
    coll = collection if collection is not None else get_collection()
    docs = [f"{a.headline}\n{a.summary}".strip() for a in articles]
    coll.upsert(
        ids=[a.id for a in articles],
        embeddings=embedder(docs),
        documents=docs,
        metadatas=[a.model_dump() for a in articles],
    )
    return len(articles)


def retrieve_news(
    ticker: str,
    query: str,
    k: int = 5,
    *,
    collection=None,
    embedder: Callable[[list[str]], list[list[float]]] = _embed_text,
) -> list[NewsArticle]:
    """Top-k articles for ticker, ranked by semantic similarity to query."""
    coll = collection if collection is not None else get_collection()
    res = coll.query(
        query_embeddings=embedder([query]),
        n_results=k,
        where={"ticker": ticker.upper()},
    )
    metas = (res.get("metadatas") or [[]])[0]
    return [NewsArticle(**m) for m in metas]
