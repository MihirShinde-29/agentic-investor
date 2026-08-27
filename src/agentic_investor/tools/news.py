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
import finnhub
from pydantic import BaseModel
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

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


# Finnhub

def _finnhub_client() -> finnhub.Client:
    s = get_settings()
    if not s.finnhub_api_key:
        raise RuntimeError("FINNHUB_API_KEY not set")
    return finnhub.Client(api_key=s.finnhub_api_key)


def _is_rate_limit(e: BaseException) -> bool:
    # Finnhub free tier: 60 req/min. Parallel picker runs blow this out easily.
    return (
        isinstance(e, finnhub.FinnhubAPIException) and getattr(e, "status_code", 0) == 429
    )


@retry(
    retry=retry_if_exception(_is_rate_limit),
    wait=wait_exponential(multiplier=1.0, min=1.0, max=30.0),
    stop=stop_after_attempt(5),
    reraise=True,
)
def _finnhub_company_news(ticker: str, frm: str, to: str) -> list[dict]:
    """Retry the raw Finnhub call with exponential backoff on 429s only."""
    return _finnhub_client().company_news(ticker.upper(), _from=frm, to=to)


@lru_cache(maxsize=512)
def _cached_company_news(ticker: str, frm: str, to: str) -> tuple[dict, ...]:
    """In-process cache keyed on (ticker, from, to). Same call re-uses result."""
    return tuple(_finnhub_company_news(ticker, frm, to))


def fetch_company_news(ticker: str, days: int = 7) -> list[NewsArticle]:
    """Pull recent company news for a ticker from Finnhub (retries on 429)."""
    to = datetime.now(UTC).date()
    frm = to - timedelta(days=days)
    raw = list(_cached_company_news(ticker.upper(), frm.isoformat(), to.isoformat()))

    out: list[NewsArticle] = []
    for item in raw:
        if not item.get("headline"):
            continue
        ts = item.get("datetime")
        published_at = (
            datetime.fromtimestamp(ts, tz=UTC).isoformat() if ts else ""
        )
        out.append(
            NewsArticle(
                id=str(item.get("id") or f"{ticker.upper()}-{ts or ''}"),
                ticker=ticker.upper(),
                headline=item["headline"],
                summary=(item.get("summary") or "").strip(),
                source=(item.get("source") or "").strip(),
                url=(item.get("url") or "").strip(),
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
