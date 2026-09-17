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


_ALPACA_NEWS_MAX_RETRIES = 3
_ALPACA_NEWS_RETRY_BACKOFF_SEC = 0.5


@lru_cache(maxsize=512)
def _cached_alpaca_news(ticker: str, frm_iso: str, to_iso: str) -> tuple[dict, ...]:
    """In-process cache keyed on (ticker, from, to). Same call reuses result.

    Retry loop covers the OSError: [Errno 22] "Invalid argument" that
    Windows raises when a socket recv gets torn down mid-read under
    concurrent httpx load (multiple tickers fetching Alpaca News in
    parallel from the graph's ThreadPoolExecutor). Rare per-call but
    frequent enough at 20-ticker batches to hurt signal coverage.
    """
    import time

    from alpaca.data.requests import NewsRequest

    client = _alpaca_news_client()
    # alpaca-py's NewsRequest.symbols is a comma-separated string, not a list.
    req = NewsRequest(
        symbols=ticker.upper(),
        start=datetime.fromisoformat(frm_iso),
        end=datetime.fromisoformat(to_iso),
        limit=50,
    )
    last_err: Exception | None = None
    for attempt in range(_ALPACA_NEWS_MAX_RETRIES):
        try:
            resp = client.get_news(req)
            break
        except OSError as e:
            # Errno 22 (EINVAL) on Windows is the "recv on torn-down
            # socket" pattern - retry with backoff. Other OSErrors
            # (auth, DNS, connection refused) also transient enough.
            last_err = e
            if attempt < _ALPACA_NEWS_MAX_RETRIES - 1:
                time.sleep(_ALPACA_NEWS_RETRY_BACKOFF_SEC * (2 ** attempt))
                continue
            raise
        except Exception as e:  # noqa: BLE001 - upstream lib may wrap OSError
            # Some alpaca-py versions wrap the httpx exception; retry a
            # small subset of message patterns that historically mask
            # the underlying socket issue.
            msg = str(e)
            if "Errno 22" in msg or "Invalid argument" in msg:
                last_err = e
                if attempt < _ALPACA_NEWS_MAX_RETRIES - 1:
                    time.sleep(_ALPACA_NEWS_RETRY_BACKOFF_SEC * (2 ** attempt))
                    continue
            raise
    else:
        # Loop exhausted without break; re-raise the last error.
        raise last_err if last_err else RuntimeError("alpaca news fetch failed")
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
    # Route through the shared ml-service if AGENTIC_ML_SERVICE_URL is set;
    # fall back to loading the local sentence-transformer model in-process.
    # See services/ml_service.py + tools/ml_client.py for the service side.
    from agentic_investor.tools import ml_client

    service_out = ml_client.embed_texts(texts)
    if service_out is not None:
        return service_out
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
