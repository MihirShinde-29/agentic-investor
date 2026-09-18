"""News tool: fetch company news, embed locally, store in sqlite-vec, retrieve by similarity.

No LLM calls here. Embeddings come from a local sentence-transformers model
(or the shared paper-ml-service when AGENTIC_ML_SERVICE_URL is set). The
vector store is a sqlite-vec file at settings.news_store_path - see
`news_store.py` for the schema. The heavy embedding model is imported
lazily so tests that mock the embedder never load it.

Migrated from chromadb on 2026-09-17 (task #146). Motivation was the
same fragility class that took down the M17 index earlier the same day.
"""

import logging
import struct
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from functools import lru_cache

from pydantic import BaseModel

from agentic_investor.config import get_settings
from agentic_investor.tools.news_store import (
    EMBED_DIM,
    get_connection,
    get_stmt_lock,
)

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


# Vector store (sqlite-vec)


def _pack_vector(vec: list[float]) -> bytes:
    """Encode one float32 vector as the byte string vec0 wants."""
    if len(vec) != EMBED_DIM:
        raise ValueError(f"embedding dim {len(vec)} != expected {EMBED_DIM}")
    return struct.pack(f"{EMBED_DIM}f", *vec)


def upsert_news_articles(
    articles: list[NewsArticle],
    *,
    conn=None,
    embedder: Callable[[list[str]], list[list[float]]] = _embed_text,
) -> int:
    """Store articles in the vector store. Idempotent by article id.

    Writes to `news_articles` (metadata) + `vec_news` (embedding) in a
    single transaction. vec_news rowid tracks news_articles.internal_id.
    """
    if not articles:
        return 0
    connection = conn if conn is not None else get_connection()
    docs = [f"{a.headline}\n{a.summary}".strip() for a in articles]
    embeddings = embedder(docs)
    # Serialize the multi-statement upsert; the shared conn now runs
    # with check_same_thread=False (news_store._init_conn) so we need
    # to prevent interleaving from concurrent LangGraph workers.
    with get_stmt_lock(), connection:  # BEGIN / COMMIT
        for art, doc, emb in zip(articles, docs, embeddings, strict=True):
            _ = doc  # doc text is what we embed; not persisted separately
            connection.execute(
                """
                INSERT INTO news_articles (
                    id, ticker, headline, summary, source, url, published_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    ticker = excluded.ticker,
                    headline = excluded.headline,
                    summary = excluded.summary,
                    source = excluded.source,
                    url = excluded.url,
                    published_at = excluded.published_at
                """,
                (
                    art.id, art.ticker.upper(), art.headline,
                    art.summary, art.source, art.url, art.published_at,
                ),
            )
            row = connection.execute(
                "SELECT internal_id FROM news_articles WHERE id = ?", (art.id,),
            ).fetchone()
            internal_id = int(row[0])
            # vec0 has no REPLACE semantics for the vector column; delete-
            # then-insert is the documented pattern for idempotent upsert.
            connection.execute(
                "DELETE FROM vec_news WHERE rowid = ?", (internal_id,),
            )
            connection.execute(
                "INSERT INTO vec_news (rowid, embedding) VALUES (?, ?)",
                (internal_id, _pack_vector(emb)),
            )
    return len(articles)


def retrieve_news(
    ticker: str,
    query: str,
    k: int = 5,
    *,
    conn=None,
    embedder: Callable[[list[str]], list[list[float]]] = _embed_text,
) -> list[NewsArticle]:
    """Top-k articles for ticker, ranked by semantic similarity to query."""
    connection = conn if conn is not None else get_connection()
    query_embedding = _pack_vector(embedder([query])[0])
    # Over-fetch a bit and post-filter by ticker so we don't miss the
    # top-K when other tickers rank higher on similarity. vec0 doesn't
    # support pre-filtering by joined-table predicates efficiently.
    fetch_k = max(k * 4, 20)
    # vec0 MATCH is not thread-safe on a shared connection; serialise
    # so concurrent per-ticker news-agent workers don't step on each
    # other's virtual-table cursors.
    with get_stmt_lock():
        rows = connection.execute(
            """
            SELECT
                a.id, a.ticker, a.headline, a.summary, a.source, a.url,
                a.published_at, v.distance
            FROM vec_news v
            JOIN news_articles a ON a.internal_id = v.rowid
            WHERE v.embedding MATCH ?
              AND v.k = ?
              AND a.ticker = ?
            ORDER BY v.distance
            LIMIT ?
            """,
            (query_embedding, fetch_k, ticker.upper(), k),
        ).fetchall()
    return [
        NewsArticle(
            id=r[0], ticker=r[1], headline=r[2], summary=r[3] or "",
            source=r[4] or "", url=r[5] or "", published_at=r[6],
        )
        for r in rows
    ]
